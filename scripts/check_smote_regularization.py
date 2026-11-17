"""Diagnostic: compare fixed versus row-count-adjusted C in SMOTE augmentation.

No reserved holdout observations are fitted or scored.

Run from the repository root after `python -m pip install -e .`:
    python scripts/check_smote_regularization.py --data-dir data

This is a small single-split development diagnostic, NOT a publication experiment.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from synthrep.data import load_dataset
from synthrep.experiment import _make_regimes, development_holdout_split
from synthrep.preprocessing import FoldPreprocessor
from synthrep.registry import make_generator, make_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="give_me_credit")
    parser.add_argument("--train-rows", type=int, default=2000)
    args = parser.parse_args()
    if args.train_rows < 20:
        parser.error("--train-rows must be at least 20")

    bundle = load_dataset(args.dataset, args.data_dir)
    X_dev, _X_holdout, y_dev, _y_holdout = development_holdout_split(
        bundle, holdout_size=0.20, random_state=42
    )
    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=43
    )
    n_real = min(args.train_rows, len(y_train))
    if n_real < len(y_train):
        X_train_raw, _, y_train, _ = train_test_split(
            X_train_raw, y_train, train_size=n_real,
            stratify=y_train, random_state=44,
        )

    preprocessor = FoldPreprocessor(
        categorical_columns=bundle.categorical_columns,
        numeric_columns=bundle.numeric_columns,
    )
    X_train = preprocessor.fit_transform(X_train_raw, y_train)
    X_valid = preprocessor.transform(X_valid_raw)
    y_train = y_train.reset_index(drop=True).rename("target")
    y_valid = y_valid.reset_index(drop=True)

    generator, _ = make_generator("smote", random_state=45)
    generator.fit(X_train, y_train)
    regimes = _make_regimes(
        X_train, y_train, generator,
        mix_amounts=(1.0, 3.0), synthetic_mode="class_conditional",
    )

    def evaluate(X_fit: pd.DataFrame, y_fit: pd.Series, C: float) -> tuple[float, float]:
        model = make_model("logreg", random_state=42)
        model.set_params(C=C, tol=1e-9)
        model.fit(X_fit, y_fit)
        pos_idx = list(model.classes_).index(bundle.positive_label)
        proba = model.predict_proba(X_valid)[:, pos_idx]
        return float(roc_auc_score(y_valid, proba)), float(
            average_precision_score(y_valid, proba)
        )

    real_auroc, real_ap = evaluate(*regimes["Real-only"], C=1.0)
    print(f"Dataset: {bundle.name}")
    print(f"Real training rows: {len(X_train)}; development validation rows: {len(X_valid)}")
    print("Reserved 20% holdout: NOT used")
    print("Generator: class-conditional SMOTE-style interpolation")
    print("Classifier: logistic regression, class_weight='balanced', tol=1e-9")
    print("All scores use the same real validation observations.\n")
    print("Regime / C policy          rows      C       AUROC     AP        ΔAUROC     ΔAP")
    print("-" * 79)
    print(f"{'Real-only':<26} {len(X_train):5d}  {1.:6.3f}   {real_auroc:.5f}  {real_ap:.5f}   {0.:+.5f}   {0.:+.5f}")

    for name in ("Synthetic-only", "Mixed +1x", "Mixed +3x"):
        X_fit, y_fit = regimes[name]
        for policy, C in ([('fixed', 1.0)] if name == 'Synthetic-only'
                          else [('fixed', 1.0), ('adjusted', len(X_train) / len(X_fit))]):
            auroc, ap = evaluate(X_fit, y_fit, C)
            label = f"{name} ({policy})"
            print(f"{label:<26} {len(X_fit):5d}  {C:6.3f}   {auroc:.5f}  {ap:.5f}   {auroc-real_auroc:+.5f}   {ap-real_ap:+.5f}")

    print("\nThis is a diagnostic only: one development split, not an inference result.")
    print("Adjusted C controls the sample-count effect, not all effects of class reweighting or synthetic data.")


if __name__ == "__main__":
    main()
