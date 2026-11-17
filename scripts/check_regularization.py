"""Diagnostic: does row duplication change logistic regression through C?

Uses only development data; never scores the project's reserved holdout.
Run from the repository root:
    python scripts/check_regularization.py --data-dir data
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from synthrep.data import load_dataset
from synthrep.experiment import development_holdout_split
from synthrep.models import make_estimator
from synthrep.preprocessing import FoldPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="htru2")
    parser.add_argument("--train-rows", type=int, default=2000)
    args = parser.parse_args()

    bundle = load_dataset(args.dataset, args.data_dir)
    X_dev, _X_holdout, y_dev, _y_holdout = development_holdout_split(
        bundle, holdout_size=0.20, random_state=42
    )
    X_train_raw, X_valid_raw, y_train_raw, y_valid = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=43
    )

    # A small real-data budget keeps this diagnostic quick. Sampling is performed
    # on the training side only; diagnostic validation stays real and unseen.
    n = min(args.train_rows, len(y_train_raw))
    if n <= 0:
        raise ValueError("--train-rows must be positive")
    if n < len(y_train_raw):
        X_train_raw, _, y_train_raw, _ = train_test_split(
            X_train_raw, y_train_raw, train_size=n,
            stratify=y_train_raw, random_state=44
        )

    prep = FoldPreprocessor(
        categorical_columns=bundle.categorical_columns,
        numeric_columns=bundle.numeric_columns,
    )
    X_train = prep.fit_transform(X_train_raw, y_train_raw)
    X_valid = prep.transform(X_valid_raw)
    y_train = y_train_raw.reset_index(drop=True)

    print(f"Dataset: {bundle.name}")
    print(f"Real training rows: {len(y_train)}; diagnostic validation rows: {len(y_valid)}")
    print("Reserved 20% holdout: not used for fitting or scoring")
    print("All runs use class_weight='balanced' and the same preprocessed rows.")
    print("For numerical precision, diagnostic fits use tol=1e-9.\n")

    baseline_prob = None
    for copies, C, label in [
        (1, 1.0, "Real-only"),
        (2, 1.0, "2 copies, C=1"),
        (2, 0.5, "2 copies, C=1/2"),
        (4, 1.0, "4 copies, C=1"),
        (4, 0.25, "4 copies, C=1/4"),
    ]:
        X_fit = pd.concat([X_train] * copies, ignore_index=True)
        y_fit = pd.concat([y_train] * copies, ignore_index=True)
        model = make_estimator("logreg", random_state=42)
        model.set_params(C=C, tol=1e-9)
        model.fit(X_fit, y_fit)
        pos_index = list(model.classes_).index(bundle.positive_label)
        probability = model.predict_proba(X_valid)[:, pos_index]
        if baseline_prob is None:
            baseline_prob = probability
        max_change = float(np.max(np.abs(probability - baseline_prob)))
        auroc = roc_auc_score(y_valid, probability)
        ap = average_precision_score(y_valid, probability)
        print(f"{label:<23} rows={len(y_fit):>5}  AUROC={auroc:.5f}  "
              f"AP={ap:.5f}  max |delta probability|={max_change:.8f}")

    print("\nExpected: adjusted-C duplicate fits closely match Real-only predictions.")
    print("Fixed-C duplicate fits may differ. This is a diagnostic, not a paper result.")


if __name__ == "__main__":
    main()
