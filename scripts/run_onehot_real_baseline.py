"""Matched real-only one-hot logistic regression control for Adult and Default Credit.

Evaluates only on repeated-CV development folds. Never fits or scores on the
reserved holdout. Uses the same split, subset, and model seeds as synthrep.

Examples (run from project root after `python -m pip install -e .`):
    python scripts/run_onehot_real_baseline.py --data-dir data --out-dir results/onehot-smoke --n-splits 2 --n-repeats 1 --training-fractions 1.0
    python scripts/run_onehot_real_baseline.py --data-dir data --out-dir results/onehot-real-only --n-splits 5 --n-repeats 10 --training-fractions 0.5 1.0
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from synthrep.data import file_sha256, load_dataset
from synthrep.experiment import (
    derive_seed,
    development_holdout_split,
    stratified_fraction_subset,
)
from synthrep.registry import make_model
from synthrep.stats import summarize_scores


def make_onehot_preprocessor(numeric_columns: tuple[str, ...], categorical_columns: tuple[str, ...]):
    """Impute and scale numerical features, and one-hot encode categoricals.

    The returned transformer must be fitted inside each training fold.
    """
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            list(numeric_columns),
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            list(categorical_columns),
        ))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def run_onehot_real_baseline(
    *, data_dir: Path, out_dir: Path, datasets: list[str],
    training_fractions: list[float], n_splits: int, n_repeats: int,
    random_state: int = 42, holdout_size: float = 0.2,
) -> pd.DataFrame:
    if not 0 < holdout_size < 1:
        raise ValueError("holdout-size must be between 0 and 1")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n-splits must be >= 2 and n-repeats must be >= 1")
    if any(not 0 < f <= 1 for f in training_fractions):
        raise ValueError("training-fractions must be in (0, 1]")
    if any(d not in ("adult", "default_credit") for d in datasets):
        raise ValueError("This focused one-hot baseline supports adult and default_credit only")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = {
        "purpose": "real_only_one_hot_baseline_development_cv",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "random_state": random_state,
            "holdout_size": holdout_size,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "training_fractions": training_fractions,
            "model": "logreg",
            "regularization_policy": "fixed",
            "logreg_c": 1.0,
            "categorical_encoding": "onehot",
            "numerical_imputation": "median",
            "numerical_scaling": "standard",
            "categorical_imputation": "most_frequent",
            "unknown_category_handling": "ignore",
        },
        "datasets": {},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "note": "Holdout is reserved and NOT evaluated. All transforms are trained within each CV fold.",
    }

    for dataset_name in datasets:
        bundle = load_dataset(dataset_name, data_dir)
        X_dev, X_holdout, y_dev, y_holdout = development_holdout_split(
            bundle, holdout_size=holdout_size, random_state=random_state
        )
        manifest["datasets"][bundle.name] = {
            "source_filename": bundle.source_path.name,
            "source_sha256": file_sha256(bundle.source_path),
            "n_rows_used": bundle.n_rows,
            "n_development": len(X_dev),
            "n_holdout_reserved": len(X_holdout),
            "positive_rate_development": float(y_dev.mean()),
            "positive_rate_holdout_reserved": float(y_holdout.mean()),
            "categorical_columns": list(bundle.categorical_columns),
            "numeric_columns": list(bundle.numeric_columns),
        }
        print(f"Dataset {bundle.name}: development={len(X_dev)}, holdout reserved={len(X_holdout)}")
        for fraction in training_fractions:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
            for split_id, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev)):
                X_available = X_dev.iloc[train_idx].reset_index(drop=True)
                y_available = y_dev.iloc[train_idx].reset_index(drop=True)
                X_validation = X_dev.iloc[val_idx].reset_index(drop=True)
                y_validation = y_dev.iloc[val_idx].reset_index(drop=True)
                subset_seed = derive_seed(random_state, bundle.name, "subset", split_id)
                X_train, y_train = stratified_fraction_subset(
                    X_available, y_available, fraction, random_state=subset_seed
                )
                model_seed = derive_seed(random_state, bundle.name, "logreg", "model", split_id)
                pipeline = Pipeline([
                    ("preprocessing", make_onehot_preprocessor(
                        bundle.numeric_columns, bundle.categorical_columns
                    )),
                    ("classifier", make_model("logreg", random_state=model_seed)),
                ])
                pipeline.fit(X_train, y_train)
                probabilities = pipeline.predict_proba(X_validation)[:, list(
                    pipeline.named_steps["classifier"].classes_
                ).index(bundle.positive_label)]
                metrics = {
                    "AUROC": float(roc_auc_score(y_validation, probabilities)),
                    "AUPRC": float(average_precision_score(y_validation, probabilities)),
                }
                base_row = {
                    "dataset": bundle.name,
                    "model": "logreg",
                    "generator": "none",
                    "synthetic_mode": "real_only",
                    "categorical_encoding": "onehot",
                    "training_fraction": fraction,
                    "split_id": split_id,
                    "repeat_id": split_id // n_splits,
                    "fold_id": split_id % n_splits,
                    "regime": "Real-only",
                    "n_train_available": len(X_available),
                    "n_train_real": len(X_train),
                    "n_train_regime": len(X_train),
                    "n_validation": len(X_validation),
                    "positive_rate_train_real": float(y_train.mean()),
                    "positive_rate_validation": float(y_validation.mean()),
                    "regularization_policy": "fixed",
                    "logreg_c": 1.0,
                    "status": "ok",
                }
                for metric, score in metrics.items():
                    records.append({**base_row, "metric": metric, "score": score})
                if split_id % n_splits == 0:
                    print(f"  fraction={fraction:g}, repeat={split_id // n_splits + 1}/{n_repeats}")

    fold_scores = pd.DataFrame.from_records(records)
    summary = summarize_scores(fold_scores)
    fold_scores.to_csv(out_dir / "fold_scores.csv", index=False)
    summary.to_csv(out_dir / "summary_scores.csv", index=False)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(fold_scores)} fold scores and {len(summary)} summary rows into {out_dir}")
    return fold_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/onehot-real-only"))
    parser.add_argument("--datasets", nargs="+", choices=["adult", "default_credit"], default=["adult", "default_credit"])
    parser.add_argument("--training-fractions", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-size", type=float, default=0.2)
    args = parser.parse_args()
    run_onehot_real_baseline(
        data_dir=args.data_dir, out_dir=args.out_dir, datasets=args.datasets,
        training_fractions=args.training_fractions,
        n_splits=args.n_splits, n_repeats=args.n_repeats,
        random_state=args.random_state, holdout_size=args.holdout_size,
    )


if __name__ == "__main__":
    main()
