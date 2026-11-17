"""Select development configurations and perform one matched holdout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from synthrep.config import ExperimentConfig, PreprocessConfig
from synthrep.holdout import evaluate_selected_on_holdout, select_best_configs


def _config_from_manifest(path: Path) -> tuple[ExperimentConfig, int | None]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw = manifest.get("config", {})
    raw_preprocessing = raw.get("preprocessor", {})
    preprocessing = PreprocessConfig(
        categorical_encoding=raw_preprocessing.get(
            "categorical_encoding", "target_order"
        ),
        numeric_impute_strategy=raw_preprocessing.get(
            "numeric_impute_strategy", "median"
        ),
        unknown_category_value=float(
            raw_preprocessing.get("unknown_category_value", 0.5)
        ),
        target_order_smoothing=float(
            raw_preprocessing.get("target_order_smoothing", 10.0)
        ),
        target_order_inner_folds=int(
            raw_preprocessing.get("target_order_inner_folds", 5)
        ),
        standardize=bool(raw_preprocessing.get("standardize", True)),
        random_state=int(
            raw_preprocessing.get("random_state", raw.get("random_state", 42))
        ),
    )
    config = ExperimentConfig(
        random_state=int(raw.get("random_state", 42)),
        holdout_size=float(raw.get("holdout_size", 0.20)),
        n_splits=int(raw.get("n_splits", 5)),
        n_repeats=int(raw.get("n_repeats", 10)),
        training_fractions=tuple(
            float(value)
            for value in raw.get("training_fractions", [0.1, 0.25, 0.5, 1.0])
        ),
        mix_amounts=tuple(float(value) for value in raw.get("mix_amounts", [1.0, 3.0])),
        synthetic_mode=raw.get("synthetic_mode", "class_conditional"),
        regularization_policy=raw.get("regularization_policy", "fixed"),
        positive_label=int(raw.get("positive_label", 1)),
        preprocessor=preprocessing,
    )
    return config, manifest.get("max_rows_per_dataset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    selected = select_best_configs(summary)
    config, max_rows = _config_from_manifest(args.manifest)
    outputs = evaluate_selected_on_holdout(
        data_dir=args.data_dir,
        selected_configs=selected,
        config=config,
        max_rows_per_dataset=max_rows,
        n_bootstrap=args.bootstrap_resamples,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "selected_development_configs.csv", index=False)
    outputs["comparisons"].to_csv(args.out_dir / "holdout_comparisons.csv", index=False)
    outputs["predictions"].to_csv(
        args.out_dir / "holdout_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    print(f"Wrote {args.out_dir / 'selected_development_configs.csv'}")
    print(f"Wrote {args.out_dir / 'holdout_comparisons.csv'}")
    print(f"Wrote {args.out_dir / 'holdout_predictions.csv.gz'}")


if __name__ == "__main__":
    main()
