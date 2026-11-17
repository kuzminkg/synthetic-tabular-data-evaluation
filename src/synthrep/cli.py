"""Command-line interface for rerunnable experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, PreprocessConfig
from .experiment import run_experiment_grid


def _float_tuple(values: list[str]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthrep",
        description="Run fold-contained synthetic tabular utility experiments.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["give_me_credit", "default_credit", "adult", "htru2"],
        help="Datasets: give_me_credit default_credit adult htru2",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],
        help="Models: logreg random_forest",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["bootstrap", "smote", "copula"],
        help="Generators: bootstrap smote copula",
    )
    parser.add_argument(
        "--training-fractions",
        nargs="+",
        default=["0.10", "0.25", "0.50", "1.00"],
    )
    parser.add_argument("--mix-amounts", nargs="+", default=["1.0", "3.0"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--holdout-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--synthetic-mode",
        choices=["class_conditional", "minority_only"],
        default="class_conditional",
    )
    parser.add_argument(
        "--regularization-policy",
        choices=["fixed", "adjusted"],
        default="fixed",
        help=(
            "Logistic regression regularization policy: "
            "'fixed' uses C=1 for all regimes; "
            "'adjusted' scales C by n_real / n_regime."
        ),
    )
    parser.add_argument(
        "--categorical-encoding",
        choices=["target_order", "ordinal", "none"],
        default="target_order",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable fold-level standardization before generation and modeling.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Stratified row limit for debugging only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching completed checkpoints in --out-dir after interruption.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test: small subsets, 2 folds, 1 repeat, one fraction.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    training_fractions = _float_tuple(args.training_fractions)
    mix_amounts = _float_tuple(args.mix_amounts)
    n_splits = args.n_splits
    n_repeats = args.n_repeats
    max_rows = args.max_rows_per_dataset

    if args.quick:
        training_fractions = (0.50,)
        mix_amounts = (1.0,)
        n_splits = 2
        n_repeats = 1
        max_rows = max_rows or 600

    preprocess_config = PreprocessConfig(
        categorical_encoding=args.categorical_encoding,
        standardize=not args.no_standardize,
        random_state=args.random_state,
    )
    config = ExperimentConfig(
        random_state=args.random_state,
        holdout_size=args.holdout_size,
        n_splits=n_splits,
        n_repeats=n_repeats,
        training_fractions=training_fractions,
        mix_amounts=mix_amounts,
        synthetic_mode=args.synthetic_mode,
        regularization_policy=args.regularization_policy,
        positive_label=1,
        preprocessor=preprocess_config,
    )

    outputs = run_experiment_grid(
        data_dir=args.data_dir,
        output_dir=args.out_dir,
        datasets=args.datasets,
        models=args.models,
        generators=args.generators,
        config=config,
        max_rows_per_dataset=max_rows,
        progress=not args.quiet,
        resume=args.resume,
    )
    print(f"Wrote fold scores: {args.out_dir / 'fold_scores.csv'}")
    print(f"Wrote summary: {args.out_dir / 'summary_scores.csv'}")
    print(f"Wrote corrected paired tests: {args.out_dir / 'paired_tests_vs_real.csv'}")
    print(
        f"Rows: fold={len(outputs['fold_scores'])}, summary={len(outputs['summary'])}"
    )


if __name__ == "__main__":
    main()
