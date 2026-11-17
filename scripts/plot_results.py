"""Create data-driven learning-curve and paired-effect figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthrep.plotting import plot_learning_curve, plot_paired_effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    tests = pd.read_csv(args.tests)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for keys, _ in summary.groupby(["dataset", "model", "generator", "metric"]):
        dataset, model, generator, metric = keys
        target = args.out_dir / f"{dataset}_{model}_{generator}_{metric}_learning_curve.png"
        plot_learning_curve(
            summary,
            dataset=dataset,
            model=model,
            generator=generator,
            metric=metric,
            output_path=target,
        )

    for keys, _ in tests.groupby(["dataset", "model", "metric"]):
        dataset, model, metric = keys
        target = args.out_dir / f"{dataset}_{model}_{metric}_paired_effects.png"
        plot_paired_effects(
            tests,
            dataset=dataset,
            model=model,
            metric=metric,
            output_path=target,
        )


if __name__ == "__main__":
    main()
