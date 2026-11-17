"""Create draft LaTeX tables from summary and paired-test CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decimals", type=int, default=4)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    tests = pd.read_csv(args.tests)
    decimals = int(args.decimals)

    summary_columns = [
        "dataset",
        "model",
        "generator",
        "training_fraction",
        "regime",
        "metric",
        "mean",
        "std",
        "n",
    ]
    test_columns = [
        "dataset",
        "model",
        "generator",
        "training_fraction",
        "metric",
        "candidate_regime",
        "mean_delta",
        "ci_low",
        "ci_high",
        "p_value_holm",
    ]
    text = ["% Descriptive summary\n"]
    text.append(summary[summary_columns].round(decimals).to_latex(index=False, escape=True))
    text.append("\n\n% Corrected paired comparisons\n")
    text.append(tests[test_columns].round(decimals).to_latex(index=False, escape=True))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(text), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
