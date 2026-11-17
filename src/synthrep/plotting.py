"""Small, data-driven plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(
    summary: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    generator: str,
    metric: str,
    output_path: str | Path,
) -> Path:
    """Plot mean validation performance against training fraction for one configuration."""

    subset = summary[
        (summary["dataset"] == dataset)
        & (summary["model"] == model)
        & (summary["generator"] == generator)
        & (summary["metric"] == metric)
    ].copy()
    if subset.empty:
        raise ValueError("No matching rows for the requested learning curve")

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for regime, group in subset.groupby("regime", sort=True):
        group = group.sort_values("training_fraction")
        axis.plot(group["training_fraction"], group["mean"], marker="o", label=regime)
    axis.set_xlabel("Training fraction")
    axis.set_ylabel(f"Mean {metric}")
    axis.set_title(f"{dataset}: {generator}, {model}")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_paired_effects(
    tests: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    metric: str,
    output_path: str | Path,
    max_rows: int = 20,
) -> Path:
    """Plot corrected paired deltas and confidence intervals for one dataset/model/metric."""

    subset = tests[
        (tests["dataset"] == dataset)
        & (tests["model"] == model)
        & (tests["metric"] == metric)
    ].copy()
    if subset.empty:
        raise ValueError("No matching rows for the requested effect plot")
    subset = subset.sort_values("mean_delta", ascending=False).head(max_rows).sort_values("mean_delta")
    subset["label"] = subset.apply(
        lambda row: (
            f"{row['generator']} | {row['candidate_regime']} | "
            f"fraction={row['training_fraction']:g}"
        ),
        axis=1,
    )

    y_positions = range(len(subset))
    lower = subset["mean_delta"] - subset["ci_low"]
    upper = subset["ci_high"] - subset["mean_delta"]
    figure, axis = plt.subplots(figsize=(8.0, max(4.0, 0.35 * len(subset) + 1.5)))
    axis.errorbar(
        subset["mean_delta"],
        list(y_positions),
        xerr=[lower, upper],
        fmt="o",
        capsize=3,
    )
    axis.axvline(0.0, linewidth=1.0)
    axis.set_yticks(list(y_positions), labels=subset["label"])
    axis.set_xlabel(f"Paired delta in {metric} (candidate - Real-only)")
    axis.set_title(f"{dataset}: corrected repeated-CV effects ({model})")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return target
