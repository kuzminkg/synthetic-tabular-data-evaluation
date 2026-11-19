#!/usr/bin/env python3
"""Regenerate the paper figures from archived development scores; no model fits.

Run from the package root: python scripts/make_paper_figures.py
Requires numpy, pandas, scipy and matplotlib. All raster outputs have transparent
backgrounds at 600 dpi; PDFs preserve vector graphics and embedded TrueType fonts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

DATASETS = ("give_me_credit", "default_credit", "adult", "htru2")
DATASET_LABELS = {"give_me_credit": "Give Me Some Credit", "default_credit": "Default Credit", "adult": "Adult", "htru2": "HTRU2"}
MODELS = ("logreg", "random_forest")
MODEL_LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest"}
MODEL_COLORS = {"logreg": "#D55E00", "random_forest": "#0072B2"}
MODEL_MARKERS = {"logreg": "o", "random_forest": "s"}
GENERATORS = ("bootstrap", "smote", "copula_quantile")
GENERATOR_LABELS = {"bootstrap": "Bootstrap", "smote": "Interpolation", "copula_quantile": "Gaussian copula"}
REGIMES = ("Synthetic-only", "Mixed +1x", "Mixed +3x")
REGIME_LABELS = {"Synthetic-only": "Synthetic-only", "Mixed +1x": r"Mixed $+1\times$", "Mixed +3x": r"Mixed $+3\times$"}
REGIME_COLORS = {"Synthetic-only": "#0072B2", "Mixed +1x": "#009E73", "Mixed +3x": "#D55E00"}
REGIME_MARKERS = {"Synthetic-only": "o", "Mixed +1x": "s", "Mixed +3x": "^"}
KEYS = ["dataset", "model", "generator", "synthetic_mode", "categorical_encoding", "training_fraction", "metric"]
FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 0.7,
        "lines.linewidth": 1.6, "lines.markersize": 4.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.transparent": True, "figure.facecolor": "none",
        "axes.facecolor": "none", "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    })


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(p_values[order] * np.arange(len(p_values), 0, -1)))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def validate_paired_scores(folds: pd.DataFrame, paired: pd.DataFrame, family_size: int) -> dict:
    """Independently verify within-generator pairs, corrected CIs and Holm tests."""
    if not folds.status.eq("ok").all() or not np.isfinite(folds.score).all():
        raise ValueError("The archived grid contains missing or failed scores.")
    identity = KEYS + ["split_id", "regime"]
    if folds.duplicated(identity).any():
        raise ValueError("Duplicate fold keys in archived input.")
    pivot = folds.pivot(index=KEYS + ["split_id"], columns="regime", values="score").sort_index()
    if pivot.isna().any().any():
        raise ValueError("A matched score is absent.")
    expected = {"mean_reference": [], "mean_candidate": [], "mean_delta": [], "corrected_se": [], "ci_low": [], "ci_high": [], "p_value": []}
    for row in paired.itertuples(index=False):
        block = pivot.loc[tuple(getattr(row, key) for key in KEYS)]
        reference = block[row.reference_regime].to_numpy()
        candidate = block[row.candidate_regime].to_numpy()
        differences = candidate - reference
        n_pairs = len(differences)
        if n_pairs != 50 or n_pairs != row.n_pairs:
            raise ValueError("Unexpected number of paired development splits.")
        # The archived implementation uses mean validation size / mean selected
        # real training size, including at fractions below the complete fold.
        se = differences.std(ddof=1) * np.sqrt(1 / n_pairs + row.n_validation / row.n_train_real)
        delta = differences.mean()
        critical = student_t.ppf(0.975, n_pairs - 1)
        p_value = float(2 * student_t.sf(abs(delta / se), n_pairs - 1)) if se > 0 else float(delta == 0)
        values = [reference.mean(), candidate.mean(), delta, se, delta - critical * se, delta + critical * se, p_value]
        for key, value in zip(expected, values):
            expected[key].append(value)
    errors = {}
    for key, values in expected.items():
        difference = np.abs(np.asarray(values) - paired[key].to_numpy())
        errors[key] = float(difference.max())
        np.testing.assert_allclose(values, paired[key], rtol=2e-10, atol=2e-12, err_msg=key)
    recomputed = paired.copy()
    recomputed["p_from_folds"] = expected["p_value"]
    max_holm_error = 0.0
    for _, family in recomputed.groupby(["dataset", "model"], sort=False):
        if len(family) != family_size or not family.holm_family_size.eq(family_size).all():
            raise ValueError("Unexpected Holm comparison family.")
        adjusted = holm_adjust(family.p_from_folds.to_numpy())
        max_holm_error = max(max_holm_error, float(np.max(np.abs(adjusted - family.p_value_holm.to_numpy()))))
        np.testing.assert_allclose(adjusted, family.p_value_holm, rtol=2e-10, atol=2e-12)
        if not np.array_equal(adjusted < 0.05, family["significant_holm_0.05"].to_numpy()):
            raise ValueError("Holm significance labels do not match scores.")
    errors["p_value_holm"] = max_holm_error
    return {"fold_records": len(folds), "paired_comparisons": len(paired), "family_size": family_size, "maximum_absolute_errors": errors}


def select_one(data: pd.DataFrame, **conditions) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for key, value in conditions.items():
        mask &= data[key].eq(value)
    selected = data.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one comparison, found {len(selected)}: {conditions}")
    return selected.iloc[0]


def finish(fig, name: str, output_dir: Path) -> None:
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04, transparent=True,
                metadata={"Creator": "make_paper_figures.py (Matplotlib)", "Title": name})
    fig.savefig(output_dir / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)


def effect_axes(ax) -> None:
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def figure_overview(paired: pd.DataFrame, output_dir: Path) -> None:
    full = paired.loc[(paired.training_fraction == 1) & paired.metric.eq("AUPRC")]
    fig, axes = plt.subplots(2, 1, figsize=(6.55, 5.2))
    fig.subplots_adjust(left=0.23, right=0.99, bottom=0.21, top=0.94, hspace=0.58)
    norm = TwoSlopeNorm(vmin=-0.12, vcenter=0, vmax=0.12)
    cmap = matplotlib.colormaps["PuOr_r"]
    columns = [(g, r) for g in GENERATORS for r in REGIMES]
    for panel, (ax, model) in enumerate(zip(axes, MODELS)):
        values = np.zeros((len(DATASETS), len(columns)))
        significance = np.zeros_like(values, dtype=bool)
        for i, dataset in enumerate(DATASETS):
            for j, (generator, regime) in enumerate(columns):
                record = select_one(full, dataset=dataset, model=model, generator=generator, candidate_regime=regime)
                values[i, j] = record.mean_delta
                significance[i, j] = bool(record["significant_holm_0.05"])
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                color = "white" if abs(value) > 0.072 else "#171717"
                # A separate asterisk gives the Holm decision without implying
                # that displayed rounded values have been used for inference.
                label = f"{value:+.3f}" + ("*" if significance[i, j] else "")
                ax.text(j, i, label, ha="center", va="center", fontsize=9.0, color=color)
        ax.set_yticks(range(4), [DATASET_LABELS[x] for x in DATASETS], fontsize=9)
        ax.set_xticks(range(9), ["Syn.", "+1×", "+3×"] * 3, fontsize=9)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.set_title(f"({chr(97 + panel)}) {MODEL_LABELS[model]}", loc="left", pad=32, fontweight="bold")
        for center, generator in zip([1, 4, 7], GENERATORS):
            ax.text(center, -0.79, GENERATOR_LABELS[generator], ha="center", va="center", fontsize=9.5, clip_on=False)
        for x in [2.5, 5.5]:
            ax.axvline(x, color="white", linewidth=3)
        ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar_ax = fig.add_axes([0.34, 0.105, 0.48, 0.025])
    cb = fig.colorbar(im, cax=colorbar_ax, orientation="horizontal", ticks=[-0.12, -0.06, 0, 0.06, 0.12])
    cb.set_label("Paired AP change from Real-only", fontsize=10, labelpad=3)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.outline.set_linewidth(0.5)
    fig.text(0.23, 0.006, "Syn.: Synthetic-only; +1× / +3×: mixed training.  * Holm-adjusted p < 0.05.", fontsize=9)
    finish(fig, "figure1_ap_overview", output_dir)


def figure_effect_intervals(paired: pd.DataFrame, output_dir: Path) -> None:
    full = paired.loc[(paired.training_fraction == 1) & paired.generator.eq("copula_quantile") & paired.candidate_regime.eq("Synthetic-only")]
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 3.65), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.17, top=0.76, wspace=0.21)
    for panel, (ax, metric) in enumerate(zip(axes, ["AUPRC", "AUROC"])):
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.85)
        for index, model in enumerate(MODELS):
            rows = pd.DataFrame([select_one(full, dataset=ds, model=model, metric=metric) for ds in DATASETS])
            y = np.arange(4) + (-0.13 if index == 0 else 0.13)
            ax.errorbar(rows.mean_delta, y,
                        xerr=np.vstack([rows.mean_delta - rows.ci_low, rows.ci_high - rows.mean_delta]),
                        fmt=MODEL_MARKERS[model], color=MODEL_COLORS[model], markersize=5.5,
                        elinewidth=1.25, capsize=2.4, capthick=1.1, label=MODEL_LABELS[model], zorder=3)
        ax.set_yticks(range(4), [DATASET_LABELS[x] for x in DATASETS])
        ax.set_ylim(3.55, -0.55)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_title(f"({chr(97 + panel)}) " + ("Average precision" if metric == "AUPRC" else "AUROC"), loc="left", pad=12, fontweight="bold")
        ax.set_xlabel("Paired change")
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_xlim(-0.13, 0.06)
    axes[1].set_xlim(-0.045, 0.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.6, 1.01), ncol=2, frameon=False, columnspacing=1.6)
    finish(fig, "figure2_copula_effects", output_dir)


def figure_training_fractions(paired: pd.DataFrame, output_dir: Path) -> None:
    copula = paired.loc[paired.generator.eq("copula_quantile") & paired.metric.eq("AUPRC")]
    fig, axes = plt.subplots(4, 2, figsize=(6.55, 7.0), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.085, top=0.91, hspace=0.55, wspace=0.29)
    for row, dataset in enumerate(DATASETS):
        ds_data = copula[copula.dataset == dataset]
        lo = min(0.0, float(ds_data.mean_delta.min()))
        hi = max(0.0, float(ds_data.mean_delta.max()))
        pad = max(0.0015, (hi - lo) * 0.12)
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            effect_axes(ax)
            for regime in REGIMES:
                rows = ds_data.loc[ds_data.model.eq(model) & ds_data.candidate_regime.eq(regime)].sort_values("training_fraction")
                ax.plot(100 * rows.training_fraction, rows.mean_delta, color=REGIME_COLORS[regime], marker=REGIME_MARKERS[regime], label=REGIME_LABELS[regime], zorder=3)
            ax.set_title(f"({chr(97 + 2 * row + col)}) {DATASET_LABELS[dataset]}", loc="left", pad=5, fontsize=9.5)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([10, 25, 50, 100])
            ax.set_xlim(6, 104)
            if col == 0:
                ax.set_ylabel("AP change")
            if row == 3:
                ax.set_xlabel("Real training fraction (%)")
    fig.text(0.35, 0.949, "Logistic regression", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.80, 0.949, "Random forest", ha="center", fontsize=11, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.011), ncol=3, frameon=False, columnspacing=1.3, handlelength=1.6)
    finish(fig, "figure3_training_fractions", output_dir)


def figure_absolute_performance(folds: pd.DataFrame, output_dir: Path) -> None:
    data = folds.loc[folds.generator.eq("copula_quantile") & folds.metric.eq("AUPRC") & folds.regime.isin(["Real-only", "Synthetic-only"])]
    means = data.groupby(["dataset", "model", "regime", "training_fraction"], as_index=False).score.mean()
    fig, axes = plt.subplots(2, 2, figsize=(6.55, 4.65), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.81, hspace=0.42, wspace=0.32)
    for panel, (ax, dataset) in enumerate(zip(axes.flat, DATASETS)):
        ds_data = means[means.dataset == dataset]
        for model in MODELS:
            for regime, linestyle in [("Real-only", "-"), ("Synthetic-only", "--")]:
                rows = ds_data.loc[ds_data.model.eq(model) & ds_data.regime.eq(regime)].sort_values("training_fraction")
                ax.plot(100 * rows.training_fraction, rows.score, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], linestyle=linestyle,
                        markerfacecolor=MODEL_COLORS[model] if regime == "Real-only" else "white", markeredgewidth=1.1, zorder=3)
        ax.set_title(f"({chr(97 + panel)}) {DATASET_LABELS[dataset]}", loc="left", fontweight="bold", pad=7)
        ax.set_xticks([10, 25, 50, 100])
        ax.set_xlim(6, 104)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        if panel % 2 == 0:
            ax.set_ylabel("Average precision")
        if panel >= 2:
            ax.set_xlabel("Real training fraction (%)")
    model_handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], label=MODEL_LABELS[m]) for m in MODELS]
    regime_handles = [Line2D([0], [0], color="#424242", linestyle="-", label="Real-only"), Line2D([0], [0], color="#424242", linestyle="--", label="Copula Synthetic-only")]
    fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.55, 1.01), ncol=2, frameon=False, columnspacing=1.8)
    fig.legend(handles=regime_handles, loc="upper center", bbox_to_anchor=(0.55, 0.947), ncol=2, frameon=False, columnspacing=1.8)
    finish(fig, "figure4_absolute_performance", output_dir)


def encoding_data(main: pd.DataFrame, ordinal: pd.DataFrame, onehot: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for data, regime, label in [(ordinal, "Real-only", "Ordinal: Real-only"), (ordinal, "Mixed +1x", "Ordinal: Mixed +1x"),
                                (main, "Real-only", "Target-order: Real-only"), (onehot, "Real-only", "One-hot: Real-only")]:
        selected = data.loc[data.dataset.isin(["adult", "default_credit"]) & data.model.eq("logreg") & data.training_fraction.eq(1) & data.regime.eq(regime)].copy()
        if label != "One-hot: Real-only":
            selected = selected.loc[selected.generator.eq("copula_quantile")]
        counts = selected.groupby(["dataset", "metric"]).size()
        if len(counts) != 4 or not counts.eq(50).all():
            raise ValueError(f"Incomplete encoding source: {label}")
        means = selected.groupby(["dataset", "metric"], as_index=False).score.mean()
        means["pipeline"] = label
        pieces.append(means)
    return pd.concat(pieces, ignore_index=True)


def figure_encoding(data: pd.DataFrame, output_dir: Path) -> None:
    pipelines = ["Ordinal: Real-only", "Ordinal: Mixed +1x", "Target-order: Real-only", "One-hot: Real-only"]
    labels = ["Ordinal: real", r"Ordinal: mixed $+1\times$", "Target-order: real", "One-hot: real"]
    colors = ["#777777", "#009E73", "#0072B2", "#D55E00"]
    markers = ["o", "^", "s", "D"]
    fig, axes = plt.subplots(2, 2, figsize=(6.55, 4.2), sharey=True)
    fig.subplots_adjust(left=0.28, right=0.985, bottom=0.14, top=0.91, hspace=0.48, wspace=0.29)
    for row, dataset in enumerate(["adult", "default_credit"]):
        for col, metric in enumerate(["AUPRC", "AUROC"]):
            ax = axes[row, col]
            values = []
            for y, (pipeline, color, marker) in enumerate(zip(pipelines, colors, markers)):
                value = float(select_one(data, dataset=dataset, metric=metric, pipeline=pipeline).score)
                values.append(value)
                ax.scatter(value, y, s=37, color=color, marker=marker, zorder=3)
            ax.set_yticks(range(4), labels)
            ax.set_ylim(3.55, -0.55)
            low, high = min(values), max(values)
            pad = max(0.005, (high - low) * 0.17)
            ax.set_xlim(low - pad, high + pad)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.set_title(f"({chr(97 + 2 * row + col)}) {DATASET_LABELS[dataset]}", loc="left", fontweight="bold", pad=8)
            ax.set_xlabel("Average precision" if metric == "AUPRC" else "AUROC")
    finish(fig, "figure5_encoding", output_dir)


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=package_root / "data")
    parser.add_argument("--out-dir", type=Path, default=package_root / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    filenames = ["supplementary_fold_scores.csv", "supplementary_paired_results.csv", "ordinal_fold_scores.csv", "ordinal_paired_results.csv", "onehot_fold_scores.csv"]
    loaded = {name: pd.read_csv(args.data_dir / name) for name in filenames}
    folds = loaded[filenames[0]]
    paired = loaded[filenames[1]]
    ordinal = loaded[filenames[2]]
    ordinal_paired = loaded[filenames[3]]
    onehot = loaded[filenames[4]]
    if len(folds) != 38400 or len(paired) != 576 or len(ordinal) != 4800 or len(ordinal_paired) != 72 or len(onehot) != 400:
        raise ValueError("Unexpected archived source dimensions.")
    report = {"main": validate_paired_scores(folds, paired, 72), "ordinal": validate_paired_scores(ordinal, ordinal_paired, 36),
              "sources": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() for name in filenames},
              "figure_notes": {"confidence_intervals": "Pointwise approximate corrected-resampling 95% intervals; not Holm-adjusted.",
                               "absolute_baselines": "Real-only scores from the copula generator loop; original records retained.",
                               "trajectory_lines": "Mean trajectories only, without confidence bands or causal interpretation.",
                               "encoding": "Descriptive pipeline comparisons; no cross-encoding p values or intervals are inferred.",
                               "metrics": "AUPRC in archived CSVs is noninterpolated average precision, labelled AP in figures."}}
    configure_style()
    figure_overview(paired, args.out_dir)
    figure_effect_intervals(paired, args.out_dir)
    figure_training_fractions(paired, args.out_dir)
    figure_absolute_performance(folds, args.out_dir)
    encodings = encoding_data(folds, ordinal, onehot)
    figure_encoding(encodings, args.out_dir)
    encodings.to_csv(args.out_dir / "encoding_figure_values.csv", index=False)
    (args.out_dir / "figure_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote five PDF/PNG figure pairs to {args.out_dir}")
    print("Verified 648 paired comparisons against 43,200 main/ordinal score records.")


if __name__ == "__main__":
    main()
