"""Summary tables, corrected repeated-CV tests, and paired holdout intervals."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def summarize_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive mean, standard deviation, naive SE, and split count.

    ``se_naive`` is retained only as a descriptive diagnostic. It assumes
    independent splits and must not be used as the inferential uncertainty for
    repeated cross-validation; ``paired_regime_tests`` provides corrected CIs.
    """

    required = {
        "dataset",
        "model",
        "generator",
        "synthetic_mode",
        "categorical_encoding",
        "training_fraction",
        "regime",
        "metric",
        "score",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    group_columns = [
        "dataset",
        "model",
        "generator",
        "synthetic_mode",
        "categorical_encoding",
        "training_fraction",
        "regime",
        "metric",
    ]
    output = (
        results.groupby(group_columns, dropna=False)["score"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    output["se_naive"] = output["std"] / np.sqrt(output["n"])
    return output.sort_values(group_columns).reset_index(drop=True)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values, preserving NaN positions."""

    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p))
    if len(finite_indices) == 0:
        return adjusted

    sorted_indices = finite_indices[np.argsort(p[finite_indices])]
    m = len(sorted_indices)
    running_max = 0.0
    for rank, original_index in enumerate(sorted_indices, start=1):
        candidate = min(1.0, (m - rank + 1) * p[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def pivot_unique(
    frame: pd.DataFrame,
    *,
    index: str | list[str],
    columns: str,
    values: str,
    context: str = "",
) -> pd.DataFrame:
    """Pivot long to wide and raise if any target cell is duplicated."""

    index_columns = [index] if isinstance(index, str) else list(index)
    key_columns = [*index_columns, columns]
    duplicated = frame.duplicated(subset=key_columns, keep=False)
    if duplicated.any():
        offenders = (
            frame.loc[duplicated, key_columns]
            .value_counts()
            .rename("n_records")
            .reset_index()
            .sort_values("n_records", ascending=False)
        )
        where = f" in {context}" if context else ""
        raise ValueError(
            f"Duplicate records{where}; long-to-wide pivot is ambiguous. "
            f"Most frequent duplicates:\n{offenders.head(5).to_string(index=False)}"
        )
    return frame.pivot(index=index_columns, columns=columns, values=values)


def corrected_stderr(
    differences: Iterable[float],
    *,
    n_train: float,
    n_test: float,
) -> float:
    """Nadeau-Bengio corrected standard error for paired resampled estimates."""

    values = np.asarray(list(differences), dtype=float)
    n = values.size
    if n < 2:
        return float("nan")
    if not (np.isfinite(n_train) and np.isfinite(n_test)) or n_train <= 0 or n_test <= 0:
        raise ValueError(f"n_train and n_test must be positive finite values, got {n_train}/{n_test}")
    corrected_variance = np.var(values, ddof=1) * (1.0 / n + float(n_test) / float(n_train))
    return float(np.sqrt(corrected_variance))


def compute_corrected_ttest(
    differences: Iterable[float],
    *,
    n_train: float,
    n_test: float,
    alternative: str = "two-sided",
) -> tuple[float, float, float, int]:
    """Test a paired mean difference using the corrected resampled t statistic."""

    values = np.asarray(list(differences), dtype=float)
    n = values.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), 0

    mean_difference = float(values.mean())
    standard_error = corrected_stderr(values, n_train=n_train, n_test=n_test)
    degrees_freedom = n - 1
    if not np.isfinite(standard_error):
        return float("nan"), float("nan"), standard_error, degrees_freedom
    if standard_error == 0:
        if mean_difference == 0:
            return 0.0, 1.0, 0.0, degrees_freedom
        return float(np.sign(mean_difference) * np.inf), 0.0, 0.0, degrees_freedom

    t_statistic = mean_difference / standard_error
    if alternative == "two-sided":
        p_value = 2.0 * stats.t.sf(abs(t_statistic), degrees_freedom)
    elif alternative == "greater":
        p_value = stats.t.sf(t_statistic, degrees_freedom)
    elif alternative == "less":
        p_value = stats.t.cdf(t_statistic, degrees_freedom)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return float(t_statistic), float(min(1.0, p_value)), standard_error, degrees_freedom


def paired_regime_tests(
    results: pd.DataFrame,
    *,
    n_splits: int,
    n_repeats: int,
    reference_regime: str = "Real-only",
    alternative: str = "two-sided",
    ci_level: float = 0.95,
    family_columns: tuple[str, ...] = (
        "dataset",
        "model",
        "synthetic_mode",
        "categorical_encoding",
    ),
) -> pd.DataFrame:
    """Compare every candidate regime with matched Real-only split scores.

    Holm correction is applied across all generator, regime, training-fraction,
    and metric comparisons inside each family specified by ``family_columns``.
    """

    required = {
        "dataset",
        "model",
        "generator",
        "synthetic_mode",
        "categorical_encoding",
        "training_fraction",
        "split_id",
        "regime",
        "metric",
        "score",
        "n_train_real",
        "n_validation",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    expected_splits = int(n_splits) * int(n_repeats)
    group_columns = [
        "dataset",
        "model",
        "generator",
        "synthetic_mode",
        "categorical_encoding",
        "training_fraction",
        "metric",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(group_columns, dropna=False):
        key_values = dict(zip(group_columns, keys))
        wide = pivot_unique(
            group,
            index="split_id",
            columns="regime",
            values="score",
            context=f"paired_regime_tests group {key_values}",
        )
        if reference_regime not in wide.columns:
            continue

        sizes = group[["split_id", "n_train_real", "n_validation"]].drop_duplicates()
        if sizes.duplicated(subset="split_id").any():
            raise ValueError(f"Split sizes disagree across regimes in group {key_values}")
        sizes = sizes.set_index("split_id")

        for candidate_regime in wide.columns:
            if candidate_regime == reference_regime:
                continue
            paired = wide[[reference_regime, candidate_regime]].dropna()
            delta = paired[candidate_regime] - paired[reference_regime]
            note = ""
            if len(delta) < 2:
                t_statistic = p_value = ci_low = ci_high = corrected_se = np.nan
                n_train = n_validation = np.nan
                degrees_freedom = 0
                note = "fewer than two paired splits"
            else:
                retained_sizes = sizes.loc[paired.index]
                n_train = float(retained_sizes["n_train_real"].mean())
                n_validation = float(retained_sizes["n_validation"].mean())
                t_statistic, p_value, corrected_se, degrees_freedom = compute_corrected_ttest(
                    delta,
                    n_train=n_train,
                    n_test=n_validation,
                    alternative=alternative,
                )
                mean_delta = float(delta.mean())
                if corrected_se == 0:
                    ci_low = ci_high = mean_delta
                elif np.isfinite(corrected_se):
                    critical = stats.t.ppf(0.5 + ci_level / 2.0, degrees_freedom)
                    ci_low = mean_delta - critical * corrected_se
                    ci_high = mean_delta + critical * corrected_se
                else:
                    ci_low = ci_high = np.nan
                if len(delta) != expected_splits:
                    note = f"{expected_splits - len(delta)} of {expected_splits} splits dropped"

            rows.append(
                {
                    **key_values,
                    "reference_regime": reference_regime,
                    "candidate_regime": candidate_regime,
                    "mean_reference": float(paired[reference_regime].mean()) if len(paired) else np.nan,
                    "mean_candidate": float(paired[candidate_regime].mean()) if len(paired) else np.nan,
                    "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                    "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                    "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    "corrected_se": float(corrected_se) if np.isfinite(corrected_se) else np.nan,
                    "t_stat": float(t_statistic) if np.isfinite(t_statistic) else t_statistic,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "alternative": alternative,
                    "n_pairs": int(len(delta)),
                    "n_train_real": n_train,
                    "n_validation": n_validation,
                    "note": note,
                }
            )

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    missing_family = set(family_columns).difference(output.columns)
    if missing_family:
        raise ValueError(f"Unknown Holm family columns: {sorted(missing_family)}")

    output["p_value_holm"] = np.nan
    output["holm_family_size"] = 0
    for _, indices in output.groupby(list(family_columns), dropna=False).groups.items():
        index_list = list(indices)
        output.loc[index_list, "p_value_holm"] = holm_adjust(
            output.loc[index_list, "p_value"].to_numpy()
        )
        output.loc[index_list, "holm_family_size"] = int(
            output.loc[index_list, "p_value"].notna().sum()
        )
    output["significant_holm_0.05"] = output["p_value_holm"] < 0.05
    return output.sort_values([*group_columns, "candidate_regime"]).reset_index(drop=True)


def _metric_value(metric: str, y_true: np.ndarray, probabilities: np.ndarray) -> float:
    normalized = metric.upper()
    if normalized == "AUROC":
        return float(roc_auc_score(y_true, probabilities))
    if normalized == "AUPRC":
        return float(average_precision_score(y_true, probabilities))
    raise ValueError("metric must be AUROC or AUPRC")


def paired_stratified_bootstrap(
    y_true: Iterable[int],
    candidate_probabilities: Iterable[float],
    reference_probabilities: Iterable[float],
    *,
    metric: str,
    n_resamples: int = 5000,
    random_state: int = 42,
    ci_level: float = 0.95,
) -> dict[str, float | int]:
    """Paired class-stratified percentile interval for a holdout score difference."""

    y = np.asarray(list(y_true))
    candidate = np.asarray(list(candidate_probabilities), dtype=float)
    reference = np.asarray(list(reference_probabilities), dtype=float)
    if not (len(y) == len(candidate) == len(reference)):
        raise ValueError("y_true and both probability arrays must have the same length")
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("paired_stratified_bootstrap requires exactly two classes")
    class_indices = [np.flatnonzero(y == label) for label in classes]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("both classes must be present")

    observed_candidate = _metric_value(metric, y, candidate)
    observed_reference = _metric_value(metric, y, reference)
    observed_delta = observed_candidate - observed_reference

    rng = np.random.default_rng(int(random_state))
    bootstrap_delta = np.empty(int(n_resamples), dtype=float)
    for index in range(int(n_resamples)):
        sampled_parts = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ]
        sampled = np.concatenate(sampled_parts)
        bootstrap_delta[index] = _metric_value(metric, y[sampled], candidate[sampled]) - _metric_value(
            metric,
            y[sampled],
            reference[sampled],
        )

    alpha = 1.0 - ci_level
    ci_low, ci_high = np.quantile(bootstrap_delta, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "candidate_score": observed_candidate,
        "reference_score": observed_reference,
        "delta": observed_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_resamples": int(n_resamples),
    }
