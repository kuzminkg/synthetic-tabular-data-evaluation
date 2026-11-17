"""Final matched holdout evaluation after development-only configuration selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_dataset, stratified_limit
from .experiment import (
    _make_regimes,
    _positive_probabilities,
    derive_seed,
    development_holdout_split,
    stratified_fraction_subset,
)
from .preprocessing import FoldPreprocessor
from .registry import make_generator, make_model
from .stats import paired_stratified_bootstrap


def _canonicalize_real_only(summary: pd.DataFrame, *, tolerance: float = 1e-12) -> pd.DataFrame:
    """Collapse Real-only rows duplicated across generator loops."""

    real = summary[summary["regime"] == "Real-only"].copy()
    non_real = summary[summary["regime"] != "Real-only"].copy()
    if real.empty:
        return summary.copy()

    keys = [
        "dataset",
        "model",
        "synthetic_mode",
        "categorical_encoding",
        "training_fraction",
        "regime",
        "metric",
    ]
    spread = real.groupby(keys, dropna=False)["mean"].agg(["min", "max"])
    if ((spread["max"] - spread["min"]) > tolerance).any():
        raise ValueError("Real-only development means differ across generators; baseline is not invariant")

    canonical = real.sort_values("generator").drop_duplicates(keys).copy()
    canonical["generator"] = "none"
    return pd.concat([canonical, non_real], ignore_index=True)


def select_best_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the highest development mean for each dataset/model/metric.

    Exact ties prefer Real-only and then use deterministic lexical ordering.
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
        "mean",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")

    candidates = _canonicalize_real_only(summary)
    candidates["_real_priority"] = (candidates["regime"] != "Real-only").astype(int)
    candidates = candidates.sort_values(
        [
            "dataset",
            "model",
            "metric",
            "mean",
            "_real_priority",
            "training_fraction",
            "generator",
            "regime",
        ],
        ascending=[True, True, True, False, True, False, True, True],
        kind="mergesort",
    )
    selected = candidates.groupby(["dataset", "model", "metric"], as_index=False).head(1).copy()
    selected = selected.drop(columns=["_real_priority"])
    selected = selected.rename(columns={"metric": "selection_metric", "mean": "development_mean"})
    return selected.sort_values(["dataset", "model", "selection_metric"]).reset_index(drop=True)


def evaluate_selected_on_holdout(
    *,
    data_dir: str | Path,
    selected_configs: pd.DataFrame,
    config: ExperimentConfig,
    max_rows_per_dataset: int | None = None,
    n_bootstrap: int = 5000,
) -> dict[str, pd.DataFrame]:
    """Evaluate selected procedures and matched Real-only comparators once on holdout."""

    comparison_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for _, selected in selected_configs.iterrows():
        dataset_name = str(selected["dataset"])
        model_name = str(selected["model"])
        generator_name = str(selected["generator"])
        regime_name = str(selected["regime"])
        training_fraction = float(selected["training_fraction"])
        synthetic_mode = str(selected["synthetic_mode"])
        selection_metric = str(selected["selection_metric"])

        bundle = load_dataset(dataset_name, data_dir)
        bundle = stratified_limit(bundle, max_rows_per_dataset, random_state=config.random_state)
        X_development, X_holdout, y_development, y_holdout = development_holdout_split(
            bundle,
            holdout_size=config.holdout_size,
            random_state=config.random_state,
        )
        subset_seed = derive_seed(config.random_state, dataset_name, "final_training_subset")
        X_train_raw, y_train = stratified_fraction_subset(
            X_development,
            y_development,
            training_fraction,
            random_state=subset_seed,
        )
        y_train = y_train.reset_index(drop=True).rename("target")

        preprocessing_seed = derive_seed(config.random_state, dataset_name, "final_preprocess")
        preprocessor = FoldPreprocessor(
            categorical_columns=bundle.categorical_columns,
            numeric_columns=bundle.numeric_columns,
            config=replace(config.preprocessor, random_state=preprocessing_seed),
        )
        X_train = preprocessor.fit_transform(X_train_raw, y_train)
        X_holdout_encoded = preprocessor.transform(X_holdout)

        model_seed = derive_seed(config.random_state, dataset_name, model_name, "final_model")
        reference_model = make_model(model_name, random_state=model_seed)
        reference_model.fit(X_train, y_train)
        reference_probabilities = _positive_probabilities(
            reference_model,
            X_holdout_encoded,
            config.positive_label,
        )

        if regime_name == "Real-only":
            selected_probabilities = reference_probabilities.copy()
            selected_generator = "none"
            selected_positive_rate = float(y_train.mean())
            selected_training_size = len(X_train)
        else:
            generator_seed = derive_seed(
                config.random_state,
                dataset_name,
                model_name,
                generator_name,
                "final_generator",
            )
            generator, selected_generator = make_generator(
                generator_name,
                random_state=generator_seed,
            )
            generator.fit(X_train, y_train)
            regimes = _make_regimes(
                X_train,
                y_train,
                generator,
                mix_amounts=config.mix_amounts,
                synthetic_mode=synthetic_mode,
            )
            if regime_name not in regimes:
                raise ValueError(
                    f"Selected regime {regime_name!r} not available; got {sorted(regimes)}"
                )
            X_selected, y_selected = regimes[regime_name]
            selected_model = make_model(model_name, random_state=model_seed)
            selected_model.fit(X_selected, y_selected)
            selected_probabilities = _positive_probabilities(
                selected_model,
                X_holdout_encoded,
                config.positive_label,
            )
            selected_positive_rate = float(y_selected.mean())
            selected_training_size = len(X_selected)

        y_holdout_array = y_holdout.reset_index(drop=True).to_numpy()
        prediction_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "selection_metric": selection_metric,
                    "row_id": np.arange(len(y_holdout_array)),
                    "y_true": y_holdout_array,
                    "selected_probability": selected_probabilities,
                    "real_only_probability": reference_probabilities,
                }
            )
        )

        for holdout_metric in ("AUROC", "AUPRC"):
            if regime_name == "Real-only":
                from sklearn.metrics import average_precision_score, roc_auc_score

                score_function = roc_auc_score if holdout_metric == "AUROC" else average_precision_score
                score = float(score_function(y_holdout_array, reference_probabilities))
                interval = {
                    "candidate_score": score,
                    "reference_score": score,
                    "delta": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n_resamples": 0,
                }
            else:
                bootstrap_seed = derive_seed(
                    config.random_state,
                    dataset_name,
                    model_name,
                    selection_metric,
                    holdout_metric,
                    "bootstrap",
                )
                interval = paired_stratified_bootstrap(
                    y_holdout_array,
                    selected_probabilities,
                    reference_probabilities,
                    metric=holdout_metric,
                    n_resamples=n_bootstrap,
                    random_state=bootstrap_seed,
                )

            comparison_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "selection_metric": selection_metric,
                    "development_mean_selected_metric": float(selected["development_mean"]),
                    "generator": selected_generator,
                    "synthetic_mode": synthetic_mode,
                    "categorical_encoding": str(selected["categorical_encoding"]),
                    "training_fraction": training_fraction,
                    "selected_regime": regime_name,
                    "reference_regime": "Real-only",
                    "holdout_metric": holdout_metric,
                    "selected_score": interval["candidate_score"],
                    "real_only_score": interval["reference_score"],
                    "delta": interval["delta"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "bootstrap_resamples": interval["n_resamples"],
                    "n_development_used": len(X_train),
                    "n_selected_regime": selected_training_size,
                    "n_holdout": len(X_holdout_encoded),
                    "positive_rate_real_training": float(y_train.mean()),
                    "positive_rate_selected_regime": selected_positive_rate,
                    "positive_rate_holdout": float(y_holdout.mean()),
                }
            )

    comparisons = pd.DataFrame(comparison_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return {"comparisons": comparisons, "predictions": predictions}
