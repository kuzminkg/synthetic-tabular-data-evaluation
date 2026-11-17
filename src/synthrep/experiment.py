"""Leakage-controlled repeated-cross-validation experiment runner."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .config import ExperimentConfig
from .data import DatasetBundle, file_sha256, load_dataset, stratified_limit
from .generators import BaseClassConditionalGenerator
from .preprocessing import FoldPreprocessor
from .registry import make_generator, make_model
from .stats import paired_regime_tests, summarize_scores


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a deterministic 32-bit seed from a base seed and stable labels."""

    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def development_holdout_split(
    bundle: DatasetBundle,
    *,
    holdout_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create one stratified development/holdout split."""

    return train_test_split(
        bundle.X.reset_index(drop=True),
        bundle.y.reset_index(drop=True),
        test_size=holdout_size,
        stratify=bundle.y,
        random_state=random_state,
    )


def stratified_fraction_subset(
    X: pd.DataFrame,
    y: pd.Series,
    fraction: float,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select a deterministic, class-stratified fraction of a training fold.

    For a fixed seed, smaller fractions are prefixes of the same per-class random
    permutations and are therefore nested within larger fractions.
    """

    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("training fraction must be in (0, 1]")
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if fraction == 1.0:
        return X.copy(), y.copy()

    rng = np.random.default_rng(int(random_state))
    selected: list[int] = []
    for class_label in sorted(y.unique(), key=str):
        class_indices = np.flatnonzero(y.to_numpy() == class_label)
        permutation = rng.permutation(class_indices)
        n_select = min(
            len(class_indices), max(1, int(round(fraction * len(class_indices))))
        )
        selected.extend(permutation[:n_select].tolist())
    selected_array = np.array(sorted(selected), dtype=int)
    return (
        X.iloc[selected_array].reset_index(drop=True),
        y.iloc[selected_array].reset_index(drop=True),
    )


def _positive_probabilities(
    estimator, X: pd.DataFrame, positive_label: int | str
) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Estimator must implement predict_proba for AUROC/AUPRC")
    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if positive_label not in classes:
        raise ValueError(
            f"positive_label={positive_label!r} not found in fitted classes {classes!r}"
        )
    return probabilities[:, classes.index(positive_label)]


def _score_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
    }


def _logreg_c_for_regime(
    policy: str,
    n_real: int,
    n_regime: int,
) -> float:
    """Calculate logistic-regression C based on training size."""

    if n_real <= 0 or n_regime <= 0:
        raise ValueError("Training row counts must be positive")

    if policy == "fixed":
        return 1.0

    if policy == "adjusted":
        return n_real / n_regime

    raise ValueError(f"Unknown regularization policy: {policy!r}")


def _make_regimes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    generator: BaseClassConditionalGenerator,
    *,
    mix_amounts: Iterable[float],
    synthetic_mode: str,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Construct the training regimes for one outer training fold."""

    n_real = len(X_train)
    regimes: dict[str, tuple[pd.DataFrame, pd.Series]] = {
        "Real-only": (X_train.copy(), y_train.copy())
    }

    if synthetic_mode == "class_conditional":
        replacement = generator.sample(n_real)
        regimes["Synthetic-only"] = (replacement.X, replacement.y)
        for amount in mix_amounts:
            n_synthetic = int(round(float(amount) * n_real))
            sample = generator.sample(n_synthetic)
            regimes[f"Mixed +{float(amount):g}x"] = (
                pd.concat([X_train, sample.X], ignore_index=True),
                pd.concat([y_train, sample.y], ignore_index=True),
            )
    elif synthetic_mode == "minority_only":
        classes, counts = np.unique(y_train, return_counts=True)
        minority_class = classes[int(np.argmin(counts))]
        majority_class = classes[int(np.argmax(counts))]
        n_minority = int(counts.min())

        majority_mask = y_train.to_numpy() == majority_class
        X_majority = X_train.loc[majority_mask].reset_index(drop=True)
        y_majority = y_train.loc[majority_mask].reset_index(drop=True)

        replacement = generator.sample_minority(n_minority)
        regimes["Minority-replaced"] = (
            pd.concat([X_majority, replacement.X], ignore_index=True),
            pd.concat([y_majority, replacement.y], ignore_index=True),
        )
        for amount in mix_amounts:
            n_synthetic = int(round(float(amount) * n_minority))
            sample = generator.sample_minority(n_synthetic)
            regimes[f"Minority +{float(amount):g}x"] = (
                pd.concat([X_train, sample.X], ignore_index=True),
                pd.concat([y_train, sample.y], ignore_index=True),
            )

        if minority_class == majority_class:
            raise RuntimeError(
                "Minority-only mode requires two classes with different counts"
            )
    else:
        raise ValueError(
            "synthetic_mode must be 'class_conditional' or 'minority_only'"
        )

    return regimes


def run_one_experiment(
    bundle: DatasetBundle,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    *,
    model_name: str,
    generator_name: str,
    training_fraction: float,
    config: ExperimentConfig,
    progress: bool = True,
) -> pd.DataFrame:
    """Run repeated CV for one dataset/model/generator/training-fraction combination.

    Repeated CV is defined once on the complete development set. The requested
    training fraction is then selected only from the training portion of each
    split, so all fractions share the same real validation rows.
    """

    X_development = X_development.reset_index(drop=True)
    y_development = y_development.reset_index(drop=True).rename("target")
    splitter = RepeatedStratifiedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )

    rows: list[dict[str, object]] = []
    for split_id, (available_train_idx, validation_idx) in enumerate(
        splitter.split(X_development, y_development)
    ):
        repeat_id = split_id // config.n_splits
        fold_id = split_id % config.n_splits
        X_train_available = X_development.iloc[available_train_idx].reset_index(
            drop=True
        )
        y_train_available = y_development.iloc[available_train_idx].reset_index(
            drop=True
        )
        X_validation_raw = X_development.iloc[validation_idx].reset_index(drop=True)
        y_validation = y_development.iloc[validation_idx].reset_index(drop=True)

        subset_seed = derive_seed(config.random_state, bundle.name, "subset", split_id)
        X_train_raw, y_train = stratified_fraction_subset(
            X_train_available,
            y_train_available,
            training_fraction,
            random_state=subset_seed,
        )
        y_train = y_train.rename("target")

        preprocessing_seed = derive_seed(
            config.random_state, bundle.name, "preprocess", split_id
        )
        preprocessor_config = replace(
            config.preprocessor, random_state=preprocessing_seed
        )
        preprocessor = FoldPreprocessor(
            categorical_columns=bundle.categorical_columns,
            numeric_columns=bundle.numeric_columns,
            config=preprocessor_config,
        )
        X_train = preprocessor.fit_transform(X_train_raw, y_train)
        X_validation = preprocessor.transform(X_validation_raw)

        generator_seed = derive_seed(
            config.random_state,
            bundle.name,
            model_name,
            generator_name,
            split_id,
        )
        generator, canonical_generator_name = make_generator(
            generator_name,
            random_state=generator_seed,
        )
        generator.fit(X_train, y_train)
        regimes = _make_regimes(
            X_train,
            y_train,
            generator,
            mix_amounts=config.mix_amounts,
            synthetic_mode=config.synthetic_mode,
        )

        if progress and fold_id == 0:
            print(
                f"[{bundle.name}] model={model_name} generator={canonical_generator_name} "
                f"fraction={training_fraction:g} repeat={repeat_id}"
            )

        model_seed = derive_seed(
            config.random_state, bundle.name, model_name, "model", split_id
        )

        for regime_name, (X_regime, y_regime) in regimes.items():
            logreg_c = _logreg_c_for_regime(
                config.regularization_policy,
                n_real=len(X_train),
                n_regime=len(X_regime),
            )
            base_row = {
                "dataset": bundle.name,
                "model": model_name,
                "generator": canonical_generator_name,
                "synthetic_mode": config.synthetic_mode,
                "categorical_encoding": config.preprocessor.categorical_encoding,
                "training_fraction": float(training_fraction),
                "split_id": split_id,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "regime": regime_name,
                "n_train_available": len(X_train_available),
                "n_train_real": len(X_train),
                "n_train_regime": len(X_regime),
                "n_validation": len(X_validation),
                "positive_rate_train_real": float(y_train.mean()),
                "positive_rate_train_regime": float(y_regime.mean()),
                "positive_rate_validation": float(y_validation.mean()),
                "regularization_policy": config.regularization_policy,
                "logreg_c": logreg_c if model_name == "logreg" else np.nan,
            }
            if y_regime.nunique() < 2:
                for metric in ("AUROC", "AUPRC"):
                    rows.append(
                        {
                            **base_row,
                            "metric": metric,
                            "score": np.nan,
                            "status": "skipped_one_class_training_regime",
                        }
                    )
                continue

            estimator = make_model(
                model_name, random_state=model_seed, logreg_c=logreg_c
            )
            estimator.fit(X_regime, y_regime)
            probabilities = _positive_probabilities(
                estimator, X_validation, config.positive_label
            )
            for metric_name, score in _score_metrics(
                y_validation, probabilities
            ).items():
                rows.append(
                    {
                        **base_row,
                        "metric": metric_name,
                        "score": score,
                        "status": "ok",
                    }
                )

    return pd.DataFrame(rows)


def _assert_real_only_invariant(
    fold_scores: pd.DataFrame, *, tolerance: float = 1e-12
) -> None:
    """Verify that duplicated Real-only baselines agree across generator loops."""

    baseline = fold_scores[fold_scores["regime"] == "Real-only"].copy()
    if baseline.empty:
        return
    keys = [
        "dataset",
        "model",
        "synthetic_mode",
        "categorical_encoding",
        "training_fraction",
        "split_id",
        "metric",
    ]
    spread = baseline.groupby(keys, dropna=False)["score"].agg(["min", "max"])
    disagreements = spread[(spread["max"] - spread["min"]) > tolerance]
    if not disagreements.empty:
        raise RuntimeError(
            "Real-only scores differ across generator loops. The paired baseline is not invariant. "
            f"First disagreements:\n{disagreements.head().to_string()}"
        )


def _installed_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _source_fingerprint() -> str:
    """Fingerprint experiment package source to prevent mixing code revisions."""

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_completed_checkpoint(
    path: Path,
    *,
    dataset: str,
    model: str,
    generator_name: str,
    fraction: float,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Refuse to reuse an incomplete or mismatched checkpoint."""

    try:
        scores = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Cannot read checkpoint {path}: {exc}") from exc

    required = {
        "dataset", "model", "training_fraction", "split_id", "regime",
        "metric", "score", "status", "regularization_policy",
        "n_train_real", "n_train_regime", "logreg_c",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing columns: {sorted(missing)}")

    if config.synthetic_mode == "class_conditional":
        expected_regimes = {"Real-only", "Synthetic-only"}
        expected_regimes.update(f"Mixed +{float(x):g}x" for x in config.mix_amounts)
    else:
        expected_regimes = {"Real-only", "Minority-replaced"}
        expected_regimes.update(f"Minority +{float(x):g}x" for x in config.mix_amounts)

    _, canonical_generator = make_generator(generator_name, random_state=config.random_state)
    n_splits = config.n_splits * config.n_repeats
    expected_rows = n_splits * len(expected_regimes) * 2
    if (
        len(scores) != expected_rows
        or set(scores["dataset"]) != {dataset}
        or set(scores["model"]) != {model}
        or set(scores["generator"]) != {canonical_generator}
        or not np.allclose(scores["training_fraction"], fraction)
        or set(scores["split_id"]) != set(range(n_splits))
        or set(scores["regime"]) != expected_regimes
        or set(scores["metric"]) != {"AUROC", "AUPRC"}
        or set(scores["regularization_policy"]) != {config.regularization_policy}
        or scores.duplicated(["split_id", "regime", "metric"]).any()
    ):
        raise ValueError(f"Checkpoint {path} does not match the requested experiment grid")

    if model == "logreg":
        expected_c = [
            _logreg_c_for_regime(config.regularization_policy, int(n_real), int(n_regime))
            for n_real, n_regime in zip(scores["n_train_real"], scores["n_train_regime"])
        ]
        if not np.allclose(scores["logreg_c"], expected_c, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Checkpoint {path} contains mismatched logistic C values")
    return scores


def run_experiment_grid(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    datasets: Iterable[str],
    models: Iterable[str],
    generators: Iterable[str],
    config: ExperimentConfig,
    max_rows_per_dataset: int | None = None,
    progress: bool = True,
    resume: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the grid; optionally resume matching per-configuration checkpoints."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_manifest_path = checkpoint_dir / "run_manifest.json"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Checkpoints already exist in {checkpoint_dir}. "
            "Use --resume to reuse compatible checkpoints, or choose a new --out-dir."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = list(datasets)
    model_names = list(models)
    generator_names = list(generators)
    existing_checkpoint_names = {p.name for p in checkpoint_dir.glob("*.csv")}

    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_datasets": dataset_names,
        "source_fingerprint": _source_fingerprint(),
        "config": json.loads(json.dumps(asdict(config))),
        "data_dir": str(data_dir),
        "datasets": {},
        "models": model_names,
        "generators": generator_names,
        "max_rows_per_dataset": max_rows_per_dataset,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _installed_versions(),
        },
    }

    previous_manifest = None
    if resume and checkpoint_manifest_path.exists():
        try:
            previous_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("Cannot read the saved checkpoint manifest") from exc
        fields = (
            "requested_datasets", "source_fingerprint", "config", "data_dir",
            "models", "generators", "max_rows_per_dataset", "environment",
        )
        different = [key for key in fields if previous_manifest.get(key) != manifest[key]]
        if different:
            raise ValueError(
                "Cannot resume: checkpoint manifest differs in " + ", ".join(different) +
                ". Use a new --out-dir for changed code or settings."
            )
        manifest["created_utc"] = previous_manifest["created_utc"]
        # Retain metadata for datasets completed before the interrupted run.
        manifest["datasets"] = dict(previous_manifest.get("datasets", {}))
    elif existing_checkpoint_names:
        raise ValueError(
            "Cannot resume: CSV checkpoints exist but their run manifest is missing. "
            "Keep the files for recovery and use a new --out-dir."
        )

    encountered_checkpoint_names: set[str] = set()

    all_results: list[pd.DataFrame] = []
    for dataset_name in dataset_names:
        bundle = load_dataset(dataset_name, data_dir)
        bundle = stratified_limit(
            bundle,
            max_rows_per_dataset,
            random_state=config.random_state,
        )
        X_development, X_holdout, y_development, y_holdout = development_holdout_split(
            bundle,
            holdout_size=config.holdout_size,
            random_state=config.random_state,
        )
        dataset_metadata = {
            "source_filename": bundle.source_path.name,
            "source_sha256": file_sha256(bundle.source_path),
            "n_rows_used": bundle.n_rows,
            "positive_rate": bundle.positive_rate,
            "n_development": len(X_development),
            "n_holdout": len(X_holdout),
            "positive_rate_development": float(y_development.mean()),
            "positive_rate_holdout": float(y_holdout.mean()),
            "categorical_columns": list(bundle.categorical_columns),
            "numeric_columns": list(bundle.numeric_columns),
        }
        if previous_manifest is not None:
            prior_metadata = previous_manifest.get("datasets", {}).get(bundle.name)
            if prior_metadata is not None and prior_metadata != dataset_metadata:
                raise ValueError(
                    f"Cannot resume: dataset {bundle.name} changed (file hash, schema, or split). "
                    "Use a new --out-dir."
                )
        manifest["datasets"][bundle.name] = dataset_metadata
        # Record the dataset checksum before any of its checkpoints are saved.
        _write_json_atomic(checkpoint_manifest_path, manifest)

        for model_name in model_names:
            for generator_name in generator_names:
                for fraction in config.training_fractions:
                    # Index and filename must be stable across interrupted reruns.
                    checkpoint_name = (
                        f"{len(all_results):04d}_{bundle.name}_{model_name}_"
                        f"{generator_name}_{float(fraction):g}.csv"
                    )
                    checkpoint_path = checkpoint_dir / checkpoint_name
                    encountered_checkpoint_names.add(checkpoint_name)
                    if checkpoint_path.exists():
                        if not resume:
                            raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
                        if previous_manifest is None:
                            raise ValueError("Checkpoint exists but no matching run manifest was loaded")
                        if bundle.name not in previous_manifest.get("datasets", {}):
                            raise ValueError(f"Missing dataset metadata for {checkpoint_path}")
                        result = _load_completed_checkpoint(
                            checkpoint_path,
                            dataset=bundle.name,
                            model=model_name,
                            generator_name=generator_name,
                            fraction=float(fraction),
                            config=config,
                        )
                        if progress:
                            print(f"Checkpoint reused: {checkpoint_path}", flush=True)
                    else:
                        result = run_one_experiment(
                            bundle,
                            X_development,
                            y_development,
                            model_name=model_name,
                            generator_name=generator_name,
                            training_fraction=float(fraction),
                            config=config,
                            progress=progress,
                        )
                        temporary_path = checkpoint_dir / f"{checkpoint_name}.tmp"
                        result.to_csv(temporary_path, index=False)
                        temporary_path.replace(checkpoint_path)
                        if progress:
                            print(f"Checkpoint saved: {checkpoint_path}", flush=True)
                    all_results.append(result)

    unexpected = existing_checkpoint_names - encountered_checkpoint_names
    if unexpected:
        raise ValueError(
            "Cannot resume: unrelated checkpoint files present: " + ", ".join(sorted(unexpected))
        )

    fold_scores = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if not fold_scores.empty:
        _assert_real_only_invariant(fold_scores)
    summary = summarize_scores(fold_scores)
    tests = paired_regime_tests(
        fold_scores,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        reference_regime="Real-only",
    )

    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)
    summary.to_csv(output_dir / "summary_scores.csv", index=False)
    tests.to_csv(output_dir / "paired_tests_vs_real.csv", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {
        "fold_scores": fold_scores,
        "summary": summary,
        "paired_tests": tests,
    }
