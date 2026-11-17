"""Resume reuses matching checkpoints and rejects mismatched runs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import synthrep.experiment as experiment
from synthrep.config import ExperimentConfig
from synthrep.data import DatasetBundle


def _setup(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("placeholder for data checksum\n", encoding="utf-8")
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x": rng.normal(size=120), "z": rng.normal(size=120)})
    y = pd.Series([0, 1] * 60, name="target")
    bundle = DatasetBundle(
        name="test_dataset", X=X, y=y,
        categorical_columns=(), numeric_columns=("x", "z"), source_path=source,
    )
    monkeypatch.setattr(experiment, "load_dataset", lambda name, data_dir: bundle)
    config = ExperimentConfig(
        n_splits=2, n_repeats=1, training_fractions=(0.5,), mix_amounts=(1.0,),
        regularization_policy="adjusted",
    )
    kwargs = dict(
        data_dir=tmp_path, datasets=["test_dataset"], models=["logreg"],
        generators=["bootstrap", "smote"], config=config, progress=False,
    )
    return source, kwargs


def _interrupt(tmp_path, monkeypatch):
    source, kwargs = _setup(tmp_path, monkeypatch)
    original = experiment.run_one_experiment
    calls = []

    def interrupt_second(*args, **named):
        calls.append(named["generator_name"])
        if named["generator_name"] == "smote":
            raise RuntimeError("simulated interruption")
        return original(*args, **named)

    monkeypatch.setattr(experiment, "run_one_experiment", interrupt_second)
    output_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.run_experiment_grid(output_dir=output_dir, **kwargs)
    monkeypatch.setattr(experiment, "run_one_experiment", original)
    return source, kwargs, output_dir


def test_resume_reuses_first_checkpoint_and_matches_fresh_run(tmp_path, monkeypatch):
    source, kwargs, output_dir = _interrupt(tmp_path, monkeypatch)
    completed = next((output_dir / "checkpoints").glob("*.csv"))
    before = completed.read_bytes()

    original = experiment.run_one_experiment
    calls = []

    def recording_runner(*args, **named):
        calls.append(named["generator_name"])
        return original(*args, **named)

    monkeypatch.setattr(experiment, "run_one_experiment", recording_runner)
    resumed = experiment.run_experiment_grid(output_dir=output_dir, resume=True, **kwargs)
    assert calls == ["smote"]
    assert completed.read_bytes() == before

    fresh = experiment.run_experiment_grid(output_dir=tmp_path / "fresh", **kwargs)
    pd.testing.assert_frame_equal(
        resumed["fold_scores"], fresh["fold_scores"], check_exact=False, atol=1e-12
    )
    assert len(list((output_dir / "checkpoints").glob("*.csv"))) == 2


def test_resume_rejects_changed_configuration(tmp_path, monkeypatch):
    _, kwargs, output_dir = _interrupt(tmp_path, monkeypatch)
    kwargs["config"] = replace(kwargs["config"], n_repeats=2)
    with pytest.raises(ValueError, match="manifest differs in config"):
        experiment.run_experiment_grid(output_dir=output_dir, resume=True, **kwargs)


def test_resume_rejects_changed_data(tmp_path, monkeypatch):
    source, kwargs, output_dir = _interrupt(tmp_path, monkeypatch)
    source.write_text("changed dataset bytes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset test_dataset changed"):
        experiment.run_experiment_grid(output_dir=output_dir, resume=True, **kwargs)


def test_resume_rejects_truncated_checkpoint(tmp_path, monkeypatch):
    _, kwargs, output_dir = _interrupt(tmp_path, monkeypatch)
    completed = next((output_dir / "checkpoints").glob("*.csv"))
    d = pd.read_csv(completed)
    d.iloc[:-1].to_csv(completed, index=False)
    with pytest.raises(ValueError, match="does not match the requested experiment"):
        experiment.run_experiment_grid(output_dir=output_dir, resume=True, **kwargs)
