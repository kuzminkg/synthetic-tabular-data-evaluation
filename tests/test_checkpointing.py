"""Checkpoint files survive an experiment failing in a later configuration."""

import pandas as pd
import pytest

import synthrep.experiment as experiment
from synthrep.config import ExperimentConfig
from synthrep.data import DatasetBundle


def test_completed_configuration_checkpoint_survives_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("x,target\n0,0\n1,1\n", encoding="utf-8")
    bundle = DatasetBundle(
        name="test_dataset",
        X=pd.DataFrame({"x": range(40)}),
        y=pd.Series([0, 1] * 20, name="target"),
        categorical_columns=(),
        numeric_columns=("x",),
        source_path=source,
    )
    monkeypatch.setattr(experiment, "load_dataset", lambda name, data_dir: bundle)

    invocations = []

    def fake_run(*args, generator_name, **kwargs):
        invocations.append(generator_name)
        if generator_name == "smote":
            raise RuntimeError("simulated interruption")
        return pd.DataFrame(
            {"dataset": ["test_dataset"], "generator": ["bootstrap"], "score": [0.5]}
        )

    monkeypatch.setattr(experiment, "run_one_experiment", fake_run)
    output_dir = tmp_path / "results"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.run_experiment_grid(
            data_dir=tmp_path,
            output_dir=output_dir,
            datasets=["test_dataset"],
            models=["logreg"],
            generators=["bootstrap", "smote"],
            config=ExperimentConfig(n_splits=2, n_repeats=1, training_fractions=(0.5,)),
            progress=False,
        )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.csv"))
    assert invocations == ["bootstrap", "smote"]
    assert len(checkpoint_files) == 1
    stored = pd.read_csv(checkpoint_files[0])
    assert stored["score"].tolist() == [0.5]
    assert (checkpoint_dir / "run_manifest.json").exists()
    assert not (output_dir / "fold_scores.csv").exists()

    # A second run must not silently overwrite the saved work.
    with pytest.raises(FileExistsError, match="Checkpoints already exist"):
        experiment.run_experiment_grid(
            data_dir=tmp_path,
            output_dir=output_dir,
            datasets=["test_dataset"],
            models=["logreg"],
            generators=["bootstrap", "smote"],
            config=ExperimentConfig(n_splits=2, n_repeats=1, training_fractions=(0.5,)),
            progress=False,
        )
