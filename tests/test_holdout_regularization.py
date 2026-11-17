"""Integration test for regularization in the final holdout evaluator."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from synthrep import holdout
from synthrep.config import ExperimentConfig
from synthrep.data import DatasetBundle


@pytest.mark.parametrize(
    ("policy", "expected_selected_c"),
    [("fixed", 1.0), ("adjusted", 0.5)],
)
def test_holdout_uses_regularization_policy(monkeypatch, policy, expected_selected_c):
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"x1": rng.normal(size=120), "x2": rng.normal(size=120)})
    y = pd.Series([0] * 60 + [1] * 60, name="target")
    bundle = DatasetBundle(
        name="htru2",
        X=X,
        y=y,
        source_path=Path("dummy.csv"),
        categorical_columns=(),
        numeric_columns=("x1", "x2"),
    )
    monkeypatch.setattr(holdout, "load_dataset", lambda name, data_dir: bundle)

    real_make_model = holdout.make_model
    c_values = []

    def recording_make_model(name, *, random_state=42, logreg_c=1.0):
        c_values.append(logreg_c)
        return real_make_model(name, random_state=random_state, logreg_c=logreg_c)

    monkeypatch.setattr(holdout, "make_model", recording_make_model)
    selected = pd.DataFrame(
        [
            {
                "dataset": "htru2",
                "model": "logreg",
                "generator": "bootstrap",
                "regime": "Mixed +1x",
                "training_fraction": 1.0,
                "synthetic_mode": "class_conditional",
                "selection_metric": "AUPRC",
                "categorical_encoding": "target_order",
                "development_mean": 0.5,
            }
        ]
    )
    outputs = holdout.evaluate_selected_on_holdout(
        data_dir="unused",
        selected_configs=selected,
        config=ExperimentConfig(regularization_policy=policy, mix_amounts=(1.0,)),
        n_bootstrap=25,
    )

    assert c_values == [1.0, expected_selected_c]
    comparisons = outputs["comparisons"]
    assert set(comparisons["regularization_policy"]) == {policy}
    assert (comparisons["reference_logreg_c"] == 1.0).all()
    assert (comparisons["selected_logreg_c"] == expected_selected_c).all()
    assert len(comparisons) == 2  # AUROC and AUPRC
    assert len(outputs["predictions"]) == 24  # 20% holdout
