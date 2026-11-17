import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from synthrep.stats import (
    corrected_stderr,
    holm_adjust,
    paired_regime_tests,
    paired_stratified_bootstrap,
)


@pytest.mark.parametrize(
    "raw",
    [
        [0.01, 0.04, 0.03, 0.005],
        [0.02, 0.03],
        [0.001, 0.008, 0.039, 0.041, 0.9],
        [0.2, 0.2, 0.2],
        [0.5],
    ],
)
def test_holm_matches_statsmodels(raw):
    values = np.array(raw, dtype=float)
    assert np.allclose(
        holm_adjust(values),
        multipletests(values, method="holm")[1],
        rtol=1e-12,
        atol=1e-15,
    )


def test_corrected_stderr_uses_actual_pair_count():
    differences = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.sqrt(np.var(differences, ddof=1) * (1 / 4 + 1 / 3))
    assert corrected_stderr(differences, n_train=30, n_test=10) == pytest.approx(expected)


def test_holm_family_spans_generators_fractions_regimes_and_metrics():
    rows = []
    for generator in ["a", "b"]:
        for fraction in [0.5, 1.0]:
            for metric in ["AUROC", "AUPRC"]:
                for split_id in range(4):
                    rows.append(
                        {
                            "dataset": "d",
                            "model": "m",
                            "generator": generator,
                            "synthetic_mode": "class_conditional",
                            "categorical_encoding": "ordinal",
                            "training_fraction": fraction,
                            "split_id": split_id,
                            "regime": "Real-only",
                            "metric": metric,
                            "score": 0.5,
                            "n_train_real": 30,
                            "n_validation": 10,
                        }
                    )
                    rows.append(
                        {
                            "dataset": "d",
                            "model": "m",
                            "generator": generator,
                            "synthetic_mode": "class_conditional",
                            "categorical_encoding": "ordinal",
                            "training_fraction": fraction,
                            "split_id": split_id,
                            "regime": "Synthetic-only",
                            "metric": metric,
                            "score": 0.51 + split_id * 0.001,
                            "n_train_real": 30,
                            "n_validation": 10,
                        }
                    )
    output = paired_regime_tests(pd.DataFrame(rows), n_splits=2, n_repeats=2)
    assert set(output["holm_family_size"]) == {8}


def test_paired_stratified_bootstrap_returns_positive_interval_for_clear_improvement():
    y = np.array([0] * 50 + [1] * 50)
    reference = np.tile([0.4, 0.6], 50)
    candidate = reference.copy()
    candidate[:50] -= 0.25
    candidate[50:] += 0.25
    result = paired_stratified_bootstrap(
        y,
        candidate,
        reference,
        metric="AUROC",
        n_resamples=200,
        random_state=2,
    )
    assert result["delta"] > 0
    assert result["ci_low"] >= 0
