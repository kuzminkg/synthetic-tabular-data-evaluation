import pandas as pd

from synthrep.experiment import _make_regimes, stratified_fraction_subset
from synthrep.generators import BootstrapGenerator


def test_fraction_subsets_are_nested_for_same_seed():
    X = pd.DataFrame({"x": range(100)})
    y = pd.Series([0] * 70 + [1] * 30)
    X_small, _ = stratified_fraction_subset(X, y, 0.25, random_state=11)
    X_large, _ = stratified_fraction_subset(X, y, 0.50, random_state=11)
    assert set(X_small["x"]).issubset(set(X_large["x"]))


def test_minority_regimes_use_minority_count_not_total_count():
    X = pd.DataFrame({"x": range(10)}, dtype=float)
    y = pd.Series([0] * 8 + [1] * 2, name="target")
    generator = BootstrapGenerator(random_state=1).fit(X, y)
    regimes = _make_regimes(
        X,
        y,
        generator,
        mix_amounts=(1.0, 3.0),
        synthetic_mode="minority_only",
    )
    replacement_X, replacement_y = regimes["Minority-replaced"]
    plus_one_X, plus_one_y = regimes["Minority +1x"]
    plus_three_X, plus_three_y = regimes["Minority +3x"]
    assert len(replacement_X) == 10
    assert replacement_y.value_counts().to_dict() == {0: 8, 1: 2}
    assert len(plus_one_X) == 12
    assert plus_one_y.value_counts().to_dict() == {0: 8, 1: 4}
    assert len(plus_three_X) == 16
    assert plus_three_y.value_counts().to_dict() == {1: 8, 0: 8}
