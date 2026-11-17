import numpy as np
import pandas as pd

from synthrep.generators import BootstrapGenerator, GaussianCopulaGenerator, SmoteInterpolationGenerator


def toy_data():
    X = pd.DataFrame(
        {
            "x1": [0, 1, 2, 3, 4, 5, 6, 7],
            "x2": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="target")
    return X, y


def test_generators_return_exact_sample_count():
    X, y = toy_data()
    for generator_class in [BootstrapGenerator, SmoteInterpolationGenerator, GaussianCopulaGenerator]:
        generator = generator_class(random_state=1).fit(X, y)
        sample = generator.sample(17)
        assert sample.X.shape == (17, 2)
        assert len(sample.y) == 17
        assert set(sample.y.unique()) == {0, 1}


def test_smote_minority_small_class_fallback_returns_labels():
    X = pd.DataFrame({"x1": [0, 1, 2, 3, 4], "x2": [0, 1, 2, 3, 4]})
    y = pd.Series([0, 0, 0, 0, 1], name="target")
    generator = SmoteInterpolationGenerator(k_neighbors=5, random_state=1).fit(X, y)
    sample = generator.sample_minority(3)
    assert sample.X.shape == (3, 2)
    assert sample.y.tolist() == [1, 1, 1]


def test_smote_uses_one_scalar_gap_per_row():
    # With two collinear points, scalar interpolation must stay on the same line.
    X = pd.DataFrame({"x1": [0.0, 1.0, 10.0, 11.0], "x2": [0.0, 2.0, 20.0, 22.0]})
    y = pd.Series([0, 0, 1, 1], name="target")
    generator = SmoteInterpolationGenerator(k_neighbors=1, random_state=3).fit(X, y)
    sample = generator.sample(20)
    assert np.allclose(sample.X["x2"], 2.0 * sample.X["x1"])
