import numpy as np
import pandas as pd

from synthrep.config import PreprocessConfig
from synthrep.preprocessing import FoldPreprocessor


def test_target_order_fit_transform_is_finite_and_transform_uses_training_map_only():
    X = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "category": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    preprocessor = FoldPreprocessor(
        categorical_columns=("category",),
        numeric_columns=("numeric",),
        config=PreprocessConfig(target_order_inner_folds=2, random_state=4),
    )
    transformed_train = preprocessor.fit_transform(X, y)
    validation = pd.DataFrame({"numeric": [3.0, np.nan], "category": ["a", "unseen"]})
    first = preprocessor.transform(validation)
    second = preprocessor.transform(validation.copy())
    assert np.isfinite(transformed_train.to_numpy()).all()
    assert np.isfinite(first.to_numpy()).all()
    assert np.allclose(first, second)
