"""Configuration objects for the synthetic replacement experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SyntheticMode = Literal["class_conditional", "minority_only"]
RegularizationPolicy = Literal["fixed", "adjusted"]
CategoricalEncoding = Literal["target_order", "ordinal", "none"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Fold-contained preprocessing settings.

    Target-order encoding is supervised. To avoid using a training row's own
    target when encoding that row, ``fit_transform`` uses inner cross-fitting.
    The mapping applied to validation and holdout rows is then fitted on the
    complete outer training fold.
    """

    categorical_encoding: CategoricalEncoding = "target_order"
    numeric_impute_strategy: str = "median"
    unknown_category_value: float = 0.5
    target_order_smoothing: float = 10.0
    target_order_inner_folds: int = 5
    standardize: bool = True
    random_state: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Core repeated-cross-validation experiment settings."""

    random_state: int = 42
    holdout_size: float = 0.20
    n_splits: int = 5
    n_repeats: int = 10
    training_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    mix_amounts: tuple[float, ...] = (1.0, 3.0)
    synthetic_mode: SyntheticMode = "class_conditional"
    positive_label: int = 1
    preprocessor: PreprocessConfig = field(default_factory=PreprocessConfig)
    regularization_policy: RegularizationPolicy = "fixed"
