"""Synthetic tabular generators used by the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler


@dataclass(frozen=True)
class SyntheticSample:
    """Synthetic feature matrix and target vector."""

    X: pd.DataFrame
    y: pd.Series

    def head(self, n: int) -> "SyntheticSample":
        return SyntheticSample(
            X=self.X.iloc[:n].reset_index(drop=True),
            y=self.y.iloc[:n].reset_index(drop=True),
        )


def _as_series(y: ArrayLike, *, name: str = "target") -> pd.Series:
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True).rename(y.name or name)
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y DataFrame must have exactly one column")
        return y.iloc[:, 0].reset_index(drop=True).rename(y.columns[0] or name)
    return pd.Series(np.asarray(y), name=name)


def _allocate_counts(classes: np.ndarray, counts: np.ndarray, n: int) -> dict[Hashable, int]:
    """Allocate exactly ``n`` synthetic rows according to empirical class prevalence."""

    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return {label: 0 for label in classes}
    expected = n * counts / counts.sum()
    base = np.floor(expected).astype(int)
    remainder = int(n - base.sum())
    if remainder:
        order = np.argsort(-(expected - base))
        for index in order[:remainder]:
            base[index] += 1
    return {label: int(value) for label, value in zip(classes, base)}


class BaseClassConditionalGenerator:
    """Shared validation and storage for class-conditional generators."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = int(random_state)
        self.rng_ = np.random.default_rng(self.random_state)
        self.feature_names_: list[str] | None = None
        self.classes_: np.ndarray | None = None
        self.class_counts_: np.ndarray | None = None
        self.indices_by_class_: dict[Hashable, np.ndarray] = {}
        self.X_: pd.DataFrame | None = None
        self.y_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "BaseClassConditionalGenerator":
        X_frame = pd.DataFrame(X).reset_index(drop=True).copy()
        y_series = _as_series(y)
        if len(X_frame) != len(y_series):
            raise ValueError("X and y have different lengths")
        if len(y_series.unique()) < 2:
            raise ValueError("At least two classes are required")
        values = X_frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Generator input must be finite after fold preprocessing")

        self.feature_names_ = list(X_frame.columns)
        self.X_ = pd.DataFrame(values, columns=self.feature_names_)
        self.y_ = y_series
        self.classes_, self.class_counts_ = np.unique(y_series, return_counts=True)
        self.indices_by_class_ = {
            label: np.flatnonzero(y_series.to_numpy() == label)
            for label in self.classes_
        }
        return self

    def _check_fitted(self) -> None:
        if self.X_ is None or self.y_ is None or self.classes_ is None or self.class_counts_ is None:
            raise RuntimeError("Generator must be fitted before sampling")

    def _class_counts_for_n(self, n: int) -> dict[Hashable, int]:
        self._check_fitted()
        assert self.classes_ is not None and self.class_counts_ is not None
        return _allocate_counts(self.classes_, self.class_counts_, n)

    def _wrap(self, X_array: np.ndarray, y_array: ArrayLike) -> SyntheticSample:
        self._check_fitted()
        X_df = pd.DataFrame(X_array, columns=self.feature_names_)
        y_name = self.y_.name if self.y_ is not None else "target"
        y_series = pd.Series(np.asarray(y_array), name=y_name)
        return SyntheticSample(X=X_df.reset_index(drop=True), y=y_series.reset_index(drop=True))

    def sample(self, n: int) -> SyntheticSample:
        raise NotImplementedError

    def sample_minority(self, n: int) -> SyntheticSample:
        """Generate ``n`` rows from the fitted minority class only."""

        self._check_fitted()
        assert self.class_counts_ is not None and self.classes_ is not None
        minority_class = self.classes_[int(np.argmin(self.class_counts_))]
        X_array = self._sample_class(minority_class, int(n))
        return self._wrap(X_array, np.full(int(n), minority_class))

    def _sample_class(self, class_label: Hashable, n: int) -> np.ndarray:
        raise NotImplementedError


class BootstrapGenerator(BaseClassConditionalGenerator):
    """Sample existing training rows with replacement within each class."""

    def sample(self, n: int) -> SyntheticSample:
        self._check_fitted()
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for class_label, n_class in self._class_counts_for_n(n).items():
            if n_class <= 0:
                continue
            X_parts.append(self._sample_class(class_label, n_class))
            y_parts.append(np.full(n_class, class_label))
        n_features = len(self.feature_names_ or [])
        X_array = np.vstack(X_parts) if X_parts else np.empty((0, n_features))
        y_array = np.concatenate(y_parts) if y_parts else np.array([])
        return self._wrap(X_array, y_array)

    def _sample_class(self, class_label: Hashable, n: int) -> np.ndarray:
        self._check_fitted()
        if n <= 0:
            return np.empty((0, len(self.feature_names_ or [])))
        assert self.X_ is not None
        indices = self.indices_by_class_[class_label]
        chosen = self.rng_.choice(indices, size=int(n), replace=True)
        return self.X_.iloc[chosen].to_numpy(dtype=float)


class SmoteInterpolationGenerator(BaseClassConditionalGenerator):
    """Class-conditional nearest-neighbor interpolation.

    Each synthetic row uses one scalar interpolation coefficient across all
    features, so it lies on the line segment between a seed and one of its
    nearest same-class neighbors. Inputs are expected to have been standardized
    by the fold preprocessor before nearest-neighbor search.
    """

    def __init__(self, *, k_neighbors: int = 5, random_state: int = 42) -> None:
        super().__init__(random_state=random_state)
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1")
        self.k_neighbors = int(k_neighbors)

    def sample(self, n: int) -> SyntheticSample:
        self._check_fitted()
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for class_label, n_class in self._class_counts_for_n(n).items():
            if n_class <= 0:
                continue
            X_parts.append(self._sample_class(class_label, n_class))
            y_parts.append(np.full(n_class, class_label))
        n_features = len(self.feature_names_ or [])
        X_array = np.vstack(X_parts) if X_parts else np.empty((0, n_features))
        y_array = np.concatenate(y_parts) if y_parts else np.array([])
        return self._wrap(X_array, y_array)

    def _sample_class(self, class_label: Hashable, n: int) -> np.ndarray:
        self._check_fitted()
        if n <= 0:
            return np.empty((0, len(self.feature_names_ or [])))
        assert self.X_ is not None
        indices = self.indices_by_class_[class_label]
        X_class = self.X_.iloc[indices].to_numpy(dtype=float)
        if len(X_class) <= self.k_neighbors:
            chosen = self.rng_.choice(len(X_class), size=int(n), replace=True)
            return X_class[chosen]

        neighbors = NearestNeighbors(n_neighbors=self.k_neighbors + 1, n_jobs=-1)
        neighbors.fit(X_class)
        seed_indices = self.rng_.choice(len(X_class), size=int(n), replace=True)
        X_seed = X_class[seed_indices]
        neighbor_matrix = neighbors.kneighbors(X_seed, return_distance=False)
        neighbor_columns = self.rng_.integers(1, self.k_neighbors + 1, size=int(n))
        neighbor_indices = neighbor_matrix[np.arange(int(n)), neighbor_columns]
        X_neighbor = X_class[neighbor_indices]
        gap = self.rng_.random(size=(int(n), 1))
        return X_seed + gap * (X_neighbor - X_seed)


class GaussianCopulaGenerator(BaseClassConditionalGenerator):
    """Class-conditional Gaussian-copula-style generator."""

    def __init__(
        self,
        *,
        transform: Literal["quantile", "standard", "minmax"] = "quantile",
        covariance_regularization: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.transform = transform
        self.covariance_regularization = float(covariance_regularization)
        self.models_: dict[Hashable, dict[str, object]] = {}

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "GaussianCopulaGenerator":
        super().fit(X, y)
        self._check_fitted()
        assert self.X_ is not None and self.classes_ is not None
        self.models_ = {}
        for class_label in self.classes_:
            X_class = self.X_.iloc[self.indices_by_class_[class_label]].to_numpy(dtype=float)
            transformer = self._make_transformer(n_samples=len(X_class))
            latent = transformer.fit_transform(X_class)
            covariance = np.atleast_2d(np.cov(latent, rowvar=False))
            covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
            covariance += np.eye(covariance.shape[0]) * self.covariance_regularization
            self.models_[class_label] = {
                "transformer": transformer,
                "mean": latent.mean(axis=0),
                "covariance": covariance,
            }
        return self

    def sample(self, n: int) -> SyntheticSample:
        self._check_fitted()
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for class_label, n_class in self._class_counts_for_n(n).items():
            if n_class <= 0:
                continue
            X_parts.append(self._sample_class(class_label, n_class))
            y_parts.append(np.full(n_class, class_label))
        n_features = len(self.feature_names_ or [])
        X_array = np.vstack(X_parts) if X_parts else np.empty((0, n_features))
        y_array = np.concatenate(y_parts) if y_parts else np.array([])
        return self._wrap(X_array, y_array)

    def _sample_class(self, class_label: Hashable, n: int) -> np.ndarray:
        self._check_fitted()
        if n <= 0:
            return np.empty((0, len(self.feature_names_ or [])))
        model = self.models_[class_label]
        latent = self.rng_.multivariate_normal(
            mean=np.asarray(model["mean"]),
            cov=np.asarray(model["covariance"]),
            size=int(n),
            check_valid="ignore",
        )
        transformer = model["transformer"]
        values = transformer.inverse_transform(latent)
        return np.asarray(values, dtype=float)

    def _make_transformer(self, n_samples: int):
        if self.transform == "quantile":
            return QuantileTransformer(
                n_quantiles=max(2, min(int(n_samples), 1000)),
                output_distribution="normal",
                subsample=100_000,
                random_state=self.random_state,
            )
        if self.transform == "standard":
            return StandardScaler()
        if self.transform == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown transform={self.transform!r}")
