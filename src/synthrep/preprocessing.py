"""Fold-contained preprocessing for generator-compatible numeric matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import PreprocessConfig

_MISSING = "__MISSING__"


@dataclass
class FoldPreprocessor:
    """Impute, encode, and optionally standardize using only an outer training fold.

    For target-order encoding, training rows are encoded out of fold through an
    inner stratified split. This prevents each training row's own target from
    contributing to the value used to encode that row. A final mapping fitted on
    the complete outer training fold is retained for validation and holdout rows.
    """

    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    _category_maps: dict[str, dict[object, float]] = field(default_factory=dict, init=False)
    _category_unknowns: dict[str, float] = field(default_factory=dict, init=False)
    _numeric_medians: dict[str, float] = field(default_factory=dict, init=False)
    _output_columns: tuple[str, ...] = field(default=tuple(), init=False)
    _scaler: StandardScaler | None = field(default=None, init=False)
    _fitted: bool = field(default=False, init=False)

    @staticmethod
    def _category_series(series: pd.Series) -> pd.Series:
        return series.astype("object").where(~series.isna(), other=_MISSING)

    def _fit_numeric_medians(self, X: pd.DataFrame) -> None:
        if self.config.numeric_impute_strategy != "median":
            raise ValueError("Only numeric_impute_strategy='median' is currently supported")
        self._numeric_medians = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(X[column], errors="coerce")
            median = values.median()
            self._numeric_medians[column] = 0.0 if pd.isna(median) else float(median)

    def _learn_target_order_map(
        self,
        series: pd.Series,
        y: pd.Series,
    ) -> tuple[dict[object, float], float]:
        categories = self._category_series(series).reset_index(drop=True)
        y = y.reset_index(drop=True)
        global_rate = float(y.mean())
        grouped = (
            pd.DataFrame({"category": categories, "target": y})
            .groupby("category", dropna=False)["target"]
            .agg(["sum", "count"])
            .reset_index()
        )
        smoothing = float(self.config.target_order_smoothing)
        grouped["rate"] = (grouped["sum"] + smoothing * global_rate) / (
            grouped["count"] + smoothing
        )
        grouped["category_text"] = grouped["category"].map(str)
        grouped = grouped.sort_values(
            ["rate", "count", "category_text"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        denominator = max(len(grouped) - 1, 1)
        mapping = {
            category: rank / denominator
            for rank, category in enumerate(grouped["category"].tolist())
        }
        return mapping, float(self.config.unknown_category_value)

    def _learn_ordinal_map(self, series: pd.Series) -> tuple[dict[object, float], float]:
        categories = self._category_series(series)
        ordered = sorted(categories.unique(), key=lambda value: str(value))
        denominator = max(len(ordered) - 1, 1)
        mapping = {category: rank / denominator for rank, category in enumerate(ordered)}
        return mapping, float(self.config.unknown_category_value)

    def _fit_full_category_maps(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._category_maps = {}
        self._category_unknowns = {}
        if self.config.categorical_encoding == "none":
            if self.categorical_columns:
                raise ValueError(
                    "categorical_encoding='none' can be used only when the dataset has no categorical columns"
                )
            return

        for column in self.categorical_columns:
            if self.config.categorical_encoding == "target_order":
                mapping, unknown = self._learn_target_order_map(X[column], y)
            elif self.config.categorical_encoding == "ordinal":
                mapping, unknown = self._learn_ordinal_map(X[column])
            else:
                raise ValueError(
                    f"Unsupported categorical_encoding={self.config.categorical_encoding!r}"
                )
            self._category_maps[column] = mapping
            self._category_unknowns[column] = unknown

    def _numeric_frame(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        output: dict[str, pd.Series] = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(X[column], errors="coerce")
            output[column] = values.fillna(self._numeric_medians[column]).astype(float)
        return output

    def _encode_with_full_maps(self, X: pd.DataFrame) -> pd.DataFrame:
        output = self._numeric_frame(X)
        for column in self.categorical_columns:
            categories = self._category_series(X[column])
            mapping = self._category_maps[column]
            unknown = self._category_unknowns[column]
            output[column] = categories.map(mapping).fillna(unknown).astype(float)
        return pd.DataFrame(output, index=X.index).loc[:, list(self._output_columns)]

    def _cross_fitted_target_order(self, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series]:
        if not self.categorical_columns:
            return {}

        class_counts = y.value_counts()
        n_inner = min(int(self.config.target_order_inner_folds), int(class_counts.min()))
        if n_inner < 2:
            # This fallback is relevant only for tiny smoke-test folds.
            return {
                column: self._category_series(X[column])
                .map(self._category_maps[column])
                .fillna(self._category_unknowns[column])
                .astype(float)
                for column in self.categorical_columns
            }

        splitter = StratifiedKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=int(self.config.random_state),
        )
        encoded = {
            column: pd.Series(np.nan, index=X.index, dtype=float)
            for column in self.categorical_columns
        }
        for inner_train_idx, inner_valid_idx in splitter.split(X, y):
            y_inner = y.iloc[inner_train_idx].reset_index(drop=True)
            for column in self.categorical_columns:
                mapping, unknown = self._learn_target_order_map(
                    X.iloc[inner_train_idx][column].reset_index(drop=True),
                    y_inner,
                )
                values = self._category_series(X.iloc[inner_valid_idx][column])
                encoded_values = values.map(mapping).fillna(unknown).astype(float).to_numpy()
                encoded[column].iloc[inner_valid_idx] = encoded_values

        if any(series.isna().any() for series in encoded.values()):
            raise RuntimeError("Inner cross-fitting failed to encode every training row")
        return encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """Fit on an outer training fold and return its leakage-controlled representation."""

        X = X.reset_index(drop=True).copy()
        y_series = pd.Series(y).reset_index(drop=True)
        if len(X) != len(y_series):
            raise ValueError("X and y have different lengths")

        self._output_columns = tuple([*self.numeric_columns, *self.categorical_columns])
        self._fit_numeric_medians(X)
        self._fit_full_category_maps(X, y_series)

        output = self._numeric_frame(X)
        if self.config.categorical_encoding == "target_order":
            output.update(self._cross_fitted_target_order(X, y_series))
        elif self.config.categorical_encoding == "ordinal":
            for column in self.categorical_columns:
                categories = self._category_series(X[column])
                output[column] = (
                    categories.map(self._category_maps[column])
                    .fillna(self._category_unknowns[column])
                    .astype(float)
                )

        transformed = pd.DataFrame(output, index=X.index).loc[:, list(self._output_columns)]
        if self.config.standardize:
            self._scaler = StandardScaler()
            transformed_array = self._scaler.fit_transform(transformed)
            transformed = pd.DataFrame(transformed_array, columns=self._output_columns)
        else:
            self._scaler = None

        self._fitted = True
        return transformed.reset_index(drop=True)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "FoldPreprocessor":
        self.fit_transform(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply full outer-training mappings to validation or holdout rows."""

        if not self._fitted:
            raise RuntimeError("FoldPreprocessor must be fitted before transform")
        X = X.reset_index(drop=True).copy()
        transformed = self._encode_with_full_maps(X)
        if self._scaler is not None:
            transformed = pd.DataFrame(
                self._scaler.transform(transformed),
                columns=self._output_columns,
            )
        return transformed.reset_index(drop=True)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return self._output_columns or tuple([*self.numeric_columns, *self.categorical_columns])

    @property
    def category_maps(self) -> Mapping[str, Mapping[object, float]]:
        return self._category_maps
