"""Model factories for the experiments."""

from __future__ import annotations

from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ModelName = Literal["logreg", "random_forest"]


def make_estimator(name: str, *, random_state: int = 42):
    """Create a fresh classifier by name.

    Fold preprocessing already returns a finite standardized numeric matrix, so
    the estimators do not repeat imputation or scaling.
    """

    normalized = name.lower().strip().replace("-", "_")
    if normalized in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=2000,
            solver="lbfgs",
            C=1.0,
        )
    if normalized in {"random_forest", "rf"}:
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
    raise ValueError("Unknown model. Valid names are: logreg, random_forest")
