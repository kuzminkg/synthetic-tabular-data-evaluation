"""Registries for model and generator names."""

from __future__ import annotations

from .generators import (
    BootstrapGenerator,
    GaussianCopulaGenerator,
    SmoteInterpolationGenerator,
)
from .models import make_estimator


def make_generator(name: str, *, random_state: int = 42):
    """Create one fresh generator and return it with its canonical name."""

    normalized = name.lower().strip().replace("-", "_")
    if normalized in {"bootstrap", "boot"}:
        return BootstrapGenerator(random_state=random_state), "bootstrap"
    if normalized in {"smote", "smote_interpolation", "interpolation"}:
        return (
            SmoteInterpolationGenerator(k_neighbors=5, random_state=random_state),
            "smote",
        )
    if normalized in {"copula", "gaussian_copula", "copula_quantile"}:
        return (
            GaussianCopulaGenerator(transform="quantile", random_state=random_state),
            "copula_quantile",
        )
    if normalized in {"copula_standard", "gaussian_standard"}:
        return (
            GaussianCopulaGenerator(transform="standard", random_state=random_state),
            "copula_standard",
        )
    if normalized in {"copula_minmax", "gaussian_minmax"}:
        return (
            GaussianCopulaGenerator(transform="minmax", random_state=random_state),
            "copula_minmax",
        )
    raise ValueError(
        "Unknown generator. Valid names include: bootstrap, smote, copula, "
        "copula_standard, copula_minmax."
    )


def make_model(name: str, *, random_state: int = 42, logreg_c: float = 1.0):
    return make_estimator(name, random_state=random_state, logreg_c=logreg_c)
