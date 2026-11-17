"""Dataset loading and lightweight cleaning utilities.

The loaders return raw feature frames plus metadata. Imputation, encoding, and
standardization happen later inside each outer cross-validation training fold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    """A cleaned binary-classification dataset and its feature metadata."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    source_path: Path
    positive_label: int = 1

    @property
    def n_rows(self) -> int:
        return len(self.X)

    @property
    def positive_rate(self) -> float:
        return float(np.mean(self.y == self.positive_label))


DEFAULT_FILENAMES = {
    "give_me_credit": "cs-training.csv",
    "default_credit": "UCI_Credit_Card.csv",
    "adult": "adult.csv",
    "htru2": "HTRU_2.csv",
}


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return a SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_path(data_dir: str | Path, filename: str) -> Path:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Pass --data-dir pointing to the folder containing "
            f"{filename}, or rename the file to match the expected name."
        )
    return path


def load_give_me_credit(data_dir: str | Path) -> DatasetBundle:
    """Load the Kaggle Give Me Some Credit training data."""

    path = _dataset_path(data_dir, DEFAULT_FILENAMES["give_me_credit"])
    df = pd.read_csv(path)
    target = "SeriousDlqin2yrs"
    if target not in df.columns:
        raise ValueError(f"{path} does not contain target column {target!r}")
    drop_cols = [column for column in ["Unnamed: 0"] if column in df.columns]
    y = df[target].astype(int).rename("target")
    X = df.drop(columns=[target, *drop_cols]).copy()
    return DatasetBundle(
        name="give_me_credit",
        X=X,
        y=y,
        categorical_columns=tuple(),
        numeric_columns=tuple(X.columns),
        source_path=path,
    )


def load_default_credit(data_dir: str | Path) -> DatasetBundle:
    """Load the UCI Default of Credit Card Clients data."""

    path = _dataset_path(data_dir, DEFAULT_FILENAMES["default_credit"])
    df = pd.read_csv(path)
    target = "default.payment.next.month"
    if target not in df.columns:
        raise ValueError(f"{path} does not contain target column {target!r}")
    drop_cols = [column for column in ["ID"] if column in df.columns]
    y = df[target].astype(int).rename("target")
    X = df.drop(columns=[target, *drop_cols]).copy()

    categorical_candidates = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    categorical_columns = tuple(column for column in categorical_candidates if column in X.columns)
    numeric_columns = tuple(column for column in X.columns if column not in categorical_columns)
    return DatasetBundle(
        name="default_credit",
        X=X,
        y=y,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        source_path=path,
    )


def load_adult(data_dir: str | Path) -> DatasetBundle:
    """Load the UCI Adult Census Income data in the project's CSV format."""

    path = _dataset_path(data_dir, DEFAULT_FILENAMES["adult"])
    df = pd.read_csv(path)
    df = df.replace({"?": np.nan, " ?": np.nan})
    target = "income"
    if target not in df.columns:
        raise ValueError(f"{path} does not contain target column {target!r}")
    labels = df[target].astype(str).str.strip().str.rstrip(".")
    y = labels.eq(">50K").astype(int).rename("target")
    X = df.drop(columns=[target]).copy()
    categorical_columns = tuple(X.select_dtypes(include=["object", "category"]).columns)
    numeric_columns = tuple(column for column in X.columns if column not in categorical_columns)
    return DatasetBundle(
        name="adult",
        X=X,
        y=y,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        source_path=path,
    )


def load_htru2(data_dir: str | Path) -> DatasetBundle:
    """Load the headerless UCI HTRU2 CSV without dropping its first observation."""

    path = _dataset_path(data_dir, DEFAULT_FILENAMES["htru2"])
    column_names = [
        "profile_mean",
        "profile_std",
        "profile_excess_kurtosis",
        "profile_skewness",
        "dm_mean",
        "dm_std",
        "dm_excess_kurtosis",
        "dm_skewness",
        "class",
    ]
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(column_names):
        raise ValueError(f"Expected 9 columns in {path}, found {df.shape[1]}")

    # Be tolerant of a locally added header while keeping the documented raw
    # headerless UCI file as the canonical format.
    first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce").notna().all()
    if not first_row_numeric:
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = column_names
    for column in column_names:
        df[column] = pd.to_numeric(df[column], errors="raise")

    y = df["class"].astype(int).rename("target")
    X = df.drop(columns=["class"]).copy()
    return DatasetBundle(
        name="htru2",
        X=X,
        y=y,
        categorical_columns=tuple(),
        numeric_columns=tuple(X.columns),
        source_path=path,
    )


def load_dataset(name: str, data_dir: str | Path) -> DatasetBundle:
    """Load one project dataset by canonical name or a supported alias."""

    normalized = name.lower().strip().replace("-", "_")
    if normalized in {"give_me_credit", "givemecredit", "gmc", "cs_training"}:
        return load_give_me_credit(data_dir)
    if normalized in {"default_credit", "default", "uci_credit", "credit_card"}:
        return load_default_credit(data_dir)
    if normalized in {"adult", "adult_census", "census_income"}:
        return load_adult(data_dir)
    if normalized == "htru2":
        return load_htru2(data_dir)
    raise ValueError(
        f"Unknown dataset {name!r}. Valid names are: give_me_credit, default_credit, adult, htru2"
    )


def load_all_datasets(
    data_dir: str | Path,
    names: Iterable[str] | None = None,
) -> dict[str, DatasetBundle]:
    """Load several project datasets."""

    requested = tuple(names or ("give_me_credit", "default_credit", "adult", "htru2"))
    return {name: load_dataset(name, data_dir) for name in requested}


def stratified_limit(
    bundle: DatasetBundle,
    max_rows: int | None,
    *,
    random_state: int,
) -> DatasetBundle:
    """Return a stratified subset for smoke tests or debugging."""

    if max_rows is None or max_rows <= 0 or max_rows >= len(bundle.X):
        return bundle
    X_sub, _, y_sub, _ = train_test_split(
        bundle.X,
        bundle.y,
        train_size=max_rows,
        stratify=bundle.y,
        random_state=random_state,
    )
    return DatasetBundle(
        name=bundle.name,
        X=X_sub.reset_index(drop=True),
        y=y_sub.reset_index(drop=True),
        categorical_columns=bundle.categorical_columns,
        numeric_columns=bundle.numeric_columns,
        source_path=bundle.source_path,
        positive_label=bundle.positive_label,
    )
