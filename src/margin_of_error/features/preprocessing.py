"""Fold-safe sklearn preprocessing pipeline for Phase 1 models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from margin_of_error.features.engineering import ORDINAL_COLS, add_derived_features
from margin_of_error.features.missingness import (
    STRUCTURAL_NONE_CATEGORICALS,
    STRUCTURAL_ZERO_NUMERICS,
    TRUE_MISSING_FREQUENT_CATEGORICALS,
    TRUE_MISSING_MEDIAN_NUMERICS,
)

logger = logging.getLogger(__name__)

SKEWED_NUMERIC_FEATURES: tuple[str, ...] = (
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "TotalSF",
    "TotalPorchSF",
)


def _as_frame(X: Any) -> pd.DataFrame:
    """Coerce transformer input to a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


class MissingnessPolicyTransformer(BaseEstimator, TransformerMixin):
    """Apply the Ames missingness policy with learned values fit inside folds."""

    neighborhood_medians_: pd.Series
    global_lot_frontage_: float
    numeric_medians_: dict[str, float]
    categorical_modes_: dict[str, str]

    def fit(self, X: pd.DataFrame, y: object | None = None) -> MissingnessPolicyTransformer:
        """Fit fold-local medians and modes for true-missing fields."""
        del y
        frame = _as_frame(X)

        if {"Neighborhood", "LotFrontage"}.issubset(frame.columns):
            self.neighborhood_medians_ = frame.groupby("Neighborhood")["LotFrontage"].median()
            self.global_lot_frontage_ = float(frame["LotFrontage"].median())
        else:
            self.neighborhood_medians_ = pd.Series(dtype=float)
            self.global_lot_frontage_ = 0.0

        self.numeric_medians_ = {}
        for col in TRUE_MISSING_MEDIAN_NUMERICS:
            if col in frame.columns:
                median = frame[col].median()
                self.numeric_medians_[col] = float(median) if pd.notna(median) else 0.0

        self.categorical_modes_ = {}
        for col in TRUE_MISSING_FREQUENT_CATEGORICALS:
            if col in frame.columns:
                modes = frame[col].dropna().mode()
                self.categorical_modes_[col] = str(modes.iloc[0]) if not modes.empty else "Missing"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill structural absence and true missingness according to policy."""
        frame = _as_frame(X)

        for col in STRUCTURAL_NONE_CATEGORICALS:
            if col in frame.columns:
                frame[col] = frame[col].fillna("None")

        for col in STRUCTURAL_ZERO_NUMERICS:
            if col in frame.columns:
                frame[col] = frame[col].fillna(0)

        if "LotFrontage" in frame.columns:
            if "Neighborhood" in frame.columns and not self.neighborhood_medians_.empty:
                mapped = frame["Neighborhood"].map(self.neighborhood_medians_)
                frame["LotFrontage"] = frame["LotFrontage"].fillna(mapped)
            frame["LotFrontage"] = frame["LotFrontage"].fillna(self.global_lot_frontage_)

        for col, median in self.numeric_medians_.items():
            if col in frame.columns:
                frame[col] = frame[col].fillna(median)

        for col, mode in self.categorical_modes_.items():
            if col in frame.columns:
                frame[col] = frame[col].fillna(mode)

        return frame


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    """Append documented Phase 1 derived features."""

    def fit(self, X: pd.DataFrame, y: object | None = None) -> DerivedFeatureTransformer:
        del X, y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_derived_features(_as_frame(X))


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop configured non-modeling columns after derived features are created."""

    def __init__(self, columns: tuple[str, ...] = ()) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object | None = None) -> DropColumnsTransformer:
        del X, y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _as_frame(X)
        present = [col for col in self.columns if col in frame.columns]
        return frame.drop(columns=present)


class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    """Encode ordered Ames quality scales as integers before one-hot encoding."""

    def fit(self, X: pd.DataFrame, y: object | None = None) -> OrdinalFeatureEncoder:
        del X, y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _as_frame(X)
        for col, order in ORDINAL_COLS.items():
            if col not in frame.columns:
                continue
            mapping = {value: rank for rank, value in enumerate(order)}
            frame[col] = frame[col].map(mapping).fillna(-1).astype(int)
        return frame


class Log1pSkewedTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to pre-declared nonnegative skewed numeric features."""

    def __init__(self, columns: tuple[str, ...] = SKEWED_NUMERIC_FEATURES) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object | None = None) -> Log1pSkewedTransformer:
        del X, y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _as_frame(X)
        for col in self.columns:
            if col in frame.columns:
                frame[col] = np.log1p(frame[col].clip(lower=0))
        return frame


def numeric_columns(X: pd.DataFrame) -> list[str]:
    """Select numeric columns after row-wise feature engineering."""
    return _as_frame(X).select_dtypes(include=[np.number, "bool"]).columns.tolist()


def categorical_columns(X: pd.DataFrame) -> list[str]:
    """Select remaining nominal categorical columns."""
    return _as_frame(X).select_dtypes(include=["object", "category", "string"]).columns.tolist()


def build_preprocessor(drop_columns: list[str] | tuple[str, ...]) -> Pipeline:
    """Build the leakage-safe preprocessing pipeline used inside CV folds."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=[
            ("missingness", MissingnessPolicyTransformer()),
            ("derived", DerivedFeatureTransformer()),
            ("drop", DropColumnsTransformer(tuple(drop_columns))),
            ("ordinal", OrdinalFeatureEncoder()),
            ("log_skewed", Log1pSkewedTransformer()),
            ("columns", columns),
        ]
    )


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return fitted transformed feature names from a Phase 1 preprocessor."""
    columns = preprocessor.named_steps["columns"]
    names = columns.get_feature_names_out()
    return [str(name) for name in names]
