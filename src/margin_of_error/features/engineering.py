"""Feature engineering pipeline.

Constructs the feature matrix used by all models. Separates raw column
selection from derived feature construction so each step is individually
testable.

PHASE 1 STATUS: Skeleton with typed signatures. Full implementation in Phase 1.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _has_positive_values(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a binary indicator for whether a numeric column is positive."""
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return (df[column] > 0).astype(int)


def _series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column when present, otherwise a zero series."""
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return df[column].fillna(0)


# Ordinal encoding mappings for quality/condition fields.
# Higher integer = better quality (consistent with OverallQual 1–10 scale).
QUALITY_ORDER = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
BSMT_FINISH_ORDER = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
FUNCTIONAL_ORDER = ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]
GARAGE_FINISH_ORDER = ["None", "Unf", "RFn", "Fin"]

ORDINAL_COLS: dict[str, list[str]] = {
    "ExterQual": QUALITY_ORDER,
    "ExterCond": QUALITY_ORDER,
    "BsmtQual": QUALITY_ORDER,
    "BsmtCond": QUALITY_ORDER,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": BSMT_FINISH_ORDER,
    "BsmtFinType2": BSMT_FINISH_ORDER,
    "HeatingQC": QUALITY_ORDER,
    "KitchenQual": QUALITY_ORDER,
    "FireplaceQu": QUALITY_ORDER,
    "GarageFinish": GARAGE_FINISH_ORDER,
    "GarageQual": QUALITY_ORDER,
    "GarageCond": QUALITY_ORDER,
    "PoolQC": QUALITY_ORDER,
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Functional": FUNCTIONAL_ORDER,
}


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ordinal string columns to integers using defined orderings.

    Args:
        df: DataFrame after cleaning (fill_none_categoricals already applied).

    Returns:
        DataFrame with ordinal string columns replaced by integer codes.
    """
    result = df.copy()
    for col, order in ORDINAL_COLS.items():
        if col not in result.columns:
            continue
        mapping = {v: i for i, v in enumerate(order)}
        result[col] = result[col].map(mapping).fillna(-1).astype(int)
    return result


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features with clear economic interpretations.

    All derived features are transformations of existing columns — no
    information leakage from the target.

    Phase 1 features:
        - TotalSF: GrLivArea + TotalBsmtSF (total finished sqft)
        - HouseAge: YrSold - YearBuilt
        - RemodAge: YrSold - YearRemodAdd
        - TotalBaths: FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
        - TotalPorchSF: deck + open/enclosed/3-season/screen porch sqft
        - HasPool, HasGarage, HasFireplace, Has2ndFloor, HasRemodeled indicators

    Args:
        df: DataFrame with cleaned and ordinal-encoded columns.

    Returns:
        DataFrame with additional derived columns appended.
    """
    result = df.copy()

    # Aggregate size features
    result["TotalSF"] = _series_or_zero(result, "GrLivArea") + _series_or_zero(
        result, "TotalBsmtSF"
    )
    result["TotalBaths"] = (
        _series_or_zero(result, "FullBath")
        + 0.5 * _series_or_zero(result, "HalfBath")
        + _series_or_zero(result, "BsmtFullBath")
        + 0.5 * _series_or_zero(result, "BsmtHalfBath")
    )
    result["TotalPorchSF"] = (
        _series_or_zero(result, "WoodDeckSF")
        + _series_or_zero(result, "OpenPorchSF")
        + _series_or_zero(result, "EnclosedPorch")
        + _series_or_zero(result, "3SsnPorch")
        + _series_or_zero(result, "ScreenPorch")
    )

    # Age features (requires YrSold)
    if {"YrSold", "YearBuilt", "YearRemodAdd"}.issubset(result.columns):
        result["HouseAge"] = result["YrSold"] - result["YearBuilt"]
        result["RemodAge"] = result["YrSold"] - result["YearRemodAdd"]
        result["HasRemodeled"] = (result["YearRemodAdd"] > result["YearBuilt"]).astype(int)

    # Binary indicators
    result["HasPool"] = _has_positive_values(result, "PoolArea")
    result["HasGarage"] = _has_positive_values(result, "GarageArea")
    result["HasFireplace"] = _has_positive_values(result, "Fireplaces")
    result["Has2ndFloor"] = _has_positive_values(result, "2ndFlrSF")

    return result


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining nominal categorical columns.

    Args:
        df: DataFrame after ordinal encoding and derived feature creation.

    Returns:
        DataFrame with dummy columns replacing nominal categoricals.
        Drop-first encoding to avoid perfect multicollinearity.
    """
    nominal_cols = [
        "MSZoning",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "Foundation",
        "Heating",
        "CentralAir",
        "Electrical",
        "GarageType",
        "PavedDrive",
        "Fence",
        "MiscFeature",
        "SaleType",
        "SaleCondition",
    ]
    present = [c for c in nominal_cols if c in df.columns]
    return pd.get_dummies(df, columns=present, drop_first=True, dtype=float)


def build_feature_matrix(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix for model training or inference.

    Applies the complete pipeline: ordinal encoding → derived features →
    one-hot encoding → column drop.

    Args:
        df: Cleaned DataFrame from data/cleaning.py.
        drop_cols: Additional columns to remove (e.g. target, IDs).

    Returns:
        Feature matrix X ready for model fitting.
    """
    drop_cols = drop_cols or []
    X = df.copy()
    X = encode_ordinals(X)
    X = add_derived_features(X)
    X = one_hot_encode(X)

    # Drop specified columns if present
    to_drop = [c for c in drop_cols if c in X.columns]
    X = X.drop(columns=to_drop)

    # Ensure fully numeric (should be after encoding)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning("Dropping %d non-numeric columns: %s", len(non_numeric), non_numeric)
        X = X.drop(columns=non_numeric)

    logger.info("Feature matrix: %d rows × %d columns", len(X), len(X.columns))
    return X
