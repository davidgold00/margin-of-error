"""Pre-modeling data cleaning transformations.

All transformations are explicit, reversible, and documented. The cleaning
pipeline is a pure function: DataFrame in, DataFrame out. No in-place mutation.

Cleaning decisions made here are separate from feature engineering (features/):
    - Cleaning: fix data quality issues (missing values, typos, impossible values)
    - Feature engineering: create new predictive signals

PHASE 0 STATUS: Stubs defined; full implementation scheduled for Phase 1.
No cleaning logic runs in Phase 0 (data-check only validates raw structure).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def impute_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing LotFrontage using median within Neighborhood.

    LotFrontage is missing for ~17% of rows. Neighborhood-median imputation
    is preferred over global median because frontage varies by subdivision layout.

    Phase 1 decision: if this materially changes model performance vs. a learned
    indicator variable, document it in docs/decisions.md.

    Args:
        df: DataFrame with LotFrontage and Neighborhood columns.

    Returns:
        DataFrame with LotFrontage NaNs filled.
    """
    result = df.copy()
    medians = result.groupby("Neighborhood")["LotFrontage"].transform("median")
    result["LotFrontage"] = result["LotFrontage"].fillna(medians)
    return result


def fill_none_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN in categorical columns where NA means 'not present'.

    Many Ames categorical features use NaN to indicate absence (e.g., no
    garage → GarageType = NaN). Replace with the string "None" so the
    absence is a distinct, learnable category rather than treated as missing.

    Affected columns: GarageType, GarageFinish, GarageQual, GarageCond,
    BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2,
    FireplaceQu, Fence, MiscFeature, Alley, PoolQC, MasVnrType.

    Args:
        df: DataFrame with raw categorical columns.

    Returns:
        DataFrame with "None" filled for structural NaNs.
    """
    absence_cols = [
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "Fence",
        "MiscFeature",
        "Alley",
        "PoolQC",
        "MasVnrType",
    ]
    result = df.copy()
    for col in absence_cols:
        if col in result.columns:
            result[col] = result[col].fillna("None")
    return result


def fill_numeric_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Fill structural NaN zeros in numeric columns (area/count features).

    These are NaN because the feature (garage, basement, etc.) does not exist,
    not because the measurement is missing. Zero is the correct imputed value.

    Affected columns: MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF,
    TotalBsmtSF, BsmtFullBath, BsmtHalfBath, GarageCars, GarageArea,
    GarageYrBlt.

    Args:
        df: DataFrame with potentially-null numeric columns.

    Returns:
        DataFrame with structural NaN numerics filled with 0.
    """
    zero_fill_cols = [
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "GarageCars",
        "GarageArea",
    ]
    result = df.copy()
    for col in zero_fill_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    if "GarageYrBlt" in result.columns:
        # Fill missing garage year with YearBuilt (no garage → built same year as house)
        result["GarageYrBlt"] = result["GarageYrBlt"].fillna(result["YearBuilt"])
    return result


def drop_low_variance_columns(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """Drop columns where a single value accounts for more than `threshold` of rows.

    Utilities is the canonical example (99.9% "AllPub"); it adds no signal.

    Args:
        df: Input DataFrame.
        threshold: Drop if most-frequent-value frequency exceeds this.

    Returns:
        DataFrame with near-constant columns removed.
    """
    result = df.copy()
    to_drop = []
    for col in result.select_dtypes(include="object").columns:
        top_freq = result[col].value_counts(normalize=True).iloc[0]
        if top_freq >= threshold:
            to_drop.append(col)
            logger.debug("Dropping near-constant column: %s (%.1f%%)", col, top_freq * 100)
    return result.drop(columns=to_drop)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline in order.

    This is the single entry point for all pre-modeling cleaning.
    Phase 1 implementation: applies all cleaning steps above.

    Args:
        df: Raw DataFrame from loaders.py (post normalize_columns).

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = impute_lot_frontage(df)
    df = fill_none_categoricals(df)
    df = fill_numeric_zeros(df)
    df = drop_low_variance_columns(df)
    logger.info("Cleaning complete: %d rows, %d columns", len(df), len(df.columns))
    return df
