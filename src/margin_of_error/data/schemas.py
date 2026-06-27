"""Pandera validation schemas for the Ames housing datasets.

Schemas enforce the structural contract expected by downstream pipelines.
They are deliberately non-exhaustive (strict=False) — they validate the columns
we care about without rejecting data that has extra columns we haven't accounted for.

Three schemas are provided:
    KAGGLE_TRAIN_SCHEMA  — validates train.csv (includes SalePrice)
    KAGGLE_TEST_SCHEMA   — validates test.csv (no SalePrice)
    AMES_FULL_SCHEMA     — validates AmesHousing.csv (full De Cock dataset)

Design notes:
- Ranges are based on the known Ames dataset distribution (De Cock 2011).
- nullable=True is set conservatively; specific NULL handling is in cleaning.py.
- Columns starting with digits (1stFlrSF, 2ndFlrSF, 3SsnPorch) cannot be used
  as attribute names in DataFrameModel, so we use the functional DataFrameSchema API.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# ── Shared column definitions (reused across schemas) ────────────────────────

_QUALITY_CODES = ["Ex", "Gd", "TA", "Fa", "Po"]

_SHARED_COLUMNS: dict[str, Column] = {
    "OverallQual": Column(int, Check.in_range(1, 10), nullable=False),
    "OverallCond": Column(int, Check.in_range(1, 10), nullable=False),
    "YearBuilt": Column(int, Check.in_range(1872, 2010), nullable=False),
    "YearRemodAdd": Column(int, Check.in_range(1950, 2010), nullable=False),
    "GrLivArea": Column(int, Check.greater_than(0), nullable=False),
    "LotArea": Column(int, Check.greater_than(0), nullable=False),
    "TotalBsmtSF": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    "FullBath": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    "BedroomAbvGr": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    "TotRmsAbvGrd": Column(int, Check.greater_than(0), nullable=False),
    "Fireplaces": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    "GarageCars": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    "GarageArea": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    "PoolArea": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    "MoSold": Column(int, Check.in_range(1, 12), nullable=False),
    "YrSold": Column(int, Check.isin([2006, 2007, 2008, 2009, 2010]), nullable=False),
    "ExterQual": Column(str, Check.isin(_QUALITY_CODES), nullable=False),
    "KitchenQual": Column(str, Check.isin(_QUALITY_CODES), nullable=False),
    "CentralAir": Column(str, Check.isin(["Y", "N"]), nullable=False),
}

# ── Kaggle train schema ──────────────────────────────────────────────────────

KAGGLE_TRAIN_SCHEMA = DataFrameSchema(
    columns={
        "Id": Column(int, Check.greater_than(0), nullable=False),
        "SalePrice": Column(int, Check.greater_than(0), nullable=False),
        **_SHARED_COLUMNS,
    },
    checks=[
        # SalePrice should be a plausible home value (sanity bounds)
        pa.Check(
            lambda df: (df["SalePrice"] >= 10_000).all() and (df["SalePrice"] <= 800_000).all(),
            error="SalePrice out of expected range [$10k, $800k]",
        ),
        # GrLivArea should not implausibly large
        pa.Check(
            lambda df: (df["GrLivArea"] <= 6_000).all(),
            error="GrLivArea exceeds 6,000 sqft — likely data error",
        ),
    ],
    coerce=False,
    strict=False,  # allow extra columns; we don't enumerate all 81
    name="KaggleTrainSchema",
)

# ── Kaggle test schema ───────────────────────────────────────────────────────

KAGGLE_TEST_SCHEMA = DataFrameSchema(
    columns={
        "Id": Column(int, Check.greater_than(0), nullable=False),
        **_SHARED_COLUMNS,
    },
    coerce=False,
    strict=False,
    name="KaggleTestSchema",
)

# ── Full Ames dataset schema (De Cock) ───────────────────────────────────────

AMES_FULL_SCHEMA = DataFrameSchema(
    columns={
        # De Cock dataset uses "PID" but normalize_columns() maps it to "Id"
        "Id": Column(int, Check.greater_than(0), nullable=True),  # nullable: some rows have no PID
        "SalePrice": Column(int, Check.greater_than(0), nullable=False),
        **_SHARED_COLUMNS,
    },
    checks=[
        pa.Check(
            lambda df: len(df) >= 2000,
            error="Full Ames dataset should have ≥ 2,000 rows; check file integrity",
        ),
    ],
    coerce=False,
    strict=False,
    name="AmesFullSchema",
)


def validate_kaggle_train(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[name-defined]
    """Validate a Kaggle training DataFrame against KAGGLE_TRAIN_SCHEMA.

    Args:
        df: DataFrame loaded via loaders.load_kaggle_train().

    Returns:
        The validated DataFrame (same object, not a copy).

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    return cast(pd.DataFrame, KAGGLE_TRAIN_SCHEMA.validate(df, lazy=True))


def validate_kaggle_test(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[name-defined]
    """Validate a Kaggle test DataFrame against KAGGLE_TEST_SCHEMA."""
    return cast(pd.DataFrame, KAGGLE_TEST_SCHEMA.validate(df, lazy=True))


def validate_ames_full(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[name-defined]
    """Validate the full Ames DataFrame against AMES_FULL_SCHEMA."""
    return cast(pd.DataFrame, AMES_FULL_SCHEMA.validate(df, lazy=True))
