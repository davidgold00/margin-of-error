"""Tests for data/schemas.py — pandera validation schemas."""

from __future__ import annotations

import pandas as pd
import pytest


def test_kaggle_train_schema_accepts_valid_data(minimal_train_df: pd.DataFrame) -> None:
    """KAGGLE_TRAIN_SCHEMA accepts the synthetic minimal training DataFrame."""
    from margin_of_error.data.schemas import validate_kaggle_train

    result = validate_kaggle_train(minimal_train_df)
    assert len(result) == len(minimal_train_df)


def test_kaggle_test_schema_accepts_valid_data(minimal_test_df: pd.DataFrame) -> None:
    """KAGGLE_TEST_SCHEMA accepts a DataFrame without SalePrice."""
    from margin_of_error.data.schemas import validate_kaggle_test

    result = validate_kaggle_test(minimal_test_df)
    assert len(result) == len(minimal_test_df)


def test_schema_rejects_invalid_overall_qual(minimal_train_df: pd.DataFrame) -> None:
    """KAGGLE_TRAIN_SCHEMA rejects OverallQual outside [1, 10]."""
    import pandera

    from margin_of_error.data.schemas import validate_kaggle_train

    bad_df = minimal_train_df.copy()
    bad_df.loc[0, "OverallQual"] = 11  # out of range

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_kaggle_train(bad_df)


def test_schema_rejects_negative_sale_price(minimal_train_df: pd.DataFrame) -> None:
    """KAGGLE_TRAIN_SCHEMA rejects non-positive SalePrice."""
    import pandera

    from margin_of_error.data.schemas import validate_kaggle_train

    bad_df = minimal_train_df.copy()
    bad_df.loc[0, "SalePrice"] = -1

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_kaggle_train(bad_df)


def test_schema_rejects_invalid_yr_sold(minimal_train_df: pd.DataFrame) -> None:
    """KAGGLE_TRAIN_SCHEMA rejects YrSold outside [2006, 2010]."""
    import pandera

    from margin_of_error.data.schemas import validate_kaggle_train

    bad_df = minimal_train_df.copy()
    bad_df.loc[0, "YrSold"] = 2020

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_kaggle_train(bad_df)


def test_schema_rejects_invalid_quality_code(minimal_train_df: pd.DataFrame) -> None:
    """KAGGLE_TRAIN_SCHEMA rejects unknown quality codes."""
    import pandera

    from margin_of_error.data.schemas import validate_kaggle_train

    bad_df = minimal_train_df.copy()
    bad_df.loc[0, "ExterQual"] = "Excellent"  # should be "Ex"

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_kaggle_train(bad_df)


def test_schema_allows_extra_columns(minimal_train_df: pd.DataFrame) -> None:
    """KAGGLE_TRAIN_SCHEMA accepts DataFrames with extra columns (strict=False)."""
    from margin_of_error.data.schemas import validate_kaggle_train

    df_with_extra = minimal_train_df.copy()
    df_with_extra["ExtraColumn"] = "extra"

    result = validate_kaggle_train(df_with_extra)
    assert "ExtraColumn" in result.columns
