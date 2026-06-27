"""Tests for data/ subpackage — loaders, dictionary parser, cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ── Data dictionary tests ─────────────────────────────────────────────────────


def test_dictionary_parser_with_real_file(repo_root: Path) -> None:
    """Parse the real data_description.txt if it exists."""
    from margin_of_error.data.dictionary import parse_description_file

    path = repo_root / "data" / "raw" / "ames" / "data_description.txt"
    if not path.exists():
        pytest.skip("data_description.txt not found — add data files and re-run")

    dictionary = parse_description_file(path)
    assert len(dictionary) > 0
    assert "OverallQual" in dictionary or "MSSubClass" in dictionary


def test_dictionary_parser_with_synthetic_content(tmp_path: Path) -> None:
    """Parse a minimal synthetic description file."""
    from margin_of_error.data.dictionary import parse_description_file

    content = """MSSubClass: Identifies the type of dwelling involved in the sale.

       020\t1-STORY 1946 & NEWER ALL STYLES
       030\t1-STORY 1945 & OLDER

OverallQual: Rates the overall material and finish of the house

       10\tVery Excellent
       9\tExcellent
       1\tVery Poor
"""
    desc_file = tmp_path / "data_description.txt"
    desc_file.write_text(content)

    dictionary = parse_description_file(desc_file)
    assert "MSSubClass" in dictionary
    assert "OverallQual" in dictionary
    assert dictionary["MSSubClass"].is_categorical


def test_dictionary_quality_ordinal_map() -> None:
    """Quality ordinal map has expected ordering."""
    from margin_of_error.data.dictionary import get_quality_ordinal_map

    mapping = get_quality_ordinal_map()
    assert mapping["Ex"] > mapping["Gd"] > mapping["TA"] > mapping["Fa"] > mapping["Po"]
    assert mapping["NA"] == 0


# ── Loader tests ──────────────────────────────────────────────────────────────


def test_normalize_columns_is_idempotent(minimal_train_df: pd.DataFrame) -> None:
    """normalize_columns applied twice returns the same columns."""
    from margin_of_error.data.loaders import normalize_columns

    once = normalize_columns(minimal_train_df)
    twice = normalize_columns(once)
    assert list(once.columns) == list(twice.columns)


def test_normalize_columns_maps_de_cock_names() -> None:
    """normalize_columns converts De Cock column names to Kaggle convention."""
    from margin_of_error.data.loaders import normalize_columns

    df = pd.DataFrame({"PID": [1, 2], "Overall Qual": [7, 8], "Gr Liv Area": [1500, 2000]})
    result = normalize_columns(df)
    assert "Id" in result.columns
    assert "OverallQual" in result.columns
    assert "GrLivArea" in result.columns
    assert "PID" not in result.columns


def test_loader_kaggle_train_not_found(tmp_path: Path) -> None:
    """load_kaggle_train raises FileNotFoundError for missing file."""
    from margin_of_error.data.loaders import load_kaggle_train

    with pytest.raises(FileNotFoundError):
        load_kaggle_train(tmp_path / "nonexistent.csv")


def test_loader_with_real_data(kaggle_train_path: Path | None) -> None:
    """Load and basic-check real Kaggle training data if present."""
    if kaggle_train_path is None:
        pytest.skip("Kaggle train.csv not found — add data files and re-run")
    assert kaggle_train_path is not None

    from margin_of_error.data.loaders import load_kaggle_train

    df = load_kaggle_train(kaggle_train_path)
    assert "SalePrice" in df.columns
    assert "OverallQual" in df.columns
    assert len(df) > 1000
    assert (df["SalePrice"] > 0).all()


# ── Cleaning tests ────────────────────────────────────────────────────────────


def test_fill_none_categoricals_handles_missing_columns() -> None:
    """fill_none_categoricals is safe when columns are absent."""
    from margin_of_error.data.cleaning import fill_none_categoricals

    df = pd.DataFrame({"Id": [1, 2], "SalePrice": [100_000, 200_000]})
    result = fill_none_categoricals(df)
    assert list(result.columns) == list(df.columns)


def test_fill_none_categoricals_replaces_nan() -> None:
    """fill_none_categoricals fills NaN with 'None' string."""

    from margin_of_error.data.cleaning import fill_none_categoricals

    df = pd.DataFrame({"GarageType": [None, "Attchd", None], "Id": [1, 2, 3]})
    result = fill_none_categoricals(df)
    assert (result["GarageType"] == "None").sum() == 2


def test_impute_lot_frontage_by_neighborhood() -> None:
    """impute_lot_frontage fills NaN with neighborhood median."""
    import numpy as np

    from margin_of_error.data.cleaning import impute_lot_frontage

    df = pd.DataFrame(
        {
            "Neighborhood": ["A", "A", "A", "B", "B"],
            "LotFrontage": [50.0, 60.0, np.nan, 80.0, np.nan],
        }
    )
    result = impute_lot_frontage(df)
    assert result["LotFrontage"].isna().sum() == 0
    # Neighborhood A median of [50, 60] = 55
    assert result.loc[2, "LotFrontage"] == 55.0


# ── Feature registry tests ────────────────────────────────────────────────────


def test_feature_registries_have_no_overlap() -> None:
    """MUTABLE_FEATURES and FIXED_FEATURES must be disjoint."""
    from margin_of_error.features.registry import FIXED_FEATURES, MUTABLE_FEATURES

    overlap = set(MUTABLE_FEATURES) & set(FIXED_FEATURES)
    assert not overlap, f"Features in both registries: {overlap}"


def test_economics_pnl_computes_correctly(economics_config) -> None:
    """compute_flip_pnl arithmetic is correct for a simple scenario."""
    from margin_of_error.economics.simulation import compute_flip_pnl

    purchase = 150_000.0
    arv = 200_000.0
    reno = 20_000.0

    result = compute_flip_pnl(purchase, arv, reno, economics_config)

    # Verify cost components are positive
    assert result.acquisition_cost > 0
    assert result.selling_cost > 0
    assert result.holding_cost > 0
    assert result.financing_cost > 0

    # Verify accounting identity
    expected_total = (
        purchase
        + reno
        + result.acquisition_cost
        + result.selling_cost
        + result.holding_cost
        + result.financing_cost
    )
    assert abs(result.total_cost - expected_total) < 0.01

    # Verify profit = arv - total_cost
    assert abs(result.profit - (arv - result.total_cost)) < 0.01
