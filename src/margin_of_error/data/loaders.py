"""Raw data loading functions.

Loads CSV files for both the Kaggle competition split and the full De Cock
Ames dataset. Both loaders normalize column names to a common schema so
downstream code does not need to handle format differences.

Column name differences between datasets:
    - Full Ames dataset uses "PID" instead of "Id"
    - Full Ames dataset uses spaces in some column names (e.g., "1st Flr SF")
      while Kaggle uses no spaces ("1stFlrSF")
    - Full Ames dataset has an "Order" column not present in Kaggle version

The `normalize_columns()` helper standardizes both to the Kaggle column naming
convention, which is the primary schema used by pandera schemas and feature
engineering code.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from De Cock full dataset column names → Kaggle convention names.
# Only entries that differ are listed; columns with identical names are unchanged.
_AMES_TO_KAGGLE_COLS: dict[str, str] = {
    "PID": "Id",
    "1st Flr SF": "1stFlrSF",
    "2nd Flr SF": "2ndFlrSF",
    "3Ssn Porch": "3SsnPorch",
    "Bsmt Full Bath": "BsmtFullBath",
    "Bsmt Half Bath": "BsmtHalfBath",
    "Bsmt Qual": "BsmtQual",
    "Bsmt Cond": "BsmtCond",
    "Bsmt Exposure": "BsmtExposure",
    "BsmtFin Type 1": "BsmtFinType1",
    "BsmtFin SF 1": "BsmtFinSF1",
    "BsmtFin Type 2": "BsmtFinType2",
    "BsmtFin SF 2": "BsmtFinSF2",
    "Bsmt Unf SF": "BsmtUnfSF",
    "Total Bsmt SF": "TotalBsmtSF",
    "Garage Type": "GarageType",
    "Garage Yr Blt": "GarageYrBlt",
    "Garage Finish": "GarageFinish",
    "Garage Cars": "GarageCars",
    "Garage Area": "GarageArea",
    "Garage Qual": "GarageQual",
    "Garage Cond": "GarageCond",
    "Lot Frontage": "LotFrontage",
    "Lot Area": "LotArea",
    "Lot Shape": "LotShape",
    "Lot Config": "LotConfig",
    "Land Contour": "LandContour",
    "Land Slope": "LandSlope",
    "Bldg Type": "BldgType",
    "House Style": "HouseStyle",
    "Overall Qual": "OverallQual",
    "Overall Cond": "OverallCond",
    "Year Built": "YearBuilt",
    "Year Remod/Add": "YearRemodAdd",
    "Roof Style": "RoofStyle",
    "Roof Matl": "RoofMatl",
    "Exter Qual": "ExterQual",
    "Exter Cond": "ExterCond",
    "Heating QC": "HeatingQC",
    "Central Air": "CentralAir",
    "Low Qual Fin SF": "LowQualFinSF",
    "Gr Liv Area": "GrLivArea",
    "Full Bath": "FullBath",
    "Half Bath": "HalfBath",
    "Bedroom AbvGr": "BedroomAbvGr",
    "Kitchen AbvGr": "KitchenAbvGr",
    "Kitchen Qual": "KitchenQual",
    "Tot Rms AbvGrd": "TotRmsAbvGrd",
    "Fireplace Qu": "FireplaceQu",
    "Wood Deck SF": "WoodDeckSF",
    "Open Porch SF": "OpenPorchSF",
    "Enclosed Porch": "EnclosedPorch",
    "Screen Porch": "ScreenPorch",
    "Pool Area": "PoolArea",
    "Pool QC": "PoolQC",
    "Misc Feature": "MiscFeature",
    "Misc Val": "MiscVal",
    "Mo Sold": "MoSold",
    "Yr Sold": "YrSold",
    "Sale Type": "SaleType",
    "Sale Condition": "SaleCondition",
    "Sale Price": "SalePrice",
    "MS SubClass": "MSSubClass",
    "MS Zoning": "MSZoning",
    "Paved Drive": "PavedDrive",
    "Mas Vnr Type": "MasVnrType",
    "Mas Vnr Area": "MasVnrArea",
    "Exter 1st": "Exterior1st",
    "Exter 2nd": "Exterior2nd",
    "Condition 1": "Condition1",
    "Condition 2": "Condition2",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename De Cock full-dataset columns to Kaggle convention.

    Idempotent: if columns are already in Kaggle format, they are unchanged.
    The "Order" column (row index in De Cock dataset) is dropped if present.

    Args:
        df: Raw DataFrame (either Kaggle or full Ames format).

    Returns:
        DataFrame with standardized column names.
    """
    df = df.rename(columns=_AMES_TO_KAGGLE_COLS)
    if "Order" in df.columns:
        df = df.drop(columns=["Order"])
    return df


def load_kaggle_train(path: Path | str) -> pd.DataFrame:
    """Load the Kaggle competition training set.

    Args:
        path: Path to train.csv.

    Returns:
        DataFrame with ~1,460 rows including SalePrice.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    logger.info("Loading Kaggle train from %s", path)
    df = pd.read_csv(path)
    df = normalize_columns(df)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def load_kaggle_test(path: Path | str) -> pd.DataFrame:
    """Load the Kaggle competition test set (no SalePrice column).

    Args:
        path: Path to test.csv.

    Returns:
        DataFrame with ~1,459 rows, no SalePrice column.
    """
    path = Path(path)
    logger.info("Loading Kaggle test from %s", path)
    df = pd.read_csv(path)
    df = normalize_columns(df)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def load_ames_full(path: Path | str) -> pd.DataFrame:
    """Load the full De Cock Ames dataset (all 2,930 sales, 2006–2010).

    This dataset is used for the Phase 4 temporal backtest. It is sorted by
    YrSold then MoSold to establish a temporal ordering before any further
    processing.

    Args:
        path: Path to AmesHousing.csv.

    Returns:
        DataFrame with ~2,930 rows sorted by sale date, with SalePrice.
    """
    path = Path(path)
    logger.info("Loading full Ames dataset from %s", path)

    # Detect delimiter: the JSE-distributed Excel-converted CSV uses tab or comma
    sample = path.read_text(encoding="utf-8", errors="replace")[:1000]
    sep = "\t" if "\t" in sample else ","

    df = pd.read_csv(path, sep=sep)
    df = normalize_columns(df)
    df = df.sort_values(["YrSold", "MoSold"], ignore_index=True)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df
