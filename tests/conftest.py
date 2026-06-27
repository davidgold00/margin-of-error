"""Shared pytest fixtures for the Margin of Error test suite.

All fixtures that synthesize DataFrames create the minimum required columns
for schema validation. The goal is to test logic, not real data — real-data
tests run only when data files are present (see the `data_present` fixture).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def economics_config(repo_root: Path):
    """Loaded EconomicsConfig for tests that need it."""
    from margin_of_error.config import load_economics

    return load_economics(repo_root / "config" / "economics.yaml")


@pytest.fixture(scope="session")
def model_config(repo_root: Path):
    """Loaded ModelConfig for tests that need it."""
    from margin_of_error.config import load_model_config

    return load_model_config(repo_root / "config" / "model.yaml")


@pytest.fixture
def minimal_train_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame that passes KAGGLE_TRAIN_SCHEMA.

    Contains exactly the columns the schema validates; extra columns
    (like Neighborhood, etc.) are not required by the schema in strict=False mode.
    """
    n = 20
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Id": range(1, n + 1),
            "OverallQual": rng.integers(1, 11, size=n),
            "OverallCond": rng.integers(1, 11, size=n),
            "YearBuilt": rng.integers(1900, 2010, size=n),
            "YearRemodAdd": rng.integers(1950, 2010, size=n),
            "GrLivArea": rng.integers(600, 3000, size=n),
            "LotArea": rng.integers(3000, 20000, size=n),
            "TotalBsmtSF": rng.uniform(0, 2000, size=n),
            "FullBath": rng.integers(0, 4, size=n),
            "BedroomAbvGr": rng.integers(0, 6, size=n),
            "TotRmsAbvGrd": rng.integers(2, 12, size=n),
            "Fireplaces": rng.integers(0, 4, size=n),
            "GarageCars": rng.uniform(0, 4, size=n),
            "GarageArea": rng.uniform(0, 1000, size=n),
            "PoolArea": rng.integers(0, 500, size=n),
            "MoSold": rng.integers(1, 13, size=n),
            "YrSold": rng.choice([2006, 2007, 2008, 2009, 2010], size=n),
            "ExterQual": rng.choice(["Ex", "Gd", "TA", "Fa", "Po"], size=n),
            "KitchenQual": rng.choice(["Ex", "Gd", "TA", "Fa", "Po"], size=n),
            "CentralAir": rng.choice(["Y", "N"], size=n),
            "SalePrice": rng.integers(50_000, 400_000, size=n),
        }
    )


@pytest.fixture
def minimal_test_df(minimal_train_df: pd.DataFrame) -> pd.DataFrame:
    """Like minimal_train_df but without SalePrice (mirrors Kaggle test set)."""
    return minimal_train_df.drop(columns=["SalePrice"])


@pytest.fixture
def kaggle_train_path(repo_root: Path) -> Path | None:
    """Return path to Kaggle train.csv if it exists, else None."""
    path = repo_root / "data" / "raw" / "kaggle" / "train.csv"
    return path if path.exists() else None


@pytest.fixture
def data_present(kaggle_train_path: Path | None) -> bool:
    """True if real data files are available."""
    return kaggle_train_path is not None
