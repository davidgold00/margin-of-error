"""Tests for config.py — pydantic configuration loaders."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_economics_config_loads(repo_root: Path) -> None:
    """EconomicsConfig loads and validates without error."""
    from margin_of_error.config import load_economics

    config = load_economics(repo_root / "config" / "economics.yaml")
    assert config.transaction.buy_side_pct > 0
    assert config.transaction.sell_side_pct > 0
    assert config.holding.typical_hold_months > 0
    assert config.financing.hard_money_rate > 0
    assert config.financing.ltv <= 1.0
    assert config.profit.minimum_margin_pct > 0
    assert config.profit.min_absolute_usd > 0


def test_model_config_loads(repo_root: Path) -> None:
    """ModelConfig loads and validates without error."""
    from margin_of_error.config import load_model_config

    config = load_model_config(repo_root / "config" / "model.yaml")
    assert config.global_seed == 42
    assert config.cross_validation.n_folds >= 2
    assert 0 < config.conformal.alpha < 1
    assert 0 < config.conformal.calibration_split < 0.5
    assert config.quantile.lower_alpha < config.quantile.upper_alpha


def test_economics_total_transaction_cost_reasonable(economics_config) -> None:
    """Total transaction cost (buy + sell) should be below 15%."""
    total = economics_config.transaction.buy_side_pct + economics_config.transaction.sell_side_pct
    assert total < 0.15, f"Combined transaction cost {total:.1%} seems too high"


def test_model_config_missing_file(tmp_path: Path) -> None:
    """load_model_config raises FileNotFoundError for a nonexistent path."""
    from margin_of_error.config import load_model_config

    with pytest.raises(FileNotFoundError):
        load_model_config(tmp_path / "nonexistent.yaml")


def test_economics_config_missing_file(tmp_path: Path) -> None:
    """load_economics raises FileNotFoundError for a nonexistent path."""
    from margin_of_error.config import load_economics

    with pytest.raises(FileNotFoundError):
        load_economics(tmp_path / "nonexistent.yaml")


def test_conformal_alpha_produces_valid_coverage(model_config) -> None:
    """1 - alpha should be between 0 and 1 (a valid probability)."""
    coverage = 1 - model_config.conformal.alpha
    assert 0 < coverage < 1


def test_quantile_ordering(model_config) -> None:
    """lower_alpha must be strictly less than upper_alpha."""
    assert model_config.quantile.lower_alpha < model_config.quantile.upper_alpha
