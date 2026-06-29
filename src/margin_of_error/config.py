"""Configuration loaders for Margin of Error.

Loads typed, validated configuration objects from YAML files in config/.
Uses pydantic v2 BaseModel for validation.

All economic assumptions live in config/economics.yaml.
All model hyperparameters live in config/model.yaml.
No magic constants should exist anywhere else in the codebase.

Environment variables (prefixed MOE_) can override file paths; see .env.example.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ── Economics configuration ──────────────────────────────────────────────────


class TransactionCosts(BaseModel):
    """Buyer- and seller-side transaction costs as fractions of price."""

    buy_side_pct: float = Field(gt=0, lt=1, description="Buyer closing costs (fraction)")
    sell_side_pct: float = Field(gt=0, lt=1, description="Seller agent commission (fraction)")


class HoldingCosts(BaseModel):
    """Monthly carry costs and expected hold duration."""

    monthly_cost_pct: float = Field(
        gt=0, lt=1, description="Monthly cost as fraction of purchase price"
    )
    typical_hold_months: int = Field(gt=0, description="Expected flip duration in months")


class FinancingConfig(BaseModel):
    """Hard-money loan parameters."""

    hard_money_rate: float = Field(gt=0, lt=1, description="Annual interest rate")
    ltv: float = Field(gt=0, le=1, description="Loan-to-value ratio at acquisition")


class RenovationCosts(BaseModel):
    """Per-unit renovation cost estimates for renovatable features.

    All values are PLACEHOLDER national averages from Remodeling Magazine 2023.
    See docs/assumptions.md for Iowa-specific adjustment guidance.
    """

    kitchen_minor_remodel_usd: float = Field(gt=0)
    bath_addition_usd: float = Field(gt=0)
    bath_remodel_usd: float = Field(gt=0)
    basement_finish_per_sqft_usd: float = Field(gt=0)


class ProfitThresholds(BaseModel):
    """Minimum profit thresholds for the underwriting decision rule."""

    minimum_margin_pct: float = Field(
        gt=0, lt=1, description="Minimum net profit as fraction of ARV"
    )
    min_absolute_usd: float = Field(gt=0, description="Minimum net profit in dollars (floor)")


class RenovationTier(BaseModel):
    """A single renovation tier: cost to execute and assumed ARV uplift.

    The uplift percentages are conservative PRIORS for Phase 2. Phase 3 (causal)
    replaces them with data-derived estimates — they are assumptions, not findings.
    """

    cost_usd: float = Field(gt=0, description="Total renovation budget for this tier")
    value_uplift_pct: float = Field(
        ge=0, lt=1, description="Assumed ARV uplift as a fraction of base value"
    )
    scope: str = Field(description="Human-readable scope of work for this tier")


class UnderwritingThresholds(BaseModel):
    """Decision thresholds for the Phase 2 underwriting rule.

    All thresholds are defensible starting points, documented in docs/decisions.md
    and intended to be swept in the Phase 5 sensitivity analysis.
    """

    minimum_underwrite_margin_buffer_usd: float = Field(
        gt=0, description="Profit floor; P(profit > this) drives APPROVE/REFER"
    )
    max_acceptable_interval_width_usd: float = Field(
        gt=0, description="If the 90% CQR interval is wider than this, DECLINE on uncertainty"
    )
    approve_prob_above_min_margin: float = Field(
        gt=0, le=1, description="Min P(profit > buffer) required to APPROVE"
    )
    approve_prob_loss_max: float = Field(gt=0, le=1, description="Max P(loss) tolerated to APPROVE")
    refer_prob_above_min_margin: float = Field(
        gt=0, le=1, description="Min P(profit > buffer) required to REFER"
    )
    refer_prob_loss_max: float = Field(gt=0, le=1, description="Max P(loss) tolerated to REFER")

    @model_validator(mode="after")
    def approve_must_be_stricter_than_refer(self) -> UnderwritingThresholds:
        if self.approve_prob_above_min_margin < self.refer_prob_above_min_margin:
            raise ValueError("APPROVE margin-probability bar must be >= REFER bar")
        if self.approve_prob_loss_max > self.refer_prob_loss_max:
            raise ValueError("APPROVE loss tolerance must be <= REFER tolerance")
        return self


class FlipConfig(BaseModel):
    """Phase 2 fix-and-flip economics: profit Monte Carlo + underwriting rule.

    This block is self-contained: it carries every parameter the Phase 2 profit
    simulation and decision rule need, so no economic constant lives in code.
    """

    acquisition_arv_factor: float = Field(
        gt=0, le=1, description="'70% rule' factor: MAO = factor*ARV - renovation cost"
    )
    transaction_cost_pct: float = Field(
        gt=0, lt=1, description="Round-trip transaction cost as a fraction of purchase price"
    )
    holding_cost_monthly_pct: float = Field(
        gt=0, lt=1, description="Monthly carry cost as a fraction of purchase price"
    )
    holding_period_months_base: float = Field(gt=0, description="Expected hold duration (months)")
    holding_period_months_std: float = Field(
        gt=0, description="Std of hold duration; flips rarely go to plan"
    )
    holding_period_months_min: float = Field(gt=0, description="Truncation floor for hold (months)")
    holding_period_months_max: float = Field(gt=0, description="Truncation cap for hold (months)")
    financing_assumption: Literal["cash", "leverage"] = Field(
        description="Capital structure; cash is the clean default for the portfolio project"
    )
    monte_carlo_samples: int = Field(gt=0, description="Profit MC draws per property per tier")
    arv_normal_z: float = Field(
        gt=0, description="z mapping a (U-L) interval half-width to a Normal std"
    )
    renovation_tiers: dict[str, RenovationTier] = Field(
        description="Named renovation tiers (minimal/moderate/substantial)"
    )
    underwriting: UnderwritingThresholds

    @model_validator(mode="after")
    def validate_tiers_and_bounds(self) -> FlipConfig:
        if not self.renovation_tiers:
            raise ValueError("renovation_tiers must define at least one tier")
        if self.holding_period_months_min >= self.holding_period_months_max:
            raise ValueError("holding_period_months_min must be < holding_period_months_max")
        return self


class EconomicsConfig(BaseModel):
    """Root economics configuration; loaded from config/economics.yaml."""

    transaction: TransactionCosts
    holding: HoldingCosts
    financing: FinancingConfig
    renovation: RenovationCosts
    profit: ProfitThresholds
    flip: FlipConfig


# ── Model configuration ──────────────────────────────────────────────────────


class DataPaths(BaseModel):
    """Filesystem paths to raw data files."""

    kaggle_train_path: Path
    kaggle_test_path: Path
    ames_full_path: Path
    description_path: Path

    @model_validator(mode="after")
    def apply_env_overrides(self) -> DataPaths:
        """Allow environment variables to override individual paths."""
        env_map = {
            "MOE_KAGGLE_TRAIN_PATH": "kaggle_train_path",
            "MOE_KAGGLE_TEST_PATH": "kaggle_test_path",
            "MOE_AMES_FULL_PATH": "ames_full_path",
            "MOE_DESCRIPTION_PATH": "description_path",
        }
        for env_var, attr in env_map.items():
            if value := os.environ.get(env_var):
                setattr(self, attr, Path(value))
        return self


class CrossValidationConfig(BaseModel):
    n_folds: int = Field(ge=2, description="Number of CV folds")
    stratify_by: str = Field(description="Column to stratify fold assignment")


class ElasticNetTuningConfig(BaseModel):
    """Hyperparameter grid for the Phase 1 regularized linear baseline."""

    alpha_grid: list[float] = Field(description="ElasticNet alpha values to search")
    l1_ratio_grid: list[float] = Field(description="ElasticNet l1_ratio values to search")
    max_iter: int = Field(gt=0, description="Maximum coordinate descent iterations")


class LightGBMTuningConfig(BaseModel):
    """Compact nested-CV grid for the Phase 1 LightGBM strawman."""

    num_leaves_grid: list[int] = Field(description="LightGBM num_leaves values")
    min_child_samples_grid: list[int] = Field(description="LightGBM min_child_samples values")
    reg_lambda_grid: list[float] = Field(description="LightGBM L2 regularization values")


class Phase1Config(BaseModel):
    """Settings specific to Phase 1 baseline modeling."""

    cv_repeats: int = Field(ge=1, description="Number of repeated CV passes")
    inner_cv_folds: int = Field(ge=2, description="Nested CV folds for tuning")
    early_stopping_validation_fraction: float = Field(
        gt=0, lt=0.5, description="Train-fold fraction held out for booster early stopping"
    )
    artifact_dir: Path = Field(description="Directory for Phase 1 trained artifacts")
    metric_card_path: Path = Field(description="JSON metric card output path")
    residuals_path: Path = Field(description="CSV residual diagnostics output path")
    random_split_note: str = Field(description="Reminder about Kaggle random split limitation")
    elastic_net: ElasticNetTuningConfig
    lightgbm_tuning: LightGBMTuningConfig


class TargetConfig(BaseModel):
    column: str = Field(description="Name of the target column in the dataset")
    transform: str = Field(description="Transformation applied before modeling (log1p or none)")


class ConformalConfig(BaseModel):
    alpha: float = Field(gt=0, lt=1, description="Miscoverage level; intervals cover 1-alpha")
    calibration_split: float = Field(
        gt=0, lt=0.5, description="Fraction of full labeled set for conformal calibration"
    )
    test_split: float = Field(
        gt=0, lt=0.5, description="Fraction of full labeled set held out as the CQR test set"
    )
    secondary_alpha: float = Field(
        gt=0, lt=1, description="Secondary miscoverage level reported alongside the primary"
    )


class LightGBMConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0, lt=1)
    num_leaves: int = Field(gt=1)
    min_child_samples: int = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_alpha: float = Field(ge=0)
    reg_lambda: float = Field(ge=0)
    early_stopping_rounds: int = Field(gt=0)


class QuantileConfig(BaseModel):
    lower_alpha: float = Field(gt=0, lt=0.5, description="Lower quantile target for CQR")
    upper_alpha: float = Field(gt=0.5, lt=1, description="Upper quantile target for CQR")

    @model_validator(mode="after")
    def lower_must_be_less_than_upper(self) -> QuantileConfig:
        if self.lower_alpha >= self.upper_alpha:
            raise ValueError("lower_alpha must be less than upper_alpha")
        return self


class FeaturesConfig(BaseModel):
    drop: list[str] = Field(description="Columns to remove before modeling")


class ModelConfig(BaseModel):
    """Root model configuration; loaded from config/model.yaml."""

    global_seed: int
    data: DataPaths
    cross_validation: CrossValidationConfig
    phase1: Phase1Config
    target: TargetConfig
    conformal: ConformalConfig
    lightgbm: LightGBMConfig
    quantile: QuantileConfig
    features: FeaturesConfig


# ── Loaders ──────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent.parent  # src/margin_of_error/../../


def _load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return the parsed dict."""
    resolved = Path(path)
    if not resolved.is_absolute():
        # Resolve relative to repo root so make targets work from any CWD
        resolved = _REPO_ROOT / resolved
    with open(resolved) as fh:
        return yaml.safe_load(fh) or {}


def load_economics(path: Path | str = "config/economics.yaml") -> EconomicsConfig:
    """Load and validate economics configuration.

    Args:
        path: Path to economics.yaml, absolute or relative to repo root.

    Returns:
        Validated EconomicsConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If any value fails validation.
    """
    raw = _load_yaml(path)
    config = EconomicsConfig.model_validate(raw)
    logger.debug("Loaded economics config from %s", path)
    return config


def load_model_config(path: Path | str = "config/model.yaml") -> ModelConfig:
    """Load and validate model configuration.

    Args:
        path: Path to model.yaml, absolute or relative to repo root.

    Returns:
        Validated ModelConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If any value fails validation.
    """
    raw = _load_yaml(path)
    config = ModelConfig.model_validate(raw)
    logger.debug("Loaded model config from %s", path)
    return config
