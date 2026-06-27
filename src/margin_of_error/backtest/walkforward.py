"""Phase 4: Walk-forward temporal backtest.

Simulates an investor deploying the Phase 2 underwriting rule month-by-month
through the 2006–2010 Ames housing market (including the crash).

Walk-forward methodology:
    For each time step t (month):
        1. Train model on all sales before t (expanding window).
        2. Apply underwriting rule to properties sold at t.
        3. Record: which deals were underwritten, actual profit/loss,
           model calibration metrics.
    Report: cumulative P&L, decision curve, and calibration stability over time.

Important caveat documented here and in docs/decisions.md:
    The KAGGLE train/test split is RANDOM and is NOT used here. This phase
    uses ONLY the full De Cock Ames dataset sorted by YrSold/MoSold.
    Mixing the two splits would cause look-ahead bias.

PHASE 4 STATUS: Skeleton. Full implementation awaiting Phase 4 approval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestPeriod:
    """Results for a single time period in the walk-forward backtest.

    Attributes:
        yr_sold: Year of the period.
        mo_sold: Month of the period.
        n_available: Number of properties available for underwriting.
        n_underwritten: Number of deals underwritten.
        n_profitable: Number of underwritten deals that were actually profitable.
        total_profit: Realized profit on underwritten deals.
        median_interval_width: Median CQR interval width (in dollars).
        empirical_coverage: Fraction of actual prices inside the predicted interval.
    """

    yr_sold: int
    mo_sold: int
    n_available: int
    n_underwritten: int
    n_profitable: int
    total_profit: float
    median_interval_width: float
    empirical_coverage: float

    @property
    def hit_rate(self) -> float:
        """Fraction of underwritten deals that were profitable."""
        if self.n_underwritten == 0:
            return float("nan")
        return self.n_profitable / self.n_underwritten


@dataclass
class BacktestResult:
    """Aggregated results from the full walk-forward backtest.

    Attributes:
        periods: List of BacktestPeriod, one per (year, month) in the data.
        total_profit: Sum of profit across all underwritten deals.
        total_deals_underwritten: Total deals underwritten across all periods.
        total_deals_available: Total deals available across all periods.
    """

    periods: list[BacktestPeriod] = field(default_factory=list)

    @property
    def total_profit(self) -> float:
        return sum(p.total_profit for p in self.periods)

    @property
    def total_deals_underwritten(self) -> int:
        return sum(p.n_underwritten for p in self.periods)

    @property
    def total_deals_available(self) -> int:
        return sum(p.n_available for p in self.periods)

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame for plotting."""
        return pd.DataFrame(
            [
                {
                    "YrSold": p.yr_sold,
                    "MoSold": p.mo_sold,
                    "n_available": p.n_available,
                    "n_underwritten": p.n_underwritten,
                    "n_profitable": p.n_profitable,
                    "total_profit": p.total_profit,
                    "hit_rate": p.hit_rate,
                    "median_interval_width": p.median_interval_width,
                    "empirical_coverage": p.empirical_coverage,
                }
                for p in self.periods
            ]
        )


def run_backtest(
    df_full: pd.DataFrame,
    economics_config: Any,  # type: ignore[name-defined]
    model_config: Any,  # type: ignore[name-defined]
    min_train_periods: int = 12,
) -> BacktestResult:
    """Run the walk-forward backtest over the full Ames temporal dataset.

    Args:
        df_full: Full De Cock Ames DataFrame sorted by YrSold/MoSold.
        economics_config: EconomicsConfig from config/economics.yaml.
        model_config: ModelConfig from config/model.yaml.
        min_train_periods: Minimum months of data before first prediction.
                           Default 12: need at least one year of training data.

    Returns:
        BacktestResult with per-period metrics and aggregate statistics.
    """
    raise NotImplementedError("Phase 4 not yet implemented — awaiting approval")
