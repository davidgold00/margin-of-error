"""Fix-and-flip P&L simulation.

Converts a CQR prediction interval (ARV distribution) into a profit
distribution for a given purchase price and renovation plan. All economic
parameters come from config/economics.yaml.

The profit model:

    Revenue  = ARV  (after-repair value — sampled from model interval)
    Costs    = Purchase price
             + Renovation cost (from renovation plan)
             + Acquisition costs (buy_side_pct × purchase)
             + Selling costs (sell_side_pct × ARV)
             + Holding costs (monthly_cost_pct × purchase × hold_months)
             + Financing costs (hard_money_rate × ltv × purchase × hold_months/12)
    Profit   = Revenue - Costs

The underwriting rule:
    UNDERWRITE if:
        - Point estimate profit > min_absolute_usd AND
        - Point estimate profit > ARV × minimum_margin_pct AND
        - (Point estimate profit - uncertainty_band) > 0
          (i.e., even in the pessimistic scenario, profit > 0)

    DECLINE otherwise, with reason reported.

PHASE 2 STATUS: Skeleton with documented P&L formula. Full implementation
awaiting Phase 2 approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import NamedTuple, cast

import numpy as np

from margin_of_error.config import EconomicsConfig, FlipConfig


class UnderwriteDecision(StrEnum):
    """Possible outcomes from the underwriting decision rule."""

    UNDERWRITE = "UNDERWRITE"
    DECLINE = "DECLINE"


class UnderwiteReason(StrEnum):
    """Human-readable reason for a DECLINE decision."""

    PROFIT_BELOW_FLOOR = "Projected profit below minimum dollar floor"
    MARGIN_BELOW_THRESHOLD = "Projected margin below minimum margin %"
    UNCERTAINTY_EXCEEDS_MARGIN = "Model uncertainty exceeds projected profit (key risk)"
    NEGATIVE_EXPECTED_PROFIT = "Expected profit is negative"


@dataclass
class FlipPnL:
    """Full P&L breakdown for a single flip scenario.

    All dollar values are in nominal dollars (not discounted).

    Attributes:
        purchase_price: Acquisition price.
        arv: After-repair value (a sampled point from the price distribution).
        renovation_cost: Total renovation budget from plan.
        acquisition_cost: buy_side_pct × purchase_price.
        selling_cost: sell_side_pct × arv.
        holding_cost: monthly_cost_pct × purchase_price × hold_months.
        financing_cost: hard_money_rate × ltv × purchase_price × (hold_months/12).
        total_cost: Sum of all costs.
        profit: arv - total_cost.
        profit_pct_arv: profit / arv.
    """

    purchase_price: float
    arv: float
    renovation_cost: float
    acquisition_cost: float
    selling_cost: float
    holding_cost: float
    financing_cost: float
    total_cost: float
    profit: float
    profit_pct_arv: float


class UnderwritingVerdict(NamedTuple):
    """Result of the underwriting decision function.

    Attributes:
        decision: UNDERWRITE or DECLINE.
        reason: Human-readable reason (None if UNDERWRITE).
        expected_profit: Dollar profit at the point estimate ARV.
        profit_at_lower_bound: Dollar profit if ARV = lower interval bound.
        uncertainty_band_dollars: Width of the CQR prediction interval in $.
        profit_distribution_p10: 10th percentile of profit distribution ($.
        profit_distribution_p50: Median profit ($.
        profit_distribution_p90: 90th percentile of profit ($.
    """

    decision: UnderwriteDecision
    reason: str | None
    expected_profit: float
    profit_at_lower_bound: float
    uncertainty_band_dollars: float
    profit_distribution_p10: float
    profit_distribution_p50: float
    profit_distribution_p90: float


def compute_flip_pnl(
    purchase_price: float,
    arv: float,
    renovation_cost: float,
    economics: EconomicsConfig,
) -> FlipPnL:
    """Compute a single P&L scenario for given ARV and purchase price.

    Args:
        purchase_price: Acquisition price in dollars.
        arv: After-repair value (modeled sale price after renovation).
        renovation_cost: Total renovation budget in dollars.
        economics: EconomicsConfig loaded from config/economics.yaml.

    Returns:
        FlipPnL with itemized cost breakdown and profit.
    """
    acq_cost = economics.transaction.buy_side_pct * purchase_price
    sell_cost = economics.transaction.sell_side_pct * arv
    hold_cost = (
        economics.holding.monthly_cost_pct * purchase_price * economics.holding.typical_hold_months
    )
    fin_cost = (
        economics.financing.hard_money_rate
        * economics.financing.ltv
        * purchase_price
        * (economics.holding.typical_hold_months / 12)
    )

    total_cost = purchase_price + renovation_cost + acq_cost + sell_cost + hold_cost + fin_cost
    profit = arv - total_cost
    profit_pct_arv = profit / arv if arv > 0 else float("-inf")

    return FlipPnL(
        purchase_price=purchase_price,
        arv=arv,
        renovation_cost=renovation_cost,
        acquisition_cost=acq_cost,
        selling_cost=sell_cost,
        holding_cost=hold_cost,
        financing_cost=fin_cost,
        total_cost=total_cost,
        profit=profit,
        profit_pct_arv=profit_pct_arv,
    )


def simulate_profit_distribution(
    purchase_price: float,
    arv_lower: float,
    arv_point: float,
    arv_upper: float,
    renovation_cost: float,
    economics: EconomicsConfig,
    n_samples: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Monte Carlo profit distribution by sampling from the ARV interval.

    Models ARV as a uniform distribution over [arv_lower, arv_upper] as a
    conservative, assumption-free approximation. The CQR interval provides
    marginal coverage guarantee; the uniform sampling is a PLACEHOLDER
    distribution — Phase 2 decision note should document if this is changed
    to something more principled (e.g., truncated normal).

    Args:
        purchase_price: Acquisition price in dollars.
        arv_lower: Lower bound of CQR prediction interval (dollars).
        arv_point: Point estimate ARV (dollars).
        arv_upper: Upper bound of CQR prediction interval (dollars).
        renovation_cost: Total renovation budget in dollars.
        economics: EconomicsConfig instance.
        n_samples: Number of Monte Carlo samples.
        seed: RNG seed for reproducibility.

    Returns:
        Array of n_samples profit values in dollars.
    """
    rng = np.random.default_rng(seed)
    arv_samples = rng.uniform(arv_lower, arv_upper, size=n_samples)
    profits = np.array(
        [
            compute_flip_pnl(purchase_price, arv, renovation_cost, economics).profit
            for arv in arv_samples
        ]
    )
    return profits


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 profit Monte Carlo (the engine the underwriter consumes)
#
# Profit model (per the Phase 2 brief), all in nominal dollars:
#     net_profit = ARV
#                - purchase_price
#                - renovation_cost
#                - purchase_price * transaction_cost_pct
#                - purchase_price * holding_cost_monthly_pct * holding_period_months
#
# ARV is sampled from the CQR interval as ~Normal(mean=arv_point, std=(U-L)/(2z)).
# holding_period_months is sampled from a truncated Normal. purchase_price is set
# by the "70% rule" Maximum Allowable Offer. See docs/decisions.md § ADR-012.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProfitSummary:
    """Summary statistics of a per-property profit distribution.

    Only summary stats are stored (never the 10,000 raw draws), so the
    dataset-wide pass stays memory-light.
    """

    purchase_price: float
    renovation_cost: float
    mean_profit: float
    std_profit: float
    profit_p10: float
    profit_p90: float
    prob_loss: float
    prob_above_min_margin: float

    def to_dict(self) -> dict[str, float]:
        """JSON/serialization-friendly mapping of the summary."""
        return asdict(self)


def maximum_allowable_offer(arv_point: float, renovation_cost: float, flip: FlipConfig) -> float:
    """Purchase price under the flip '70% rule': factor*ARV - renovation cost.

    Returns a non-negative offer (clamped at 0 for homes where renovation cost
    alone exceeds the allowable basis).
    """
    return max(flip.acquisition_arv_factor * arv_point - renovation_cost, 0.0)


def arv_std_from_interval(arv_lower: float, arv_upper: float, flip: FlipConfig) -> float:
    """Map a (U - L) interval width to a working Normal std for ARV sampling.

    std = (U - L) / (2 * arv_normal_z). Acknowledged simplification: the CQR
    interval is distribution-free, but a Normal is a convenient sampling proxy.
    """
    return max((arv_upper - arv_lower) / (2.0 * flip.arv_normal_z), 0.0)


def _sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Sample a truncated Normal by resampling out-of-bounds draws.

    With the configured holding-period parameters the truncation mass is small,
    so a bounded resampling loop converges in a couple of passes.
    """
    if std <= 0:
        return cast(np.ndarray, np.clip(np.full(size, mean), low, high))
    out = rng.normal(mean, std, size=size)
    for _ in range(64):
        bad = (out < low) | (out > high)
        n_bad = int(np.count_nonzero(bad))
        if n_bad == 0:
            break
        out[bad] = rng.normal(mean, std, size=n_bad)
    return cast(np.ndarray, np.clip(out, low, high))


def simulate_flip_profit(
    arv_point: float,
    arv_lower: float,
    arv_upper: float,
    renovation_cost: float,
    economics: EconomicsConfig,
    purchase_price: float | None = None,
    seed: int = 42,
) -> ProfitSummary:
    """Monte Carlo profit distribution for one property at one renovation tier.

    Args:
        arv_point: Point estimate of after-repair value (dollars).
        arv_lower: Lower bound of the 90% CQR interval (dollars).
        arv_upper: Upper bound of the 90% CQR interval (dollars).
        renovation_cost: Renovation budget for the chosen tier (dollars).
        economics: Loaded EconomicsConfig (uses the ``flip`` block).
        purchase_price: Explicit acquisition price. If None, the '70% rule'
            Maximum Allowable Offer is used.
        seed: RNG seed for reproducibility.

    Returns:
        ProfitSummary with mean/std/p10/p90/P(loss)/P(profit > buffer).
    """
    flip = economics.flip
    n = flip.monte_carlo_samples
    buffer = flip.underwriting.minimum_underwrite_margin_buffer_usd
    if purchase_price is None:
        purchase_price = maximum_allowable_offer(arv_point, renovation_cost, flip)

    rng = np.random.default_rng(seed)
    arv_std = arv_std_from_interval(arv_lower, arv_upper, flip)
    arv_samples = rng.normal(arv_point, arv_std, size=n) if arv_std > 0 else np.full(n, arv_point)
    hold_samples = _sample_truncated_normal(
        rng,
        flip.holding_period_months_base,
        flip.holding_period_months_std,
        flip.holding_period_months_min,
        flip.holding_period_months_max,
        size=n,
    )

    transaction_cost = purchase_price * flip.transaction_cost_pct
    holding_cost = purchase_price * flip.holding_cost_monthly_pct * hold_samples
    profit = arv_samples - purchase_price - renovation_cost - transaction_cost - holding_cost

    return ProfitSummary(
        purchase_price=float(purchase_price),
        renovation_cost=float(renovation_cost),
        mean_profit=float(np.mean(profit)),
        std_profit=float(np.std(profit)),
        profit_p10=float(np.percentile(profit, 10)),
        profit_p90=float(np.percentile(profit, 90)),
        prob_loss=float(np.mean(profit < 0)),
        prob_above_min_margin=float(np.mean(profit > buffer)),
    )


def underwrite(
    purchase_price: float,
    arv_lower: float,
    arv_point: float,
    arv_upper: float,
    renovation_cost: float,
    economics: EconomicsConfig,
    seed: int = 42,
) -> UnderwritingVerdict:
    """Apply the underwriting decision rule to a single property.

    This is the core deliverable of Phase 2. Returns both a binary decision
    and a full P&L breakdown for reporting.

    Decision rule (all three must pass to UNDERWRITE):
        1. Expected profit > min_absolute_usd
        2. Expected profit > arv_point × minimum_margin_pct
        3. Profit at lower ARV bound > 0 (uncertainty-adjusted check)

    Args:
        purchase_price: Acquisition price in dollars.
        arv_lower: Lower bound of CQR prediction interval (dollars).
        arv_point: Point estimate ARV (dollars).
        arv_upper: Upper bound of CQR prediction interval (dollars).
        renovation_cost: Total renovation budget in dollars.
        economics: EconomicsConfig instance.
        seed: RNG seed for profit distribution sampling.

    Returns:
        UnderwritingVerdict with decision, reason, and P&L statistics.
    """
    raise NotImplementedError("Phase 2 not yet implemented — awaiting approval")
