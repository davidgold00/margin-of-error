"""Phase 2 underwriting decision rule.

The core deliverable of Phase 2: turn a calibrated ARV interval and the flip
economics into an APPROVE / REFER / DECLINE verdict for a single property.

The guiding principle — the anti-Zillow guardrail — is that a flip is only
underwritable when the expected margin clears the model's own uncertainty by a
meaningful buffer. Concretely:

    1. If the 90% CQR interval is wider than the configured cap, DECLINE on
       *uncertainty* regardless of the point estimate. A model that doesn't know
       a home's value within that band cannot safely underwrite capital on it.
    2. Otherwise, APPROVE only when the profit distribution clears the margin
       buffer with high probability AND has low loss probability.
    3. REFER the in-between cases to a human.

All thresholds live in config/economics.yaml (``flip.underwriting``); none are
hardcoded here. See docs/decisions.md § ADR-013 for the threshold rationale.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal, cast

import pandas as pd

from margin_of_error.config import EconomicsConfig
from margin_of_error.economics.simulation import (
    ProfitSummary,
    maximum_allowable_offer,
    simulate_flip_profit,
)

logger = logging.getLogger(__name__)

Verdict = Literal["APPROVE", "DECLINE", "REFER"]
UpliftMode = Literal["none", "correlational", "causal", "config"]


class DeclineReason(StrEnum):
    """Mutually exclusive primary reasons a deal fails the underwriting gate."""

    EXCESSIVE_UNCERTAINTY = "Model uncertainty exceeds the acceptable interval width"
    HIGH_LOSS_PROBABILITY = "Probability of an outright loss is too high"
    INSUFFICIENT_MARGIN = "Too little chance of clearing the minimum profit buffer"


@dataclass(frozen=True)
class UnderwriteResult:
    """Typed, serializable outcome of underwriting one property at one tier."""

    verdict: Verdict
    renovation_tier: str
    purchase_price: float
    base_predicted_arv: float
    predicted_arv: float
    uplift_mode: str
    uplift_amount: float
    interval_low: float
    interval_high: float
    interval_width_dollars: float
    expected_profit: float
    profit_p10: float
    profit_p90: float
    prob_loss: float
    prob_above_min_margin: float
    primary_decline_reason: str | None
    decision_note: str

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly mapping (all fields are primitives)."""
        return asdict(self)


def _decide(
    interval_width: float,
    prob_above_min_margin: float,
    prob_loss: float,
    economics: EconomicsConfig,
) -> tuple[Verdict, DeclineReason | None]:
    """Apply the verdict logic. Excessive width is a hard DECLINE override."""
    uw = economics.flip.underwriting

    # Guardrail: if the model is too unsure about this home, decline regardless
    # of the point estimate. This is the core anti-Zillow rule and it overrides
    # an otherwise-acceptable margin (a deliberate strengthening of REFER).
    if interval_width > uw.max_acceptable_interval_width_usd:
        return "DECLINE", DeclineReason.EXCESSIVE_UNCERTAINTY

    approve = (
        prob_above_min_margin >= uw.approve_prob_above_min_margin
        and prob_loss <= uw.approve_prob_loss_max
    )
    if approve:
        return "APPROVE", None

    refer = (
        prob_above_min_margin >= uw.refer_prob_above_min_margin
        and prob_loss <= uw.refer_prob_loss_max
    )
    if refer:
        return "REFER", None

    # DECLINE: report the binding reason (loss risk takes priority over margin).
    if prob_loss > uw.refer_prob_loss_max:
        return "DECLINE", DeclineReason.HIGH_LOSS_PROBABILITY
    return "DECLINE", DeclineReason.INSUFFICIENT_MARGIN


def _decision_note(
    verdict: Verdict,
    reason: DeclineReason | None,
    tier: str,
    width: float,
    prob_above: float,
    prob_loss: float,
    expected_profit: float,
) -> str:
    """Human-readable, numbers-filled explanation of the verdict."""
    margin_pct = f"{prob_above:.0%}"
    loss_pct = f"{prob_loss:.0%}"
    width_str = f"${width:,.0f}"
    profit_str = f"${expected_profit:,.0f}"
    if verdict == "APPROVE":
        return (
            f"APPROVE ({tier} reno): {margin_pct} chance of clearing the profit buffer, "
            f"only {loss_pct} chance of loss, and the 90% value interval ({width_str}) is "
            f"tight enough to bet on. Expected profit {profit_str}."
        )
    if verdict == "REFER":
        return (
            f"REFER ({tier} reno): borderline — {margin_pct} chance of clearing the buffer "
            f"and {loss_pct} loss probability. Interval {width_str}. A human should review "
            f"before committing capital. Expected profit {profit_str}."
        )
    assert reason is not None
    if reason is DeclineReason.EXCESSIVE_UNCERTAINTY:
        return (
            f"DECLINE ({tier} reno): the model's 90% value interval is {width_str} — wider "
            f"than the acceptable band. The point estimate may look fine, but the model does "
            f"not know this home's value precisely enough to underwrite it. {reason.value}."
        )
    if reason is DeclineReason.HIGH_LOSS_PROBABILITY:
        return (
            f"DECLINE ({tier} reno): {loss_pct} probability of an outright loss is too high "
            f"despite a {width_str} interval. {reason.value}."
        )
    return (
        f"DECLINE ({tier} reno): only {margin_pct} chance of clearing the profit buffer "
        f"(expected profit {profit_str}). {reason.value}."
    )


def _resolve_uplift_mode(mode: UpliftMode, economics: EconomicsConfig) -> UpliftMode:
    """Resolve config-driven uplift mode into an explicit mode."""
    if mode != "config":
        return mode
    return "causal" if economics.flip.use_causal_uplifts else "correlational"


def causal_uplift_for_tier(renovation_tier: str, economics: EconomicsConfig) -> float:
    """Sum Phase 3 DML treatment uplifts for one renovation tier."""
    tier_features = economics.flip.causal_tier_uplift_features
    if renovation_tier not in tier_features:
        raise KeyError(
            f"No causal_tier_uplift_features entry for '{renovation_tier}'. "
            f"Known tiers: {sorted(tier_features)}"
        )
    values = economics.flip.causal_renovation_uplifts
    missing = [key for key in tier_features[renovation_tier] if values.get(key) is None]
    if missing:
        raise ValueError(
            f"Causal uplift values are not populated for tier '{renovation_tier}': {missing}"
        )
    total = 0.0
    for key in tier_features[renovation_tier]:
        value = values[key]
        if value is None:
            raise ValueError(f"Causal uplift value is not populated for {key}")
        total += float(value)
    return total


def _apply_renovation_uplift(
    arv_point: float,
    arv_lower: float,
    arv_upper: float,
    renovation_tier: str,
    economics: EconomicsConfig,
    uplift_mode: UpliftMode,
) -> tuple[float, float, float, str, float]:
    """Return ARV point/interval after applying the requested renovation uplift."""
    mode = _resolve_uplift_mode(uplift_mode, economics)
    if mode == "none":
        return arv_point, arv_lower, arv_upper, mode, 0.0

    tier = economics.flip.renovation_tiers[renovation_tier]
    if mode == "correlational":
        multiplier = 1.0 + tier.value_uplift_pct
        adjusted_point = arv_point * multiplier
        return (
            adjusted_point,
            arv_lower * multiplier,
            arv_upper * multiplier,
            mode,
            adjusted_point - arv_point,
        )

    uplift = causal_uplift_for_tier(renovation_tier, economics)
    return arv_point + uplift, arv_lower + uplift, arv_upper + uplift, mode, uplift


def underwrite(
    arv_point: float,
    arv_lower: float,
    arv_upper: float,
    renovation_tier: str,
    economics: EconomicsConfig,
    purchase_price: float | None = None,
    uplift_mode: UpliftMode = "none",
    seed: int = 42,
) -> UnderwriteResult:
    """Underwrite one property at one renovation tier.

    Pure function: given a calibrated ARV interval and the economics config, it
    returns a fully typed verdict. The ARV interval is expected to come from the
    primary (90%) CQR model; ``arv_point`` is the bias-corrected point estimate.

    Args:
        arv_point: Bias-corrected point estimate of after-repair value (dollars).
        arv_lower: Lower bound of the 90% CQR interval (dollars).
        arv_upper: Upper bound of the 90% CQR interval (dollars).
        renovation_tier: Key into ``economics.flip.renovation_tiers``.
        economics: Loaded EconomicsConfig.
        purchase_price: Explicit acquisition price; defaults to the '70% rule' MAO.
        seed: RNG seed for the profit Monte Carlo.

    Returns:
        UnderwriteResult with verdict, profit statistics, and a decision note.

    Raises:
        KeyError: If ``renovation_tier`` is not defined in the config.
    """
    tiers = economics.flip.renovation_tiers
    if renovation_tier not in tiers:
        raise KeyError(f"Unknown renovation tier '{renovation_tier}'. Known tiers: {sorted(tiers)}")
    renovation_cost = tiers[renovation_tier].cost_usd
    base_arv = arv_point
    arv_point, arv_lower, arv_upper, resolved_mode, uplift_amount = _apply_renovation_uplift(
        arv_point, arv_lower, arv_upper, renovation_tier, economics, uplift_mode
    )

    if purchase_price is None:
        purchase_price = maximum_allowable_offer(arv_point, renovation_cost, economics.flip)

    summary: ProfitSummary = simulate_flip_profit(
        arv_point=arv_point,
        arv_lower=arv_lower,
        arv_upper=arv_upper,
        renovation_cost=renovation_cost,
        economics=economics,
        purchase_price=purchase_price,
        seed=seed,
    )

    interval_width = arv_upper - arv_lower
    verdict, reason = _decide(
        interval_width, summary.prob_above_min_margin, summary.prob_loss, economics
    )
    note = _decision_note(
        verdict,
        reason,
        renovation_tier,
        interval_width,
        summary.prob_above_min_margin,
        summary.prob_loss,
        summary.mean_profit,
    )

    return UnderwriteResult(
        verdict=verdict,
        renovation_tier=renovation_tier,
        purchase_price=float(purchase_price),
        base_predicted_arv=float(base_arv),
        predicted_arv=float(arv_point),
        uplift_mode=resolved_mode,
        uplift_amount=float(uplift_amount),
        interval_low=float(arv_lower),
        interval_high=float(arv_upper),
        interval_width_dollars=float(interval_width),
        expected_profit=summary.mean_profit,
        profit_p10=summary.profit_p10,
        profit_p90=summary.profit_p90,
        prob_loss=summary.prob_loss,
        prob_above_min_margin=summary.prob_above_min_margin,
        primary_decline_reason=None if reason is None else reason.value,
        decision_note=note,
    )


def underwrite_best_tier(
    arv_point: float,
    arv_lower: float,
    arv_upper: float,
    economics: EconomicsConfig,
    uplift_mode: UpliftMode = "none",
    seed: int = 42,
) -> UnderwriteResult:
    """Underwrite a property at the renovation tier that maximizes expected profit.

    A rational investor chooses the best renovation plan; the dataset-wide pass
    uses this to assign one verdict per home. Ties break toward the cheaper tier.
    """
    candidates = [
        underwrite(
            arv_point,
            arv_lower,
            arv_upper,
            tier,
            economics,
            uplift_mode=uplift_mode,
            seed=seed,
        )
        for tier in economics.flip.renovation_tiers
    ]
    return max(candidates, key=lambda r: r.expected_profit)


def detect_verdict_flips(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows where correlational and causal underwriting verdicts differ."""
    required = {"correlational_verdict", "causal_verdict"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing verdict comparison columns: {sorted(missing)}")

    flips = frame.loc[frame["correlational_verdict"] != frame["causal_verdict"]].copy()
    if flips.empty:
        flips["flip_direction"] = pd.Series(dtype=str)
        return flips
    flips["flip_direction"] = [
        f"correlational_{correlational}_to_causal_{causal}"
        for correlational, causal in zip(
            flips["correlational_verdict"].astype(str),
            flips["causal_verdict"].astype(str),
            strict=False,
        )
    ]
    return flips


def build_underwriting_comparison(
    frame: pd.DataFrame,
    economics: EconomicsConfig,
    seed: int = 42,
) -> pd.DataFrame:
    """Score homes under Phase 2 correlational and Phase 3 causal uplift modes."""
    records: list[dict[str, object]] = []
    for offset, (_row_index, row) in enumerate(frame.reset_index(drop=True).iterrows()):
        arv_point = float(row["predicted_arv"])
        arv_lower = float(row["interval_low_90"])
        arv_upper = float(row["interval_high_90"])
        row_seed = seed + offset
        correlational = underwrite_best_tier(
            arv_point,
            arv_lower,
            arv_upper,
            economics,
            uplift_mode="correlational",
            seed=row_seed,
        )
        causal = underwrite_best_tier(
            arv_point,
            arv_lower,
            arv_upper,
            economics,
            uplift_mode="causal",
            seed=row_seed,
        )
        records.append(
            {
                "Id": int(cast(Any, row["Id"])) if "Id" in row.index else offset,
                "Neighborhood": row.get("Neighborhood", "Unknown"),
                "base_predicted_arv": float(row["predicted_arv"]),
                "interval_low_90": float(row["interval_low_90"]),
                "interval_high_90": float(row["interval_high_90"]),
                "correlational_verdict": correlational.verdict,
                "causal_verdict": causal.verdict,
                "correlational_tier": correlational.renovation_tier,
                "causal_tier": causal.renovation_tier,
                "correlational_uplift": correlational.uplift_amount,
                "causal_uplift": causal.uplift_amount,
                "correlational_expected_profit": correlational.expected_profit,
                "causal_expected_profit": causal.expected_profit,
                "expected_profit_delta": causal.expected_profit - correlational.expected_profit,
                "correlational_prob_loss": correlational.prob_loss,
                "causal_prob_loss": causal.prob_loss,
                "verdict_changed": correlational.verdict != causal.verdict,
            }
        )
    comparison = pd.DataFrame(records)
    flips = detect_verdict_flips(comparison)
    if not flips.empty:
        comparison = comparison.merge(flips[["Id", "flip_direction"]], on="Id", how="left")
    else:
        comparison["flip_direction"] = pd.NA
    return comparison
