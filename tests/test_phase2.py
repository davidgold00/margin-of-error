"""Phase 2 economics tests: profit Monte Carlo, the underwriting rule, config keys.

These are fast (no model fitting) — they exercise the decision logic and the
'no economic constant lives outside config' guarantee.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from margin_of_error.economics.simulation import (
    ProfitSummary,
    maximum_allowable_offer,
    simulate_flip_profit,
)
from margin_of_error.economics.underwriter import (
    UnderwriteResult,
    underwrite,
    underwrite_best_tier,
)

# ── Profit Monte Carlo ────────────────────────────────────────────────────────


def test_profit_monte_carlo_probabilities_are_valid(economics_config) -> None:
    """P(loss) and P(above buffer) are valid probabilities in [0, 1]."""
    summary = simulate_flip_profit(
        arv_point=200_000,
        arv_lower=185_000,
        arv_upper=215_000,
        renovation_cost=25_000,
        economics=economics_config,
        seed=1,
    )
    assert isinstance(summary, ProfitSummary)
    assert 0.0 <= summary.prob_loss <= 1.0
    assert 0.0 <= summary.prob_above_min_margin <= 1.0
    assert summary.profit_p10 <= summary.mean_profit <= summary.profit_p90


def test_mean_prob_loss_across_homes_is_a_probability(economics_config) -> None:
    """Sanity: averaging P(loss) over several homes stays within [0, 1]."""
    rng = np.random.default_rng(0)
    probs = []
    for _ in range(8):
        point = float(rng.uniform(120_000, 300_000))
        half = float(rng.uniform(15_000, 60_000))
        probs.append(
            simulate_flip_profit(
                point, point - half, point + half, 25_000, economics_config, seed=2
            ).prob_loss
        )
    assert 0.0 <= float(np.mean(probs)) <= 1.0


def test_wider_interval_increases_loss_probability(economics_config) -> None:
    """More model uncertainty (a wider interval) must not reduce loss probability."""
    tight = simulate_flip_profit(200_000, 190_000, 210_000, 25_000, economics_config, seed=3)
    wide = simulate_flip_profit(200_000, 150_000, 250_000, 25_000, economics_config, seed=3)
    assert wide.prob_loss >= tight.prob_loss


def test_maximum_allowable_offer_clamps_at_zero(economics_config) -> None:
    """A renovation cost above the allowable basis yields a non-negative offer."""
    offer = maximum_allowable_offer(50_000, 200_000, economics_config.flip)
    assert offer == 0.0


# ── The underwriting decision rule ────────────────────────────────────────────


def test_wide_interval_declines_despite_positive_point_estimate(economics_config) -> None:
    """THE anti-Zillow scenario: a wide interval declines even with a great point.

    The point estimate looks very profitable, but the model's 90% interval is far
    wider than the acceptable band, so the deal must DECLINE on uncertainty.
    """
    result = underwrite(
        arv_point=250_000,
        arv_lower=150_000,  # width = 150k, far above the configured cap
        arv_upper=300_000,
        renovation_tier="moderate",
        economics=economics_config,
    )
    assert result.verdict == "DECLINE"
    assert result.interval_width_dollars > (
        economics_config.flip.underwriting.max_acceptable_interval_width_usd
    )
    assert "uncertainty" in (result.primary_decline_reason or "").lower()


def test_approve_requires_both_margin_and_uncertainty_conditions(economics_config) -> None:
    """APPROVE demands BOTH a tight interval AND a strong profit distribution.

    A tight-interval, healthy-margin home approves; widening the interval past the
    cap (holding the point estimate fixed) must flip the verdict away from APPROVE.
    """
    tight = underwrite(220_000, 205_000, 235_000, "moderate", economics_config)
    assert tight.verdict == "APPROVE"

    cap = economics_config.flip.underwriting.max_acceptable_interval_width_usd
    widened = underwrite(
        220_000,
        220_000 - cap,  # width strictly greater than the cap
        220_000 + cap,
        "moderate",
        economics_config,
    )
    assert widened.verdict != "APPROVE"


def test_underwrite_result_is_fully_typed_and_serializable(economics_config) -> None:
    """UnderwriteResult must round-trip through JSON with only primitive fields."""
    result = underwrite_best_tier(200_000, 188_000, 212_000, economics_config)
    assert isinstance(result, UnderwriteResult)
    payload = result.to_dict()
    encoded = json.dumps(payload)  # raises if any field is non-serializable
    decoded = json.loads(encoded)
    assert decoded["verdict"] in {"APPROVE", "REFER", "DECLINE"}
    assert isinstance(decoded["expected_profit"], float)
    assert isinstance(decoded["decision_note"], str)


def test_unknown_renovation_tier_raises(economics_config) -> None:
    with pytest.raises(KeyError):
        underwrite(200_000, 190_000, 210_000, "platinum", economics_config)


# ── Config completeness and the no-hardcoded-constants guarantee ──────────────

REQUIRED_FLIP_KEYS = {
    "acquisition_arv_factor",
    "transaction_cost_pct",
    "holding_cost_monthly_pct",
    "holding_period_months_base",
    "holding_period_months_std",
    "holding_period_months_min",
    "holding_period_months_max",
    "financing_assumption",
    "monte_carlo_samples",
    "arv_normal_z",
    "renovation_tiers",
    "underwriting",
}
REQUIRED_UNDERWRITING_KEYS = {
    "minimum_underwrite_margin_buffer_usd",
    "max_acceptable_interval_width_usd",
    "approve_prob_above_min_margin",
    "approve_prob_loss_max",
    "refer_prob_above_min_margin",
    "refer_prob_loss_max",
}


def test_economics_config_loads_all_required_keys(economics_config) -> None:
    """Every key the Phase 2 engine relies on is present and typed."""
    flip = economics_config.flip
    dumped = flip.model_dump()
    assert REQUIRED_FLIP_KEYS.issubset(dumped)
    assert REQUIRED_UNDERWRITING_KEYS.issubset(flip.underwriting.model_dump())
    assert set(flip.renovation_tiers) == {"minimal", "moderate", "substantial"}
    for tier in flip.renovation_tiers.values():
        assert tier.cost_usd > 0
        assert 0 <= tier.value_uplift_pct < 1


# Distinctive economic literals that must live ONLY in config.py, never inline.
_FORBIDDEN_ECONOMIC_LITERALS = {0.008, 0.06, 8000.0, 25000.0, 60000.0, 15000.0}


def _numeric_literals(path: Path) -> set[float]:
    tree = ast.parse(path.read_text())
    found: set[float] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            found.add(float(node.value))
    return found


def test_no_economic_constants_outside_config(repo_root: Path) -> None:
    """Fail if any distinctive economic value is hardcoded in src/ (except config.py).

    Parses each module's AST for numeric literals and checks none match a known
    economic constant — those belong exclusively in config/economics.yaml.
    """
    src = repo_root / "src" / "margin_of_error"
    offenders: dict[str, set[float]] = {}
    for py in src.rglob("*.py"):
        if py.name == "config.py":
            continue
        hits = _numeric_literals(py) & _FORBIDDEN_ECONOMIC_LITERALS
        if hits:
            offenders[str(py.relative_to(repo_root))] = hits
    assert not offenders, f"Hardcoded economic constants found: {offenders}"
