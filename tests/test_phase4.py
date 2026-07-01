"""Tests for the Phase 4 walk-forward crash backtest.

Fast tests exercise the aggregation math (drawdown, hit rate, coverage weighting,
cumulative P&L) on hand-built results — no model fitting. One `slow` test runs the
real backtest on the full Ames dataset and asserts the no-look-ahead invariant plus
structural sanity checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from margin_of_error.backtest.walkforward import (
    STRATEGY_KEYS,
    STRATEGY_LABELS,
    BacktestPeriod,
    BacktestResult,
    StrategyPeriod,
    _flip_fixed_costs,
    _max_drawdown,
    _period_key,
)

# ── Pure-math unit tests ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        ([], 0.0),
        ([10.0, 20.0, 30.0], 0.0),  # monotonically rising → no drawdown
        ([0.0, 50.0, 30.0, 80.0], 20.0),  # dip of 20 from the 50 peak
        ([0.0, -10.0, -25.0], 25.0),  # straight decline
        ([100.0, 0.0, 150.0, 50.0], 100.0),  # two 100-deep troughs
    ],
)
def test_max_drawdown(series: list[float], expected: float) -> None:
    assert _max_drawdown(np.array(series, dtype=float)) == pytest.approx(expected)


def test_flip_fixed_costs_matches_config(economics_config) -> None:
    """Fixed costs = transaction + expected-hold carry, both on the purchase price."""
    flip = economics_config.flip
    purchase = 120_000.0
    expected = purchase * flip.transaction_cost_pct + (
        purchase * flip.holding_cost_monthly_pct * flip.holding_period_months_base
    )
    assert _flip_fixed_costs(purchase, flip) == pytest.approx(expected)


def test_strategy_labels_cover_all_keys() -> None:
    assert set(STRATEGY_KEYS) == set(STRATEGY_LABELS)
    assert STRATEGY_KEYS[0] == "buy_all"


def test_period_key_orders_chronologically() -> None:
    assert _period_key(2007, 1) < _period_key(2007, 12) < _period_key(2008, 1)


# ── Aggregation tests on synthetic results ─────────────────────────────────────


def _strategy(n_bought: int, n_profitable: int, realized: float) -> StrategyPeriod:
    return StrategyPeriod(n_bought=n_bought, n_profitable=n_profitable, realized_profit=realized)


_REGIME = "flip"


def _period(
    yr: int,
    mo: int,
    *,
    is_crash: bool,
    coverage: float,
    n_available: int,
    buy_all: StrategyPeriod,
    naive: StrategyPeriod,
    uncertainty: StrategyPeriod,
) -> BacktestPeriod:
    return BacktestPeriod(
        yr_sold=yr,
        mo_sold=mo,
        n_available=n_available,
        median_interval_width=50_000.0,
        empirical_coverage=coverage,
        mean_predicted_arv=200_000.0,
        mean_actual_price=190_000.0,
        is_crash=is_crash,
        regimes={
            _REGIME: {
                "buy_all": buy_all,
                "naive_point": naive,
                "uncertainty_aware": uncertainty,
            }
        },
    )


def _sample_result() -> BacktestResult:
    p1 = _period(
        2007,
        1,
        is_crash=False,
        coverage=0.90,
        n_available=100,
        buy_all=_strategy(2, 2, 100.0),
        naive=_strategy(2, 2, 100.0),
        uncertainty=_strategy(1, 1, 60.0),
    )
    p2 = _period(
        2008,
        1,
        is_crash=True,
        coverage=0.70,
        n_available=100,
        buy_all=_strategy(2, 0, -50.0),
        naive=_strategy(2, 0, -50.0),
        uncertainty=_strategy(0, 0, 0.0),
    )
    return BacktestResult(
        periods=[p1, p2],
        nominal_coverage=0.90,
        crash_window=(2008, 2010),
        regime_factors={_REGIME: 0.70},
    )


def test_to_dataframe_has_cumulative_and_strategy_columns() -> None:
    frame = _sample_result().to_dataframe()
    assert list(frame["period_index"]) == [0, 1]
    for key in STRATEGY_KEYS:
        assert f"{_REGIME}__{key}_cum_profit" in frame.columns
    # buy_all realized 100 then -50 → cumulative 100 then 50
    assert list(frame[f"{_REGIME}__buy_all_cum_profit"]) == pytest.approx([100.0, 50.0])


def test_strategy_summary_drawdown_and_hit_rate() -> None:
    summary = _sample_result().strategy_summary()[_REGIME]
    buy_all = summary["buy_all"]
    assert buy_all["total_realized_profit"] == pytest.approx(50.0)
    assert buy_all["total_deals_bought"] == pytest.approx(4.0)
    assert buy_all["overall_hit_rate"] == pytest.approx(0.5)  # 2 of 4 profitable
    assert buy_all["max_drawdown"] == pytest.approx(50.0)  # 100 → 50 dip
    # The disciplined rule bought fewer deals and avoided the crash-month loss.
    assert summary["uncertainty_aware"]["total_deals_bought"] == pytest.approx(1.0)
    assert summary["uncertainty_aware"]["max_drawdown"] == pytest.approx(0.0)


def test_coverage_summary_weighting_and_collapse_flag() -> None:
    cov = _sample_result().coverage_summary()
    assert cov["nominal"] == pytest.approx(0.90)
    # Equal weights (100 each): overall = mean(0.90, 0.70) = 0.80
    assert cov["overall_empirical"] == pytest.approx(0.80)
    assert cov["pre_crash_empirical"] == pytest.approx(0.90)
    assert cov["crash_empirical"] == pytest.approx(0.70)
    assert cov["coverage_collapsed"] is True


def test_coverage_not_collapsed_when_crash_coverage_holds() -> None:
    p1 = _period(
        2007,
        1,
        is_crash=False,
        coverage=0.91,
        n_available=50,
        buy_all=_strategy(1, 1, 10.0),
        naive=_strategy(1, 1, 10.0),
        uncertainty=_strategy(1, 1, 10.0),
    )
    p2 = _period(
        2008,
        1,
        is_crash=True,
        coverage=0.92,
        n_available=50,
        buy_all=_strategy(1, 1, 10.0),
        naive=_strategy(1, 1, 10.0),
        uncertainty=_strategy(1, 1, 10.0),
    )
    cov = BacktestResult(
        periods=[p1, p2], nominal_coverage=0.90, regime_factors={_REGIME: 0.70}
    ).coverage_summary()
    assert cov["coverage_collapsed"] is False


# ── Slow end-to-end test on real data (no-look-ahead guardrail) ────────────────


@pytest.mark.slow
def test_run_backtest_end_to_end_no_lookahead(
    repo_root: Path, model_config, economics_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real backtest; assert generations never train on their own/future year."""
    ames_path = repo_root / "data" / "raw" / "ames" / "AmesHousing.csv"
    if not ames_path.exists():
        pytest.skip("full Ames dataset not present; run data download per data/README.md")

    from margin_of_error.backtest import walkforward as wf
    from margin_of_error.data.loaders import load_ames_full
    from margin_of_error.data.schemas import validate_ames_full

    raw = validate_ames_full(load_ames_full(ames_path))

    original = wf.train_generation

    def spy(X_past, y_past_log, config, trained_through_key):  # type: ignore[no-untyped-def]
        # Annual generations key on the year; training data must be strictly earlier.
        max_train_year = int(X_past["YrSold"].max())
        assert max_train_year < trained_through_key, (
            f"look-ahead: generation {trained_through_key} trained on year {max_train_year}"
        )
        return original(X_past, y_past_log, config, trained_through_key)

    monkeypatch.setattr(wf, "train_generation", spy)
    result = wf.run_backtest(raw, model_config, economics_config)

    assert len(result.periods) > 0
    summary = result.strategy_summary()
    assert set(summary) == set(economics_config.flip.backtest_acquisition_regimes)
    for regime in summary:
        per = summary[regime]
        assert set(per) == set(STRATEGY_KEYS)
        # Buy-All underwrites everything, so it buys >= any gated strategy.
        assert (
            per["buy_all"]["total_deals_bought"] >= per["uncertainty_aware"]["total_deals_bought"]
        )
        assert per["buy_all"]["total_deals_bought"] >= per["naive_point"]["total_deals_bought"]
        assert all(per[k]["max_drawdown"] >= 0 for k in STRATEGY_KEYS)

    cov = result.coverage_summary()
    assert {"nominal", "overall_empirical", "pre_crash_empirical", "crash_empirical"} <= set(cov)
    assert 0.0 <= float(cov["overall_empirical"]) <= 1.0
