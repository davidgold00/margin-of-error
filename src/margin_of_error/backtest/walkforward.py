"""Phase 4: Walk-forward temporal backtest through the 2006–2010 Ames market.

Simulates an investor deploying an AVM-driven flip-underwriting rule month by
month across the full De Cock Ames dataset — including the post-2007 downturn.

Walk-forward methodology (see docs/decisions.md § ADR-022..ADR-026):
    * Sort every sale by (YrSold, MoSold) and never use a future sale to inform a
      past decision. The RANDOM Kaggle split is deliberately NOT used here
      (ADR-005); mixing it in would leak future information.
    * Retrain the point + CQR stack on an EXPANDING window of past-only data
      (annually by default). Each model generation trained on years < Y scores the
      homes selling in year Y — a strict out-of-time evaluation.
    * At each retrain the past pool is split into a fit fold (quantile arms + point
      model) and a calibration fold (conformal Q̂), so the intervals are
      calibrated only on data the arms never saw.

Two ACQUISITION REGIMES are compared to show WHEN uncertainty discipline pays
(ADR-025): a conservative 70%-rule flip (fat margin) and an aggressive iBuyer that
buys near model value (thin margin, ≈ how Zillow Offers operated). Within each
regime, three strategies with identical pricing differ only in the buy gate:
    1. buy_all           — underwrite every available home (undisciplined baseline).
    2. naive_point       — buy whenever the POINT estimate shows a profit (trusts the
                           number; ignores the interval — the Zillow rule).
    3. uncertainty_aware — buy only when the model is confident enough: the 90% interval
                           is within the width cap AND the modeled loss probability is
                           within tolerance (the regime-agnostic core of ADR-013).

Realized P&L treats the observed sale price as the realized after-repair value and
the acquisition as synthetic (the dataset has no true buy/resell pairs), so this is
an underwriting-rule STRESS TEST of valuation/regime risk, not a live trading P&L.
Renovation is out of scope for the backtest (ADR-024): with no renovated-resale
counterfactual, folding in an uplift would require inventing a sale price.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from margin_of_error.config import (
    BacktestConfig,
    EconomicsConfig,
    FlipConfig,
    ModelConfig,
    load_economics,
    load_model_config,
)
from margin_of_error.data.loaders import load_ames_full
from margin_of_error.data.schemas import validate_ames_full
from margin_of_error.economics.simulation import simulate_flip_profit
from margin_of_error.features.preprocessing import build_preprocessor
from margin_of_error.models.baseline import (
    EarlyStoppingLightGBMRegressor,
    duan_smearing_factor,
    log_predictions_to_dollars,
    make_target,
    resolve_repo_path,
)
from margin_of_error.models.conformal import CQRModel, CQRResult

logger = logging.getLogger(__name__)

# Strategy identifiers and their human-readable labels (used in charts + reports).
STRATEGY_KEYS: tuple[str, ...] = ("buy_all", "naive_point", "uncertainty_aware")
STRATEGY_LABELS: dict[str, str] = {
    "buy_all": "Buy-All (no gate)",
    "naive_point": "Naive point-estimate",
    "uncertainty_aware": "Uncertainty-aware (ours)",
}
# Which regime is the signature (Figure 4B) vs. the "margin dominates" contrast.
SIGNATURE_REGIME = "ibuyer"
CONTRAST_REGIME = "conservative_flip"


# ── Result containers ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyPeriod:
    """One strategy's outcome within a single (year, month) period and regime."""

    n_bought: int
    n_profitable: int
    realized_profit: float

    @property
    def hit_rate(self) -> float:
        """Fraction of bought deals that were actually profitable (NaN if none)."""
        return self.n_profitable / self.n_bought if self.n_bought else float("nan")


@dataclass(frozen=True)
class BacktestPeriod:
    """Aggregated metrics for a single (year, month) evaluation period.

    ``regimes`` maps acquisition-regime name → strategy key → StrategyPeriod.
    Coverage/width/bias are regime-independent (they describe the model, not the
    purchase price) and stored once.
    """

    yr_sold: int
    mo_sold: int
    n_available: int
    median_interval_width: float
    empirical_coverage: float
    mean_predicted_arv: float
    mean_actual_price: float
    is_crash: bool
    regimes: dict[str, dict[str, StrategyPeriod]]

    @property
    def period_key(self) -> int:
        """Sortable integer key (year * 12 + month)."""
        return self.yr_sold * 12 + self.mo_sold

    @property
    def date_label(self) -> str:
        """ISO-ish YYYY-MM label for the period."""
        return f"{self.yr_sold:04d}-{self.mo_sold:02d}"

    @property
    def prediction_bias_pct(self) -> float:
        """(mean predicted ARV − mean actual price) / mean actual price."""
        if self.mean_actual_price <= 0:
            return float("nan")
        return (self.mean_predicted_arv - self.mean_actual_price) / self.mean_actual_price


@dataclass
class BacktestResult:
    """Full walk-forward result with per-period metrics and aggregate summaries."""

    periods: list[BacktestPeriod] = field(default_factory=list)
    nominal_coverage: float = 0.90
    crash_window: tuple[int, int] = (2008, 2010)
    regime_factors: dict[str, float] = field(default_factory=dict)

    @property
    def regime_names(self) -> list[str]:
        return list(self.regime_factors)

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy per-period frame with cumulative P&L per (regime, strategy)."""
        rows: list[dict[str, Any]] = []
        for idx, period in enumerate(sorted(self.periods, key=lambda p: p.period_key)):
            row: dict[str, Any] = {
                "period_index": idx,
                "date": period.date_label,
                "YrSold": period.yr_sold,
                "MoSold": period.mo_sold,
                "n_available": period.n_available,
                "median_interval_width": period.median_interval_width,
                "empirical_coverage": period.empirical_coverage,
                "is_crash": period.is_crash,
                "mean_predicted_arv": period.mean_predicted_arv,
                "mean_actual_price": period.mean_actual_price,
                "prediction_bias_pct": period.prediction_bias_pct,
            }
            for regime, per_strategy in period.regimes.items():
                for key in STRATEGY_KEYS:
                    sp = per_strategy[key]
                    row[f"{regime}__{key}_n_bought"] = sp.n_bought
                    row[f"{regime}__{key}_n_profitable"] = sp.n_profitable
                    row[f"{regime}__{key}_realized_profit"] = sp.realized_profit
            rows.append(row)
        frame = pd.DataFrame(rows)
        for regime in self.regime_names:
            for key in STRATEGY_KEYS:
                col = f"{regime}__{key}_realized_profit"
                frame[f"{regime}__{key}_cum_profit"] = frame[col].cumsum()
        return frame

    def strategy_summary(self) -> dict[str, dict[str, dict[str, float]]]:
        """Per-regime, per-strategy aggregate P&L: totals, hit rate, max drawdown."""
        frame = self.to_dataframe()
        crash_mask = frame["is_crash"].to_numpy(dtype=bool)
        summary: dict[str, dict[str, dict[str, float]]] = {}
        for regime in self.regime_names:
            summary[regime] = {}
            for key in STRATEGY_KEYS:
                realized = frame[f"{regime}__{key}_realized_profit"].to_numpy(dtype=float)
                cum = frame[f"{regime}__{key}_cum_profit"].to_numpy(dtype=float)
                bought = frame[f"{regime}__{key}_n_bought"].to_numpy(dtype=float)
                profitable = frame[f"{regime}__{key}_n_profitable"].to_numpy(dtype=float)
                total_bought = float(bought.sum())
                crash_bought = float(bought[crash_mask].sum())
                crash_profitable = float(profitable[crash_mask].sum())
                summary[regime][key] = {
                    "label": STRATEGY_LABELS[key],  # type: ignore[dict-item]
                    "total_realized_profit": float(realized.sum()),
                    "final_cumulative_profit": float(cum[-1]) if len(cum) else 0.0,
                    "total_deals_bought": total_bought,
                    "overall_hit_rate": (
                        float(profitable.sum()) / total_bought if total_bought else float("nan")
                    ),
                    "max_drawdown": _max_drawdown(cum),
                    "crash_realized_profit": float(realized[crash_mask].sum()),
                    "crash_deals_bought": crash_bought,
                    "crash_hit_rate": (
                        crash_profitable / crash_bought if crash_bought else float("nan")
                    ),
                    "profit_per_deal": (
                        float(realized.sum()) / total_bought if total_bought else float("nan")
                    ),
                }
        return summary

    def coverage_summary(self) -> dict[str, float | bool]:
        """Realized 90% coverage overall, pre-crash, and in the crash window."""
        frame = self.to_dataframe()
        crash_mask = frame["is_crash"].to_numpy(dtype=bool)
        weights = frame["n_available"].to_numpy(dtype=float)
        coverage = frame["empirical_coverage"].to_numpy(dtype=float)

        def _weighted(mask: np.ndarray) -> float:
            w = weights[mask]
            if w.sum() == 0:
                return float("nan")
            return float(np.average(coverage[mask], weights=w))

        crash = _weighted(crash_mask)
        return {
            "nominal": self.nominal_coverage,
            "overall_empirical": _weighted(np.ones(len(frame), dtype=bool)),
            "pre_crash_empirical": _weighted(~crash_mask),
            "crash_empirical": crash,
            "coverage_collapsed": bool(crash < self.nominal_coverage),
        }


def _max_drawdown(cumulative: np.ndarray) -> float:
    """Maximum peak-to-trough decline of a cumulative-P&L series (positive $)."""
    if len(cumulative) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max  # <= 0 everywhere
    return float(-drawdown.min())


# ── Model generation (a point + CQR stack trained on a past-only window) ───────


@dataclass
class ModelGeneration:
    """A point model + smearing factor + calibrated CQR model for one window."""

    point_model: Pipeline
    smearing_factor: float
    cqr_model: CQRModel
    n_train_rows: int
    trained_through_key: int


def _build_point_pipeline(config: ModelConfig, seed: int) -> Pipeline:
    """Phase 1-style LightGBM point regressor on the shared feature preprocessor."""
    lgbm = config.lightgbm
    model = EarlyStoppingLightGBMRegressor(
        n_estimators=lgbm.n_estimators,
        learning_rate=lgbm.learning_rate,
        num_leaves=lgbm.num_leaves,
        min_child_samples=lgbm.min_child_samples,
        subsample=lgbm.subsample,
        colsample_bytree=lgbm.colsample_bytree,
        reg_alpha=lgbm.reg_alpha,
        reg_lambda=lgbm.reg_lambda,
        early_stopping_rounds=lgbm.early_stopping_rounds,
        validation_fraction=config.phase1.early_stopping_validation_fraction,
        random_state=seed,
    )
    return Pipeline(
        steps=[("preprocess", build_preprocessor(config.features.drop)), ("model", model)]
    )


def train_generation(
    X_past: pd.DataFrame,
    y_past_log: pd.Series,
    config: ModelConfig,
    trained_through_key: int,
) -> ModelGeneration:
    """Fit the point + CQR stack on a past-only pool (no future leakage).

    The point model is fit on the full past pool (its predictions are used only for
    valuation, not coverage). The CQR quantile arms are fit on a fit fold and the
    conformal Q̂ is calibrated on a held-out calibration fold, so interval coverage
    is never measured on rows the arms trained on.
    """
    seed = config.global_seed
    idx = np.arange(len(X_past))
    fit_idx, cal_idx = train_test_split(
        idx, test_size=config.backtest.calibration_fraction, random_state=seed
    )

    point_model = _build_point_pipeline(config, seed)
    point_model.fit(X_past, np.asarray(y_past_log))
    smearing = duan_smearing_factor(y_past_log.to_numpy(), np.asarray(point_model.predict(X_past)))

    cqr_model = CQRModel.fit(
        X_past.iloc[fit_idx],
        y_past_log.iloc[fit_idx],
        X_past.iloc[cal_idx],
        y_past_log.iloc[cal_idx],
        config,
        alpha=config.conformal.alpha,
    )
    logger.info(
        "Trained generation through key %d on %d rows (fit=%d, cal=%d, Q̂=%.5f)",
        trained_through_key,
        len(X_past),
        len(fit_idx),
        len(cal_idx),
        cqr_model.q_hat,
    )
    return ModelGeneration(
        point_model=point_model,
        smearing_factor=smearing,
        cqr_model=cqr_model,
        n_train_rows=len(X_past),
        trained_through_key=trained_through_key,
    )


# ── The walk-forward loop ──────────────────────────────────────────────────────


def _period_key(year: int, month: int) -> int:
    return year * 12 + month


def _flip_fixed_costs(purchase_price: float, flip: FlipConfig) -> float:
    """Deterministic transaction + expected-hold carry cost on the purchase price."""
    transaction = purchase_price * flip.transaction_cost_pct
    holding = purchase_price * flip.holding_cost_monthly_pct * flip.holding_period_months_base
    return transaction + holding


def _uncertainty_gate(interval_width: float, prob_loss: float, economics: EconomicsConfig) -> bool:
    """Regime-agnostic core of the underwriting rule (ADR-013): buy only when the
    model is confident enough — the 90% interval fits the width cap AND the modeled
    loss probability is within the APPROVE tolerance. The flip-specific $15k
    margin-buffer probability is intentionally omitted: it is meaningless at thin
    iBuyer margins (ADR-026)."""
    uw = economics.flip.underwriting
    return (
        interval_width <= uw.max_acceptable_interval_width_usd
        and prob_loss <= uw.approve_prob_loss_max
    )


def _score_period(
    generation: ModelGeneration, X_period: pd.DataFrame
) -> tuple[np.ndarray, CQRResult]:
    """Return (bias-corrected point ARV in dollars, log-scale CQR result)."""
    point_log = np.asarray(generation.point_model.predict(X_period))
    arv_point = log_predictions_to_dollars(point_log, generation.smearing_factor)
    cqr_log = generation.cqr_model.predict(X_period)
    return arv_point, cqr_log


def _evaluate_period(
    yr: int,
    mo: int,
    generation: ModelGeneration,
    X_period: pd.DataFrame,
    actual_price: np.ndarray,
    actual_log: np.ndarray,
    economics: EconomicsConfig,
    regimes: dict[str, float],
    is_crash: bool,
    seed_base: int,
) -> BacktestPeriod:
    """Score one month's homes and tally each (regime, strategy) realized outcome."""
    flip = economics.flip
    buffer = flip.underwriting.minimum_underwrite_margin_buffer_usd
    arv_point, cqr_log = _score_period(generation, X_period)
    cqr_dollars = cqr_log.to_dollars()
    lower, upper = cqr_dollars.y_lower, cqr_dollars.y_upper
    widths = upper - lower
    inside = (actual_log >= cqr_log.y_lower) & (actual_log <= cqr_log.y_upper)

    # Accumulators: regime -> strategy -> {bought, profitable, realized}.
    acc = {
        regime: {key: {"n_bought": 0, "n_profitable": 0, "realized": 0.0} for key in STRATEGY_KEYS}
        for regime in regimes
    }

    for i in range(len(X_period)):
        point_i = float(arv_point[i])
        width_i = float(widths[i])
        for regime, factor in regimes.items():
            mao = max(factor * point_i, 0.0)  # renovation excluded (ADR-024)
            fixed_costs = _flip_fixed_costs(mao, flip)
            realized = float(actual_price[i]) - mao - fixed_costs
            point_profit = point_i - mao - fixed_costs  # what a naive point model expects

            summary = simulate_flip_profit(
                arv_point=point_i,
                arv_lower=float(lower[i]),
                arv_upper=float(upper[i]),
                renovation_cost=0.0,
                economics=economics,
                purchase_price=mao,
                seed=seed_base + i,
            )
            bought = {
                "buy_all": True,
                "naive_point": point_profit >= buffer,  # trusts the point estimate's margin
                "uncertainty_aware": _uncertainty_gate(width_i, summary.prob_loss, economics),
            }
            for key, did_buy in bought.items():
                if did_buy:
                    acc[regime][key]["n_bought"] += 1  # type: ignore[operator]
                    acc[regime][key]["realized"] += realized  # type: ignore[operator]
                    if realized > 0:
                        acc[regime][key]["n_profitable"] += 1  # type: ignore[operator]

    regime_strategies = {
        regime: {
            key: StrategyPeriod(
                n_bought=int(acc[regime][key]["n_bought"]),
                n_profitable=int(acc[regime][key]["n_profitable"]),
                realized_profit=float(acc[regime][key]["realized"]),
            )
            for key in STRATEGY_KEYS
        }
        for regime in regimes
    }
    return BacktestPeriod(
        yr_sold=yr,
        mo_sold=mo,
        n_available=len(X_period),
        median_interval_width=float(np.median(widths)) if len(widths) else float("nan"),
        empirical_coverage=float(np.mean(inside)) if len(inside) else float("nan"),
        mean_predicted_arv=float(np.mean(arv_point)) if len(arv_point) else float("nan"),
        mean_actual_price=float(np.mean(actual_price)) if len(actual_price) else float("nan"),
        is_crash=is_crash,
        regimes=regime_strategies,
    )


def run_backtest(
    df_full: pd.DataFrame,
    model_config: ModelConfig,
    economics_config: EconomicsConfig,
) -> BacktestResult:
    """Run the walk-forward backtest over the full Ames temporal dataset.

    Args:
        df_full: Full De Cock Ames DataFrame (column-normalized). It is validated
            and sorted by (YrSold, MoSold) internally.
        model_config: ModelConfig with the ``backtest`` block and model settings.
        economics_config: EconomicsConfig; the ``flip`` block supplies all economics
            including the ``backtest_acquisition_regimes``.

    Returns:
        BacktestResult with one BacktestPeriod per evaluated (year, month).
    """
    bt: BacktestConfig = model_config.backtest
    target = model_config.target.column
    regimes = economics_config.flip.backtest_acquisition_regimes
    if not regimes:
        raise ValueError(
            "economics.flip.backtest_acquisition_regimes is empty; define at least one regime."
        )
    raw = validate_ames_full(df_full).sort_values(["YrSold", "MoSold"]).reset_index(drop=True)
    y_log = make_target(raw[target], model_config.target.transform)
    X = raw.drop(columns=[target])

    crash_start, crash_end = bt.crash_window_years
    nominal = round(1.0 - model_config.conformal.alpha, 4)

    eval_periods = sorted(
        {
            (int(yr), int(mo))
            for yr, mo in zip(raw["YrSold"], raw["MoSold"], strict=False)
            if int(yr) >= bt.eval_start_year
        }
    )
    logger.info(
        "Backtest: %d periods from %s over regimes %s", len(eval_periods), eval_periods[0], regimes
    )

    generations: dict[int, ModelGeneration] = {}
    periods: list[BacktestPeriod] = []

    for order, (yr, mo) in enumerate(eval_periods):
        current_key = _period_key(yr, mo)
        if bt.retrain_frequency == "annual":
            gen_key = yr
            past_mask = raw["YrSold"] < yr
        else:  # monthly expanding window
            gen_key = current_key
            past_mask = (raw["YrSold"] * 12 + raw["MoSold"]) < current_key

        n_past = int(past_mask.sum())
        if n_past < bt.warmup_min_train_rows:
            logger.warning("Skipping %04d-%02d: only %d past rows (< warmup)", yr, mo, n_past)
            continue

        if gen_key not in generations:
            generations[gen_key] = train_generation(
                X.loc[past_mask], y_log.loc[past_mask], model_config, gen_key
            )
        generation = generations[gen_key]

        period_mask = (raw["YrSold"] == yr) & (raw["MoSold"] == mo)
        if not period_mask.any():
            continue
        periods.append(
            _evaluate_period(
                yr,
                mo,
                generation,
                X.loc[period_mask],
                raw.loc[period_mask, target].to_numpy(dtype=float),
                y_log.loc[period_mask].to_numpy(dtype=float),
                economics_config,
                regimes,
                crash_start <= yr <= crash_end,
                seed_base=model_config.global_seed + order * 10_000,
            )
        )

    return BacktestResult(
        periods=periods,
        nominal_coverage=nominal,
        crash_window=(crash_start, crash_end),
        regime_factors=dict(regimes),
    )


# ── Metric card + persistence ──────────────────────────────────────────────────


def build_metric_card(
    result: BacktestResult,
    raw: pd.DataFrame,
    model_config: ModelConfig,
    economics_config: EconomicsConfig,
) -> dict[str, Any]:
    """Assemble the Phase 4 metric card from the backtest result and inputs."""
    coverage = result.coverage_summary()
    strategies = result.strategy_summary()
    price_by_year = {
        int(str(year)): float(median)
        for year, median in raw.groupby("YrSold")[model_config.target.column]
        .median()
        .round(0)
        .items()
    }
    peak = max(price_by_year.values())
    trough = min(price_by_year.values())
    n_generations = len({p.yr_sold for p in result.periods})

    return {
        "phase": "4",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data": {
            "rows": int(len(raw)),
            "years": sorted({int(y) for y in raw["YrSold"].unique()}),
            "median_price_by_year": {str(year): value for year, value in price_by_year.items()},
            "peak_to_trough_median_pct": float((trough - peak) / peak),
            "source": "De Cock (2011) full Ames dataset, sorted by YrSold/MoSold",
        },
        "method": {
            "retrain_frequency": model_config.backtest.retrain_frequency,
            "eval_start_year": model_config.backtest.eval_start_year,
            "calibration_fraction": model_config.backtest.calibration_fraction,
            "n_evaluation_periods": len(result.periods),
            "n_model_generations": n_generations,
            "nominal_coverage": result.nominal_coverage,
            "acquisition_regimes": result.regime_factors,
            "disciplined_rule": "interval-width cap + loss-probability cap (ADR-026)",
            "realized_pnl_basis": "observed sale price as realized ARV; synthetic acquisition; "
            "renovation excluded (ADR-024)",
        },
        "crash_window_years": list(result.crash_window),
        "coverage": coverage,
        "strategies": strategies,
        "headline": _headline(result, coverage, strategies),
        "backtest_config_snapshot": model_config.backtest.model_dump(mode="json"),
        "economics_config_snapshot": economics_config.flip.model_dump(mode="json"),
    }


def _headline(
    result: BacktestResult,
    coverage: dict[str, float | bool],
    strategies: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """Plain-English headline numbers for the memo/explainer to quote verbatim."""
    sig = strategies.get(SIGNATURE_REGIME, {})
    contrast = strategies.get(CONTRAST_REGIME, {})
    naive = sig.get("naive_point", {})
    ours = sig.get("uncertainty_aware", {})
    return {
        "nominal_coverage": coverage["nominal"],
        "coverage_pre_crash": coverage["pre_crash_empirical"],
        "coverage_in_crash": coverage["crash_empirical"],
        "coverage_collapsed": coverage["coverage_collapsed"],
        "signature_regime": SIGNATURE_REGIME,
        "ibuyer_naive_max_drawdown": naive.get("max_drawdown"),
        "ibuyer_uncertainty_max_drawdown": ours.get("max_drawdown"),
        "ibuyer_drawdown_reduction": (
            naive.get("max_drawdown", float("nan")) - ours.get("max_drawdown", float("nan"))
        ),
        "ibuyer_naive_hit_rate": naive.get("overall_hit_rate"),
        "ibuyer_uncertainty_hit_rate": ours.get("overall_hit_rate"),
        "ibuyer_naive_total_profit": naive.get("total_realized_profit"),
        "ibuyer_uncertainty_total_profit": ours.get("total_realized_profit"),
        "conservative_naive_max_drawdown": contrast.get("naive_point", {}).get("max_drawdown"),
        "conservative_uncertainty_max_drawdown": contrast.get("uncertainty_aware", {}).get(
            "max_drawdown"
        ),
    }


def run_and_save(
    model_config_path: Path | str = "config/model.yaml",
    economics_config_path: Path | str = "config/economics.yaml",
) -> dict[str, Any]:
    """Run the backtest end-to-end and persist metric card, periods CSV, and figures."""
    mpl_config_dir = Path(tempfile.gettempdir()) / "margin-of-error-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir))

    model_config = load_model_config(resolve_repo_path(model_config_path))
    economics_config = load_economics(economics_config_path)

    ames_path = resolve_repo_path(model_config.data.ames_full_path)
    if not ames_path.exists():
        raise FileNotFoundError(
            f"Full Ames dataset not found at {ames_path}. Download it per data/README.md "
            "(De Cock AmesHousing.xls → data/raw/ames/AmesHousing.csv) before the Phase 4 backtest."
        )
    raw = (
        validate_ames_full(load_ames_full(ames_path))
        .sort_values(["YrSold", "MoSold"])
        .reset_index(drop=True)
    )

    result = run_backtest(raw, model_config, economics_config)
    metric_card = build_metric_card(result, raw, model_config, economics_config)

    reports_dir = resolve_repo_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase4_metric_card.json").write_text(
        json.dumps(metric_card, indent=2, sort_keys=True) + "\n"
    )
    frame = result.to_dataframe()
    frame.to_csv(reports_dir / "phase4_backtest_periods.csv", index=False)
    logger.info("Phase 4 metric card written to %s", reports_dir / "phase4_metric_card.json")

    from margin_of_error.viz.charts import plot_backtest_equity, plot_coverage_drift

    summary = result.strategy_summary()
    plot_coverage_drift(frame, result.nominal_coverage, result.crash_window)
    plot_backtest_equity(
        frame,
        SIGNATURE_REGIME,
        result.crash_window,
        summary[SIGNATURE_REGIME],
        save_as="04b_three_strategies_pnl.png",
    )
    if CONTRAST_REGIME in result.regime_names:
        plot_backtest_equity(
            frame,
            CONTRAST_REGIME,
            result.crash_window,
            summary[CONTRAST_REGIME],
            save_as="04c_conservative_regime_pnl.png",
        )
    return metric_card


def main() -> None:
    """CLI entrypoint for `python -m margin_of_error.backtest.walkforward`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    card = run_and_save()
    cov = card["coverage"]
    print("\n=== Phase 4 headline ===")
    print(
        f"Realized 90% coverage — pre-crash {cov['pre_crash_empirical']:.1%}, "
        f"in-crash {cov['crash_empirical']:.1%} "
        f"({'below target' if cov['coverage_collapsed'] else 'held'})"
    )
    for regime in card["strategies"]:
        print(f"\n[{regime}]  (ARV factor {card['method']['acquisition_regimes'][regime]})")
        for key in STRATEGY_KEYS:
            s = card["strategies"][regime][key]
            print(
                f"  {STRATEGY_LABELS[key]:<26} deals={s['total_deals_bought']:>5.0f} "
                f"P&L=${s['total_realized_profit']:>13,.0f} "
                f"maxDD=${s['max_drawdown']:>12,.0f} hit={s['overall_hit_rate']:.1%}"
            )


if __name__ == "__main__":
    main()
