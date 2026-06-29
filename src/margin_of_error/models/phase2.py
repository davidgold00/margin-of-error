"""Phase 2 orchestrator: CQR intervals → flip economics → underwriting verdicts.

Runs the full Phase 2 pipeline end-to-end and persists:
    reports/phase2_metric_card.json       — coverage, widths, headline findings
    reports/phase2_test_underwriting.csv  — per-test-home verdict + economics
    reports/phase2_calibration.csv        — empirical vs. nominal coverage curve

Data discipline (see docs/decisions.md § ADR-010):
    The 1,460 labeled Kaggle rows are split THREE ways — train / calibration /
    test — with no overlap. The CQR quantile models are fit only on the train
    fold; Q̂ is calibrated only on the calibration fold; coverage and width are
    measured only on the test fold.

The Phase 1 artifact supplies the bias-corrected point ARV (its smearing factor)
but is NOT retrained. Note its point predictions on the test rows are in-sample
(Phase 1 trained on all 1,460 rows); the CQR interval that drives the coverage
guarantee and the underwriting verdict is properly out-of-sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from margin_of_error.config import ModelConfig, load_economics, load_model_config
from margin_of_error.data.loaders import load_kaggle_train
from margin_of_error.data.schemas import validate_kaggle_train
from margin_of_error.economics.underwriter import underwrite_best_tier
from margin_of_error.models.baseline import (
    EarlyStoppingLightGBMRegressor,  # noqa: F401 — needed to unpickle the artifact
    log_predictions_to_dollars,
    make_target,
    resolve_repo_path,
)
from margin_of_error.models.conformal import CQRModel, evaluate_coverage

logger = logging.getLogger(__name__)

# Nominal coverage levels for the calibration curve (Figure 2D). The primary
# (0.90) and secondary (0.80) intervals are taken from this same sweep.
CALIBRATION_LEVELS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
PRIMARY_LEVEL = 0.90
SECONDARY_LEVEL = 0.80


@dataclass(frozen=True)
class Phase2RunResult:
    """Paths and headline payload from a Phase 2 run."""

    metric_card_path: Path
    underwriting_path: Path
    calibration_path: Path
    metric_card: dict[str, Any]


def load_phase1_artifact(path: Path | str) -> tuple[Any, float, dict[str, Any]]:
    """Load and validate the Phase 1 model artifact.

    Returns (fitted_point_model, smearing_factor, metric_card). Raises a clear
    error if the artifact is missing or missing required keys — we never silently
    retrain a substitute (the Phase 1 → Phase 2 chain must stay honest).
    """
    artifact_path = resolve_repo_path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Phase 1 artifact not found at {artifact_path}. Run Phase 1 first "
            "(`python -m margin_of_error.models.baseline`); do not substitute a new model."
        )
    # Compatibility shim: Phase 1 was run via `python -m`, so its custom estimator
    # was pickled as `__main__.EarlyStoppingLightGBMRegressor`. Alias that name to
    # the identical class in baseline.py so the EXACT fitted model unpickles cleanly
    # regardless of entry point (we do not retrain Phase 1).
    import __main__

    if not hasattr(__main__, "EarlyStoppingLightGBMRegressor"):
        __main__.EarlyStoppingLightGBMRegressor = EarlyStoppingLightGBMRegressor  # type: ignore[attr-defined]
    payload = joblib.load(artifact_path)
    required = {"model", "smearing_factor", "metric_card"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Phase 1 artifact at {artifact_path} is missing keys: {sorted(missing)}")
    return payload["model"], float(payload["smearing_factor"]), payload["metric_card"]


def three_way_split(
    X: pd.DataFrame, config: ModelConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-overlapping (train_idx, cal_idx, test_idx) row indices.

    Sizes come from config: test_split and calibration_split are fractions of the
    full labeled set; the remainder is the training fold.
    """
    n = len(X)
    idx = np.arange(n)
    test_frac = config.conformal.test_split
    cal_frac = config.conformal.calibration_split
    seed = config.global_seed

    train_cal_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=seed)
    # calibration as a fraction of the remaining train+cal pool
    cal_frac_of_remainder = cal_frac / (1.0 - test_frac)
    train_idx, cal_idx = train_test_split(
        train_cal_idx, test_size=cal_frac_of_remainder, random_state=seed
    )
    logger.info(
        "Three-way split: train=%d, calibration=%d, test=%d",
        len(train_idx),
        len(cal_idx),
        len(test_idx),
    )
    return train_idx, cal_idx, test_idx


def _deterministic_point_profit(
    arv_point: float, purchase_price: float, renovation_cost: float, economics: Any
) -> float:
    """Profit at the point estimate with the expected hold (no uncertainty).

    This is what a 'naive' point model would report — used only to rank the
    top-N opportunities and contrast them with the uncertainty-aware verdict.
    """
    flip = economics.flip
    transaction = purchase_price * flip.transaction_cost_pct
    holding = purchase_price * flip.holding_cost_monthly_pct * flip.holding_period_months_base
    return float(arv_point - purchase_price - renovation_cost - transaction - holding)


def fit_calibration_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    y_test_log: np.ndarray,
    config: ModelConfig,
) -> tuple[dict[float, CQRModel], pd.DataFrame]:
    """Fit a CQR model at each nominal level; measure test coverage and width.

    Returns the fitted models keyed by nominal level and a tidy calibration
    DataFrame (nominal_coverage, empirical_coverage, median_width_dollars).
    """
    models: dict[float, CQRModel] = {}
    rows: list[dict[str, float]] = []
    for level in CALIBRATION_LEVELS:
        alpha = round(1.0 - level, 4)
        logger.info("Fitting CQR for nominal coverage %.2f (alpha=%.2f)", level, alpha)
        model = CQRModel.fit(X_train, y_train, X_cal, y_cal, config, alpha=alpha)
        result = model.predict(X_test)
        empirical = evaluate_coverage(result, y_test_log)
        dollars = result.to_dollars()
        median_width = float(np.median(dollars.interval_width()))
        models[level] = model
        rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": empirical,
                "median_width_dollars": median_width,
            }
        )
        logger.info(
            "  → empirical coverage %.3f, median width %s",
            empirical,
            f"${median_width:,.0f}",
        )
    return models, pd.DataFrame(rows)


def build_test_underwriting_frame(
    raw_test: pd.DataFrame,
    arv_point: np.ndarray,
    primary: CQRModel,
    secondary: CQRModel,
    X_test: pd.DataFrame,
    economics: Any,
    config: ModelConfig,
) -> pd.DataFrame:
    """Score every test home: CQR interval + best-tier underwriting verdict."""
    primary_dollars = primary.predict(X_test).to_dollars()
    secondary_dollars = secondary.predict(X_test).to_dollars()
    l90, u90 = primary_dollars.y_lower, primary_dollars.y_upper
    l80, u80 = secondary_dollars.y_lower, secondary_dollars.y_upper

    records: list[dict[str, Any]] = []
    for i in range(len(X_test)):
        result = underwrite_best_tier(
            arv_point=float(arv_point[i]),
            arv_lower=float(l90[i]),
            arv_upper=float(u90[i]),
            economics=economics,
            seed=config.global_seed + i,
        )
        point_profit = _deterministic_point_profit(
            float(arv_point[i]), result.purchase_price, _tier_cost(economics, result), economics
        )
        records.append(
            {
                "Id": int(raw_test["Id"].iloc[i]) if "Id" in raw_test.columns else i,
                "Neighborhood": raw_test["Neighborhood"].iloc[i]
                if "Neighborhood" in raw_test.columns
                else "Unknown",
                "actual_sale_price": float(np.expm1(np.log1p(raw_test["SalePrice"].iloc[i]))),
                "predicted_arv": result.predicted_arv,
                "interval_low_90": float(l90[i]),
                "interval_high_90": float(u90[i]),
                "interval_width_90": float(u90[i] - l90[i]),
                "interval_low_80": float(l80[i]),
                "interval_high_80": float(u80[i]),
                "interval_width_80": float(u80[i] - l80[i]),
                "renovation_tier": result.renovation_tier,
                "purchase_price": result.purchase_price,
                "verdict": result.verdict,
                "expected_profit": result.expected_profit,
                "naive_point_profit": point_profit,
                "profit_p10": result.profit_p10,
                "profit_p90": result.profit_p90,
                "prob_loss": result.prob_loss,
                "prob_above_min_margin": result.prob_above_min_margin,
                "primary_decline_reason": result.primary_decline_reason,
            }
        )
    return pd.DataFrame(records)


def _tier_cost(economics: Any, result: Any) -> float:
    return float(economics.flip.renovation_tiers[result.renovation_tier].cost_usd)


def compute_headline(frame: pd.DataFrame, top_n: int = 50) -> dict[str, Any]:
    """Compute the plain-English Phase 2 headline numbers from the scored frame."""
    n = len(frame)
    verdict_counts = frame["verdict"].value_counts().to_dict()
    approve = int(verdict_counts.get("APPROVE", 0))
    refer = int(verdict_counts.get("REFER", 0))
    decline = int(verdict_counts.get("DECLINE", 0))

    top = frame.nlargest(top_n, "naive_point_profit")
    top_not_approved = int((top["verdict"] != "APPROVE").sum())
    top_declined = int((top["verdict"] == "DECLINE").sum())

    return {
        "n_test_homes": n,
        "mean_interval_width_90_dollars": float(frame["interval_width_90"].mean()),
        "median_interval_width_90_dollars": float(frame["interval_width_90"].median()),
        "approve_rate": approve / n,
        "refer_rate": refer / n,
        "decline_rate": decline / n,
        "approve_count": approve,
        "refer_count": refer,
        "decline_count": decline,
        f"top_{top_n}_naive_picks_not_approved": top_not_approved,
        f"top_{top_n}_naive_picks_not_approved_frac": top_not_approved / max(len(top), 1),
        f"top_{top_n}_naive_picks_declined": top_declined,
        f"top_{top_n}_naive_picks_declined_frac": top_declined / max(len(top), 1),
        "decline_reason_counts": (
            frame.loc[frame["verdict"] == "DECLINE", "primary_decline_reason"]
            .value_counts()
            .to_dict()
        ),
    }


def run_phase2(config_path: Path | str = "config/model.yaml") -> Phase2RunResult:
    """Run Phase 2 end-to-end and persist the metric card, verdicts, and curve."""
    config = load_model_config(resolve_repo_path(config_path))
    economics = load_economics("config/economics.yaml")

    artifact_path = resolve_repo_path(config.phase1.artifact_dir) / "baseline_lightgbm.joblib"
    point_model, smearing, phase1_card = load_phase1_artifact(artifact_path)
    logger.info("Loaded Phase 1 artifact (smearing=%.4f)", smearing)

    train_path = resolve_repo_path(config.data.kaggle_train_path)
    raw = validate_kaggle_train(load_kaggle_train(train_path))
    y_log = make_target(raw[config.target.column], config.target.transform)
    X = raw.drop(columns=[config.target.column])

    train_idx, cal_idx, test_idx = three_way_split(X, config)
    X_train, y_train = X.iloc[train_idx], y_log.iloc[train_idx]
    X_cal, y_cal = X.iloc[cal_idx], y_log.iloc[cal_idx]
    X_test = X.iloc[test_idx]
    y_test_log = y_log.iloc[test_idx].to_numpy()
    raw_test = raw.iloc[test_idx].reset_index(drop=True)

    models, calibration_df = fit_calibration_curve(
        X_train, y_train, X_cal, y_cal, X_test, y_test_log, config
    )
    primary = models[PRIMARY_LEVEL]
    secondary = models[SECONDARY_LEVEL]

    primary_coverage = float(
        calibration_df.loc[
            calibration_df["nominal_coverage"] == PRIMARY_LEVEL, "empirical_coverage"
        ].iloc[0]
    )
    # NON-NEGOTIABLE GATE: a miscalibrated interval makes all downstream economics wrong.
    if primary_coverage < PRIMARY_LEVEL:
        raise RuntimeError(
            f"CQR calibration FAILED: empirical coverage {primary_coverage:.3f} < "
            f"{PRIMARY_LEVEL:.2f} on the test set. Stopping before economics — debug first."
        )
    logger.info("Calibration gate PASSED: empirical coverage %.3f >= 0.90", primary_coverage)

    # Bias-corrected point ARV from the Phase 1 model (in-sample on test rows; documented).
    arv_point = log_predictions_to_dollars(np.asarray(point_model.predict(X_test)), smearing)

    frame = build_test_underwriting_frame(
        raw_test, arv_point, primary, secondary, X_test, economics, config
    )
    headline = compute_headline(frame)

    metric_card = {
        "phase": "2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase1_typical_error_dollars": phase1_card["residual_diagnostics"]["spread"][
            "median_abs_error_dollars"
        ],
        "splits": {
            "train": int(len(train_idx)),
            "calibration": int(len(cal_idx)),
            "test": int(len(test_idx)),
        },
        "cqr": {
            "primary_nominal_coverage": PRIMARY_LEVEL,
            "primary_empirical_coverage": primary_coverage,
            "primary_q_hat_log": primary.q_hat,
            "secondary_nominal_coverage": SECONDARY_LEVEL,
            "secondary_empirical_coverage": float(
                calibration_df.loc[
                    calibration_df["nominal_coverage"] == SECONDARY_LEVEL, "empirical_coverage"
                ].iloc[0]
            ),
            "calibration_curve": calibration_df.to_dict(orient="records"),
        },
        "headline": headline,
        "economics_config_snapshot": economics.flip.model_dump(mode="json"),
    }

    reports_dir = resolve_repo_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metric_card_path = reports_dir / "phase2_metric_card.json"
    underwriting_path = reports_dir / "phase2_test_underwriting.csv"
    calibration_path = reports_dir / "phase2_calibration.csv"

    import json

    metric_card_path.write_text(json.dumps(metric_card, indent=2, sort_keys=True) + "\n")
    frame.to_csv(underwriting_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)
    logger.info("Phase 2 metric card written to %s", metric_card_path)

    return Phase2RunResult(
        metric_card_path=metric_card_path,
        underwriting_path=underwriting_path,
        calibration_path=calibration_path,
        metric_card=metric_card,
    )


def main() -> None:
    """CLI entrypoint for `python -m margin_of_error.models.phase2`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    result = run_phase2()
    headline = result.metric_card["headline"]
    print("\n=== Phase 2 headline ===")
    print(f"Test homes: {headline['n_test_homes']}")
    print(f"Median 90% interval width: ${headline['median_interval_width_90_dollars']:,.0f}")
    print(
        f"APPROVE {headline['approve_rate']:.0%} | "
        f"REFER {headline['refer_rate']:.0%} | "
        f"DECLINE {headline['decline_rate']:.0%}"
    )
    print(f"Top-50 naive picks not approved: {headline['top_50_naive_picks_not_approved']}/50")


if __name__ == "__main__":
    main()
