"""Phase 2 CQR tests: conformal math, interval validity, and empirical coverage.

The coverage test fits real CQR models on the Kaggle data and is skipped when the
data file is absent. It is the single most important Phase 2 guarantee: if the
intervals are not calibrated, every downstream economic number is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from margin_of_error.models.conformal import (
    compute_conformal_quantile,
    compute_conformity_scores,
    quantile_arms,
)


def test_quantile_arms_are_symmetric() -> None:
    lo, hi = quantile_arms(0.10)
    assert lo == pytest.approx(0.05)
    assert hi == pytest.approx(0.95)
    lo80, hi80 = quantile_arms(0.20)
    assert (lo80, hi80) == pytest.approx((0.10, 0.90))


def test_conformity_score_positive_when_outside_interval() -> None:
    """A target above the upper quantile yields a positive nonconformity score."""
    y = np.array([100.0])
    lower = np.array([80.0])
    upper = np.array([90.0])
    scores = compute_conformity_scores(y, lower, upper)
    assert scores[0] == pytest.approx(10.0)  # y - q_high = 100 - 90


def test_conformity_score_negative_when_inside_interval() -> None:
    """A target comfortably inside yields a negative score (room to shrink)."""
    y = np.array([85.0])
    lower = np.array([80.0])
    upper = np.array([90.0])
    scores = compute_conformity_scores(y, lower, upper)
    assert scores[0] < 0


def test_conformal_quantile_uses_correct_finite_sample_rank() -> None:
    """Q̂ is the ceil((1-alpha)(n+1))-th smallest score, NOT a collapsed 1/n level.

    Regression guard for the calibration bug: the naive ``ceil(.)/n`` form
    collapsed the level to ~1/n and produced hugely negative Q̂.
    """
    scores = np.arange(1.0, 101.0)  # 1..100, n=100
    # alpha=0.1 -> rank = ceil(0.9 * 101) = ceil(90.9) = 91 -> 91st smallest = 91.0
    assert compute_conformal_quantile(scores, 0.10) == pytest.approx(91.0)
    # alpha=0.2 -> rank = ceil(0.8 * 101) = ceil(80.8) = 81 -> 81.0
    assert compute_conformal_quantile(scores, 0.20) == pytest.approx(81.0)


def test_conformal_quantile_infinite_when_calibration_too_small() -> None:
    """If the requested rank exceeds n, no finite correction guarantees coverage."""
    scores = np.arange(1.0, 6.0)  # n=5
    # alpha=0.01 -> rank = ceil(0.99 * 6) = ceil(5.94) = 6 > 5 -> +inf
    assert compute_conformal_quantile(scores, 0.01) == float("inf")


@pytest.mark.slow
def test_cqr_empirical_coverage_meets_nominal_on_real_data(kaggle_train_path, model_config) -> None:
    """Primary 90% CQR must achieve >= 90% empirical coverage on the held-out test set."""
    if kaggle_train_path is None:
        pytest.skip("Kaggle train.csv not present")

    from margin_of_error.data.loaders import load_kaggle_train
    from margin_of_error.data.schemas import validate_kaggle_train
    from margin_of_error.models.baseline import make_target
    from margin_of_error.models.conformal import CQRModel, evaluate_coverage
    from margin_of_error.models.phase2 import three_way_split

    raw = validate_kaggle_train(load_kaggle_train(kaggle_train_path))
    y_log = make_target(raw[model_config.target.column], model_config.target.transform)
    X = raw.drop(columns=[model_config.target.column])

    train_idx, cal_idx, test_idx = three_way_split(X, model_config)
    model = CQRModel.fit(
        X.iloc[train_idx],
        y_log.iloc[train_idx],
        X.iloc[cal_idx],
        y_log.iloc[cal_idx],
        model_config,
        alpha=model_config.conformal.alpha,
    )
    result = model.predict(X.iloc[test_idx])
    coverage = evaluate_coverage(result, y_log.iloc[test_idx].to_numpy())
    assert coverage >= 0.90, f"empirical coverage {coverage:.3f} < 0.90"

    # Every interval must be finite and strictly positive in width (dollar scale).
    widths = result.to_dollars().interval_width()
    assert np.all(np.isfinite(widths))
    assert np.all(widths > 0)
