"""Phase 2: Conformalized Quantile Regression (CQR) prediction intervals.

Implements split-conformal CQR as described in:
    Romano, Sesia, Candès (2019). "Conformalized Quantile Regression."
    NeurIPS 2019. https://arxiv.org/abs/1905.03222

The CQR algorithm:
    1. Split training data: model-training set (85%) + calibration set (15%).
       The calibration set is NEVER seen during model fitting.
    2. Train two LightGBM quantile models:
       - q_low(x): predicts the (lower_alpha)-th quantile
       - q_high(x): predicts the (upper_alpha)-th quantile
    3. On the calibration set, compute conformity scores:
       s_i = max(q_low(x_i) - y_i,  y_i - q_high(x_i))
    4. Let Q̂ = the ceil[(1-alpha)(1 + 1/n_cal)]-th smallest score.
    5. For a new point x: interval = [q_low(x) - Q̂,  q_high(x) + Q̂]

Guarantee: Under exchangeability, P(y ∈ interval) ≥ 1 - alpha in finite samples.

We also provide a MAPIE-based implementation as a cross-check. Results should
agree within Monte Carlo noise; MAPIE is used for production, the manual
implementation for auditability.

PHASE 2 STATUS: Skeleton with typed signatures and algorithm documentation.
Full implementation awaiting Phase 2 approval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CQRResult:
    """Output from a CQR conformal interval model.

    Attributes:
        y_pred: Point predictions (midpoint of interval, in log scale).
        y_lower: Lower bound of prediction interval (log scale).
        y_upper: Upper bound of prediction interval (log scale).
        conformity_score: The calibration-set quantile Q̂ used to adjust intervals.
        empirical_coverage: Fraction of calibration-set targets inside the interval.
        nominal_coverage: Target coverage (1 - alpha from config).
    """

    y_pred: np.ndarray
    y_lower: np.ndarray
    y_upper: np.ndarray
    conformity_score: float
    empirical_coverage: float
    nominal_coverage: float

    def to_dollars(self) -> CQRResult:
        """Return a new CQRResult with all predictions in dollar scale."""
        return CQRResult(
            y_pred=np.expm1(self.y_pred),
            y_lower=np.expm1(self.y_lower),
            y_upper=np.expm1(self.y_upper),
            conformity_score=self.conformity_score,
            empirical_coverage=self.empirical_coverage,
            nominal_coverage=self.nominal_coverage,
        )

    def interval_width(self) -> np.ndarray:
        """Return element-wise interval widths (in whatever scale y is in)."""
        return np.asarray(self.y_upper - self.y_lower)


def compute_conformity_scores(
    y_true: np.ndarray,
    y_lower_pred: np.ndarray,
    y_upper_pred: np.ndarray,
) -> np.ndarray:
    """Compute CQR conformity scores for a calibration set.

    s_i = max(q_low(x_i) - y_i,  y_i - q_high(x_i))

    Positive score means the true value is OUTSIDE the quantile interval.
    The conformal adjustment Q̂ expands the interval so a (1-alpha) fraction
    of calibration scores fall at or below Q̂.

    Args:
        y_true: Observed target values on the calibration set.
        y_lower_pred: Lower quantile predictions on calibration set.
        y_upper_pred: Upper quantile predictions on calibration set.

    Returns:
        Array of conformity scores (one per calibration point).
    """
    return np.asarray(np.maximum(y_lower_pred - y_true, y_true - y_upper_pred))


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile Q̂ from calibration scores.

    Q̂ = quantile(scores, ceil[(1-alpha)(1 + 1/n)] / n)

    This is the finite-sample correction from Romano et al. (2019).

    Args:
        scores: Conformity scores from compute_conformity_scores().
        alpha: Miscoverage level (e.g., 0.10 for 90% coverage).

    Returns:
        Scalar Q̂ used to expand prediction intervals.
    """
    n = len(scores)
    level = np.ceil((1 - alpha) * (1 + 1 / n)) / n
    level = min(level, 1.0)  # clamp for tiny calibration sets
    return float(np.quantile(scores, level))


def train_cqr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    config: Any,  # type: ignore[name-defined]
) -> tuple[Any, Any, float]:
    """Train CQR quantile models and compute the conformal quantile.

    Args:
        X_train: Feature matrix for model fitting.
        y_train: Log-SalePrice targets for model fitting.
        X_cal: Calibration-set features (never seen during training).
        y_cal: Calibration-set targets.
        config: ModelConfig with quantile and conformal settings.

    Returns:
        Tuple of (lower_model, upper_model, Q̂).
        Lower and upper models are fitted LightGBM quantile regressors.

    Phase 2 implementation note:
        Both models use LightGBM's 'quantile' objective.
        Q̂ is computed via compute_conformal_quantile().
    """
    raise NotImplementedError("Phase 2 not yet implemented — awaiting approval")


def predict_intervals(
    lower_model: Any,  # type: ignore[name-defined]
    upper_model: Any,  # type: ignore[name-defined]
    X: pd.DataFrame,
    q_hat: float,
) -> CQRResult:
    """Apply CQR to generate prediction intervals for new observations.

    Args:
        lower_model: Fitted lower-quantile LightGBM model.
        upper_model: Fitted upper-quantile LightGBM model.
        X: Feature matrix for new observations.
        q_hat: Conformal quantile from the calibration step.

    Returns:
        CQRResult with intervals in log scale (use .to_dollars() to convert).
    """
    raise NotImplementedError("Phase 2 not yet implemented — awaiting approval")
