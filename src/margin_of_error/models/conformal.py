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
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from margin_of_error.features.preprocessing import build_preprocessor

if TYPE_CHECKING:
    from margin_of_error.config import ModelConfig

logger = logging.getLogger(__name__)


def quantile_arms(alpha: float) -> tuple[float, float]:
    """Lower/upper quantile targets for a symmetric (1 - alpha) interval.

    For alpha = 0.10 this returns (0.05, 0.95); for alpha = 0.20, (0.10, 0.90).
    The conformal step later expands these to achieve exact finite-sample coverage.
    """
    return alpha / 2.0, 1.0 - alpha / 2.0


class QuantileLightGBM(BaseEstimator, RegressorMixin):
    """LightGBM in quantile-loss mode with train-fold early stopping.

    Mirrors the Phase 1 ``EarlyStoppingLightGBMRegressor`` but optimizes the
    pinball (quantile) loss at a fixed quantile level. The early-stopping split
    is carved from the training fold only — it never sees calibration or test data.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        validation_fraction: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantileLightGBM:
        """Fit a quantile regressor with a train-fold-only early-stopping split."""
        try:
            from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "LightGBM could not load. On macOS this usually means libomp is missing; "
                "install it with `brew install libomp`, then rerun Phase 2."
            ) from exc

        y_array = np.asarray(y)
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X,
            y_array,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )
        self.model_ = LGBMRegressor(
            objective="quantile",
            alpha=self.quantile,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
        )
        self.model_.fit(
            X_fit,
            y_fit,
            eval_set=[(X_eval, y_eval)],
            eval_metric="quantile",
            callbacks=[
                early_stopping(self.early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the configured conditional quantile (log scale)."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return np.asarray(self.model_.predict(X))


def build_quantile_pipeline(
    drop_columns: list[str] | tuple[str, ...],
    quantile: float,
    config: ModelConfig,
    seed: int,
) -> Pipeline:
    """Build a leakage-safe quantile pipeline: Phase 1 preprocessor + quantile LGBM.

    The preprocessor is an *unfitted* clone of the Phase 1 feature pipeline; it is
    refit on whatever training fold this pipeline is fit on (never inherited from
    the Phase 1 artifact), so there is no cross-split leakage.
    """
    lgbm = config.lightgbm
    model = QuantileLightGBM(
        quantile=quantile,
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
        steps=[
            ("preprocess", build_preprocessor(list(drop_columns))),
            ("model", model),
        ]
    )


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

    Romano et al. (2019): Q̂ is the k-th smallest conformity score, where
        k = ceil((1 - alpha) * (n + 1)).
    If k > n (calibration set too small for the requested level), there is no
    finite correction that guarantees coverage, so Q̂ = +inf (interval = the
    whole real line). Q̂ may be negative — that correctly *shrinks* a quantile
    interval that already over-covers.

    Args:
        scores: Conformity scores from compute_conformity_scores().
        alpha: Miscoverage level (e.g., 0.10 for 90% coverage).

    Returns:
        Scalar Q̂ added symmetrically to the quantile interval.
    """
    n = len(scores)
    rank = int(np.ceil((1 - alpha) * (n + 1)))
    if rank > n:
        return float("inf")
    sorted_scores = np.sort(scores)
    return float(sorted_scores[rank - 1])  # rank-th smallest (1-indexed)


def train_cqr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    config: ModelConfig,
    alpha: float | None = None,
) -> tuple[Pipeline, Pipeline, float]:
    """Train CQR quantile models and compute the conformal quantile Q̂.

    Args:
        X_train: Feature matrix for model fitting (raw columns; the pipeline
            refits the Phase 1 preprocessor on this fold only).
        y_train: Log-SalePrice targets for model fitting.
        X_cal: Calibration-set features (never seen during training).
        y_cal: Calibration-set targets.
        config: ModelConfig with lightgbm, conformal, and feature settings.
        alpha: Miscoverage level. Defaults to config.conformal.alpha (the
            primary 90% interval). Pass config.conformal.secondary_alpha for 80%.

    Returns:
        Tuple of (lower_model, upper_model, Q̂). Both models are fitted quantile
        pipelines; Q̂ is the finite-sample conformal correction from Romano (2019).
    """
    target_alpha = config.conformal.alpha if alpha is None else alpha
    lower_q, upper_q = quantile_arms(target_alpha)
    drop_cols = config.features.drop
    seed = config.global_seed

    logger.info("Fitting CQR quantile arms at q=%.3f and q=%.3f", lower_q, upper_q)
    lower_model = build_quantile_pipeline(drop_cols, lower_q, config, seed)
    upper_model = build_quantile_pipeline(drop_cols, upper_q, config, seed)
    lower_model.fit(X_train, np.asarray(y_train))
    upper_model.fit(X_train, np.asarray(y_train))

    cal_lower = np.asarray(lower_model.predict(X_cal))
    cal_upper = np.asarray(upper_model.predict(X_cal))
    scores = compute_conformity_scores(np.asarray(y_cal), cal_lower, cal_upper)
    q_hat = compute_conformal_quantile(scores, target_alpha)
    logger.info("Conformal Q̂ = %.5f (log scale) from %d calibration points", q_hat, len(scores))
    return lower_model, upper_model, q_hat


def predict_intervals(
    lower_model: Pipeline,
    upper_model: Pipeline,
    X: pd.DataFrame,
    q_hat: float,
    alpha: float,
) -> CQRResult:
    """Apply CQR to generate prediction intervals for new observations.

    The conformal correction Q̂ is added symmetrically: the interval is
    [q_low(x) - Q̂, q_high(x) + Q̂]. All arithmetic is in log space; use
    ``.to_dollars()`` to back-transform each bound with expm1.

    Args:
        lower_model: Fitted lower-quantile pipeline.
        upper_model: Fitted upper-quantile pipeline.
        X: Feature matrix for new observations.
        q_hat: Conformal quantile from the calibration step.
        alpha: Miscoverage level these intervals target (for reporting).

    Returns:
        CQRResult with intervals in log scale.
    """
    raw_lower = np.asarray(lower_model.predict(X))
    raw_upper = np.asarray(upper_model.predict(X))
    y_lower = raw_lower - q_hat
    y_upper = raw_upper + q_hat
    return CQRResult(
        y_pred=(y_lower + y_upper) / 2.0,
        y_lower=y_lower,
        y_upper=y_upper,
        conformity_score=q_hat,
        empirical_coverage=float("nan"),  # filled in by evaluate_coverage on a labeled set
        nominal_coverage=1.0 - alpha,
    )


def evaluate_coverage(result: CQRResult, y_true_log: np.ndarray) -> float:
    """Empirical coverage: fraction of true (log) targets inside the interval."""
    inside = (y_true_log >= result.y_lower) & (y_true_log <= result.y_upper)
    return float(np.mean(inside))


@dataclass
class CQRModel:
    """A fitted CQR model: two quantile arms plus the conformal correction.

    Ergonomic wrapper around train_cqr/predict_intervals for the runner, the
    notebook, and the underwriter. Holds everything needed to score new homes.
    """

    lower_model: Pipeline
    upper_model: Pipeline
    q_hat: float
    alpha: float

    @classmethod
    def fit(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        config: ModelConfig,
        alpha: float | None = None,
    ) -> CQRModel:
        """Fit quantile arms on the train fold and calibrate Q̂ on the calibration fold."""
        target_alpha = config.conformal.alpha if alpha is None else alpha
        lower_model, upper_model, q_hat = train_cqr(
            X_train, y_train, X_cal, y_cal, config, alpha=target_alpha
        )
        return cls(
            lower_model=lower_model, upper_model=upper_model, q_hat=q_hat, alpha=target_alpha
        )

    def predict(self, X: pd.DataFrame) -> CQRResult:
        """Return log-scale CQR intervals for the given homes."""
        return predict_intervals(self.lower_model, self.upper_model, X, self.q_hat, self.alpha)
