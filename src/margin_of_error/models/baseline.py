"""Phase 1 baseline: gradient boosting point model.

This model is the deliberate "strawman" of the project. It represents the
conventional approach — optimize for RMSE — and is built well precisely so
we can attack it honestly in Phase 2.

The baseline is NOT the deliverable. It is the foil that motivates the need
for uncertainty quantification. After this model is trained and evaluated, we
will show:
  (a) its dollar error on a median Ames home
  (b) that this error is larger than the typical flip margin
  (c) why trusting a point estimate for underwriting is reckless

PHASE 1 STATUS: Skeleton with typed signatures. Full implementation awaiting
Phase 1 approval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Output from a baseline model training run.

    Attributes:
        oof_predictions: Out-of-fold predictions on the training set (log scale).
        oof_rmse: Cross-validated RMSE in log scale.
        oof_rmse_dollars: Dollar-scale RMSE at the median home price.
        feature_importance: Series of feature → importance score.
        model: Trained model object (implementation-specific).
    """

    oof_predictions: np.ndarray
    oof_rmse: float
    oof_rmse_dollars: float
    feature_importance: pd.Series
    model: Any = field(repr=False)


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    config: Any,
    seed: int = 42,
) -> BaselineResult:
    """Train the LightGBM baseline model with k-fold cross-validation.

    Args:
        X: Feature matrix from features/engineering.py.
        y: Log-transformed SalePrice (log1p).
        config: ModelConfig instance from config.py.
        seed: Random seed (should come from config.global_seed).

    Returns:
        BaselineResult with OOF metrics and feature importance.

    Phase 1 implementation note:
        Uses LightGBM with early stopping and stratified neighborhood folds.
        Final model is a full re-fit on all training data.
    """
    raise NotImplementedError("Phase 1 not yet implemented — awaiting approval")


def evaluate_baseline(
    result: BaselineResult,
    y_true: pd.Series,
    median_price: float,
) -> dict[str, float]:
    """Compute dollar-scale evaluation metrics from OOF predictions.

    Converts from log scale to dollars and reports:
        - RMSE in log scale (the leaderboard metric)
        - RMSE in dollars (the economically meaningful metric)
        - Median absolute error in dollars
        - 90th percentile absolute error in dollars

    Args:
        result: Output of train_baseline().
        y_true: True log(SalePrice) values for the training set.
        median_price: Median SalePrice in dollars (for context).

    Returns:
        Dict of metric name → value.
    """
    raise NotImplementedError("Phase 1 not yet implemented — awaiting approval")
