"""Signature charts for the Margin of Error project.

Each function produces one named chart. Charts are saved to reports/figures/
and embedded in notebooks. All chart functions are pure: they take data and
return a matplotlib Figure (no side effects, no global state).

Planned charts (implemented by phase):

Phase 1:
    plot_residual_distribution  — error histogram showing dollar error vs. flip margin
    plot_feature_importance     — horizontal bar chart of top 20 features

Phase 2 (CORE):
    plot_margin_vs_uncertainty  — THE signature chart: scatter of predicted margin
                                  vs. interval width, with the diagonal "danger zone"
                                  where uncertainty > margin (PLACEHOLDER until Phase 2)
    plot_calibration            — reliability diagram: nominal vs. empirical coverage

Phase 3:
    plot_naive_vs_causal        — coefficient comparison: naive OLS vs. DML estimates

Phase 4:
    plot_backtest_equity        — cumulative P&L over 2006–2010 with recession overlay
    plot_coverage_drift         — interval calibration stability over time
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")


def _save_or_show(fig: plt.Figure, filename: str | None) -> plt.Figure:
    """Save figure to reports/figures/ if filename given, otherwise return it."""
    if filename is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", out)
    return fig


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    median_home_price: float,
    save_as: str | None = "01_residual_distribution.png",
) -> plt.Figure:
    """Phase 1: histogram of dollar prediction errors vs. typical flip margin.

    The chart has two vertical lines:
        - Median dollar error (the model's typical miss)
        - The minimum profit threshold (from economics config)

    If the error line is to the right of the profit threshold, the model
    cannot reliably underwrite at that margin level.

    Args:
        y_true: True log(SalePrice) values.
        y_pred: Predicted log(SalePrice) values.
        median_home_price: Median SalePrice in dollars (for context annotation).
        save_as: Filename to save under reports/figures/. None returns only.

    Returns:
        matplotlib Figure.
    """
    raise NotImplementedError("Phase 1 not yet implemented — awaiting approval")


def plot_margin_vs_uncertainty(
    predicted_margins: np.ndarray,
    interval_widths: np.ndarray,
    underwrite_mask: np.ndarray,
    save_as: str | None = "02_margin_vs_uncertainty.png",
) -> plt.Figure:
    """Phase 2: THE signature chart — margin vs. model uncertainty.

    X-axis: Predicted gross profit margin (% of ARV)
    Y-axis: CQR prediction interval width (% of predicted ARV)

    Each point is a property. Color: UNDERWRITE (green) vs DECLINE (red/grey).

    The diagonal line y = x marks where uncertainty = margin. Points above
    this line have uncertainty > margin and cannot be safely underwritten.
    The chart's central finding: many 'profitable' deals (positive x) are
    above the line (high uncertainty) and should be declined.

    Args:
        predicted_margins: Predicted profit as fraction of ARV, per property.
        interval_widths: CQR interval width as fraction of ARV, per property.
        underwrite_mask: Boolean array; True = UNDERWRITE decision.
        save_as: Filename to save under reports/figures/.

    Returns:
        matplotlib Figure.
    """
    raise NotImplementedError("Phase 2 not yet implemented — awaiting approval")


def plot_calibration(
    coverages: pd.Series,
    nominal_levels: list[float],
    save_as: str | None = "02_calibration.png",
) -> plt.Figure:
    """Phase 2: reliability diagram for interval calibration.

    Args:
        coverages: Empirical coverage at each nominal level.
        nominal_levels: List of alpha levels tested.
        save_as: Filename to save under reports/figures/.

    Returns:
        matplotlib Figure.
    """
    raise NotImplementedError("Phase 2 not yet implemented — awaiting approval")


def plot_naive_vs_causal(
    effects_df: pd.DataFrame,
    save_as: str | None = "03_naive_vs_causal.png",
) -> plt.Figure:
    """Phase 3: comparison chart of naive OLS vs. DML causal effect estimates.

    Args:
        effects_df: DataFrame from causal/dml.py compare_naive_vs_causal().
        save_as: Filename to save under reports/figures/.

    Returns:
        matplotlib Figure.
    """
    raise NotImplementedError("Phase 3 not yet implemented — awaiting approval")


def plot_backtest_equity(
    backtest_df: pd.DataFrame,
    save_as: str | None = "04_backtest_equity.png",
) -> plt.Figure:
    """Phase 4: cumulative P&L over the 2006–2010 backtest period.

    Overlays a recession shading (December 2007 – June 2009).

    Args:
        backtest_df: DataFrame from backtest/walkforward.py BacktestResult.to_dataframe().
        save_as: Filename to save under reports/figures/.

    Returns:
        matplotlib Figure.
    """
    raise NotImplementedError("Phase 4 not yet implemented — awaiting approval")
