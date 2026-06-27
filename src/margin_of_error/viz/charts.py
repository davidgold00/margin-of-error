"""Signature charts for the Margin of Error project.

Each function produces one named chart. Charts are saved to reports/figures/
and embedded in notebooks. All chart functions are pure: they take data and
return a matplotlib Figure (no side effects, no global state).

Planned charts (implemented by phase):

Phase 1:
    plot_residual_distribution  — error histogram showing dollar error vs. flip margin
    plot_price_error_vs_home_price — dollar error vs. actual home price
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
from xml.sax.saxutils import escape

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
    import matplotlib.pyplot as plt
    import numpy as np

    actual = np.expm1(y_true)
    predicted = np.expm1(y_pred)
    abs_errors = np.abs(predicted - actual)
    median_error = float(np.median(abs_errors))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(abs_errors, bins=35, color="#4C78A8", alpha=0.85)
    ax.axvspan(10_000, 20_000, color="#F58518", alpha=0.18, label="$10K-$20K margin range")
    ax.axvline(
        median_error,
        color="#111111",
        linewidth=2,
        label=f"Median error ${median_error:,.0f}",
    )
    ax.set_title("Phase 1 OOF Dollar Prediction Error")
    ax.set_xlabel("Absolute prediction error ($)")
    ax.set_ylabel("Homes")
    ax.text(
        0.98,
        0.92,
        f"Median home price: ${median_home_price:,.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.legend()
    return _save_or_show(fig, save_as)


def plot_price_error_vs_home_price(
    sale_prices: np.ndarray,
    abs_errors: np.ndarray,
    save_as: str | None = "01_price_error_vs_home_price.png",
) -> plt.Figure:
    """Phase 1 signature chart: dollar error against actual home price.

    The shaded horizontal band marks a plausible $10K-$20K flip-margin range.
    The plot intentionally does not model flip economics; it only shows whether
    point-model error is on the same order as that margin.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(sale_prices, abs_errors, s=28, alpha=0.62, color="#4C78A8", edgecolor="none")
    ax.axhspan(10_000, 20_000, color="#F58518", alpha=0.18, label="$10K-$20K margin range")

    order = np.argsort(sale_prices)
    if len(order) >= 20:
        window = max(20, len(order) // 12)
        sorted_prices = sale_prices[order]
        sorted_errors = abs_errors[order]
        rolling = np.array(
            [
                np.median(sorted_errors[max(0, i - window) : min(len(order), i + window)])
                for i in range(len(order))
            ]
        )
        ax.plot(sorted_prices, rolling, color="#111111", linewidth=2, label="Rolling median error")

    ax.set_title("Phase 1: OOF Dollar Error vs. Home Price")
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("Absolute prediction error ($)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    return _save_or_show(fig, save_as)


def write_price_error_svg(
    sale_prices: np.ndarray,
    abs_errors: np.ndarray,
    save_as: str = "01_price_error_vs_home_price.svg",
) -> Path:
    """Write the Phase 1 signature chart as dependency-light SVG."""
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / save_as

    width, height = 920, 560
    left, right, top, bottom = 86, 28, 52, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = float(np.min(sale_prices)), float(np.max(sale_prices))
    y_min = 0.0
    y_max = float(max(np.max(abs_errors), 20_000) * 1.08)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

    def money(value: float) -> str:
        return f"${value:,.0f}"

    band_top = sy(20_000)
    band_bottom = sy(10_000)
    order = np.argsort(sale_prices)
    window = max(20, len(order) // 12)
    rolling_points = []
    if len(order) >= 20:
        sorted_prices = sale_prices[order]
        sorted_errors = abs_errors[order]
        step = max(1, len(order) // 160)
        for i in range(0, len(order), step):
            lo = max(0, i - window)
            hi = min(len(order), i + window)
            rolling_points.append(
                f"{sx(float(sorted_prices[i])):.1f},{sy(float(np.median(sorted_errors[lo:hi]))):.1f}"
            )

    circles = "\n".join(
        f'<circle cx="{sx(float(price)):.1f}" cy="{sy(float(err)):.1f}" r="2.3" '
        'fill="#4C78A8" fill-opacity="0.56" />'
        for price, err in zip(sale_prices, abs_errors, strict=False)
    )

    x_ticks = np.linspace(x_min, x_max, 5)
    y_ticks = np.linspace(0, y_max, 6)
    x_tick_svg = "\n".join(
        f'<g><line x1="{sx(float(tick)):.1f}" y1="{top + plot_h:.1f}" '
        f'x2="{sx(float(tick)):.1f}" y2="{top + plot_h + 6:.1f}" stroke="#333" />'
        f'<text x="{sx(float(tick)):.1f}" y="{top + plot_h + 26:.1f}" '
        f'text-anchor="middle" font-size="12">{escape(money(float(tick)))}</text></g>'
        for tick in x_ticks
    )
    y_tick_svg = "\n".join(
        f'<g><line x1="{left - 6:.1f}" y1="{sy(float(tick)):.1f}" '
        f'x2="{left:.1f}" y2="{sy(float(tick)):.1f}" stroke="#333" />'
        f'<line x1="{left:.1f}" y1="{sy(float(tick)):.1f}" '
        f'x2="{left + plot_w:.1f}" y2="{sy(float(tick)):.1f}" stroke="#E5E7EB" />'
        f'<text x="{left - 10:.1f}" y="{sy(float(tick)) + 4:.1f}" '
        f'text-anchor="end" font-size="12">{escape(money(float(tick)))}</text></g>'
        for tick in y_ticks
    )
    polyline = (
        f'<polyline points="{" ".join(rolling_points)}" fill="none" '
        'stroke="#111111" stroke-width="2.4" />'
        if rolling_points
        else ""
    )

    svg_lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
        "  <style>",
        "    text { font-family: Arial, Helvetica, sans-serif; fill: #111827; }",
        "  </style>",
        (
            f'  <text x="{left}" y="30" font-size="20" font-weight="700">'
            "Phase 1: OOF Dollar Error vs. Home Price</text>"
        ),
        (
            f'  <rect x="{left}" y="{band_top:.1f}" width="{plot_w}" '
            f'height="{band_bottom - band_top:.1f}" fill="#F58518" '
            'fill-opacity="0.18"/>'
        ),
        f"  {y_tick_svg}",
        f"  {x_tick_svg}",
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        (
            f'  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
            f'y2="{top + plot_h}" stroke="#111827"/>'
        ),
        f"  {circles}",
        f"  {polyline}",
        (
            f'  <rect x="{left + 12}" y="{top + 12}" width="208" height="48" '
            'fill="#ffffff" fill-opacity="0.82" stroke="#D1D5DB"/>'
        ),
        (
            f'  <rect x="{left + 24}" y="{top + 28}" width="28" height="12" '
            'fill="#F58518" fill-opacity="0.18"/>'
        ),
        (f'  <text x="{left + 60}" y="{top + 39}" font-size="12">$10K-$20K margin range</text>'),
        (
            f'  <line x1="{left + 24}" y1="{top + 52}" x2="{left + 52}" '
            f'y2="{top + 52}" stroke="#111111" stroke-width="2.4"/>'
        ),
        f'  <text x="{left + 60}" y="{top + 56}" font-size="12">',
        "Rolling median error</text>",
        (
            f'  <text x="{left + plot_w / 2}" y="{height - 22}" '
            'text-anchor="middle" font-size="14">Actual sale price</text>'
        ),
        (
            f'  <text x="22" y="{top + plot_h / 2}" '
            f'transform="rotate(-90 22 {top + plot_h / 2})" '
            'text-anchor="middle" font-size="14">Absolute prediction error</text>'
        ),
        "</svg>",
    ]
    svg = "\n".join(svg_lines) + "\n"
    out.write_text(svg)
    logger.info("Saved figure: %s", out)
    return out


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
