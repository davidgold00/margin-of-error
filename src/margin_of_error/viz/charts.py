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

    from margin_of_error.config import EconomicsConfig

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")


def _save_or_show(fig: plt.Figure, filename: str | None) -> plt.Figure:
    """Save figure to reports/figures/ if filename given, otherwise return it."""
    if filename is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / filename
        fig.savefig(out, dpi=200, bbox_inches="tight")
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


# Verdict palette shared across all Phase 2 figures.
VERDICT_COLORS: dict[str, str] = {
    "APPROVE": "#2CA02C",  # green
    "REFER": "#F2C744",  # yellow
    "DECLINE": "#D62728",  # red
}


def plot_confrontation(
    frame: pd.DataFrame,
    flip_margin_low: float,
    flip_margin_high: float,
    save_as: str | None = "02a_confrontation.png",
) -> plt.Figure:
    """Figure 2A — The Confrontation Chart (the hero visual).

    X = actual sale price ($); Y = 90% CQR interval width ($). Each point is a
    test home, colored by underwriting verdict. The shaded horizontal band marks
    the plausible flip-margin range. The visual argument: most interval widths sit
    far above the flip-margin band — the model doesn't know enough to underwrite.

    Args:
        frame: phase2_test_underwriting DataFrame (needs actual_sale_price,
            interval_width_90, verdict).
        flip_margin_low / flip_margin_high: flip-margin band bounds ($).
        save_as: filename under reports/figures/.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.axhspan(
        flip_margin_low,
        flip_margin_high,
        color="#F58518",
        alpha=0.20,
        label=f"Flip-margin band (${flip_margin_low:,.0f}-${flip_margin_high:,.0f})",
    )
    for verdict, color in VERDICT_COLORS.items():
        sub = frame[frame["verdict"] == verdict]
        if sub.empty:
            continue
        ax.scatter(
            sub["actual_sale_price"],
            sub["interval_width_90"],
            s=30,
            alpha=0.7,
            color=color,
            edgecolor="none",
            label=f"{verdict} (n={len(sub)})",
        )
    median_width = float(frame["interval_width_90"].median())
    ax.axhline(median_width, color="#111111", linestyle="--", linewidth=1.3)
    ax.text(
        ax.get_xlim()[1],
        median_width,
        f"  median width ${median_width:,.0f}",
        va="center",
        ha="left",
        fontsize=9,
    )
    ax.set_title(
        "Model Uncertainty vs. Flip Margin:\nMost 'Profitable' Homes Are Ununderwritable",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("90% prediction-interval width ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_or_show(fig, save_as)


def plot_zillow_trap(
    frame: pd.DataFrame,
    top_n: int = 50,
    save_as: str | None = "02b_zillow_trap.png",
) -> plt.Figure:
    """Figure 2B — The Zillow Trap.

    Take the top-N homes by naive point-estimate profit (what a point model would
    call the best buys). Draw each home's profit uncertainty bar (p10–p90), a
    marker at expected profit, colored by verdict, sorted by expected profit. Many
    'best buys' have profit intervals straddling zero or fail the gate on width.

    Args:
        frame: phase2_test_underwriting DataFrame.
        top_n: number of top naive picks to show.
        save_as: filename under reports/figures/.
    """
    import matplotlib.pyplot as plt

    top = frame.nlargest(top_n, "naive_point_profit").sort_values("expected_profit")
    y = range(len(top))
    fig, ax = plt.subplots(figsize=(9.5, 10))
    for yi, (_, row) in zip(y, top.iterrows(), strict=False):
        color = VERDICT_COLORS.get(row["verdict"], "#888888")
        ax.plot(
            [row["profit_p10"], row["profit_p90"]],
            [yi, yi],
            color=color,
            linewidth=2.4,
            alpha=0.85,
        )
        ax.scatter(row["expected_profit"], yi, color=color, s=22, zorder=3)
    ax.axvline(0, color="#111111", linewidth=1.2, linestyle="-")
    ax.text(0, len(top) + 0.5, " break-even", fontsize=9, color="#111111")

    handles = [plt.Line2D([0], [0], color=c, lw=3, label=v) for v, c in VERDICT_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9, title="Verdict")
    ax.set_title(
        f"The Top {top_n} 'Deals': Point Estimates vs. What the Model Actually Knows",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel("Profit ($) — bar = 10th–90th percentile, dot = expected")
    ax.set_ylabel(f"Top {top_n} homes by naive point-estimate profit")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_or_show(fig, save_as)


def plot_approval_by_neighborhood(
    frame: pd.DataFrame,
    min_homes: int = 4,
    save_as: str | None = "02c_approval_by_neighborhood.png",
) -> plt.Figure:
    """Figure 2C — Approval Rate by Neighborhood.

    Bar chart of underwriting approval rate (% APPROVE + REFER) by neighborhood,
    annotated with median interval width. Model uncertainty is not uniform: some
    neighborhoods the model understands, others it doesn't.

    Args:
        frame: phase2_test_underwriting DataFrame.
        min_homes: drop neighborhoods with fewer than this many test homes.
        save_as: filename under reports/figures/.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    grp = frame.groupby("Neighborhood")
    stats = grp.agg(
        n=("verdict", "size"),
        approve_rate=("verdict", lambda s: float(np.mean(s.isin(["APPROVE", "REFER"])))),
        median_width=("interval_width_90", "median"),
    )
    stats = stats[stats["n"] >= min_homes].sort_values("approve_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.34 * len(stats))))
    colors = ["#2CA02C" if r >= 0.5 else "#D62728" for r in stats["approve_rate"]]
    ax.barh(stats.index, stats["approve_rate"], color=colors, alpha=0.85)
    for yi, (_, row) in enumerate(stats.iterrows()):
        label = (
            f"{row['approve_rate']:.0%}  "
            f"(med. width ${row['median_width']:,.0f}, n={int(row['n'])})"
        )
        ax.text(row["approve_rate"] + 0.01, yi, label, va="center", fontsize=8)
    ax.set_xlim(0, 1.35)
    ax.set_title(
        "Underwriting Approval Rate by Neighborhood\n(model certainty is not uniform)",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel("Approve + Refer rate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_or_show(fig, save_as)


def plot_calibration(
    empirical_coverages: list[float] | np.ndarray | pd.Series,
    nominal_levels: list[float] | np.ndarray | pd.Series,
    save_as: str | None = "02d_calibration.png",
) -> plt.Figure:
    """Figure 2D — calibration reliability diagram.

    Plots empirical vs. nominal coverage. Points on the diagonal mean the
    intervals deliver exactly the coverage they promise; deviation is
    miscalibration. Required for the scientific credibility of every downstream
    economic number.

    Args:
        empirical_coverages: measured coverage at each nominal level.
        nominal_levels: the nominal coverage targets (e.g., 0.5 … 0.95).
        save_as: filename under reports/figures/.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    nominal = np.asarray(nominal_levels, dtype=float)
    empirical = np.asarray(empirical_coverages, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", label="Perfect calibration")
    ax.plot(nominal, empirical, marker="o", color="#4C78A8", linewidth=2, label="CQR (test set)")
    for x, y in zip(nominal, empirical, strict=False):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_title(
        "CQR Calibration: Empirical vs. Nominal Coverage", fontsize=12.5, fontweight="bold"
    )
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Empirical coverage (test set)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save_or_show(fig, save_as)


def plot_naive_vs_causal(
    effects_df: pd.DataFrame,
    save_as: str | None = "03a_confounding_gap.png",
) -> plt.Figure:
    """Figure 3A — confounding gap: naive OLS vs. DML causal estimates.

    Args:
        effects_df: DataFrame from causal/dml.py compare_naive_vs_causal().
        save_as: Filename to save under reports/figures/.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    frame = effects_df.sort_values("DML Causal ($)", ascending=True).reset_index(drop=True)
    y = np.arange(len(frame))
    height = 0.36

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.55 * len(frame))))
    ax.barh(
        y - height / 2,
        frame["Naive OLS ($)"],
        height=height,
        color="#C44E52",
        alpha=0.86,
        label="Naive OLS",
    )
    causal = frame["DML Causal ($)"].to_numpy(dtype=float)
    ci_low = frame["DML CI Low ($)"].to_numpy(dtype=float)
    ci_high = frame["DML CI High ($)"].to_numpy(dtype=float)
    ax.barh(
        y + height / 2,
        causal,
        height=height,
        color="#4C78A8",
        alpha=0.9,
        label="DML causal",
        xerr=np.vstack([causal - ci_low, ci_high - causal]),
        error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#2A2A2A"},
    )

    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["Feature"])
    ax.set_xlabel("Estimated value per treatment unit ($)")
    ax.set_title(
        "What Renovations Are Actually Worth: Causal vs. Naive Estimates",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.22)

    annotate = frame.assign(abs_bias=frame["Bias ($)"].abs()).nlargest(3, "abs_bias")
    for _, row in annotate.iterrows():
        idx = int(frame.index[frame["Feature"] == row["Feature"]][0])
        x = float(row["DML Causal ($)"])
        bias = float(row["Bias ($)"])
        word = "overstates" if bias > 0 else "understates"
        ax.annotate(
            f"Naive {word} by ${abs(bias):,.0f}",
            xy=(x, idx + height / 2),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color="#222222",
        )

    fig.tight_layout()
    return _save_or_show(fig, save_as)


def plot_renovation_decision_matrix(
    effects_df: pd.DataFrame,
    save_as: str | None = "03b_renovation_decision_matrix.png",
) -> plt.Figure:
    """Figure 3B — causal return vs. typical renovation cost."""
    import matplotlib.pyplot as plt

    frame = effects_df.dropna(subset=["Treatment Cost ($)"]).copy()
    if frame.empty:
        raise ValueError("Treatment Cost ($) is required for the decision matrix")

    cost_cutoff = float(frame["Treatment Cost ($)"].median())
    fig, ax = plt.subplots(figsize=(10.5, 7))
    significant = frame["Statistically Significant?"].astype(bool)
    colors = significant.map({True: "#4C78A8", False: "#9C755F"})
    ax.scatter(
        frame["DML Causal ($)"],
        frame["Treatment Cost ($)"],
        s=95,
        c=colors,
        edgecolor="#222222",
        linewidth=0.7,
        alpha=0.9,
    )
    for _, row in frame.iterrows():
        ax.annotate(
            str(row["Feature"]),
            (float(row["DML Causal ($)"]), float(row["Treatment Cost ($)"])),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8.5,
        )

    ax.axvline(0, color="#333333", linewidth=1)
    ax.axhline(cost_cutoff, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("DML causal effect per unit ($)")
    ax.set_ylabel("Typical renovation cost from config ($)")
    ax.set_title("The Renovation Decision Matrix", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.22)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin, ymax, "High cost, low return -- avoid", va="top", ha="left", fontsize=9)
    ax.text(xmax, ymax, "High cost, high return -- selective", va="top", ha="right", fontsize=9)
    ax.text(xmin, ymin, "Low cost, low return -- skip", va="bottom", ha="left", fontsize=9)
    ax.text(xmax, ymin, "Low cost, high return -- always do", va="bottom", ha="right", fontsize=9)
    fig.tight_layout()
    return _save_or_show(fig, save_as)


def plot_verdict_flip_distributions(
    comparison_df: pd.DataFrame,
    economics: EconomicsConfig,
    save_as: str | None = "03c_verdict_flip_distributions.png",
) -> plt.Figure:
    """Figure 3C — before/after profit distributions for changed decisions."""
    import matplotlib.pyplot as plt
    import numpy as np

    from margin_of_error.economics.simulation import sample_flip_profit
    from margin_of_error.economics.underwriter import underwrite_best_tier

    frame = comparison_df.copy()
    if "verdict_changed" in frame.columns and frame["verdict_changed"].any():
        frame = frame.loc[frame["verdict_changed"]].copy()
    elif "expected_profit_delta" in frame.columns:
        ordered_index = frame["expected_profit_delta"].abs().sort_values(ascending=False).index
        frame = frame.reindex(ordered_index)
    frame = frame.head(10).reset_index(drop=True)
    if frame.empty:
        raise ValueError("At least one underwriting comparison row is required")

    n = len(frame)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows * 2.7)), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), frame.iterrows(), strict=False):
        base = float(row["base_predicted_arv"])
        low = float(row["interval_low_90"])
        high = float(row["interval_high_90"])
        corr = underwrite_best_tier(base, low, high, economics, uplift_mode="correlational")
        causal = underwrite_best_tier(base, low, high, economics, uplift_mode="causal")
        corr_cost = economics.flip.renovation_tiers[corr.renovation_tier].cost_usd
        causal_cost = economics.flip.renovation_tiers[causal.renovation_tier].cost_usd
        corr_profit = sample_flip_profit(
            corr.predicted_arv,
            corr.interval_low,
            corr.interval_high,
            corr_cost,
            economics,
            purchase_price=corr.purchase_price,
            seed=17,
        )
        causal_profit = sample_flip_profit(
            causal.predicted_arv,
            causal.interval_low,
            causal.interval_high,
            causal_cost,
            economics,
            purchase_price=causal.purchase_price,
            seed=23,
        )
        bins = np.linspace(
            min(float(corr_profit.min()), float(causal_profit.min())),
            max(float(corr_profit.max()), float(causal_profit.max())),
            34,
        )
        ax.hist(
            corr_profit,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#C44E52",
            label=f"Prior: {corr.verdict}",
        )
        ax.hist(
            causal_profit,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#4C78A8",
            label=f"Causal: {causal.verdict}",
        )
        ax.axvline(0, color="#222222", linewidth=0.9)
        ax.set_title(f"Id {int(row['Id'])}", fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        "Decisions That Change When You Get the Causality Right",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_or_show(fig, save_as)


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
