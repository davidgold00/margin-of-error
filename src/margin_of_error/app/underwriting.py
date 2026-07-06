"""Phase 5 Streamlit underwriting tool and pure underwriting functions.

The module has two halves:

1. Pure functions (no Streamlit): ``build_property_frame``, ``underwrite_property``,
   ``renovation_guidance``, ``gate_checks``. These are imported by tests and
   contain all decision logic. Every economic threshold comes from
   ``config/economics.yaml`` via the loaded ``EconomicsConfig`` — never from
   constants in this file.
2. The Streamlit presentation layer (``main`` and ``_render_*`` helpers). It is
   written for a reader who has never seen the project: the verdict is
   explained in plain English, the three checks behind it are shown with their
   actual numbers against their configured thresholds, and every chart carries
   a caption saying how to read it.

Chart colors follow the project's validated palette (blue #2a78d6 for value,
red #e34948 for loss; status green/amber/red for verdicts) — see
docs/APP_GUIDE.md for the design notes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from margin_of_error.app.artifacts import AppArtifacts, ArtifactLoadError, load_app_artifacts
from margin_of_error.config import EconomicsConfig
from margin_of_error.economics.simulation import maximum_allowable_offer, sample_flip_profit
from margin_of_error.economics.underwriter import underwrite
from margin_of_error.models.baseline import log_predictions_to_dollars


@dataclass(frozen=True)
class ValuationResult:
    """Point value and calibrated interval for one property."""

    point_value: float
    interval_low: float
    interval_high: float
    interval_width: float
    nominal_coverage: float


@dataclass(frozen=True)
class GuidanceRow:
    """One causal-renovation row shown in the app."""

    feature: str
    current_value: str
    estimated_lift: float
    estimated_cost: float
    net_value: float
    payback_ratio: float
    verdict: str


@dataclass(frozen=True)
class UnderwritingVerdict:
    """Complete app-level verdict object used by tests and Streamlit."""

    valuation: ValuationResult
    verdict: str
    reason: str
    renovation_tier: str
    purchase_price: float
    expected_profit: float
    profit_p10: float
    profit_p90: float
    prob_loss: float
    prob_above_min_margin: float
    profit_draws: np.ndarray
    causal_guidance: list[GuidanceRow]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly payload, excluding raw draws."""
        payload = asdict(self)
        payload.pop("profit_draws", None)
        return payload


@dataclass(frozen=True)
class GateCheck:
    """One of the three underwriting checks, evaluated for one property.

    ``status`` is ``"pass"`` (meets the approve threshold), ``"borderline"``
    (meets only the refer threshold), or ``"fail"``. The statuses mirror
    ``margin_of_error.economics.underwriter.classify_verdict`` exactly — they
    are computed from the same config thresholds, so the panel the app renders
    can never disagree with the verdict the engine issued.
    """

    name: str
    question: str
    status: str
    detail: str


ORDINAL_MAX: dict[str, int] = {
    "KitchenQual": 5,
    "BsmtFinType1": 6,
    "HeatingQC": 5,
    "FireplaceQu": 5,
    "GarageFinish": 3,
    "ExterQual": 5,
}

TREATMENT_LABELS: dict[str, tuple[str, str]] = {
    "KitchenQual_per_step": ("Kitchen quality", "KitchenQual"),
    "BsmtFinType1_per_step": ("Basement finish", "BsmtFinType1"),
    "HeatingQC_per_step": ("Heating quality", "HeatingQC"),
    "FireplaceQu_per_step": ("Fireplace quality", "FireplaceQu"),
    "GarageFinish_per_step": ("Garage finish", "GarageFinish"),
    "ExterQual_per_step": ("Exterior quality", "ExterQual"),
    "FullBath_per_unit": ("Full bathroom", "FullBath"),
    "HalfBath_per_unit": ("Half bathroom", "HalfBath"),
    "BsmtFullBath_per_unit": ("Basement full bathroom", "BsmtFullBath"),
}


def build_property_frame(
    artifacts: AppArtifacts,
    overrides: dict[str, Any],
) -> pd.DataFrame:
    """Create the one-row model input using saved defaults plus user overrides."""
    row = dict(artifacts.defaults)
    for key, value in overrides.items():
        if key in row and value is not None:
            row[key] = value
    if "YearBuilt" in row and "YearRemodAdd" in row:
        row["YearRemodAdd"] = max(int(row["YearBuilt"]), int(row["YearRemodAdd"]))
    if "GarageCars" in overrides and "GarageArea" in row:
        row["GarageArea"] = max(float(row.get("GarageCars") or 0.0) * 240.0, 0.0)
    return pd.DataFrame([{column: row.get(column) for column in artifacts.feature_columns}])


def _score_valuation(artifacts: AppArtifacts, property_frame: pd.DataFrame) -> ValuationResult:
    """Run Phase 1 point model plus Phase 2 CQR interval."""
    point_log = np.asarray(artifacts.point_model.predict(property_frame))
    point_value = float(log_predictions_to_dollars(point_log, artifacts.smearing_factor)[0])
    interval = artifacts.cqr_model.predict(property_frame).to_dollars()
    low = float(interval.y_lower[0])
    high = float(interval.y_upper[0])
    return ValuationResult(
        point_value=point_value,
        interval_low=low,
        interval_high=high,
        interval_width=high - low,
        nominal_coverage=float(interval.nominal_coverage),
    )


def _current_value(property_frame: pd.DataFrame, column: str) -> str:
    if column not in property_frame.columns:
        return "not available"
    value = property_frame.iloc[0][column]
    if pd.isna(value):
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def renovation_guidance(
    artifacts: AppArtifacts,
    property_frame: pd.DataFrame,
) -> list[GuidanceRow]:
    """Per-property causal renovation guidance from Phase 3 DML estimates."""
    economics = artifacts.economics
    effects = economics.flip.causal_renovation_uplifts
    costs = economics.renovation.treatment_costs_usd
    rows: list[GuidanceRow] = []
    for treatment_key, effect in effects.items():
        if effect is None or treatment_key not in TREATMENT_LABELS:
            continue
        label, source_column = TREATMENT_LABELS[treatment_key]
        current = _current_value(property_frame, source_column)
        cost = float(costs.get(treatment_key, 0.0))
        lift = float(effect)
        net = lift - cost
        payback = lift / cost if cost > 0 else float("nan")
        if source_column in ORDINAL_MAX:
            numeric_current = property_frame.iloc[0].get(source_column)
            if isinstance(numeric_current, str):
                headroom_note = "review"
            elif (
                isinstance(numeric_current, (int, float, np.integer, np.floating))
                and float(numeric_current) >= ORDINAL_MAX[source_column]
            ):
                headroom_note = "already high"
            else:
                headroom_note = "review"
        else:
            headroom_note = "review"
        if net > 0 and headroom_note != "already high":
            verdict = "Pays on estimate"
        elif headroom_note == "already high":
            verdict = "Already high"
        else:
            verdict = "Cost exceeds lift"
        rows.append(
            GuidanceRow(
                feature=label,
                current_value=current,
                estimated_lift=lift,
                estimated_cost=cost,
                net_value=net,
                payback_ratio=payback,
                verdict=verdict,
            )
        )
    return sorted(rows, key=lambda row: row.net_value, reverse=True)


def gate_checks(verdict: UnderwritingVerdict, economics: EconomicsConfig) -> list[GateCheck]:
    """Evaluate the three underwriting checks for display.

    Mirrors ``classify_verdict``: the uncertainty cap is a hard override, the
    profit and loss checks each have an approve threshold and a looser refer
    (borderline) threshold. All thresholds come from the loaded economics
    config, so this panel and the engine's verdict share one source of truth.
    """
    uw = economics.flip.underwriting
    width = verdict.valuation.interval_width
    prob_above = verdict.prob_above_min_margin
    prob_loss = verdict.prob_loss

    width_status = "pass" if width <= uw.max_acceptable_interval_width_usd else "fail"
    if prob_above >= uw.approve_prob_above_min_margin:
        profit_status = "pass"
    elif prob_above >= uw.refer_prob_above_min_margin:
        profit_status = "borderline"
    else:
        profit_status = "fail"
    if prob_loss <= uw.approve_prob_loss_max:
        loss_status = "pass"
    elif prob_loss <= uw.refer_prob_loss_max:
        loss_status = "borderline"
    else:
        loss_status = "fail"

    return [
        GateCheck(
            name="Uncertainty check",
            question="Is the model sure enough about what this house is worth?",
            status=width_status,
            detail=(
                f"The model's 90% value range for this house is {_money(width)} wide; "
                f"the cap is {_money(uw.max_acceptable_interval_width_usd)}. "
                "If the range is wider than the cap the deal is declined automatically — "
                "a model that cannot pin down the value should not be trusted with capital, "
                "no matter how good the profit looks."
            ),
        ),
        GateCheck(
            name="Profit check",
            question="Is a genuinely good outcome likely enough?",
            status=profit_status,
            detail=(
                f"{_percent(prob_above)} of the simulated outcomes clear the "
                f"{_money(uw.minimum_underwrite_margin_buffer_usd)} profit floor. "
                f"Approval needs at least {_percent(uw.approve_prob_above_min_margin)}; "
                f"{_percent(uw.refer_prob_above_min_margin)} or better is borderline "
                "(sent to a human)."
            ),
        ),
        GateCheck(
            name="Loss check",
            question="Is losing money unlikely enough?",
            status=loss_status,
            detail=(
                f"{_percent(prob_loss)} of the simulated outcomes lose money. "
                f"Approval allows at most {_percent(uw.approve_prob_loss_max)}; "
                f"up to {_percent(uw.refer_prob_loss_max)} is borderline."
            ),
        ),
    ]


def underwrite_property(
    artifacts: AppArtifacts,
    feature_overrides: dict[str, Any],
    purchase_price: float,
    renovation_tier: str,
    seed: int = 42,
) -> UnderwritingVerdict:
    """Return a full uncertainty-aware verdict for one property."""
    if purchase_price <= 0:
        raise ValueError("Purchase price must be positive.")
    if renovation_tier not in artifacts.economics.flip.renovation_tiers:
        raise ValueError(f"Unknown renovation tier: {renovation_tier}")

    property_frame = build_property_frame(artifacts, feature_overrides)
    valuation = _score_valuation(artifacts, property_frame)
    result = underwrite(
        arv_point=valuation.point_value,
        arv_lower=valuation.interval_low,
        arv_upper=valuation.interval_high,
        renovation_tier=renovation_tier,
        economics=artifacts.economics,
        purchase_price=purchase_price,
        uplift_mode="config",
        seed=seed,
    )
    renovation_cost = artifacts.economics.flip.renovation_tiers[renovation_tier].cost_usd
    draws = sample_flip_profit(
        arv_point=result.predicted_arv,
        arv_lower=result.interval_low,
        arv_upper=result.interval_high,
        renovation_cost=renovation_cost,
        economics=artifacts.economics,
        purchase_price=purchase_price,
        seed=seed,
    )
    return UnderwritingVerdict(
        valuation=ValuationResult(
            point_value=result.predicted_arv,
            interval_low=result.interval_low,
            interval_high=result.interval_high,
            interval_width=result.interval_width_dollars,
            nominal_coverage=valuation.nominal_coverage,
        ),
        verdict=result.verdict,
        reason=result.decision_note,
        renovation_tier=renovation_tier,
        purchase_price=result.purchase_price,
        expected_profit=result.expected_profit,
        profit_p10=result.profit_p10,
        profit_p90=result.profit_p90,
        prob_loss=result.prob_loss,
        prob_above_min_margin=result.prob_above_min_margin,
        profit_draws=draws,
        causal_guidance=renovation_guidance(artifacts, property_frame),
    )


# ── Formatting & design tokens ───────────────────────────────────────────────

# Chart chrome and series colors. The blue/red pair and the status colors were
# checked with the palette validator (CVD separation and 3:1 surface contrast).
_TOKENS: dict[str, str] = {
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
    "value_blue": "#2a78d6",
    "loss_red": "#e34948",
    "status_good": "#0ca30c",
    "status_warn": "#b97900",
    "status_bad": "#d03b3b",
}

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_VERDICT_STYLE: dict[str, dict[str, str]] = {
    "APPROVE": {
        "icon": "✓",
        "color": _TOKENS["status_good"],
        "tint": "rgba(12, 163, 12, 0.08)",
        "headline": "Worth a serious look",
        "plain": (
            "This deal clears all three checks: the model is confident enough about the "
            "value, a good profit is likely, and a loss is unlikely — on these assumptions."
        ),
    },
    "REFER": {
        "icon": "⚠",
        "color": _TOKENS["status_warn"],
        "tint": "rgba(250, 178, 25, 0.12)",
        "headline": "Borderline — needs a human",
        "plain": (
            "The numbers don't clearly fail, but they aren't strong enough to approve "
            "automatically. A person should weigh in before any money moves."
        ),
    },
    "DECLINE": {
        "icon": "✕",
        "color": _TOKENS["status_bad"],
        "tint": "rgba(208, 59, 59, 0.08)",
        "headline": "Walk away",
        "plain": (
            "At least one check fails. The discipline of this tool is that no point "
            "estimate, however attractive, overrides a failed check."
        ),
    },
}

_GATE_CHIP: dict[str, tuple[str, str, str]] = {
    "pass": ("✓ PASS", _TOKENS["status_good"], "rgba(12, 163, 12, 0.10)"),
    "borderline": ("⚠ BORDERLINE", _TOKENS["status_warn"], "rgba(250, 178, 25, 0.14)"),
    "fail": ("✕ FAIL", _TOKENS["status_bad"], "rgba(208, 59, 59, 0.10)"),
}


def _money(value: float) -> str:
    """Format dollars for display, with a proper minus sign for losses."""
    if value < 0:
        return f"−${abs(value):,.0f}"
    return f"${value:,.0f}"


def _percent(value: float) -> str:
    """Format a probability for display; extremes read as bounds, not 0/100%."""
    if 0.0 < value < 0.01:
        return "<1%"
    if 0.99 < value < 1.0:
        return ">99%"
    return f"{value:.0%}"


def _load_for_streamlit() -> AppArtifacts:
    """Streamlit-cached wrapper around the artifact loader."""
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _cached() -> AppArtifacts:
        return load_app_artifacts()

    return cast(AppArtifacts, _cached())


# ── Charts ───────────────────────────────────────────────────────────────────


def _base_layout(fig: Any, height: int) -> None:
    """Shared chart chrome: quiet axes, transparent surface, system font."""
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 44, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": _FONT, "color": _TOKENS["ink_secondary"], "size": 13},
        hoverlabel={"font": {"family": _FONT, "size": 13}},
    )


def _interval_chart(verdict: UnderwritingVerdict) -> Any:
    """One horizontal band: the 90% value range, the best guess, and your price.

    Single series, so identity is carried by direct labels rather than a
    legend; the purchase price is a labeled reference line, not a series.
    """
    import plotly.graph_objects as go

    v = verdict.valuation
    lo, hi, point, price = v.interval_low, v.interval_high, v.point_value, verdict.purchase_price
    span_lo = min(lo, price)
    span_hi = max(hi, price)
    pad = (span_hi - span_lo) * 0.10 or 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[""],
            x=[hi - lo],
            base=[lo],
            orientation="h",
            width=0.28,
            marker={"color": _TOKENS["value_blue"], "cornerradius": 4},
            hovertemplate=(f"90% value range<br>{_money(lo)} to {_money(hi)}<extra></extra>"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[point],
            y=[""],
            mode="markers",
            marker={
                "color": _TOKENS["ink"],
                "size": 14,
                "line": {"color": _TOKENS["surface"], "width": 2},
            },
            hovertemplate=f"Model's best guess: {_money(point)}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=point,
        y=0.34,
        yref="y",
        text=f"Best guess {_money(point)}",
        showarrow=False,
        font={"color": _TOKENS["ink"], "size": 13},
        yanchor="bottom",
    )
    fig.add_annotation(
        x=lo,
        y=-0.36,
        yref="y",
        text=f"Could be as low as {_money(lo)}",
        showarrow=False,
        font={"color": _TOKENS["ink_secondary"], "size": 12},
        xanchor="left",
        yanchor="top",
    )
    fig.add_annotation(
        x=hi,
        y=-0.36,
        yref="y",
        text=f"or as high as {_money(hi)}",
        showarrow=False,
        font={"color": _TOKENS["ink_secondary"], "size": 12},
        xanchor="right",
        yanchor="top",
    )
    fig.add_vline(
        x=price,
        line_color=_TOKENS["ink_secondary"],
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"You pay {_money(price)}",
        annotation_position="top",
        annotation_font={"color": _TOKENS["ink"], "size": 13},
    )
    fig.update_xaxes(
        range=[span_lo - pad, span_hi + pad],
        tickformat="$,.0f",
        nticks=5,
        showgrid=False,
        zeroline=False,
        linecolor=_TOKENS["axis"],
        tickfont={"color": _TOKENS["muted"], "size": 12},
    )
    fig.update_yaxes(visible=False, range=[-0.8, 0.9])
    _base_layout(fig, height=190)
    return fig


def _profit_chart(verdict: UnderwritingVerdict) -> Any:
    """Histogram of the simulated outcomes, split at break-even.

    Bins are computed once with numpy and split by sign, so the loss (red) and
    profit (blue) traces sit on one shared grid and zero is always a bin edge.
    """
    import plotly.graph_objects as go

    draws = np.asarray(verdict.profit_draws, dtype=float)
    n = len(draws)
    lo, hi = float(draws.min()), float(draws.max())
    raw_step = (hi - lo) / 48 if hi > lo else 1.0
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = float(np.ceil(raw_step / magnitude) * magnitude)
    start = float(np.floor(lo / step) * step)
    stop = float(np.ceil(hi / step) * step)
    edges = np.arange(start, stop + step / 2, step)
    counts, edges = np.histogram(draws, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure()
    for label, color, mask in (
        ("Loses money", _TOKENS["loss_red"], centers < 0),
        ("Makes money", _TOKENS["value_blue"], centers >= 0),
    ):
        if not mask.any():
            continue
        custom = [
            (_money(float(left)), _money(float(right)))
            for left, right in zip(edges[:-1][mask], edges[1:][mask], strict=True)
        ]
        fig.add_trace(
            go.Bar(
                x=centers[mask],
                y=counts[mask],
                width=step * 0.92,
                name=label,
                marker={"color": color, "cornerradius": 2},
                customdata=custom,
                hovertemplate=(
                    "%{customdata[0]} to %{customdata[1]}<br>"
                    f"%{{y:,}} of {n:,} simulations<extra>%{{fullData.name}}</extra>"
                ),
            )
        )
    # Break-even needs no text label: the red/blue split, the $0 tick, and the
    # legend already say it. Label only the average, on whichever side of its
    # line has more room, so the two reference lines never collide.
    fig.add_vline(x=0, line_color=_TOKENS["ink"], line_width=1.5)
    mean_profit = verdict.expected_profit
    mean_on_left_half = mean_profit <= (float(edges[0]) + float(edges[-1])) / 2
    fig.add_vline(
        x=mean_profit,
        line_color=_TOKENS["ink_secondary"],
        line_dash="dash",
        line_width=1.5,
    )
    fig.add_annotation(
        x=mean_profit,
        y=1.0,
        yref="paper",
        yanchor="bottom",
        xanchor="left" if mean_on_left_half else "right",
        text=f"Average {_money(mean_profit)}",
        showarrow=False,
        font={"color": _TOKENS["ink"], "size": 13},
    )
    fig.update_xaxes(
        tickformat="$,.0f",
        showgrid=False,
        zeroline=False,
        linecolor=_TOKENS["axis"],
        tickfont={"color": _TOKENS["muted"], "size": 12},
        title={"text": "Profit or loss on this deal", "font": {"size": 13}},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_TOKENS["grid"],
        gridwidth=1,
        zeroline=False,
        tickfont={"color": _TOKENS["muted"], "size": 12},
        title={"text": "Simulations", "font": {"size": 13}},
    )
    fig.update_layout(
        barmode="overlay",
        bargap=0,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 13},
        },
    )
    _base_layout(fig, height=320)
    return fig


def _guidance_frame(verdict: UnderwritingVerdict) -> pd.DataFrame:
    """Guidance rows as a display frame with icon-carrying verdict chips."""
    frame = pd.DataFrame([asdict(row) for row in verdict.causal_guidance])
    if frame.empty:
        return frame
    chip = {
        "Pays on estimate": "✓ Pays for itself",
        "Already high": "— Already top quality",
        "Cost exceeds lift": "✕ Costs more than it adds",
    }
    frame["verdict"] = frame["verdict"].map(lambda v: chip.get(str(v), str(v)))
    return frame.rename(
        columns={
            "feature": "Upgrade",
            "current_value": "This house now",
            "estimated_lift": "Value added",
            "estimated_cost": "Typical cost",
            "net_value": "Net gain",
            "payback_ratio": "Added per $1 spent",
            "verdict": "Bottom line",
        }
    )


# ── Streamlit layout ─────────────────────────────────────────────────────────


def _inject_css() -> None:
    """Small, scoped style block for the custom banner and check rows."""
    import streamlit as st

    st.markdown(
        """
        <style>
        .moe-banner {
            display: flex; align-items: flex-start; gap: 14px;
            padding: 18px 20px; border-radius: 10px; margin-bottom: 4px;
        }
        .moe-banner .moe-icon {
            font-size: 26px; line-height: 1; font-weight: 700; margin-top: 2px;
        }
        .moe-banner .moe-verdict {
            font-size: 15px; font-weight: 700; letter-spacing: 0.08em;
        }
        .moe-banner .moe-headline {
            font-size: 22px; font-weight: 700; color: #0b0b0b; margin: 2px 0 6px 0;
        }
        .moe-banner .moe-plain { font-size: 14px; color: #52514e; margin: 0; }
        .moe-check {
            display: flex; align-items: flex-start; gap: 12px;
            padding: 12px 14px; border-radius: 8px; margin-bottom: 8px;
            background: #f7f6f3;
        }
        .moe-chip {
            font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
            padding: 3px 10px; border-radius: 999px; white-space: nowrap;
            margin-top: 1px;
        }
        .moe-check .moe-q { font-size: 14px; font-weight: 600; color: #0b0b0b; margin: 0; }
        .moe-check .moe-d { font-size: 13px; color: #52514e; margin: 3px 0 0 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_verdict_banner(verdict: UnderwritingVerdict) -> None:
    """Big, unambiguous verdict with a plain-English meaning and the engine's note."""
    import streamlit as st

    style = _VERDICT_STYLE.get(verdict.verdict, _VERDICT_STYLE["DECLINE"])
    st.markdown(
        f"""
        <div class="moe-banner" style="background:{style["tint"]};
             border-left: 5px solid {style["color"]};">
          <div class="moe-icon" style="color:{style["color"]};">{style["icon"]}</div>
          <div>
            <div class="moe-verdict" style="color:{style["color"]};">{verdict.verdict}</div>
            <div class="moe-headline">{style["headline"]}</div>
            <p class="moe-plain">{style["plain"]}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Engine note: {verdict.reason}")


def _render_gate_checks(verdict: UnderwritingVerdict, economics: EconomicsConfig) -> None:
    """The three checks behind the verdict, each with its numbers and threshold."""
    import streamlit as st

    checks = gate_checks(verdict, economics)
    st.subheader("Why? The three checks")
    st.caption(
        "Every verdict comes from these three checks — nothing else. "
        "All thresholds live in `config/economics.yaml`, not in code."
    )
    for check in checks:
        label, color, tint = _GATE_CHIP[check.status]
        st.markdown(
            f"""
            <div class="moe-check">
              <span class="moe-chip" style="color:{color}; background:{tint};">{label}</span>
              <div>
                <p class="moe-q">{check.name} — {check.question}</p>
                <p class="moe-d">{check.detail}</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if checks[0].status == "fail":
        st.caption(
            "The uncertainty check is a hard override: when it fails, the deal is "
            "declined even if the other two checks look fine. That refusal to buy "
            "what the model cannot price is the entire point of this project."
        )


def _render_assumptions(economics: EconomicsConfig) -> None:
    """Human-readable economics, with the raw config one tab away."""
    import streamlit as st

    flip = economics.flip
    uw = flip.underwriting
    with st.expander("Every assumption behind these numbers"):
        plain, raw = st.tabs(["Plain English", "Raw config"])
        with plain:
            st.markdown(
                f"""
**Deal costs** — selling costs of {flip.transaction_cost_pct:.0%} of the purchase
price, carrying costs of {flip.holding_cost_monthly_pct:.1%} per month
(taxes, insurance, utilities, upkeep), and an all-cash purchase (no loan
interest). The holding period is random in each simulation: about
{flip.holding_period_months_base:.0f} months on average, give or take
{flip.holding_period_months_std:.1f}, never less than
{flip.holding_period_months_min:.0f} or more than
{flip.holding_period_months_max:.0f} — because flips rarely close on schedule.

**Renovation plans** — each tier has a budget and a scope:
"""
            )
            tier_rows = [
                {
                    "Tier": name,
                    "Budget": _money(tier.cost_usd),
                    "Scope": tier.scope,
                }
                for name, tier in flip.renovation_tiers.items()
            ]
            st.dataframe(pd.DataFrame(tier_rows), hide_index=True, width="stretch")
            st.markdown(
                f"""
The value a renovation adds comes from the project's Phase 3 **causal**
estimates (double machine learning on Ames sales), not from simple
correlations — see the renovation table above.

**Verdict thresholds** — approve needs a ≥{_percent(uw.approve_prob_above_min_margin)}
chance of clearing a {_money(uw.minimum_underwrite_margin_buffer_usd)} profit floor
and a ≤{_percent(uw.approve_prob_loss_max)} chance of loss; borderline (refer) is
≥{_percent(uw.refer_prob_above_min_margin)} and ≤{_percent(uw.refer_prob_loss_max)};
and the 90% value range may be at most
{_money(uw.max_acceptable_interval_width_usd)} wide.

Every number above is loaded from `config/economics.yaml`, where each has a
written rationale. Change the file, restart the app, and the tool runs under
your assumptions. The full catalog with sources and sensitivity flags is in
`docs/assumptions.md`.
"""
            )
        with raw:
            st.json(flip.model_dump(mode="json"))


def _render_intro() -> None:
    """First-visit explainer: what the tool is and how to read it."""
    import streamlit as st

    with st.expander("First time here? How this works (60 seconds)"):
        st.markdown(
            """
**The one-line idea:** a price prediction is not a buying decision. This tool
prices a house *with honest uncertainty*, simulates the flip 10,000 times, and
only approves the deal when the numbers clear three explicit checks.

**What happens when you change an input:**

1. **The house is priced.** A model trained on 1,460 Ames, Iowa sales produces
   a best-guess value — and, more importantly, a *90% value range* that has
   been calibrated so that ranges like it contain the true price about 90% of
   the time. Wide range = the model honestly doesn't know.
2. **The deal is simulated 10,000 times.** Each simulation draws a plausible
   resale value from that range, a random holding period, and subtracts
   purchase price, renovation budget, selling costs, and carrying costs.
   The result is a full distribution of outcomes, not one hopeful number.
3. **Three checks issue the verdict.** Is the model sure enough? Is a good
   profit likely enough? Is a loss unlikely enough? APPROVE needs all three;
   borderline cases go to REFER; anything else is DECLINE.

**Things worth trying:** nudge the purchase price up until the verdict flips;
switch neighborhoods and watch the value range widen or tighten; give the
house a top-quality kitchen and see how little it changes a wide range.
The deep guide lives in `docs/APP_GUIDE.md`.

*This is a portfolio decision-system built on 2006–2010 Ames, Iowa data —
a demonstration, not investment advice.*
"""
        )


def _input_overrides(artifacts: AppArtifacts) -> tuple[dict[str, Any], float, str]:
    """Sidebar inputs: describe the house, then set the deal."""
    import streamlit as st

    defaults = artifacts.defaults
    options = artifacts.options
    st.sidebar.title("Underwrite a house")
    st.sidebar.caption(
        "Set the facts below; the verdict updates live. Anything you don't see "
        "here is filled with the most typical value in the Ames dataset."
    )

    st.sidebar.header("1 · The house")
    neighborhood_options = options.get("Neighborhood", ["NAmes"])
    kitchen_options = options.get("KitchenQual", ["TA", "Gd", "Ex"])
    garage_finish_options = options.get("GarageFinish", ["None", "Unf", "RFn", "Fin"])

    overrides = {
        "Neighborhood": st.sidebar.selectbox(
            "Neighborhood",
            neighborhood_options,
            index=neighborhood_options.index(defaults.get("Neighborhood"))
            if defaults.get("Neighborhood") in neighborhood_options
            else 0,
            help="Location is the strongest driver of both value and model certainty. "
            "Codes are Ames district abbreviations, e.g. NAmes = North Ames.",
        ),
        "GrLivArea": st.sidebar.number_input(
            "Living area (sq ft)",
            400,
            6000,
            int(defaults.get("GrLivArea") or 1500),
            step=50,
            help="Finished above-ground living area — one of the model's top value drivers.",
        ),
        "OverallQual": st.sidebar.slider(
            "Overall quality (1–10)",
            1,
            10,
            int(defaults.get("OverallQual") or 6),
            help="The assessor's overall material and finish rating. "
            "5–6 is average; 8+ is high-end.",
        ),
        "YearBuilt": st.sidebar.number_input(
            "Year built",
            1870,
            2010,
            int(defaults.get("YearBuilt") or 1970),
            step=1,
            help="The dataset covers sales from 2006–2010, so 2010 is the newest possible build.",
        ),
        "FullBath": st.sidebar.number_input(
            "Full baths",
            0,
            5,
            int(defaults.get("FullBath") or 2),
            step=1,
            help="Above-ground full bathrooms.",
        ),
        "HalfBath": st.sidebar.number_input(
            "Half baths",
            0,
            4,
            int(defaults.get("HalfBath") or 0),
            step=1,
            help="Above-ground half bathrooms (no shower/tub).",
        ),
        "KitchenQual": st.sidebar.selectbox(
            "Kitchen quality",
            kitchen_options,
            index=kitchen_options.index(defaults.get("KitchenQual"))
            if defaults.get("KitchenQual") in kitchen_options
            else 0,
            help="Ames quality codes: Po=poor, Fa=fair, TA=typical/average, Gd=good, Ex=excellent.",
        ),
        "TotalBsmtSF": st.sidebar.number_input(
            "Basement area (sq ft)",
            0,
            4000,
            int(defaults.get("TotalBsmtSF") or 900),
            step=50,
            help="Total basement footprint, finished or not. 0 means no basement.",
        ),
        "GarageCars": st.sidebar.number_input(
            "Garage spaces",
            0,
            5,
            int(defaults.get("GarageCars") or 2),
            step=1,
            help="Car capacity of the garage. 0 means no garage.",
        ),
        "GarageFinish": st.sidebar.selectbox(
            "Garage finish",
            garage_finish_options,
            index=garage_finish_options.index(defaults.get("GarageFinish"))
            if defaults.get("GarageFinish") in garage_finish_options
            else 0,
            help="Interior finish of the garage: Unf=unfinished, RFn=rough-finished, Fin=finished.",
        ),
    }

    st.sidebar.header("2 · The deal")
    purchase_price = st.sidebar.number_input(
        "Purchase price ($)",
        10_000,
        1_000_000,
        int(defaults.get("SalePrice") or 140_000),
        step=5_000,
        help="What you would actually pay for the house, before renovation. "
        "This is the number the whole verdict hinges on — try nudging it.",
    )
    tiers = artifacts.economics.flip.renovation_tiers
    tier = st.sidebar.selectbox(
        "Renovation plan",
        list(tiers),
        format_func=lambda name: f"{name} — {_money(tiers[name].cost_usd)}",
        help="Each plan has a budget and scope (see the assumptions panel). "
        "The value it adds comes from the project's causal Phase 3 estimates.",
    )
    st.sidebar.caption(f"Scope: {tiers[str(tier)].scope}")
    return overrides, float(purchase_price), str(tier)


def main() -> None:
    """Streamlit application entry point."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - import-time user path
        raise ImportError("Streamlit is required for the app. Install with `make setup`.") from exc

    st.set_page_config(
        page_title="Margin of Error — Flip Underwriter",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    st.title("Margin of Error")
    st.markdown(
        "**Should you buy this house to flip?** This tool doesn't just predict the "
        "price — it prices its own uncertainty, simulates the deal 10,000 times, "
        "and refuses to approve what the model can't confidently value."
    )
    _render_intro()

    try:
        artifacts = _load_for_streamlit()
    except ArtifactLoadError as exc:
        st.error(str(exc))
        st.code("make train uncertainty app-artifacts", language="bash")
        return

    overrides, purchase_price, tier = _input_overrides(artifacts)

    try:
        verdict = underwrite_property(artifacts, overrides, purchase_price, tier)
    except Exception as exc:  # pragma: no cover - defensive user-facing path
        st.error(f"Could not underwrite this property: {exc}")
        return

    _render_verdict_banner(verdict)
    _render_gate_checks(verdict, artifacts.economics)

    st.divider()
    st.subheader("What is the house worth?")
    st.caption(
        "The blue band is the model's calibrated 90% value range *after* your chosen "
        "renovation — ranges like this contain the true price about 9 times in 10. "
        "The dot is the single best guess; the dashed line is what you'd pay. "
        "A wide band is the model saying, honestly, that it does not know."
    )
    v = verdict.valuation
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Best-guess value (after reno)",
        _money(v.point_value),
        help="The point estimate — the number a naive tool would stop at.",
    )
    c2.metric(
        "90% value range",
        f"{_money(v.interval_low)} – {_money(v.interval_high)}",
        help="Calibrated so ranges like this contain the true price ~90% of the time.",
    )
    c3.metric(
        "Range width",
        _money(v.interval_width),
        help="The number the uncertainty check gates on. Narrow = confident model.",
    )
    st.plotly_chart(_interval_chart(verdict), width="stretch", config={"displayModeBar": False})

    st.divider()
    st.subheader(f"If you pay {_money(verdict.purchase_price)}, how does the flip go?")
    reno_cost = artifacts.economics.flip.renovation_tiers[verdict.renovation_tier].cost_usd
    mao = maximum_allowable_offer(v.point_value, reno_cost, artifacts.economics.flip)
    st.caption(
        f"10,000 simulated flips at the **{verdict.renovation_tier}** renovation tier "
        f"({_money(reno_cost)} budget). Each simulation draws a resale value from the "
        "range above and a realistic holding period, then subtracts every cost. "
        f"For reference, the traditional “70% rule” would cap the offer for this house "
        f"at about {_money(mao)}."
    )
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Average outcome",
        _money(verdict.expected_profit),
        help="The mean profit across all 10,000 simulations.",
    )
    p2.metric(
        "Bad case (P10)",
        _money(verdict.profit_p10),
        help="1 in 10 simulations ends worse than this.",
    )
    p3.metric(
        "Good case (P90)",
        _money(verdict.profit_p90),
        help="1 in 10 simulations ends better than this.",
    )
    p4.metric(
        "Chance of losing money",
        _percent(verdict.prob_loss),
        help="Share of simulations that end below break-even.",
    )
    st.plotly_chart(_profit_chart(verdict), width="stretch", config={"displayModeBar": False})

    guidance = _guidance_frame(verdict)
    if not guidance.empty:
        st.divider()
        st.subheader("Which upgrades actually pay?")
        st.caption(
            "Value-added figures are the project's *causal* estimates (double machine "
            "learning on Ames sales) — what the upgrade itself adds, after stripping out "
            "the fact that nicer houses tend to have nicer everything. Costs are "
            "documented national-average assumptions. Market-wide averages, not quotes "
            "for this specific house — and note the project's core finding: no upgrade "
            "story rescues a deal the uncertainty check has already declined."
        )
        st.dataframe(
            guidance,
            hide_index=True,
            width="stretch",
            column_config={
                "Value added": st.column_config.NumberColumn(
                    format="$%d", help="Causal (DML) estimate of resale value added."
                ),
                "Typical cost": st.column_config.NumberColumn(
                    format="$%d", help="National-average cost assumption for the work."
                ),
                "Net gain": st.column_config.NumberColumn(
                    format="$%d", help="Value added minus typical cost."
                ),
                "Added per $1 spent": st.column_config.NumberColumn(
                    format="%.2f", help="Above 1.00, the upgrade adds more than it costs."
                ),
            },
        )

    st.divider()
    _render_assumptions(artifacts.economics)
    st.caption(
        "Margin of Error is a portfolio decision-system built on 2006–2010 Ames, Iowa "
        "data — a demonstration of uncertainty-aware underwriting, not investment "
        "advice. How to use it well: `docs/APP_GUIDE.md` · the full story: "
        "`docs/PROJECT_EXPLAINER.md`."
    )


if __name__ == "__main__":
    main()
