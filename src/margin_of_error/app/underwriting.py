"""Phase 5 Streamlit underwriting tool and pure underwriting functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from margin_of_error.app.artifacts import AppArtifacts, ArtifactLoadError, load_app_artifacts
from margin_of_error.economics.simulation import sample_flip_profit
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


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _load_for_streamlit() -> AppArtifacts:
    """Streamlit-cached wrapper around the artifact loader."""
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _cached() -> AppArtifacts:
        return load_app_artifacts()

    return cast(AppArtifacts, _cached())


def _interval_chart(verdict: UnderwritingVerdict) -> Any:
    import plotly.graph_objects as go

    v = verdict.valuation
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[v.interval_low, v.interval_high],
            y=["90% value range", "90% value range"],
            mode="lines",
            line={"color": "#2563eb", "width": 16},
            hovertemplate="%{x:$,.0f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[v.point_value],
            y=["90% value range"],
            mode="markers",
            marker={"color": "#111827", "size": 13},
            name="Point value",
            hovertemplate="%{x:$,.0f}<extra></extra>",
        )
    )
    fig.add_vline(x=verdict.purchase_price, line_color="#dc2626", line_dash="dash")
    fig.update_layout(
        height=180,
        margin={"l": 10, "r": 10, "t": 20, "b": 30},
        xaxis_title="Dollars",
        yaxis_title=None,
    )
    return fig


def _profit_chart(verdict: UnderwritingVerdict) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=verdict.profit_draws,
            nbinsx=45,
            marker={"color": "#0f766e"},
            opacity=0.82,
            hovertemplate="%{x:$,.0f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_color="#dc2626", line_dash="dash")
    fig.add_vline(x=verdict.expected_profit, line_color="#111827")
    fig.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 20, "b": 35},
        xaxis_title="Simulated profit",
        yaxis_title="Draws",
        bargap=0.03,
    )
    return fig


def _guidance_frame(verdict: UnderwritingVerdict) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in verdict.causal_guidance])
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "feature": "Renovation",
            "current_value": "Current",
            "estimated_lift": "DML Lift",
            "estimated_cost": "Cost",
            "net_value": "Net",
            "payback_ratio": "Lift/Cost",
            "verdict": "Guidance",
        }
    )


def _input_overrides(artifacts: AppArtifacts) -> tuple[dict[str, Any], float, str]:
    import streamlit as st

    defaults = artifacts.defaults
    options = artifacts.options
    st.sidebar.title("Underwrite")
    st.sidebar.caption(
        "The interval is the model's honest 90% value range. The verdict declines deals when "
        "that uncertainty is too wide, loss probability is too high, or margin is too thin."
    )
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
        ),
        "GrLivArea": st.sidebar.number_input(
            "Living area", 400, 6000, int(defaults.get("GrLivArea") or 1500), step=50
        ),
        "OverallQual": st.sidebar.slider(
            "Overall quality", 1, 10, int(defaults.get("OverallQual") or 6)
        ),
        "YearBuilt": st.sidebar.number_input(
            "Year built", 1870, 2010, int(defaults.get("YearBuilt") or 1970), step=1
        ),
        "FullBath": st.sidebar.number_input(
            "Full baths", 0, 5, int(defaults.get("FullBath") or 2), step=1
        ),
        "HalfBath": st.sidebar.number_input(
            "Half baths", 0, 4, int(defaults.get("HalfBath") or 0), step=1
        ),
        "KitchenQual": st.sidebar.selectbox(
            "Kitchen quality",
            kitchen_options,
            index=kitchen_options.index(defaults.get("KitchenQual"))
            if defaults.get("KitchenQual") in kitchen_options
            else 0,
        ),
        "TotalBsmtSF": st.sidebar.number_input(
            "Basement area", 0, 4000, int(defaults.get("TotalBsmtSF") or 900), step=50
        ),
        "GarageCars": st.sidebar.number_input(
            "Garage spaces", 0, 5, int(defaults.get("GarageCars") or 2), step=1
        ),
        "GarageFinish": st.sidebar.selectbox(
            "Garage finish",
            garage_finish_options,
            index=garage_finish_options.index(defaults.get("GarageFinish"))
            if defaults.get("GarageFinish") in garage_finish_options
            else 0,
        ),
    }
    purchase_price = st.sidebar.number_input(
        "Purchase price", 10_000, 1_000_000, int(defaults.get("SalePrice") or 140_000), step=5_000
    )
    tier = st.sidebar.selectbox("Renovation tier", list(artifacts.economics.flip.renovation_tiers))
    return overrides, float(purchase_price), str(tier)


def main() -> None:
    """Streamlit application entry point."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - import-time user path
        raise ImportError("Streamlit is required for the app. Install with `make setup`.") from exc

    st.set_page_config(page_title="Margin of Error", page_icon="MOE", layout="wide")
    st.title("Margin of Error")
    st.caption("Uncertainty-aware fix-and-flip underwriting")

    try:
        artifacts = _load_for_streamlit()
    except ArtifactLoadError as exc:
        st.error(str(exc))
        return

    overrides, purchase_price, tier = _input_overrides(artifacts)

    try:
        verdict = underwrite_property(artifacts, overrides, purchase_price, tier)
    except Exception as exc:  # pragma: no cover - defensive user-facing path
        st.error(f"Could not underwrite this property: {exc}")
        return

    if verdict.verdict == "APPROVE":
        st.success(verdict.reason)
    elif verdict.verdict == "REFER":
        st.warning(verdict.reason)
    else:
        st.error(verdict.reason)

    v = verdict.valuation
    c1, c2, c3 = st.columns(3)
    c1.metric("Point valuation", _money(v.point_value))
    c2.metric("90% interval", f"{_money(v.interval_low)} to {_money(v.interval_high)}")
    c3.metric("Interval width", _money(v.interval_width))
    st.plotly_chart(_interval_chart(verdict), use_container_width=True)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Expected profit", _money(verdict.expected_profit))
    p2.metric("Downside P10", _money(verdict.profit_p10))
    p3.metric("Upside P90", _money(verdict.profit_p90))
    p4.metric("Probability of loss", _percent(verdict.prob_loss))
    st.plotly_chart(_profit_chart(verdict), use_container_width=True)

    guidance = _guidance_frame(verdict)
    if not guidance.empty:
        st.subheader("Causal Renovation Guidance")
        st.dataframe(
            guidance,
            hide_index=True,
            use_container_width=True,
            column_config={
                "DML Lift": st.column_config.NumberColumn(format="$%d"),
                "Cost": st.column_config.NumberColumn(format="$%d"),
                "Net": st.column_config.NumberColumn(format="$%d"),
                "Lift/Cost": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with st.expander("Assumptions"):
        st.json(artifacts.economics.flip.model_dump(mode="json"))


if __name__ == "__main__":
    main()
