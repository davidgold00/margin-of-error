"""Phase 5: Interactive underwriting tool (Streamlit app).

Launch with: make app  (or: streamlit run src/margin_of_error/app/underwriting.py)

Workflow:
    1. User inputs property characteristics (OverallQual, GrLivArea, Neighborhood, etc.)
    2. User inputs proposed purchase price and renovation plan
    3. App calls the CQR model to get a prediction interval for ARV
    4. App runs the P&L simulation from economics/simulation.py
    5. App displays:
        - Interval valuation: "This home is worth $X ± $Y (90% CI)"
        - Profit distribution chart
        - Underwriting verdict (UNDERWRITE / DECLINE) with reason
        - Key drivers of the price estimate (SHAP values)
        - Sensitivity table: what if renovation costs run 20% over?

PHASE 5 STATUS: Placeholder. Full implementation awaiting Phase 5 approval.
"""

from __future__ import annotations


def main() -> None:
    """Streamlit application entry point.

    Phase 5 implementation:
        - Load pre-trained CQR model from models/ directory
        - Load economics config from config/economics.yaml
        - Build Streamlit UI
        - Wire inputs → model → P&L simulation → display
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "Streamlit is required for the app. Install with: pip install streamlit"
        ) from exc

    st.title("Margin of Error — Fix-and-Flip Underwriting Tool")
    st.warning(
        "Phase 5 not yet implemented. This placeholder confirms the app scaffolding is correct. "
        "Run `make app` after Phase 5 is approved and implemented."
    )

    st.markdown(
        """
    **What this tool will do (Phase 5):**
    - Accept property characteristics and a proposed purchase price
    - Return a calibrated 90% prediction interval for after-repair value (ARV)
    - Run a P&L simulation and display the profit distribution
    - Issue an UNDERWRITE or DECLINE verdict with a plain-English reason
    - Show the key drivers of the price estimate
    """
    )


if __name__ == "__main__":
    main()
