"""Phase 3: Double Machine Learning (DML) for causal renovation effect estimation.

Uses EconML's LinearDML / CausalForestDML to estimate the causal effect of
MUTABLE features (from features/registry.py) on SalePrice, controlling for
FIXED features as nuisance confounders.

Why DML vs. naive regression:
    Naive OLS coefficient on KitchenQual → SalePrice is confounded: homes with
    expensive kitchens also tend to have high OverallQual, better neighborhoods,
    larger square footage, etc. The naive coefficient mixes causal effect with
    sorting effect (wealthy owners build better kitchens AND choose expensive
    neighborhoods). DML isolates the causal channel by:
        1. Predicting KitchenQual from FIXED features (residualizing treatment)
        2. Predicting SalePrice from FIXED features (residualizing outcome)
        3. Regressing residual-SalePrice on residual-KitchenQual

    The coefficient from step 3 is the causal effect, purged of confounding.

Key deliverable:
    A table comparing naive vs. DML-estimated effects with 95% CI for each
    mutable feature. The confounding bias (naive - causal) is the "renovation
    effect illusion" caused by OverallQual and neighborhood sorting.

PHASE 3 STATUS: Skeleton. Full implementation awaiting Phase 3 approval.
Requires: pip install -e '.[causal]'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TreatmentEffect:
    """Estimated causal effect for a single treatment variable.

    Attributes:
        feature: Name of the treatment variable (a MUTABLE feature).
        naive_coef: Naive OLS coefficient (confounded).
        naive_se: Standard error of naive coefficient.
        causal_coef: DML-estimated causal coefficient.
        causal_se: Standard error of causal estimate.
        bias: naive_coef - causal_coef (the confounding bias).
        confounding_ratio: abs(bias / causal_coef) — how much bias distorts naive estimate.
    """

    feature: str
    naive_coef: float
    naive_se: float
    causal_coef: float
    causal_se: float

    @property
    def bias(self) -> float:
        return self.naive_coef - self.causal_coef

    @property
    def confounding_ratio(self) -> float:
        if self.causal_coef == 0:
            return float("inf")
        return abs(self.bias / self.causal_coef)


def estimate_causal_effects(
    df: pd.DataFrame,
    treatments: list[str],
    controls: list[str],
    outcome: str = "SalePrice",
    seed: int = 42,
) -> list[TreatmentEffect]:
    """Estimate causal effects of treatment features using LinearDML.

    For each treatment in `treatments`:
        - Naive: OLS(log(outcome) ~ treatment + controls)
        - DML: EconML LinearDML with LightGBM nuisance models

    Args:
        df: Cleaned DataFrame with all features.
        treatments: List of MUTABLE feature names (treatments).
        controls: List of FIXED feature names (confounders / controls).
        outcome: Target column name (SalePrice or log1p(SalePrice)).
        seed: Random seed for nuisance model fitting.

    Returns:
        List of TreatmentEffect objects, one per treatment.

    Requires:
        econml package. Install with: pip install -e '.[causal]'
    """
    try:
        import econml  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Phase 3 requires EconML. Install with: pip install -e '.[causal]'"
        ) from exc

    raise NotImplementedError("Phase 3 not yet implemented — awaiting approval")


def compare_naive_vs_causal(effects: list[TreatmentEffect]) -> pd.DataFrame:
    """Format a comparison table of naive vs. DML effect estimates.

    Args:
        effects: List of TreatmentEffect from estimate_causal_effects().

    Returns:
        DataFrame with columns: Feature, NaiveEffect, CausalEffect, Bias,
        ConfoundingRatio, Significant (95% CI excludes 0).
    """
    rows = []
    for e in effects:
        rows.append(
            {
                "Feature": e.feature,
                "Naive ($/unit)": round(e.naive_coef, 0),
                "Naive SE": round(e.naive_se, 0),
                "Causal ($/unit)": round(e.causal_coef, 0),
                "Causal SE": round(e.causal_se, 0),
                "Bias (Naive - Causal)": round(e.bias, 0),
                "Confounding Ratio": round(e.confounding_ratio, 2),
            }
        )
    return pd.DataFrame(rows).set_index("Feature")
