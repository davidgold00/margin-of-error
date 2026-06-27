"""Missingness policy for Ames features.

The Ames data uses NA in two different ways:

- structural absence, such as no garage or no basement;
- true missingness, such as an unrecorded lot frontage.

Phase 1 model preprocessing consumes this table inside sklearn transformers so
learned imputations are fit only on the current training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MissingnessKind = Literal["structural_absence", "true_missing", "not_expected"]
MissingnessStrategy = Literal[
    "constant_none",
    "constant_zero",
    "neighborhood_median",
    "median",
    "most_frequent",
    "none",
]


@dataclass(frozen=True)
class MissingnessRule:
    """Policy row for one source feature."""

    feature: str
    kind: MissingnessKind
    strategy: MissingnessStrategy
    fill_value: str | int | float | None
    rationale: str


STRUCTURAL_NONE_CATEGORICALS: tuple[str, ...] = (
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
)

STRUCTURAL_ZERO_NUMERICS: tuple[str, ...] = (
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
)

TRUE_MISSING_MEDIAN_NUMERICS: tuple[str, ...] = ("GarageYrBlt",)

TRUE_MISSING_FREQUENT_CATEGORICALS: tuple[str, ...] = (
    "Electrical",
    "MSZoning",
    "Utilities",
    "Exterior1st",
    "Exterior2nd",
    "KitchenQual",
    "Functional",
)


def _structural_none_rule(feature: str) -> MissingnessRule:
    return MissingnessRule(
        feature=feature,
        kind="structural_absence",
        strategy="constant_none",
        fill_value="None",
        rationale="Ames data dictionary uses NA to mean the amenity is absent.",
    )


def _structural_zero_rule(feature: str) -> MissingnessRule:
    return MissingnessRule(
        feature=feature,
        kind="structural_absence",
        strategy="constant_zero",
        fill_value=0,
        rationale="A missing area/count occurs when the corresponding structure is absent.",
    )


MISSINGNESS_POLICY: dict[str, MissingnessRule] = {
    **{feature: _structural_none_rule(feature) for feature in STRUCTURAL_NONE_CATEGORICALS},
    **{feature: _structural_zero_rule(feature) for feature in STRUCTURAL_ZERO_NUMERICS},
    "LotFrontage": MissingnessRule(
        feature="LotFrontage",
        kind="true_missing",
        strategy="neighborhood_median",
        fill_value=None,
        rationale="Frontage varies by subdivision; fit neighborhood medians inside each fold.",
    ),
    **{
        feature: MissingnessRule(
            feature=feature,
            kind="true_missing",
            strategy="median",
            fill_value=None,
            rationale="Numeric field is genuinely missing; median is learned inside the fold.",
        )
        for feature in TRUE_MISSING_MEDIAN_NUMERICS
    },
    **{
        feature: MissingnessRule(
            feature=feature,
            kind="true_missing",
            strategy="most_frequent",
            fill_value=None,
            rationale="Categorical field is genuinely missing; mode is learned inside the fold.",
        )
        for feature in TRUE_MISSING_FREQUENT_CATEGORICALS
    },
}


def get_missingness_policy_table() -> list[MissingnessRule]:
    """Return the versioned missingness policy as sorted table rows."""
    return [MISSINGNESS_POLICY[key] for key in sorted(MISSINGNESS_POLICY)]
