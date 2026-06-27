"""Feature registry: mutable versus fixed source features.

This registry is the Phase 3 down payment required by Phase 1. The point model
uses all eligible features, but every source feature is tagged now so later
causal work can separate investor-changeable treatments from fixed confounders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureTag = Literal["mutable", "fixed"]

REGISTRY_VERSION = "phase1.0"


@dataclass(frozen=True)
class FeatureRegistryEntry:
    """One versioned feature tag with a short causal rationale."""

    feature: str
    tag: FeatureTag
    rationale: str


EXCLUDED_FEATURES: dict[str, str] = {
    "Id": "Identifier only; no valuation signal.",
    "SalePrice": "Target variable.",
    "SaleType": "Sale-specific field; excluded to avoid transaction-process leakage.",
    "SaleCondition": "Sale-specific field; excluded to avoid transaction-process leakage.",
    "YrSold": "Used only to derive age features, then dropped for random-split Phase 1.",
    "MoSold": "Sale timing field reserved for Phase 4 temporal backtesting.",
}


_MUTABLE_RATIONALES: dict[str, str] = {
    "OverallCond": "Overall condition can improve through repairs and maintenance.",
    "ExterQual": "Exterior material quality can change through siding/finish upgrades.",
    "ExterCond": "Exterior condition is maintenance-dependent.",
    "BsmtFinType1": "Basement finish quality is renovatable.",
    "BsmtFinSF1": "Finished basement area can be added or improved.",
    "BsmtFinType2": "Second basement finish quality is renovatable.",
    "BsmtFinSF2": "Second finished basement area can be added or improved.",
    "BsmtUnfSF": "Unfinished basement area is potential renovation scope.",
    "HeatingQC": "Heating quality/condition can be upgraded.",
    "CentralAir": "Central air can be added or replaced.",
    "Electrical": "Electrical system can be upgraded.",
    "BsmtFullBath": "Basement bathrooms can be added/remodeled.",
    "BsmtHalfBath": "Basement half bathrooms can be added/remodeled.",
    "FullBath": "Full bathrooms are investor-changeable renovation scope.",
    "HalfBath": "Half bathrooms are investor-changeable renovation scope.",
    "KitchenAbvGr": "Kitchen count is changeable, though uncommon in single-family flips.",
    "KitchenQual": "Kitchen quality is a canonical renovation treatment.",
    "Functional": "Functional defects can often be repaired.",
    "FireplaceQu": "Fireplace quality can be cosmetically improved.",
    "GarageFinish": "Garage finish can be improved.",
    "GarageQual": "Garage quality can be improved.",
    "GarageCond": "Garage condition can be repaired.",
    "WoodDeckSF": "Deck area can be added or rebuilt.",
    "OpenPorchSF": "Open porch area can be added or rebuilt.",
    "EnclosedPorch": "Enclosed porch area can be added or rebuilt.",
    "3SsnPorch": "Three-season porch area can be added or rebuilt.",
    "ScreenPorch": "Screen porch area can be added or rebuilt.",
    "Fence": "Fencing can be added or replaced.",
    "TotalBaths": "Derived bathroom count is investor-changeable.",
    "TotalPorchSF": "Derived porch/deck area is investor-changeable.",
    "HasRemodeled": (
        "Derived indicator of past renovation, not a treatment for Phase 1 but mutable history."
    ),
}


_FIXED_RATIONALES: dict[str, str] = {
    "MSSubClass": "Dwelling class is structural.",
    "MSZoning": "Zoning is location/regulatory context.",
    "LotFrontage": "Lot frontage is fixed without subdivision.",
    "LotArea": "Lot size is fixed without subdivision.",
    "Street": "Street access is location context.",
    "Alley": "Alley access is location context.",
    "LotShape": "Parcel geometry is fixed.",
    "LandContour": "Land contour is fixed.",
    "Utilities": "Utility availability is location infrastructure.",
    "LotConfig": "Lot configuration is fixed.",
    "LandSlope": "Land slope is fixed.",
    "Neighborhood": "Neighborhood is the primary location confounder.",
    "Condition1": "Proximity condition is location context.",
    "Condition2": "Secondary proximity condition is location context.",
    "BldgType": "Dwelling type is structural.",
    "HouseStyle": "House style is structural.",
    "OverallQual": "Overall material/build quality is a confounder, not a simple treatment.",
    "YearBuilt": "Construction year is fixed.",
    "YearRemodAdd": "Most recent remodel year is observed history at purchase.",
    "RoofStyle": "Roof style is structural.",
    "RoofMatl": "Roof material is structural and expensive to alter.",
    "Exterior1st": "Exterior covering is treated as fixed for causal control.",
    "Exterior2nd": "Secondary exterior covering is treated as fixed for causal control.",
    "MasVnrType": "Masonry veneer type is structural/cosmetic history.",
    "MasVnrArea": "Masonry veneer area is structural/cosmetic history.",
    "Foundation": "Foundation is fixed.",
    "BsmtQual": "Basement height/quality is largely structural.",
    "BsmtCond": "Basement condition is retained as fixed confounding history.",
    "BsmtExposure": "Basement walkout/exposure is lot/structure dependent.",
    "TotalBsmtSF": "Basement size is structural.",
    "Heating": "Heating system type is treated as fixed infrastructure for Phase 3.",
    "1stFlrSF": "First-floor area is structural.",
    "2ndFlrSF": "Second-floor area is structural.",
    "LowQualFinSF": "Low-quality finished area is structural.",
    "GrLivArea": "Above-grade living area is structural.",
    "BedroomAbvGr": "Bedroom count is treated as structural layout.",
    "TotRmsAbvGrd": "Room count is treated as structural layout.",
    "Fireplaces": "Fireplace count is structural.",
    "GarageType": "Garage type is treated as structural.",
    "GarageYrBlt": "Garage construction year is fixed history.",
    "GarageCars": "Garage capacity is structural.",
    "GarageArea": "Garage area is structural.",
    "PavedDrive": "Driveway paving is treated as site infrastructure.",
    "PoolArea": "Pool area is treated as fixed major site improvement.",
    "PoolQC": "Pool quality is rare and tied to fixed pool presence.",
    "MiscFeature": "Rare miscellaneous amenity is treated as fixed for Phase 1.",
    "MiscVal": "Miscellaneous amenity value is treated as fixed context.",
    "TotalSF": "Derived total square footage is structural.",
    "HouseAge": "Derived age at sale is fixed history.",
    "RemodAge": "Derived remodel recency is fixed observed history.",
    "HasPool": "Derived pool presence is fixed major site improvement.",
    "HasGarage": "Derived garage presence is structural.",
    "HasFireplace": "Derived fireplace presence is structural.",
    "Has2ndFloor": "Derived second-floor presence is structural.",
}


FEATURE_REGISTRY: dict[str, FeatureRegistryEntry] = {
    **{
        feature: FeatureRegistryEntry(feature, "mutable", rationale)
        for feature, rationale in _MUTABLE_RATIONALES.items()
    },
    **{
        feature: FeatureRegistryEntry(feature, "fixed", rationale)
        for feature, rationale in _FIXED_RATIONALES.items()
    },
}

MUTABLE_FEATURES: list[str] = sorted(_MUTABLE_RATIONALES)
FIXED_FEATURES: list[str] = sorted(_FIXED_RATIONALES)

ENGINEERED_FEATURES: tuple[str, ...] = (
    "TotalSF",
    "TotalBaths",
    "HouseAge",
    "RemodAge",
    "TotalPorchSF",
    "HasPool",
    "HasGarage",
    "HasFireplace",
    "Has2ndFloor",
    "HasRemodeled",
)

MODELED_RAW_FEATURES: tuple[str, ...] = (
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "MasVnrArea",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinSF1",
    "BsmtFinType2",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Functional",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MiscVal",
)


def get_all_registered_features() -> set[str]:
    """Return every mutable/fixed feature source."""
    return set(FEATURE_REGISTRY)


def get_model_feature_sources() -> set[str]:
    """Return raw plus engineered source features used by Phase 1 models."""
    return set(MODELED_RAW_FEATURES) | set(ENGINEERED_FEATURES)


def assert_no_overlap() -> None:
    """Raise ValueError if a feature appears in both registries."""
    overlap = set(MUTABLE_FEATURES) & set(FIXED_FEATURES)
    if overlap:
        raise ValueError(f"Feature(s) appear in both registries: {overlap}")


def assert_registry_covers_model_features() -> None:
    """Raise ValueError unless every modeled source feature has exactly one tag."""
    assert_no_overlap()
    missing = get_model_feature_sources() - get_all_registered_features()
    extra = get_all_registered_features() - get_model_feature_sources()
    if missing or extra:
        raise ValueError(
            f"Feature registry coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def source_feature_from_transformed_name(feature_name: str) -> str:
    """Map a transformed sklearn feature name back to its source feature."""
    if "__" in feature_name:
        feature_name = feature_name.split("__", 1)[1]
    for source in sorted(get_model_feature_sources(), key=len, reverse=True):
        if feature_name == source or feature_name.startswith(f"{source}_"):
            return source
    return feature_name
