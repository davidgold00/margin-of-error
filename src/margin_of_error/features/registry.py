"""Feature registries: mutable (renovatable) vs. fixed (structural) features.

This is a critical design artifact for Phase 3 causal analysis. The causal DML
model estimates the effect of MUTABLE features on price while controlling for
FIXED features as confounders. Mixing the two lists would contaminate causal
estimates.

Definitions:
    MUTABLE_FEATURES  — features an investor can change (bathroom additions,
                        kitchen upgrades, basement finishing, etc.)
    FIXED_FEATURES    — features set at construction and location
                        (lot size, neighborhood, year built, etc.)

When in doubt, a feature goes in FIXED. Wrongly classifying a fixed feature
as mutable would inflate estimated renovation returns.

Phase 1 note: both lists are used in feature engineering (all features are
used for the point model). The split only matters for Phase 3 causal analysis.
"""

from __future__ import annotations

# ── Mutable features (investor can change these) ─────────────────────────────

MUTABLE_FEATURES: list[str] = [
    # Kitchen
    "KitchenQual",  # Quality of kitchen (upgrade via remodel)
    "KitchenAbvGr",  # Number of kitchens above grade (rarely changed)
    # Bathrooms
    "FullBath",  # Full bathrooms above grade
    "HalfBath",  # Half baths above grade
    "BsmtFullBath",  # Basement full baths
    "BsmtHalfBath",  # Basement half baths
    # Basement
    "BsmtFinType1",  # Rating of basement finished area type 1
    "BsmtFinSF1",  # Finished basement area type 1 (sqft)
    "BsmtFinType2",  # Rating of basement finished area type 2
    "BsmtFinSF2",  # Finished basement area type 2 (sqft)
    "BsmtUnfSF",  # Unfinished basement area (can be finished)
    # Exterior / finish
    "ExterQual",  # Exterior material quality (can reside)
    "ExterCond",  # Exterior condition (maintenance-dependent)
    # Heating / cooling
    "HeatingQC",  # Heating quality and condition
    "CentralAir",  # Central air conditioning (can be added)
    # Garage
    "GarageType",  # Type of garage
    "GarageFinish",  # Interior finish of garage (can be improved)
    "GarageQual",  # Garage quality
    "GarageCond",  # Garage condition
    # Interior
    "FireplaceQu",  # Fireplace quality (cosmetic)
    "Functional",  # Home functionality (deductions for damage/defects)
    # Deck / porch
    "WoodDeckSF",  # Wood deck area (can be added)
    "OpenPorchSF",  # Open porch area
    "EnclosedPorch",  # Enclosed porch area
    "ScreenPorch",  # Screen porch area
]

# ── Fixed features (cannot be changed by the investor) ───────────────────────

FIXED_FEATURES: list[str] = [
    # Location (strongest confounder)
    "Neighborhood",  # Physical location
    "Condition1",  # Proximity to various conditions (railroad, etc.)
    "Condition2",
    "MSZoning",  # General zoning classification
    # Lot / land (quasi-fixed; lot size can't be changed without subdivision)
    "LotArea",
    "LotFrontage",
    "LotShape",
    "LotConfig",
    "LandContour",
    "LandSlope",
    "Street",
    "Alley",
    # Structure (fixed at construction)
    "MSSubClass",  # Building class / dwelling type
    "BldgType",  # Type of dwelling
    "HouseStyle",  # Style of dwelling
    "OverallQual",  # IMPORTANT: overall material and finish quality.
    # This is the key confounder for all renovation effects.
    # High-quality homes tend to have better kitchens, bathrooms, etc.
    # by construction, not renovation. Must be in FIXED for Phase 3.
    "OverallCond",  # Overall condition rating (partially mutable but treated as fixed
    # because it reflects cumulative history, not a single renovation)
    "YearBuilt",
    "YearRemodAdd",  # Year of most recent remodel (included as fixed because it
    # reflects past investment history, not current investor choice)
    "RoofStyle",
    "RoofMatl",
    "Foundation",
    "Exterior1st",  # Exterior covering on house (could be redone but treated as fixed
    # because cost is prohibitive and correlated with original quality)
    "Exterior2nd",
    "MasVnrType",  # Masonry veneer type
    "MasVnrArea",
    # Basement structure (fixed)
    "BsmtQual",  # Height of basement (cannot be changed)
    "BsmtCond",  # General condition of basement
    "BsmtExposure",  # Walkout or garden-level walls
    "TotalBsmtSF",  # Total basement area
    # Floor areas (structural)
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    # Rooms (structural; major additions are major renovations counted separately)
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",  # Number of fireplaces (structural)
    # Garage (structural)
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    # Other
    "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MiscVal",
    "Electrical",
    "Heating",
    "PavedDrive",
    "Utilities",
]

# ── Validation helper ─────────────────────────────────────────────────────────


def get_all_registered_features() -> set[str]:
    """Return the union of mutable and fixed feature names."""
    return set(MUTABLE_FEATURES) | set(FIXED_FEATURES)


def assert_no_overlap() -> None:
    """Raise ValueError if a feature appears in both registries."""
    overlap = set(MUTABLE_FEATURES) & set(FIXED_FEATURES)
    if overlap:
        raise ValueError(f"Feature(s) appear in both registries: {overlap}")
