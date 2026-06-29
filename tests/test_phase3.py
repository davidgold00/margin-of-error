"""Phase 3 causal-layer tests: cross-fitting, inference, config, and underwriting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from margin_of_error.causal.dml import (
    TreatmentSpec,
    cross_fit_residuals,
    estimate_treatment_effect,
    select_treatment_specs,
)
from margin_of_error.economics.underwriter import (
    build_underwriting_comparison,
    detect_verdict_flips,
    underwrite_best_tier,
)
from margin_of_error.features.engineering import ORDINAL_COLS
from margin_of_error.features.registry import MUTABLE_FEATURES


def _fast_model_config(model_config):
    lightgbm = model_config.lightgbm.model_copy(
        update={
            "n_estimators": 80,
            "num_leaves": 15,
            "min_child_samples": 5,
            "early_stopping_rounds": 10,
        }
    )
    phase1 = model_config.phase1.model_copy(update={"early_stopping_validation_fraction": 0.2})
    return model_config.model_copy(update={"lightgbm": lightgbm, "phase1": phase1})


def _synthetic_causal_frame(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    overall = rng.integers(2, 10, size=n_rows)
    kitchen_score = np.clip(overall // 2 + rng.integers(-1, 2, size=n_rows), 1, 5)
    quality_order = ORDINAL_COLS["KitchenQual"]
    kitchen = [quality_order[int(score)] for score in kitchen_score]
    price = 70_000 + 18_000 * overall + 7_500 * kitchen_score + rng.normal(0, 9_000, n_rows)
    return pd.DataFrame(
        {
            "Id": np.arange(1, n_rows + 1),
            "OverallQual": overall,
            "OverallCond": rng.integers(3, 9, size=n_rows),
            "YearBuilt": rng.integers(1940, 2008, size=n_rows),
            "YearRemodAdd": rng.integers(1960, 2010, size=n_rows),
            "YrSold": rng.choice([2006, 2007, 2008, 2009, 2010], size=n_rows),
            "LotArea": rng.integers(4_000, 18_000, size=n_rows),
            "GrLivArea": rng.integers(800, 3_200, size=n_rows),
            "TotalBsmtSF": rng.integers(0, 1_800, size=n_rows),
            "1stFlrSF": rng.integers(600, 1_800, size=n_rows),
            "2ndFlrSF": rng.integers(0, 1_400, size=n_rows),
            "GarageArea": rng.integers(0, 900, size=n_rows),
            "Fireplaces": rng.integers(0, 3, size=n_rows),
            "GarageCars": rng.integers(0, 4, size=n_rows),
            "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown"], size=n_rows),
            "BldgType": rng.choice(["1Fam", "TwnhsE"], size=n_rows),
            "HouseStyle": rng.choice(["1Story", "2Story"], size=n_rows),
            "Foundation": rng.choice(["PConc", "CBlock"], size=n_rows),
            "KitchenQual": kitchen,
            "SalePrice": np.clip(price, 50_000, None),
        }
    )


@pytest.fixture(scope="module")
def synthetic_effect(model_config):
    raw = _synthetic_causal_frame()
    fast_config = _fast_model_config(model_config)
    y_log = np.log1p(raw["SalePrice"])
    spec = TreatmentSpec(
        feature="KitchenQual",
        label="Kitchen quality",
        treatment_key="KitchenQual_per_step",
        kind="ordinal",
        unit="one quality step",
        rationale="Synthetic test treatment.",
    )
    return estimate_treatment_effect(
        raw,
        pd.Series(y_log),
        spec,
        fast_config,
        median_sale_price=float(raw["SalePrice"].median()),
        n_folds=5,
        seed=99,
    )


def test_cross_fitting_uses_no_test_fold_in_training(model_config) -> None:
    raw = _synthetic_causal_frame()
    fast_config = _fast_model_config(model_config)
    residuals = cross_fit_residuals(
        raw,
        pd.Series(np.log1p(raw["SalePrice"])),
        "KitchenQual",
        fast_config,
        n_folds=5,
        seed=42,
    )
    assert len(residuals.fold_records) == 5
    assert all(record.is_disjoint for record in residuals.fold_records)
    assert set(residuals.fold_id) == {1, 2, 3, 4, 5}


def test_dml_coefficients_have_finite_standard_errors(synthetic_effect) -> None:
    assert np.isfinite(synthetic_effect.causal_log_se)
    assert np.isfinite(synthetic_effect.causal_ci_low_dollars)
    assert np.isfinite(synthetic_effect.causal_ci_high_dollars)


def test_naive_ols_and_dml_produce_different_estimates(synthetic_effect) -> None:
    gap = abs(synthetic_effect.naive_dollars - synthetic_effect.causal_dollars)
    assert gap > 1e-6


def test_causal_config_populated_after_phase3(economics_config) -> None:
    uplifts = economics_config.flip.causal_renovation_uplifts
    specs, _excluded = select_treatment_specs()
    expected = {spec.treatment_key for spec in specs}
    assert expected.issubset(uplifts)
    assert all(uplifts[key] is not None for key in expected)


def test_underwriter_runs_in_both_causal_and_correlational_mode(economics_config) -> None:
    correlational = underwrite_best_tier(
        200_000, 178_000, 222_000, economics_config, uplift_mode="correlational"
    )
    causal = underwrite_best_tier(200_000, 178_000, 222_000, economics_config, uplift_mode="causal")
    assert correlational.verdict in {"APPROVE", "REFER", "DECLINE"}
    assert causal.verdict in {"APPROVE", "REFER", "DECLINE"}
    assert correlational.uplift_mode == "correlational"
    assert causal.uplift_mode == "causal"


def test_verdict_flip_detection_identifies_changed_decisions(economics_config) -> None:
    manual = pd.DataFrame(
        {
            "correlational_verdict": ["APPROVE", "DECLINE"],
            "causal_verdict": ["DECLINE", "DECLINE"],
        }
    )
    flips = detect_verdict_flips(manual)
    assert len(flips) == 1
    assert flips.iloc[0]["flip_direction"] == "correlational_APPROVE_to_causal_DECLINE"

    frame = pd.DataFrame(
        {
            "Id": [1, 2],
            "Neighborhood": ["NAmes", "OldTown"],
            "predicted_arv": [180_000.0, 260_000.0],
            "interval_low_90": [160_000.0, 225_000.0],
            "interval_high_90": [200_000.0, 295_000.0],
        }
    )
    comparison = build_underwriting_comparison(frame, economics_config, seed=3)
    assert "verdict_changed" in comparison.columns
    assert comparison["verdict_changed"].dtype == bool


def test_mutable_registry_covers_all_treatment_variables() -> None:
    specs, excluded = select_treatment_specs()
    assert specs
    assert all(spec.feature in MUTABLE_FEATURES for spec in specs)
    excluded_features = {item.feature for item in excluded}
    assert {"BsmtQual", "Fireplaces", "GarageCars"}.issubset(excluded_features)
    assert "OverallQual" not in {spec.feature for spec in specs}
    assert "OverallCond" not in {spec.feature for spec in specs}
