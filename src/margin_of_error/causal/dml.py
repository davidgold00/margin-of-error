"""Phase 3: causal renovation effects via manual Double Machine Learning.

The estimator follows the partially linear DML recipe:

1. Cross-fit an outcome model E[Y | W] with LightGBM.
2. Cross-fit a treatment model E[T | W] with LightGBM.
3. Regress residualized Y on residualized T with HC3 robust standard errors.

Y is log1p(SalePrice), T is one renovatable feature at a time, and W is the
fixed/confounding feature set from the Phase 1 registry plus OverallQual and
OverallCond. Dollar effects use the local delta-method approximation:
log coefficient * median(SalePrice).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from margin_of_error.config import EconomicsConfig, ModelConfig, load_economics, load_model_config
from margin_of_error.data.loaders import load_kaggle_train
from margin_of_error.data.schemas import validate_kaggle_train
from margin_of_error.economics.underwriter import (
    build_underwriting_comparison,
    detect_verdict_flips,
)
from margin_of_error.features.engineering import ORDINAL_COLS
from margin_of_error.features.missingness import (
    STRUCTURAL_NONE_CATEGORICALS,
    STRUCTURAL_ZERO_NUMERICS,
    TRUE_MISSING_FREQUENT_CATEGORICALS,
    TRUE_MISSING_MEDIAN_NUMERICS,
)
from margin_of_error.features.preprocessing import build_preprocessor
from margin_of_error.features.registry import (
    ENGINEERED_FEATURES,
    FEATURE_REGISTRY,
    FIXED_FEATURES,
    MUTABLE_FEATURES,
    assert_registry_covers_model_features,
)
from margin_of_error.models.baseline import (
    EarlyStoppingLightGBMRegressor,
    make_target,
    resolve_repo_path,
)
from margin_of_error.models.phase2 import load_phase1_artifact
from margin_of_error.viz.charts import (
    plot_naive_vs_causal,
    plot_renovation_decision_matrix,
    plot_verdict_flip_distributions,
)

logger = logging.getLogger(__name__)

DML_FOLDS = 5
Z_95 = 1.96
PRIMARY_PHASE2_UNDERWRITING = Path("reports/phase2_test_underwriting.csv")
PRIMARY_PHASE2_CARD = Path("reports/phase2_metric_card.json")
PRIMARY_PHASE2_CALIBRATION = Path("reports/phase2_calibration.csv")

TreatmentKind = Literal["ordinal", "count"]


@dataclass(frozen=True)
class TreatmentSpec:
    """One causal treatment definition sourced from the mutable registry."""

    feature: str
    label: str
    treatment_key: str
    kind: TreatmentKind
    unit: str
    rationale: str


@dataclass(frozen=True)
class ExcludedTreatment:
    """Brief-recommended treatment excluded after registry verification."""

    feature: str
    reason: str


@dataclass(frozen=True)
class CrossFitFoldRecord:
    """Training/test indices used to generate residuals for one fold."""

    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    @property
    def is_disjoint(self) -> bool:
        """True when no test row was used to fit that fold's nuisance models."""
        return set(self.train_indices).isdisjoint(self.test_indices)


@dataclass(frozen=True)
class DMLResiduals:
    """Cross-fitted residuals for one treatment."""

    y_residual: np.ndarray
    t_residual: np.ndarray
    fold_id: np.ndarray
    fold_records: tuple[CrossFitFoldRecord, ...]


@dataclass(frozen=True)
class TreatmentEffect:
    """Naive and DML estimates for one treatment, in log and dollar units."""

    feature: str
    label: str
    treatment_key: str
    unit: str
    kind: TreatmentKind
    naive_log_coef: float
    naive_log_se: float
    causal_log_coef: float
    causal_log_se: float
    p_value: float
    median_sale_price: float
    treatment_cost_usd: float | None = None

    @property
    def naive_dollars(self) -> float:
        return self.naive_log_coef * self.median_sale_price

    @property
    def naive_se_dollars(self) -> float:
        return self.naive_log_se * self.median_sale_price

    @property
    def causal_dollars(self) -> float:
        return self.causal_log_coef * self.median_sale_price

    @property
    def causal_se_dollars(self) -> float:
        return self.causal_log_se * self.median_sale_price

    @property
    def causal_ci_low_dollars(self) -> float:
        return (self.causal_log_coef - Z_95 * self.causal_log_se) * self.median_sale_price

    @property
    def causal_ci_high_dollars(self) -> float:
        return (self.causal_log_coef + Z_95 * self.causal_log_se) * self.median_sale_price

    @property
    def bias_dollars(self) -> float:
        return self.naive_dollars - self.causal_dollars

    @property
    def bias_pct(self) -> float:
        denom = abs(self.causal_dollars)
        if denom < 1e-9:
            return float("inf")
        return self.bias_dollars / denom

    @property
    def statistically_significant(self) -> bool:
        return self.causal_ci_low_dollars > 0 or self.causal_ci_high_dollars < 0

    @property
    def sensitivity_ratio(self) -> float | None:
        """Informal Kling-Manski-style ratio: effect / naive coefficient."""
        if abs(self.naive_dollars) < 1e-9:
            return None
        return self.causal_dollars / self.naive_dollars


@dataclass(frozen=True)
class Phase3RunResult:
    """Paths and headline payload from a Phase 3 run."""

    metric_card_path: Path
    effects_path: Path
    underwriting_comparison_path: Path
    metric_card: dict[str, Any]


_BRIEF_TREATMENT_SPECS: tuple[TreatmentSpec, ...] = (
    TreatmentSpec(
        "KitchenQual",
        "Kitchen quality",
        "KitchenQual_per_step",
        "ordinal",
        "one ordinal quality step",
        "Kitchen quality is directly changed by a kitchen renovation.",
    ),
    TreatmentSpec(
        "BsmtQual",
        "Basement quality",
        "BsmtQual_per_step",
        "ordinal",
        "one ordinal quality step",
        "Brief-recommended, but registry decides whether this is treatment or confounder.",
    ),
    TreatmentSpec(
        "BsmtFinType1",
        "Basement finish type",
        "BsmtFinType1_per_step",
        "ordinal",
        "one basement finish step",
        "Basement finish quality is a plausible investor renovation scope.",
    ),
    TreatmentSpec(
        "HeatingQC",
        "Heating quality",
        "HeatingQC_per_step",
        "ordinal",
        "one ordinal quality step",
        "Heating quality can be improved through HVAC replacement or repair.",
    ),
    TreatmentSpec(
        "FireplaceQu",
        "Fireplace quality",
        "FireplaceQu_per_step",
        "ordinal",
        "one ordinal quality step",
        "Missing fireplace quality is encoded as no fireplace before ordinal encoding.",
    ),
    TreatmentSpec(
        "GarageFinish",
        "Garage finish",
        "GarageFinish_per_step",
        "ordinal",
        "one garage finish step",
        "Garage finish can be improved cosmetically.",
    ),
    TreatmentSpec(
        "ExterQual",
        "Exterior quality",
        "ExterQual_per_step",
        "ordinal",
        "one ordinal quality step",
        "Exterior quality is partly renovatable and documented with a caveat.",
    ),
    TreatmentSpec(
        "FullBath",
        "Full bathrooms",
        "FullBath_per_unit",
        "count",
        "one full bathroom",
        "A full bathroom can be added or remodeled as renovation scope.",
    ),
    TreatmentSpec(
        "HalfBath",
        "Half bathrooms",
        "HalfBath_per_unit",
        "count",
        "one half bathroom",
        "A half bathroom can be added or remodeled as renovation scope.",
    ),
    TreatmentSpec(
        "BsmtFullBath",
        "Basement full bathrooms",
        "BsmtFullBath_per_unit",
        "count",
        "one basement full bathroom",
        "A basement full bathroom can be added during basement renovation.",
    ),
    TreatmentSpec(
        "Fireplaces",
        "Fireplaces",
        "Fireplaces_per_unit",
        "count",
        "one fireplace",
        "Brief-recommended, but registry decides whether this is treatment or confounder.",
    ),
    TreatmentSpec(
        "GarageCars",
        "Garage capacity",
        "GarageCars_per_unit",
        "count",
        "one garage bay",
        "Brief-recommended, but registry decides whether this is treatment or confounder.",
    ),
)

_FORCED_CONFOUNDERS = ("OverallQual", "OverallCond")


def select_treatment_specs() -> tuple[list[TreatmentSpec], list[ExcludedTreatment]]:
    """Return mutable treatment specs and brief-recommended exclusions."""
    assert_registry_covers_model_features()
    included: list[TreatmentSpec] = []
    excluded: list[ExcludedTreatment] = []

    for spec in _BRIEF_TREATMENT_SPECS:
        entry = FEATURE_REGISTRY.get(spec.feature)
        if entry is None:
            excluded.append(ExcludedTreatment(spec.feature, "missing from feature registry"))
            continue
        if spec.feature in _FORCED_CONFOUNDERS:
            excluded.append(ExcludedTreatment(spec.feature, "forced confounder per Phase 3 brief"))
            continue
        if entry.tag != "mutable":
            excluded.append(
                ExcludedTreatment(spec.feature, f"registry tag is {entry.tag}, not mutable")
            )
            continue
        included.append(spec)

    return included, excluded


def treatment_keys() -> list[str]:
    """Keys expected in config/economics.yaml causal_renovation_uplifts."""
    specs, _ = select_treatment_specs()
    return [spec.treatment_key for spec in specs]


def _nuisance_estimator(config: ModelConfig, seed: int) -> BaseEstimator:
    """Build the LightGBM nuisance model used inside each cross-fit fold."""
    return EarlyStoppingLightGBMRegressor(
        n_estimators=config.lightgbm.n_estimators,
        learning_rate=config.lightgbm.learning_rate,
        num_leaves=config.lightgbm.num_leaves,
        min_child_samples=config.lightgbm.min_child_samples,
        subsample=config.lightgbm.subsample,
        colsample_bytree=config.lightgbm.colsample_bytree,
        reg_alpha=config.lightgbm.reg_alpha,
        reg_lambda=config.lightgbm.reg_lambda,
        early_stopping_rounds=config.lightgbm.early_stopping_rounds,
        validation_fraction=config.phase1.early_stopping_validation_fraction,
        random_state=seed,
    )


def _nuisance_pipeline(config: ModelConfig, seed: int, y: pd.Series | np.ndarray) -> Pipeline:
    """Build a fold-local preprocessing + nuisance-model pipeline."""
    y_arr = np.asarray(y, dtype=float)
    if np.nanstd(y_arr) < 1e-12:
        model: BaseEstimator = DummyRegressor(strategy="mean")
    else:
        model = _nuisance_estimator(config, seed)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(config.features.drop)),
            ("model", model),
        ]
    )


def _confounder_source_columns(raw: pd.DataFrame) -> list[str]:
    """Raw columns needed to construct W, including YrSold for age features."""
    engineered = set(ENGINEERED_FEATURES)
    columns = [feature for feature in FIXED_FEATURES if feature not in engineered]
    columns.extend(feature for feature in _FORCED_CONFOUNDERS if feature not in columns)
    if {"YearBuilt", "YearRemodAdd"}.intersection(columns) and "YrSold" in raw.columns:
        columns.append("YrSold")
    return [column for column in dict.fromkeys(columns) if column in raw.columns]


def build_confounder_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the raw source columns used to build the confounder matrix W."""
    return raw[_confounder_source_columns(raw)].copy()


def encode_treatment(raw: pd.DataFrame, treatment: str) -> pd.Series:
    """Encode one treatment as a numeric series using Phase 1 ordinal policy."""
    if treatment not in raw.columns:
        raise KeyError(f"Treatment '{treatment}' is missing from the data")

    series = raw[treatment].copy()
    if treatment in ORDINAL_COLS:
        if treatment in STRUCTURAL_NONE_CATEGORICALS:
            series = series.fillna("None")
        elif treatment in TRUE_MISSING_FREQUENT_CATEGORICALS:
            mode = series.dropna().mode()
            series = series.fillna(mode.iloc[0] if not mode.empty else "None")
        mapping = {value: rank for rank, value in enumerate(ORDINAL_COLS[treatment])}
        return series.map(mapping).fillna(-1).astype(float)

    numeric = pd.to_numeric(series, errors="coerce")
    if treatment in STRUCTURAL_ZERO_NUMERICS:
        numeric = numeric.fillna(0)
    elif treatment in TRUE_MISSING_MEDIAN_NUMERICS:
        numeric = numeric.fillna(float(numeric.median()))
    else:
        numeric = numeric.fillna(0)
    return numeric.astype(float)


def cross_fit_residuals(
    raw: pd.DataFrame,
    y_log: pd.Series,
    treatment: str,
    config: ModelConfig,
    n_folds: int = DML_FOLDS,
    seed: int = 42,
) -> DMLResiduals:
    """Generate out-of-fold residuals for Y and T with no fold leakage."""
    if len(raw) != len(y_log):
        raise ValueError("raw and y_log must have the same number of rows")

    y_resid = np.empty(len(raw), dtype=float)
    t_resid = np.empty(len(raw), dtype=float)
    fold_id = np.empty(len(raw), dtype=int)
    fold_records: list[CrossFitFoldRecord] = []

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(raw), start=1):
        W_train = build_confounder_frame(raw.iloc[train_idx])
        W_test = build_confounder_frame(raw.iloc[test_idx])
        y_train = y_log.iloc[train_idx]
        t_train = encode_treatment(raw.iloc[train_idx], treatment)
        t_test = encode_treatment(raw.iloc[test_idx], treatment)

        y_model = _nuisance_pipeline(config, seed + fold * 101, y_train)
        t_model = _nuisance_pipeline(config, seed + fold * 211, t_train)

        y_model.fit(W_train, y_train)
        t_model.fit(W_train, t_train)

        y_hat = np.asarray(y_model.predict(W_test), dtype=float)
        t_hat = np.asarray(t_model.predict(W_test), dtype=float)
        y_resid[test_idx] = y_log.iloc[test_idx].to_numpy(dtype=float) - y_hat
        t_resid[test_idx] = t_test.to_numpy(dtype=float) - t_hat
        fold_id[test_idx] = fold
        fold_records.append(
            CrossFitFoldRecord(
                fold=fold,
                train_indices=tuple(int(i) for i in train_idx),
                test_indices=tuple(int(i) for i in test_idx),
            )
        )

    return DMLResiduals(
        y_residual=y_resid,
        t_residual=t_resid,
        fold_id=fold_id,
        fold_records=tuple(fold_records),
    )


def _fit_ols(y: np.ndarray, x: np.ndarray) -> Any:
    """Fit HC3-robust OLS with an intercept."""
    design = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    return sm.OLS(np.asarray(y, dtype=float), design).fit(cov_type="HC3")


def _fit_naive_ols(
    raw: pd.DataFrame,
    y_log: pd.Series,
    treatment: str,
    config: ModelConfig,
) -> tuple[float, float]:
    """Fit naive OLS: log price on treatment plus full confounder controls."""
    W = build_confounder_frame(raw)
    preprocessor = build_preprocessor(config.features.drop)
    W_design = np.asarray(preprocessor.fit_transform(W), dtype=float)
    treatment_values = encode_treatment(raw, treatment).to_numpy(dtype=float).reshape(-1, 1)
    design = np.hstack([treatment_values, W_design])
    result = _fit_ols(y_log.to_numpy(dtype=float), design)
    return float(result.params[1]), float(result.bse[1])


def estimate_treatment_effect(
    raw: pd.DataFrame,
    y_log: pd.Series,
    spec: TreatmentSpec,
    config: ModelConfig,
    median_sale_price: float,
    treatment_cost_usd: float | None = None,
    n_folds: int = DML_FOLDS,
    seed: int = 42,
) -> TreatmentEffect:
    """Estimate naive and DML effects for one treatment."""
    naive_coef, naive_se = _fit_naive_ols(raw, y_log, spec.feature, config)
    residuals = cross_fit_residuals(raw, y_log, spec.feature, config, n_folds=n_folds, seed=seed)
    final = _fit_ols(residuals.y_residual, residuals.t_residual.reshape(-1, 1))
    causal_coef = float(final.params[1])
    causal_se = float(final.bse[1])
    p_value = float(final.pvalues[1])

    if not np.isfinite(causal_se):
        raise ValueError(f"DML standard error for {spec.feature} is not finite")

    return TreatmentEffect(
        feature=spec.feature,
        label=spec.label,
        treatment_key=spec.treatment_key,
        unit=spec.unit,
        kind=spec.kind,
        naive_log_coef=naive_coef,
        naive_log_se=naive_se,
        causal_log_coef=causal_coef,
        causal_log_se=causal_se,
        p_value=p_value,
        median_sale_price=median_sale_price,
        treatment_cost_usd=treatment_cost_usd,
    )


def estimate_causal_effects(
    raw: pd.DataFrame,
    treatments: list[TreatmentSpec],
    config: ModelConfig,
    economics: EconomicsConfig,
    seed: int = 42,
    n_folds: int = DML_FOLDS,
) -> list[TreatmentEffect]:
    """Run the full DML pipeline independently for each treatment variable."""
    y_log = make_target(raw[config.target.column], config.target.transform)
    median_sale_price = float(raw[config.target.column].median())
    costs = economics.renovation.treatment_costs_usd
    effects: list[TreatmentEffect] = []
    for spec in treatments:
        logger.info("Estimating Phase 3 DML effect for %s", spec.feature)
        effects.append(
            estimate_treatment_effect(
                raw=raw,
                y_log=y_log,
                spec=spec,
                config=config,
                median_sale_price=median_sale_price,
                treatment_cost_usd=costs.get(spec.treatment_key),
                n_folds=n_folds,
                seed=seed,
            )
        )
    return effects


def _interpret_effect(effect: TreatmentEffect) -> str:
    naive = effect.naive_dollars
    causal = effect.causal_dollars
    bias = effect.bias_dollars
    verb = "overstates" if bias > 0 else "understates"
    significance = (
        "statistically clear at 95%"
        if effect.statistically_significant
        else "statistically inconclusive at 95%"
    )
    return (
        f"{effect.label}: naive analysis suggests {naive:,.0f} dollars per "
        f"{effect.unit}; DML estimates {causal:,.0f}. The naive estimate "
        f"{verb} by about {abs(bias):,.0f}; {significance}."
    )


def compare_naive_vs_causal(effects: list[TreatmentEffect]) -> pd.DataFrame:
    """Return the Phase 3 centrepiece naive-vs-causal comparison table."""
    rows: list[dict[str, object]] = []
    for effect in effects:
        rows.append(
            {
                "Feature": effect.feature,
                "Treatment Key": effect.treatment_key,
                "Unit": effect.unit,
                "Naive OLS ($)": effect.naive_dollars,
                "DML Causal ($)": effect.causal_dollars,
                "DML CI Low ($)": effect.causal_ci_low_dollars,
                "DML CI High ($)": effect.causal_ci_high_dollars,
                "DML SE ($)": effect.causal_se_dollars,
                "Bias ($)": effect.bias_dollars,
                "Bias (%)": effect.bias_pct,
                "P Value": effect.p_value,
                "Statistically Significant?": effect.statistically_significant,
                "Treatment Cost ($)": effect.treatment_cost_usd,
                "Sensitivity Ratio": effect.sensitivity_ratio,
                "Practical Interpretation": _interpret_effect(effect),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("Bias ($)", key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )


def causal_uplift_mapping(effects: list[TreatmentEffect]) -> dict[str, float]:
    """Map config uplift keys to DML point estimates in dollars."""
    return {effect.treatment_key: float(effect.causal_dollars) for effect in effects}


def _format_yaml_float(value: float) -> str:
    return f"{value:.2f}"


def update_economics_config_with_uplifts(
    config_path: Path | str,
    uplifts: dict[str, float],
    use_causal_uplifts: bool = True,
) -> None:
    """Populate the Phase 3 causal uplift block while preserving YAML comments."""
    path = resolve_repo_path(config_path)
    lines = path.read_text().splitlines()
    next_lines: list[str] = []
    in_causal_uplifts = False
    causal_indent = ""
    for line in lines:
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("causal_renovation_uplifts:"):
            in_causal_uplifts = True
            causal_indent = indent
            next_lines.append(line)
            continue
        if (
            in_causal_uplifts
            and stripped
            and not line.startswith(f"{causal_indent}  ")
            and not stripped.startswith("#")
        ):
            in_causal_uplifts = False
        if stripped.startswith("use_causal_uplifts:"):
            value = "true" if use_causal_uplifts else "false"
            next_lines.append(f"{indent}use_causal_uplifts: {value}")
            continue
        replaced = False
        if in_causal_uplifts:
            for key, uplift in uplifts.items():
                if stripped.startswith(f"{key}:"):
                    comment = ""
                    if "#" in line:
                        comment = "  #" + line.split("#", 1)[1]
                    next_lines.append(f"{indent}{key}: {_format_yaml_float(uplift)}{comment}")
                    replaced = True
                    break
        if not replaced:
            next_lines.append(line)
    path.write_text("\n".join(next_lines) + "\n")


def _load_required_json(path: Path | str) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required prior phase artifact is missing: {resolved}")
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Required JSON artifact is not an object: {resolved}")
    return payload


def _load_required_csv(path: Path | str) -> pd.DataFrame:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required prior phase artifact is missing: {resolved}")
    frame = pd.read_csv(resolved)
    if frame.empty:
        raise ValueError(f"Required prior phase artifact is empty: {resolved}")
    return frame


def verify_prior_artifacts(config: ModelConfig) -> dict[str, Any]:
    """Load Phase 1 and Phase 2 artifacts; raise if any are missing or corrupt."""
    phase1_path = resolve_repo_path(config.phase1.artifact_dir) / "baseline_lightgbm.joblib"
    _model, smearing, phase1_card = load_phase1_artifact(phase1_path)
    phase2_card = _load_required_json(PRIMARY_PHASE2_CARD)
    phase2_underwriting = _load_required_csv(PRIMARY_PHASE2_UNDERWRITING)
    phase2_calibration = _load_required_csv(PRIMARY_PHASE2_CALIBRATION)
    return {
        "phase1_smearing": smearing,
        "phase1_selected_model": phase1_card.get("selected_model"),
        "phase2_primary_coverage": phase2_card["cqr"]["primary_empirical_coverage"],
        "phase2_underwriting_rows": int(len(phase2_underwriting)),
        "phase2_calibration_rows": int(len(phase2_calibration)),
    }


def _representative_rows(frame: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """Pick deterministic representative homes across the predicted-ARV range."""
    ordered = frame.sort_values("predicted_arv").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, num=min(n_rows, len(ordered))).round().astype(int)
    return ordered.iloc[positions].reset_index(drop=True)


def _top_sensitivity_rows(table: pd.DataFrame, n_rows: int = 3) -> list[dict[str, object]]:
    ordered = table.sort_values("DML Causal ($)", key=lambda s: s.abs(), ascending=False).head(
        n_rows
    )
    rows: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        rows.append(
            {
                "Feature": row["Feature"],
                "Naive OLS ($)": row["Naive OLS ($)"],
                "DML Causal ($)": row["DML Causal ($)"],
                "Sensitivity Ratio": row["Sensitivity Ratio"],
            }
        )
    return rows


def run_phase3(
    model_config_path: Path | str = "config/model.yaml",
    economics_config_path: Path | str = "config/economics.yaml",
) -> Phase3RunResult:
    """Run Phase 3 end-to-end and persist all causal artifacts."""
    config = load_model_config(model_config_path)
    economics = load_economics(economics_config_path)
    prior = verify_prior_artifacts(config)

    train_path = resolve_repo_path(config.data.kaggle_train_path)
    raw = validate_kaggle_train(load_kaggle_train(train_path))
    treatments, excluded = select_treatment_specs()
    logger.info("Phase 3 treatments: %s", [spec.feature for spec in treatments])

    effects = estimate_causal_effects(raw, treatments, config, economics, seed=config.global_seed)
    table = compare_naive_vs_causal(effects)
    uplifts = causal_uplift_mapping(effects)

    update_economics_config_with_uplifts(economics_config_path, uplifts)
    economics = load_economics(economics_config_path)

    phase2_frame = _load_required_csv(PRIMARY_PHASE2_UNDERWRITING)
    representative = _representative_rows(phase2_frame)
    underwriting_comparison = build_underwriting_comparison(
        representative, economics, seed=config.global_seed
    )
    flips = detect_verdict_flips(underwriting_comparison)

    reports_dir = resolve_repo_path("reports")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    effects_path = reports_dir / "phase3_causal_effects.csv"
    underwriting_path = reports_dir / "phase3_underwriting_comparison.csv"
    metric_card_path = reports_dir / "phase3_metric_card.json"
    table.to_csv(effects_path, index=False)
    underwriting_comparison.to_csv(underwriting_path, index=False)

    plot_naive_vs_causal(table, save_as="03a_confounding_gap.png")
    plot_renovation_decision_matrix(table, save_as="03b_renovation_decision_matrix.png")
    plot_verdict_flip_distributions(
        flips if not flips.empty else underwriting_comparison,
        economics,
        save_as="03c_verdict_flip_distributions.png",
    )

    flip_counts = (
        flips["flip_direction"].value_counts().to_dict() if "flip_direction" in flips else {}
    )
    metric_card: dict[str, Any] = {
        "phase": "3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "estimator": {
            "type": "manual_cross_fitted_dml",
            "nuisance_models": "LightGBM regressors",
            "final_stage": "OLS on residuals with HC3 robust standard errors",
            "folds": DML_FOLDS,
            "seed": config.global_seed,
            "outcome": "log1p(SalePrice)",
            "dollar_backtransform": "coefficient * median(SalePrice)",
        },
        "prior_artifacts": prior,
        "registry": {
            "mutable_count": len(MUTABLE_FEATURES),
            "fixed_count": len(FIXED_FEATURES),
            "included_treatments": [asdict(spec) for spec in treatments],
            "excluded_brief_treatments": [asdict(item) for item in excluded],
            "forced_confounders": list(_FORCED_CONFOUNDERS),
        },
        "top_bias_features": table.head(5).to_dict(orient="records"),
        "sensitivity_check_top_3": _top_sensitivity_rows(table),
        "underwriting_comparison": {
            "representative_homes": int(len(underwriting_comparison)),
            "verdict_flips": int(len(flips)),
            "flip_direction_counts": flip_counts,
        },
        "outputs": {
            "effects_csv": str(effects_path.relative_to(resolve_repo_path("."))),
            "underwriting_comparison_csv": str(
                underwriting_path.relative_to(resolve_repo_path("."))
            ),
            "figures": [
                "reports/figures/03a_confounding_gap.png",
                "reports/figures/03b_renovation_decision_matrix.png",
                "reports/figures/03c_verdict_flip_distributions.png",
            ],
        },
    }
    metric_card_path.write_text(json.dumps(metric_card, indent=2, sort_keys=True) + "\n")
    logger.info("Phase 3 metric card written to %s", metric_card_path)

    return Phase3RunResult(
        metric_card_path=metric_card_path,
        effects_path=effects_path,
        underwriting_comparison_path=underwriting_path,
        metric_card=metric_card,
    )


def main() -> None:
    """CLI entrypoint for `python -m margin_of_error.causal.dml`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    result = run_phase3()
    headline = result.metric_card["underwriting_comparison"]
    print("\n=== Phase 3 headline ===")
    print(f"Effects table: {result.effects_path}")
    print(f"Representative homes: {headline['representative_homes']}")
    print(f"Verdict flips: {headline['verdict_flips']}")
    print(f"Flip directions: {headline['flip_direction_counts']}")


if __name__ == "__main__":
    main()
