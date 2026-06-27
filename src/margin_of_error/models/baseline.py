"""Phase 1 baseline valuation model.

The baseline is a rigorous strawman: median predictor, ElasticNet, and a tuned
LightGBM point model. All reported residuals are out-of-fold. Log-space
predictions are retransformed with Duan's smearing estimator before dollar
metrics are computed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from margin_of_error.config import ModelConfig, load_model_config
from margin_of_error.data.loaders import load_kaggle_train
from margin_of_error.data.schemas import validate_kaggle_train
from margin_of_error.features.preprocessing import build_preprocessor, get_feature_names
from margin_of_error.features.registry import source_feature_from_transformed_name
from margin_of_error.viz.charts import write_price_error_svg

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FoldScore:
    """Metrics for one outer CV fold."""

    model: str
    repeat: int
    fold: int
    rmse_log: float
    mae_log: float
    rmse_dollars: float
    mae_dollars: float
    smearing_factor: float


@dataclass
class BaselineResult:
    """Output from repeated-CV model evaluation."""

    name: str
    oof_predictions_log: np.ndarray
    oof_predictions_dollars: np.ndarray
    prediction_counts: np.ndarray
    fold_scores: pd.DataFrame
    summary: dict[str, float]
    estimator: Any = field(repr=False)
    feature_importance: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass(frozen=True)
class Phase1RunResult:
    """All persisted Phase 1 outputs."""

    metric_card_path: Path
    residuals_path: Path
    artifact_path: Path
    figure_path: Path
    metric_card: dict[str, Any]


class EarlyStoppingLightGBMRegressor(BaseEstimator, RegressorMixin):
    """Small sklearn-compatible LightGBM wrapper with internal validation split."""

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        validation_fraction: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> EarlyStoppingLightGBMRegressor:
        """Fit LightGBM with a train-fold-only early-stopping validation split."""
        try:
            from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "LightGBM could not load. On macOS this usually means libomp is missing; "
                "install it with `brew install libomp`, then rerun Phase 1."
            ) from exc

        y_array = np.asarray(y)
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X,
            y_array,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )
        self.model_ = LGBMRegressor(
            objective="regression",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
        )
        self.model_.fit(
            X_fit,
            y_fit,
            eval_set=[(X_eval, y_eval)],
            eval_metric="rmse",
            callbacks=[
                early_stopping(self.early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log-price values."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return np.asarray(self.model_.predict(X))


def resolve_repo_path(path: Path | str) -> Path:
    """Resolve a config path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate


def make_target(sale_price: pd.Series, transform: str = "log1p") -> pd.Series:
    """Create the modeling target from SalePrice."""
    if transform != "log1p":
        raise ValueError(f"Unsupported target transform: {transform}")
    return pd.Series(np.log1p(sale_price.to_numpy()), index=sale_price.index, name="log_sale_price")


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared error."""
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def duan_smearing_factor(y_true_log: np.ndarray | pd.Series, y_pred_log: np.ndarray) -> float:
    """Estimate retransformation bias correction using Duan smearing."""
    residuals = np.asarray(y_true_log) - np.asarray(y_pred_log)
    return float(np.mean(np.exp(residuals)))


def log_predictions_to_dollars(y_pred_log: np.ndarray, smearing_factor: float) -> np.ndarray:
    """Back-transform log1p predictions to dollars with smearing correction."""
    dollars = np.maximum(np.exp(np.asarray(y_pred_log)) * smearing_factor - 1.0, 0.0)
    return cast(np.ndarray, dollars)


def make_cv_splits(
    X: pd.DataFrame,
    config: ModelConfig,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Create repeated outer folds, stratified by configured column when feasible."""
    n_folds = config.cross_validation.n_folds
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    stratify_col = config.cross_validation.stratify_by

    for repeat in range(config.phase1.cv_repeats):
        seed = config.global_seed + repeat
        if stratify_col in X.columns:
            labels = X[stratify_col].fillna("Missing").astype(str)
            counts = labels.value_counts()
            rare_labels = counts[counts < n_folds].index.tolist()
            rare_total = int(counts.loc[rare_labels].sum()) if rare_labels else 0
            if rare_labels and rare_total < n_folds:
                for label, count in counts.drop(index=rare_labels).sort_values().items():
                    rare_labels.append(str(label))
                    rare_total += int(count)
                    if rare_total >= n_folds:
                        break
            if len(rare_labels) > 0:
                labels = labels.mask(labels.isin(rare_labels), "__rare_neighborhood__")
                logger.info(
                    "Bucketed %d rare/small %s levels for stratified CV",
                    len(rare_labels),
                    stratify_col,
                )
            if labels.value_counts().min() >= n_folds:
                splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                fold_iter = splitter.split(X, labels)
            else:
                logger.warning(
                    "Falling back to KFold because %s has a class with fewer than %d rows",
                    stratify_col,
                    n_folds,
                )
                splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                fold_iter = splitter.split(X)
        else:
            logger.warning("Falling back to KFold because %s is absent", stratify_col)
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_iter = splitter.split(X)

        for fold, (train_idx, valid_idx) in enumerate(fold_iter, start=1):
            splits.append((repeat + 1, fold, train_idx, valid_idx))

    return splits


def make_dummy_estimator(_config: ModelConfig, _seed: int) -> BaseEstimator:
    """Median log-price predictor."""
    return DummyRegressor(strategy="median")


def make_elastic_net_estimator(config: ModelConfig, seed: int) -> GridSearchCV:
    """Regularized linear model with nested CV tuning on each outer train fold."""
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(config.features.drop)),
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    random_state=seed,
                    max_iter=config.phase1.elastic_net.max_iter,
                    selection="cyclic",
                ),
            ),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__alpha": config.phase1.elastic_net.alpha_grid,
            "model__l1_ratio": config.phase1.elastic_net.l1_ratio_grid,
        },
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=config.phase1.inner_cv_folds, shuffle=True, random_state=seed),
        n_jobs=1,
        refit=True,
    )


def make_lightgbm_estimator(config: ModelConfig, seed: int) -> GridSearchCV:
    """Primary Phase 1 gradient booster with nested CV hyperparameter search."""
    model = EarlyStoppingLightGBMRegressor(
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
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(config.features.drop)),
            ("model", model),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__num_leaves": config.phase1.lightgbm_tuning.num_leaves_grid,
            "model__min_child_samples": config.phase1.lightgbm_tuning.min_child_samples_grid,
            "model__reg_lambda": config.phase1.lightgbm_tuning.reg_lambda_grid,
        },
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=config.phase1.inner_cv_folds, shuffle=True, random_state=seed),
        n_jobs=1,
        refit=True,
    )


EstimatorFactory = Callable[[ModelConfig, int], BaseEstimator]


def cross_validate_model(
    name: str,
    estimator_factory: EstimatorFactory,
    X: pd.DataFrame,
    y_log: pd.Series,
    config: ModelConfig,
) -> BaselineResult:
    """Evaluate one estimator with repeated outer CV and OOF residuals."""
    splits = make_cv_splits(X, config)
    pred_log_sum = np.zeros(len(X), dtype=float)
    pred_dollar_sum = np.zeros(len(X), dtype=float)
    pred_counts = np.zeros(len(X), dtype=float)
    fold_scores: list[FoldScore] = []

    for repeat, fold, train_idx, valid_idx in splits:
        seed = config.global_seed + repeat * 100 + fold
        estimator = estimator_factory(config, seed)
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y_log.iloc[train_idx]
        y_valid = y_log.iloc[valid_idx]

        logger.info("Fitting %s repeat=%d fold=%d", name, repeat, fold)
        estimator.fit(X_train, y_train)
        pred_valid_log = np.asarray(estimator.predict(X_valid))
        pred_train_log = np.asarray(estimator.predict(X_train))
        smearing = duan_smearing_factor(y_train, pred_train_log)

        pred_valid_dollars = log_predictions_to_dollars(pred_valid_log, smearing)
        actual_valid_dollars = np.expm1(y_valid.to_numpy())

        pred_log_sum[valid_idx] += pred_valid_log
        pred_dollar_sum[valid_idx] += pred_valid_dollars
        pred_counts[valid_idx] += 1

        fold_scores.append(
            FoldScore(
                model=name,
                repeat=repeat,
                fold=fold,
                rmse_log=rmse(y_valid, pred_valid_log),
                mae_log=mae(y_valid, pred_valid_log),
                rmse_dollars=rmse(actual_valid_dollars, pred_valid_dollars),
                mae_dollars=mae(actual_valid_dollars, pred_valid_dollars),
                smearing_factor=smearing,
            )
        )

    if np.any(pred_counts == 0):
        raise RuntimeError("OOF prediction coverage failed: at least one row was never predicted")

    fold_df = pd.DataFrame([score.__dict__ for score in fold_scores])
    summary = summarize_fold_scores(fold_df)
    fitted_estimator = clone(estimator_factory(config, config.global_seed))
    fitted_estimator.fit(X, y_log)
    feature_importance = extract_feature_importance(fitted_estimator)

    return BaselineResult(
        name=name,
        oof_predictions_log=pred_log_sum / pred_counts,
        oof_predictions_dollars=pred_dollar_sum / pred_counts,
        prediction_counts=pred_counts,
        fold_scores=fold_df,
        summary=summary,
        estimator=fitted_estimator,
        feature_importance=feature_importance,
    )


def summarize_fold_scores(fold_scores: pd.DataFrame) -> dict[str, float]:
    """Summarize repeated-CV fold metrics with means and standard deviations."""
    metrics = ["rmse_log", "mae_log", "rmse_dollars", "mae_dollars", "smearing_factor"]
    summary: dict[str, float] = {}
    for metric in metrics:
        summary[f"{metric}_mean"] = float(fold_scores[metric].mean())
        summary[f"{metric}_std"] = float(fold_scores[metric].std(ddof=1))
    return summary


def extract_feature_importance(estimator: BaseEstimator) -> pd.Series:
    """Extract feature importances from a fitted LightGBM GridSearchCV pipeline."""
    if not hasattr(estimator, "best_estimator_"):
        return pd.Series(dtype=float)
    pipeline = estimator.best_estimator_
    if not isinstance(pipeline, Pipeline):
        return pd.Series(dtype=float)
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocess")
    if model is None or preprocessor is None or not hasattr(model, "model_"):
        return pd.Series(dtype=float)
    names = get_feature_names(preprocessor)
    importances = np.asarray(model.model_.feature_importances_)
    if len(names) != len(importances):
        return pd.Series(dtype=float)
    return pd.Series(importances, index=names).sort_values(ascending=False)


def build_residual_frame(
    raw: pd.DataFrame, y_log: pd.Series, result: BaselineResult
) -> pd.DataFrame:
    """Build row-level OOF residual diagnostics for the selected model."""
    actual_dollars = np.expm1(y_log.to_numpy())
    residual_dollars = result.oof_predictions_dollars - actual_dollars
    frame = pd.DataFrame(
        {
            "Id": raw["Id"].to_numpy() if "Id" in raw.columns else np.arange(len(raw)),
            "SalePrice": actual_dollars,
            "predicted_dollars": result.oof_predictions_dollars,
            "predicted_log": result.oof_predictions_log,
            "residual_dollars": residual_dollars,
            "abs_error_dollars": np.abs(residual_dollars),
            "residual_log": result.oof_predictions_log - y_log.to_numpy(),
            "Neighborhood": raw.get("Neighborhood", pd.Series("Unknown", index=raw.index)),
            "OverallQual": raw.get("OverallQual", pd.Series(np.nan, index=raw.index)),
            "recently_remodeled": (
                raw["YearRemodAdd"].to_numpy() > raw["YearBuilt"].to_numpy()
                if {"YearRemodAdd", "YearBuilt"}.issubset(raw.columns)
                else np.zeros(len(raw), dtype=bool)
            ),
        }
    )
    return frame


def price_percentile_errors(
    residuals: pd.DataFrame,
    percentiles: tuple[int, ...] = (50, 80, 95),
    window: float = 0.05,
) -> dict[str, float]:
    """Median absolute error near selected home-price percentiles."""
    ranks = residuals["SalePrice"].rank(pct=True)
    values: dict[str, float] = {}
    for percentile in percentiles:
        center = percentile / 100
        mask = ranks.between(max(center - window, 0), min(center + window, 1))
        if not mask.any():
            idx = int((ranks - center).abs().idxmin())
            errors = residuals["abs_error_dollars"].to_numpy(dtype=float)
            values[f"p{percentile}_home_error_dollars"] = float(errors[idx])
        else:
            errors = residuals.loc[mask, "abs_error_dollars"].to_numpy(dtype=float)
            values[f"p{percentile}_home_error_dollars"] = float(np.median(errors))
    return values


def residual_diagnostics(residuals: pd.DataFrame) -> dict[str, Any]:
    """Compute Phase 1 residual diagnostics that seed the uncertainty thesis."""
    by_neighborhood = (
        residuals.groupby("Neighborhood")["abs_error_dollars"]
        .agg(["count", "median", "mean"])
        .sort_values("median", ascending=False)
        .head(10)
    )
    by_quality = residuals.groupby("OverallQual")["abs_error_dollars"].median()
    by_remodel = residuals.groupby("recently_remodeled")["abs_error_dollars"].median()
    corr = residuals[["SalePrice", "abs_error_dollars"]].corr().to_numpy(dtype=float)
    price_error_corr = float(corr[0, 1])
    spread = {
        "median_abs_error_dollars": float(residuals["abs_error_dollars"].median()),
        "p80_abs_error_dollars": float(residuals["abs_error_dollars"].quantile(0.80)),
        "p95_abs_error_dollars": float(residuals["abs_error_dollars"].quantile(0.95)),
        "price_abs_error_correlation": price_error_corr,
        **price_percentile_errors(residuals),
    }
    return {
        "spread": spread,
        "worst_neighborhoods_by_median_abs_error": by_neighborhood.reset_index().to_dict(
            orient="records"
        ),
        "median_abs_error_by_overall_quality": {
            str(key): float(value) for key, value in by_quality.items()
        },
        "median_abs_error_by_remodel_status": {
            str(key): float(value) for key, value in by_remodel.items()
        },
    }


def make_framing_sentence(typical_error: float) -> str:
    """Create the Phase 1 hypothesis sentence with the observed dollar error."""
    return (
        "Hypothesis to be tested in Phase 2: A typical fix-and-flip net margin is "
        "on the order of $10-20K; this model's typical dollar error is "
        f"${typical_error:,.0f}. If that error is comparable to or larger than that "
        "margin, point predictions cannot safely underwrite a flip."
    )


def make_metric_card(
    raw: pd.DataFrame,
    results: list[BaselineResult],
    selected: BaselineResult,
    diagnostics: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable Phase 1 metric card."""
    typical_error = diagnostics["spread"]["median_abs_error_dollars"]
    feature_importance = selected.feature_importance.head(20)
    return {
        "phase": "1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data": {
            "rows": int(len(raw)),
            "columns": int(len(raw.columns)),
            "target": f"{config.target.transform}({config.target.column})",
            "median_sale_price": float(raw[config.target.column].median()),
            "random_split_note": config.phase1.random_split_note,
        },
        "cv": {
            "outer_folds": config.cross_validation.n_folds,
            "repeats": config.phase1.cv_repeats,
            "stratify_by": config.cross_validation.stratify_by,
            "inner_cv_folds": config.phase1.inner_cv_folds,
        },
        "models": {result.name: result.summary for result in results},
        "selected_model": selected.name,
        "residual_diagnostics": diagnostics,
        "top_feature_importances": [
            {
                "feature": str(feature),
                "source_feature": source_feature_from_transformed_name(str(feature)),
                "importance": float(importance),
            }
            for feature, importance in feature_importance.items()
        ],
        "framing_sentence": make_framing_sentence(typical_error),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_phase1(config_path: Path | str = "config/model.yaml") -> Phase1RunResult:
    """Run Phase 1 end-to-end and persist metrics, residuals, figure, and artifact."""
    config = load_model_config(resolve_repo_path(config_path))
    mpl_config_dir = Path(tempfile.gettempdir()) / "margin-of-error-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir))
    train_path = resolve_repo_path(config.data.kaggle_train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Kaggle train.csv not found at {train_path}. "
            "Place the Kaggle House Prices train.csv there or set MOE_KAGGLE_TRAIN_PATH."
        )

    raw = validate_kaggle_train(load_kaggle_train(train_path))
    y_log = make_target(raw[config.target.column], config.target.transform)
    X = raw.drop(columns=[config.target.column])

    model_specs: list[tuple[str, EstimatorFactory]] = [
        ("dumb_median", make_dummy_estimator),
        ("elastic_net", make_elastic_net_estimator),
        ("lightgbm", make_lightgbm_estimator),
    ]
    results = [
        cross_validate_model(name, factory, X, y_log, config) for name, factory in model_specs
    ]
    selected = next(result for result in results if result.name == "lightgbm")
    residuals = build_residual_frame(raw, y_log, selected)
    diagnostics = residual_diagnostics(residuals)
    metric_card = make_metric_card(raw, results, selected, diagnostics, config)

    artifact_dir = resolve_repo_path(config.phase1.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "baseline_lightgbm.joblib"
    final_smearing = duan_smearing_factor(y_log, np.asarray(selected.estimator.predict(X)))
    joblib.dump(
        {
            "model": selected.estimator,
            "smearing_factor": final_smearing,
            "config": config.model_dump(mode="json"),
            "metric_card": metric_card,
        },
        artifact_path,
    )

    metric_card_path = resolve_repo_path(config.phase1.metric_card_path)
    residuals_path = resolve_repo_path(config.phase1.residuals_path)
    save_json(metric_card_path, metric_card)
    residuals_path.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(residuals_path, index=False)

    figure_path = write_price_error_svg(
        residuals["SalePrice"].to_numpy(),
        residuals["abs_error_dollars"].to_numpy(),
        save_as="01_price_error_vs_home_price.svg",
    )

    logger.info("Phase 1 metric card written to %s", metric_card_path)
    logger.info("Phase 1 residuals written to %s", residuals_path)
    logger.info("Phase 1 artifact written to %s", artifact_path)

    return Phase1RunResult(
        metric_card_path=metric_card_path,
        residuals_path=residuals_path,
        artifact_path=artifact_path,
        figure_path=figure_path,
        metric_card=metric_card,
    )


def main() -> None:
    """CLI entrypoint for `python -m margin_of_error.models.baseline`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_phase1()


if __name__ == "__main__":
    main()
