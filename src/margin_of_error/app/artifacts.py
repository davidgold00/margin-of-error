"""Phase 5 app artifacts: saved CQR model, feature defaults, and loaders.

The Streamlit app must not retrain models at runtime. This module provides the
offline artifact builder (`make app-artifacts`) plus small pure loaders used by
the app and tests.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "margin-of-error-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CONFIG_DIR))

from margin_of_error.config import (  # noqa: E402
    EconomicsConfig,
    ModelConfig,
    load_economics,
    load_model_config,
)
from margin_of_error.data.loaders import load_kaggle_train  # noqa: E402
from margin_of_error.data.schemas import validate_kaggle_train  # noqa: E402
from margin_of_error.models.baseline import make_target, resolve_repo_path  # noqa: E402
from margin_of_error.models.conformal import CQRModel  # noqa: E402
from margin_of_error.models.phase2 import load_phase1_artifact, three_way_split  # noqa: E402

logger = logging.getLogger(__name__)

PHASE2_CQR_ARTIFACT = Path("models/phase2/cqr_90.joblib")
FEATURE_DEFAULTS_ARTIFACT = Path("models/phase5/feature_defaults.json")

EXPOSED_FEATURES: tuple[dict[str, str], ...] = (
    {
        "column": "Neighborhood",
        "label": "Neighborhood",
        "why": "Strongest location proxy and a major driver of valuation uncertainty.",
    },
    {
        "column": "GrLivArea",
        "label": "Living area",
        "why": "Finished living area is one of the top Phase 1 value drivers.",
    },
    {
        "column": "OverallQual",
        "label": "Overall quality",
        "why": "A broad quality signal that strongly predicts price and confounds renovations.",
    },
    {
        "column": "YearBuilt",
        "label": "Year built",
        "why": "Age shapes both value and renovation risk.",
    },
    {
        "column": "FullBath",
        "label": "Full baths",
        "why": "Bathroom count is visible to buyers and part of the causal renovation layer.",
    },
    {
        "column": "HalfBath",
        "label": "Half baths",
        "why": "Half-bath changes appear in the Phase 3 treatment set.",
    },
    {
        "column": "KitchenQual",
        "label": "Kitchen quality",
        "why": "Kitchen quality has a positive DML-estimated causal lift.",
    },
    {
        "column": "TotalBsmtSF",
        "label": "Basement area",
        "why": "Basement size and finish affect usable space and renovation options.",
    },
    {
        "column": "GarageCars",
        "label": "Garage spaces",
        "why": "Garage capacity is a fixed amenity that affects value and feasibility.",
    },
    {
        "column": "GarageFinish",
        "label": "Garage finish",
        "why": "Garage finish is one of the statistically clear DML renovation effects.",
    },
)

OPTION_COLUMNS: tuple[str, ...] = (
    "Neighborhood",
    "KitchenQual",
    "BsmtQual",
    "BsmtFinType1",
    "GarageFinish",
    "CentralAir",
)


@dataclass(frozen=True)
class AppArtifacts:
    """All objects needed to underwrite one app property."""

    point_model: Any
    smearing_factor: float
    cqr_model: CQRModel
    defaults: dict[str, Any]
    feature_columns: list[str]
    options: dict[str, list[Any]]
    exposed_features: tuple[dict[str, str], ...]
    model_config: ModelConfig
    economics: EconomicsConfig


class ArtifactLoadError(RuntimeError):
    """Raised when the app cannot load a required saved artifact."""


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy scalars to stable JSON primitives."""
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value.item() if hasattr(value, "item") else value


def build_feature_defaults(raw: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Build a one-row default profile from dataset medians and modes."""
    X = raw.drop(columns=[target_column])
    defaults: dict[str, Any] = {}
    for column in X.columns:
        series = X[column]
        value: Any
        if pd.api.types.is_numeric_dtype(series):
            value = series.median()
        else:
            modes = series.dropna().mode()
            value = modes.iloc[0] if not modes.empty else None
        defaults[column] = _json_value(value)

    options: dict[str, list[Any]] = {}
    for column in OPTION_COLUMNS:
        if column in X.columns:
            values = sorted(_json_value(value) for value in X[column].dropna().unique())
            options[column] = values

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "Kaggle train.csv medians/modes; target SalePrice excluded",
        "feature_columns": list(X.columns),
        "defaults": defaults,
        "options": options,
        "exposed_features": list(EXPOSED_FEATURES),
        "median_sale_price": float(raw[target_column].median()),
    }


def build_app_artifacts(
    model_config_path: Path | str = "config/model.yaml",
    economics_config_path: Path | str = "config/economics.yaml",
) -> dict[str, str]:
    """Fit and persist the CQR app artifact plus feature-default profile.

    This is an offline build step. The Streamlit app only loads the resulting
    files, so user interaction remains fast and deterministic.
    """
    config = load_model_config(resolve_repo_path(model_config_path))
    load_economics(resolve_repo_path(economics_config_path))

    raw = validate_kaggle_train(load_kaggle_train(resolve_repo_path(config.data.kaggle_train_path)))
    y_log = make_target(raw[config.target.column], config.target.transform)
    X = raw.drop(columns=[config.target.column])
    train_idx, cal_idx, _test_idx = three_way_split(X, config)

    logger.info("Fitting app CQR artifact from Phase 2 split")
    cqr_model = CQRModel.fit(
        X.iloc[train_idx],
        y_log.iloc[train_idx],
        X.iloc[cal_idx],
        y_log.iloc[cal_idx],
        config,
        alpha=config.conformal.alpha,
    )

    cqr_path = resolve_repo_path(PHASE2_CQR_ARTIFACT)
    cqr_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": cqr_model,
            "alpha": config.conformal.alpha,
            "nominal_coverage": 1.0 - config.conformal.alpha,
            "q_hat": cqr_model.q_hat,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        cqr_path,
        compress=3,
    )

    defaults_path = resolve_repo_path(FEATURE_DEFAULTS_ARTIFACT)
    defaults_path.parent.mkdir(parents=True, exist_ok=True)
    defaults_path.write_text(
        json.dumps(build_feature_defaults(raw, config.target.column), indent=2, sort_keys=True)
        + "\n"
    )
    return {"cqr": str(cqr_path), "feature_defaults": str(defaults_path)}


def load_app_artifacts(
    model_config_path: Path | str = "config/model.yaml",
    economics_config_path: Path | str = "config/economics.yaml",
    cqr_artifact_path: Path | str = PHASE2_CQR_ARTIFACT,
    feature_defaults_path: Path | str = FEATURE_DEFAULTS_ARTIFACT,
) -> AppArtifacts:
    """Load every saved artifact the underwriting app requires."""
    config = load_model_config(resolve_repo_path(model_config_path))
    economics = load_economics(resolve_repo_path(economics_config_path))

    phase1_path = resolve_repo_path(config.phase1.artifact_dir) / "baseline_lightgbm.joblib"
    cqr_path = resolve_repo_path(cqr_artifact_path)
    defaults_path = resolve_repo_path(feature_defaults_path)

    missing = [path for path in (phase1_path, cqr_path, defaults_path) if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise ArtifactLoadError(
            f"Missing app artifact(s): {formatted}. Run `make train uncertainty app-artifacts` "
            "before launching the underwriting tool."
        )

    try:
        point_model, smearing, _phase1_card = load_phase1_artifact(phase1_path)
        cqr_payload = joblib.load(cqr_path)
        defaults_payload = json.loads(defaults_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive user-facing path
        raise ArtifactLoadError(f"Could not load app artifacts cleanly: {exc}") from exc

    cqr_model = cqr_payload.get("model")
    if not isinstance(cqr_model, CQRModel):
        raise ArtifactLoadError(f"CQR artifact at {cqr_path} does not contain a CQRModel")

    return AppArtifacts(
        point_model=point_model,
        smearing_factor=smearing,
        cqr_model=cqr_model,
        defaults=dict(defaults_payload["defaults"]),
        feature_columns=list(defaults_payload["feature_columns"]),
        options={k: list(v) for k, v in defaults_payload.get("options", {}).items()},
        exposed_features=tuple(defaults_payload.get("exposed_features", EXPOSED_FEATURES)),
        model_config=config,
        economics=economics,
    )


def main() -> None:
    """CLI entrypoint for `python -m margin_of_error.app.artifacts`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    paths = build_app_artifacts()
    print("Built app artifacts:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
