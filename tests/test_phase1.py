"""Phase 1 tests: preprocessing, retransformation, and registry guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


def _synthetic_phase1_frame(n_rows: int = 18) -> pd.DataFrame:
    neighborhoods = ["NAmes", "CollgCr", "OldTown"]
    quality = ["TA", "Gd", "Ex"]
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "Id": i + 1,
                "SalePrice": 120_000 + i * 5_000,
                "Neighborhood": neighborhoods[i % len(neighborhoods)],
                "LotFrontage": np.nan if i in {2, 7} else 55 + i,
                "LotArea": 8_000 + i * 120,
                "YearBuilt": 1970 + (i % 35),
                "YearRemodAdd": 1980 + (i % 30),
                "YrSold": 2008,
                "MoSold": 6,
                "GrLivArea": 1_100 + i * 20,
                "TotalBsmtSF": 700 + i * 10,
                "1stFlrSF": 800 + i * 10,
                "2ndFlrSF": 300 if i % 2 else 0,
                "FullBath": 1 + (i % 2),
                "HalfBath": i % 2,
                "BsmtFullBath": i % 2,
                "BsmtHalfBath": 0,
                "BedroomAbvGr": 3,
                "KitchenAbvGr": 1,
                "TotRmsAbvGrd": 6,
                "OverallQual": 5 + (i % 4),
                "OverallCond": 5 + (i % 3),
                "ExterQual": quality[i % len(quality)],
                "KitchenQual": quality[(i + 1) % len(quality)],
                "CentralAir": "Y" if i % 3 else "N",
                "GarageArea": 240 + i * 5,
                "GarageCars": 1,
                "GarageType": None if i % 5 == 0 else "Attchd",
                "GarageFinish": None if i % 5 == 0 else "RFn",
                "GarageQual": None if i % 5 == 0 else "TA",
                "GarageCond": None if i % 5 == 0 else "TA",
                "Fireplaces": i % 2,
                "FireplaceQu": None if i % 2 == 0 else "TA",
                "PoolArea": 0,
                "PoolQC": None,
                "WoodDeckSF": i * 3,
                "OpenPorchSF": i * 2,
                "EnclosedPorch": 0,
                "3SsnPorch": 0,
                "ScreenPorch": 0,
                "SaleType": "WD",
                "SaleCondition": "Normal",
                "MSZoning": "RL",
            }
        )
    return pd.DataFrame(rows)


def test_neighborhood_lot_frontage_imputation_is_fit_on_train_only() -> None:
    """Holdout values must not influence learned neighborhood medians."""
    from margin_of_error.features.preprocessing import MissingnessPolicyTransformer

    train = pd.DataFrame(
        {
            "Neighborhood": ["A", "A", "B", "B"],
            "LotFrontage": [50.0, 70.0, 100.0, 120.0],
        }
    )
    holdout = pd.DataFrame({"Neighborhood": ["A"], "LotFrontage": [np.nan]})

    transformer = MissingnessPolicyTransformer().fit(train)
    result = transformer.transform(holdout)

    assert result.loc[0, "LotFrontage"] == 60.0


def test_ordinal_encoder_preserves_quality_order() -> None:
    """Ex/Gd/TA/Fa/Po must remain ordered, not one-hot nominal."""
    from margin_of_error.features.preprocessing import OrdinalFeatureEncoder

    df = pd.DataFrame({"KitchenQual": ["Po", "TA", "Gd", "Ex", "unknown"]})
    result = OrdinalFeatureEncoder().transform(df)
    encoded = result["KitchenQual"].astype(int).tolist()

    assert encoded[3] > encoded[2]
    assert encoded[2] > encoded[1]
    assert encoded[1] > encoded[0]
    assert encoded[4] == -1


def test_duan_smearing_corrects_low_log_retransformation_bias() -> None:
    """When residuals are positive, smearing raises dollar predictions."""
    from margin_of_error.models.baseline import duan_smearing_factor, log_predictions_to_dollars

    y_pred_log = np.log1p(np.array([100_000.0, 100_000.0]))
    y_true_log = np.log1p(np.array([110_000.0, 120_000.0]))

    smearing = duan_smearing_factor(y_true_log, y_pred_log)
    corrected = log_predictions_to_dollars(y_pred_log, smearing)
    naive = np.expm1(y_pred_log)

    assert smearing > 1.0
    assert np.all(corrected > naive)


def test_mutable_fixed_registry_covers_model_sources_once() -> None:
    """Every modeled raw/engineered source feature has exactly one registry tag."""
    from margin_of_error.features.registry import (
        FIXED_FEATURES,
        MUTABLE_FEATURES,
        assert_registry_covers_model_features,
    )

    assert not (set(MUTABLE_FEATURES) & set(FIXED_FEATURES))
    assert_registry_covers_model_features()


def test_preprocessing_pipeline_is_fittable_inside_cv() -> None:
    """The model preprocessing pipeline can be fit independently per CV fold."""
    from margin_of_error.features.preprocessing import build_preprocessor

    df = _synthetic_phase1_frame()
    X = df.drop(columns=["SalePrice"])
    y = np.log1p(df["SalePrice"])
    drop_cols = ["Id", "SalePrice", "SaleType", "SaleCondition", "YrSold", "MoSold"]
    splitter = KFold(n_splits=3, shuffle=True, random_state=42)

    for train_idx, valid_idx in splitter.split(X):
        estimator = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(drop_cols)),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
            ]
        )
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = estimator.predict(X.iloc[valid_idx])
        assert np.isfinite(preds).all()
