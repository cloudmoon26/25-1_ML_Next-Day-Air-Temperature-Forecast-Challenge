"""Model definitions for the stacking ensemble."""

from __future__ import annotations

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from .config import CAT_PARAMS, LGBM_PARAMS, RIDGE_ALPHA, STACKING_CV_FOLDS, XGB_PARAMS


def make_stack() -> StackingRegressor:
    """Create LightGBM/XGBoost/CatBoost stacking ensemble."""
    try:
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "Missing modeling packages. Install them with: "
            "pip install -r requirements.txt"
        ) from exc

    base_models = [
        ("lgbm", LGBMRegressor(**LGBM_PARAMS)),
        ("xgb", XGBRegressor(**XGB_PARAMS)),
        ("cat", CatBoostRegressor(**CAT_PARAMS)),
    ]

    return StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=RIDGE_ALPHA),
        cv=STACKING_CV_FOLDS,
        n_jobs=-1,
        passthrough=False,
    )
