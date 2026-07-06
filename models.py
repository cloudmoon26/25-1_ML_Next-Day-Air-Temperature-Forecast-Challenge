from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from config import CAT_PARAMS, LGBM_PARAMS, RIDGE_ALPHA, XGB_PARAMS


def make_lgbm() -> LGBMRegressor:
    """Create LightGBM model with original notebook parameters."""
    return LGBMRegressor(**LGBM_PARAMS)


def make_stack() -> StackingRegressor:
    """Create stacking ensemble with original notebook structure."""
    return StackingRegressor(
        estimators=[
            ("lgb", make_lgbm()),
            ("xgb", XGBRegressor(**XGB_PARAMS)),
            ("cat", CatBoostRegressor(**CAT_PARAMS)),
        ],
        final_estimator=Ridge(alpha=RIDGE_ALPHA),
        n_jobs=-1,
    )
