from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from config import CV_FOLDS
from models import make_stack


def run_cross_validation(X, y, name: str) -> np.ndarray:
    """Run 5-fold RMSE CV using the same scoring as the notebook."""
    scores = cross_val_score(
        make_stack(),
        X,
        y,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
    )
    rmse_scores = -scores
    print(f"{name} RMSE 평균: {rmse_scores.mean():.4f}")
    return rmse_scores


def train_models(split_data: dict):
    """Train wet/dry stacking models."""
    model_wet = make_stack().fit(split_data["X_wet"], split_data["y_wet"])
    model_dry = make_stack().fit(split_data["X_dry"], split_data["y_dry"])
    return model_wet, model_dry


def predict_submission(
    test: pd.DataFrame,
    split_data: dict,
    model_wet,
    model_dry,
) -> pd.DataFrame:
    """Predict wet/dry test rows separately and return submission DataFrame."""
    preds = pd.Series(index=test.index, dtype=float)

    preds[split_data["wet_te_idx"]] = model_wet.predict(split_data["X_test_wet"])
    preds[split_data["dry_te_idx"]] = model_dry.predict(split_data["X_test_dry"])

    submission = pd.DataFrame(
        {
            "id": test["id"],
            "target_pred": preds,
        }
    )
    return submission
