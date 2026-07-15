"""Cross-validation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from .config import CV_FOLDS, RANDOM_STATE
from .models import make_stack


def run_cross_validation(X, y, name: str, n_splits: int = CV_FOLDS) -> np.ndarray:
    """Run RMSE cross-validation for a split-specific model."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        make_stack(),
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=None,
    )
    rmse_scores = -scores
    print(f"[{name}] RMSE mean: {rmse_scores.mean():.4f}")
    print(f"[{name}] RMSE folds: {np.round(rmse_scores, 4)}")
    return rmse_scores
