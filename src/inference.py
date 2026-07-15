"""Inference and submission generation."""

from __future__ import annotations

import pandas as pd

from .config import ID_COL, TARGET_COL


def make_submission(test: pd.DataFrame, split_data: dict, model_wet, model_dry) -> pd.DataFrame:
    """Predict wet/dry test rows separately and merge into original order."""
    preds = pd.Series(index=test.index, dtype=float)

    if len(split_data["X_test_wet"]) > 0:
        preds.loc[split_data["wet_test_idx"]] = model_wet.predict(split_data["X_test_wet"])

    if len(split_data["X_test_dry"]) > 0:
        preds.loc[split_data["dry_test_idx"]] = model_dry.predict(split_data["X_test_dry"])

    return pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds})
