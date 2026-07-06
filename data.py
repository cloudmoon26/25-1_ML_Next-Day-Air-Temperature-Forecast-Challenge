from __future__ import annotations

import pandas as pd

from config import PRECIPITATION_FEATURE, SNOW_FEATURE
from preprocess import preprocess, scale_train_test


def load_data(
    train_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read CSV files and apply preprocessing/scaling."""
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train = preprocess(train_raw, is_train=True)
    test = preprocess(test_raw, is_train=False)

    train, test = scale_train_test(train, test)

    return train, test


def split_wet_dry(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Split train/test data into wet and dry subsets.

    Wet rows are rows where any hourly precipitation or snow-depth value is
    greater than 0, exactly as in the original notebook.
    """
    prec_cols = [f"{PRECIPITATION_FEATURE}_{hour}" for hour in range(24)]
    snow_cols = [f"{SNOW_FEATURE}_{hour}" for hour in range(24)]

    wet_tr_idx = (train[prec_cols + snow_cols] > 0).any(axis=1)
    wet_te_idx = (test[prec_cols + snow_cols] > 0).any(axis=1)

    dry_tr_idx = ~wet_tr_idx
    dry_te_idx = ~wet_te_idx

    wet_tr = train[wet_tr_idx]
    dry_tr = train[dry_tr_idx]

    X_wet = wet_tr.drop(columns=["target", "id"])
    y_wet = wet_tr["target"]

    X_dry = dry_tr.drop(columns=["target", "id"])
    y_dry = dry_tr["target"]

    X_test_wet = test.loc[wet_te_idx].drop(columns=["id"])
    X_test_dry = test.loc[dry_te_idx].drop(columns=["id"])

    return {
        "wet_tr_idx": wet_tr_idx,
        "dry_tr_idx": dry_tr_idx,
        "wet_te_idx": wet_te_idx,
        "dry_te_idx": dry_te_idx,
        "X_wet": X_wet,
        "y_wet": y_wet,
        "X_dry": X_dry,
        "y_dry": y_dry,
        "X_test_wet": X_test_wet,
        "X_test_dry": X_test_dry,
    }
