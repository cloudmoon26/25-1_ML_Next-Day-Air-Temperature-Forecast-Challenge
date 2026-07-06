from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    AVG_FILL_FEATURES,
    CLOUD_FEATURE,
    DROP_COLS,
    SCALE_BASE_FEATURES,
    TIME_DERIVED_FEATURES,
    ZERO_FILL_FEATURES,
    is_warm,
)


def fill_block_interp(values: np.ndarray, *, integer: bool = False) -> np.ndarray:
    """Fill consecutive missing blocks by linear interpolation.

    This function preserves the original notebook logic:
    - if both neighboring values are missing, keep NaN
    - if only previous value exists, fill with previous value
    - if only next value exists, fill with next value
    - if both exist, linearly interpolate
    - for cloud_cover, floor interpolated values
    """
    n, i = len(values), 0

    while i < n:
        if np.isnan(values[i]):
            start = i

            while i < n and np.isnan(values[i]):
                i += 1

            end = i - 1
            length = end - start + 1

            prev_val = values[start - 1] if start - 1 >= 0 else np.nan
            next_val = values[end + 1] if end + 1 < n else np.nan

            if np.isnan(prev_val) and np.isnan(next_val):
                fill_vals = [np.nan] * length
            elif np.isnan(prev_val):
                fill_vals = [next_val] * length
            elif np.isnan(next_val):
                fill_vals = [prev_val] * length
            else:
                step = (next_val - prev_val) / (length + 1)
                fill_vals = [prev_val + step * (k + 1) for k in range(length)]

            if integer:
                fill_vals = [
                    np.floor(value) if not np.isnan(value) else value
                    for value in fill_vals
                ]

            values[start : end + 1] = fill_vals
        else:
            i += 1

    return values


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Preprocess raw train/test data.

    This function follows the original notebook's preprocessing order:
    1. Replace -9999 with NaN
    2. Interpolate cloud/average-fill features
    3. Add mean/std/max/min features
    4. Fill zero-fill features with 0
    5. Add time-derived features
    6. Add month/is_warm
    7. Drop non-model columns
    """
    df = df.replace(-9999, np.nan).copy()

    # 1. Interpolation + summary statistics
    for feat in [CLOUD_FEATURE] + AVG_FILL_FEATURES:
        cols = [f"{feat}_{hour}" for hour in range(24)]

        if is_train:
            df = df[df[cols].isna().sum(axis=1) < 12].reset_index(drop=True)

        df[cols] = df[cols].apply(
            lambda row: fill_block_interp(
                row.values.copy(),
                integer=(feat == CLOUD_FEATURE),
            ),
            axis=1,
            result_type="expand",
        )

        df[f"{feat}_mean"] = df[cols].mean(axis=1)
        df[f"{feat}_std"] = df[cols].std(axis=1)
        df[f"{feat}_max"] = df[cols].max(axis=1)
        df[f"{feat}_min"] = df[cols].min(axis=1)

    # 2. Fill selected missing features with zero
    for feat in ZERO_FILL_FEATURES:
        cols = [f"{feat}_{hour}" for hour in range(24)]
        df[cols] = df[cols].fillna(0)

    # 3. Time-based derived features
    for feat in TIME_DERIVED_FEATURES:
        cols = [
            f"{feat}_{hour}"
            for hour in range(24)
            if f"{feat}_{hour}" in df.columns
        ]

        if len(cols) == 24:
            df[f"{feat}_diff"] = df[cols[-1]] - df[cols[0]]
            df[f"{feat}_am_mean"] = df[cols[:12]].mean(axis=1)
            df[f"{feat}_pm_mean"] = df[cols[12:]].mean(axis=1)
            df[f"{feat}_trend"] = df[cols].diff(axis=1).mean(axis=1)

    # 4. Date features
    df["month"] = pd.to_datetime(
        "2023-" + df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    ).dt.month
    df["is_warm"] = df["month"].apply(is_warm)

    drop_cols = DROP_COLS.copy()
    if not is_train and "target" in df.columns:
        drop_cols.append("target")

    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def scale_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize hourly weather features using train statistics only."""
    train = train.copy()
    test = test.copy()

    scale_cols = [
        f"{base}_{hour}"
        for base in SCALE_BASE_FEATURES
        for hour in range(24)
        if f"{base}_{hour}" in train.columns
    ]

    mean_std = {
        col: (train[col].mean(skipna=True), train[col].std(skipna=True))
        for col in scale_cols
    }

    for col, (mu, sig) in mean_std.items():
        sig = sig if (sig != 0 and not np.isnan(sig)) else 1.0
        train[col] = (train[col] - mu) / sig
        test[col] = (test[col] - mu) / sig

    return train, test
