"""Data cleaning, interpolation, preprocessing, and scaling."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from .config import (
    CLOUD_FEATURE,
    DROP_COLS,
    DROP_HOURLY_BASE_FEATURES,
    INTERP_FEATURES,
    MISSING_VALUE_CODE,
    SCALE_BASE_FEATURES,
    TARGET_COL,
    ZERO_FILL_FEATURES,
)
from .features import (
    add_advanced_weather_features,
    add_date_features,
    add_hourly_summary_features,
    add_station_metadata,
    add_time_derived_features,
    add_wind_direction_features,
    get_hourly_cols,
)


def fill_block_interp(values: np.ndarray, *, integer: bool = False) -> np.ndarray:
    """Fill consecutive missing blocks by linear interpolation.

    If both boundaries are missing, the block remains missing. If only one
    boundary exists, that boundary value is propagated.
    """
    values = values.astype(float, copy=True)
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
                fill_vals = [np.floor(v) if not np.isnan(v) else v for v in fill_vals]

            values[start : end + 1] = fill_vals
        else:
            i += 1

    return values


def clean_and_interpolate(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    """Replace abnormal codes, interpolate selected variables, and zero-fill events."""
    df = df.replace(MISSING_VALUE_CODE, np.nan).copy()

    for feat in INTERP_FEATURES:
        cols = get_hourly_cols(df, feat)
        if not cols:
            continue

        # Original strategy: remove train rows with too many missing hourly values
        # in key interpolated variables. Do not remove test rows.
        if is_train:
            df = df[df[cols].isna().sum(axis=1) < 12].reset_index(drop=True)

        df[cols] = df[cols].apply(
            lambda row: fill_block_interp(row.values.copy(), integer=(feat == CLOUD_FEATURE)),
            axis=1,
            result_type="expand",
        )
        df = add_hourly_summary_features(df, feat)

    for feat in ZERO_FILL_FEATURES:
        cols = get_hourly_cols(df, feat)
        if not cols:
            continue
        df[cols] = df[cols].fillna(0)
        df = add_hourly_summary_features(df, feat)

    return df


def drop_unwanted_columns(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    """Drop text identifiers and selected raw hourly blocks."""
    drop_cols = list(DROP_COLS)

    for base in DROP_HOURLY_BASE_FEATURES:
        drop_cols.extend(get_hourly_cols(df, base))

    if not is_train and TARGET_COL in df.columns:
        drop_cols.append(TARGET_COL)

    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def preprocess_raw_data(
    df: pd.DataFrame,
    *,
    is_train: bool,
    station_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full preprocessing pipeline for one raw train/test dataframe."""
    df = df.copy()
    df = add_station_metadata(df, station_info)
    df = clean_and_interpolate(df, is_train=is_train)
    df = add_date_features(df)
    df = add_time_derived_features(df)
    df = add_wind_direction_features(df)
    df = add_advanced_weather_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = drop_unwanted_columns(df, is_train=is_train)
    return df


def align_train_test_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make train and test feature columns identical except for target."""
    train = train.copy()
    test = test.copy()

    for col in train.columns:
        if col != TARGET_COL and col not in test.columns:
            test[col] = 0

    for col in test.columns:
        if col not in train.columns:
            train[col] = 0

    ordered_test_cols = [c for c in train.columns if c != TARGET_COL]
    test = test[ordered_test_cols]
    return train, test


def scale_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize raw hourly features using training statistics only."""
    train = train.copy()
    test = test.copy()

    scale_cols = [
        f"{base}_{hour}"
        for base in SCALE_BASE_FEATURES
        for hour in range(24)
        if f"{base}_{hour}" in train.columns and f"{base}_{hour}" in test.columns
    ]

    for col in scale_cols:
        mu = train[col].mean(skipna=True)
        sigma = train[col].std(skipna=True)
        sigma = sigma if sigma != 0 and not pd.isna(sigma) else 1.0
        train[col] = (train[col] - mu) / sigma
        test[col] = (test[col] - mu) / sigma

    return train, test
