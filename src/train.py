"""Data loading, split-specific final training, and optional validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ID_COL, PRECIPITATION_FEATURE, SNOW_FEATURE, TARGET_COL
from .features import get_hourly_cols
from .models import make_stack
from .preprocess import align_train_test_columns, preprocess_raw_data, scale_train_test
from .validation import run_cross_validation


def find_station_info_path(train_path: str, station_info_path: str | None) -> str | None:
    """Use explicit station_info path or auto-detect station_info.csv near train."""
    if station_info_path is not None and Path(station_info_path).exists():
        return station_info_path

    candidate = Path(train_path).parent / "station_info.csv"
    if candidate.exists():
        return str(candidate)
    return None


def load_processed_data(
    train_path: str,
    test_path: str,
    station_info_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs and return fully processed train/test dataframes."""
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    station_info = None
    resolved_station_info_path = find_station_info_path(train_path, station_info_path)
    if resolved_station_info_path is not None:
        station_info = pd.read_csv(resolved_station_info_path)
        print(f"Using station_info: {resolved_station_info_path}")
    else:
        print("station_info.csv was not found. Continuing without station metadata.")

    train = preprocess_raw_data(train_raw, is_train=True, station_info=station_info)
    test = preprocess_raw_data(test_raw, is_train=False, station_info=station_info)
    train, test = align_train_test_columns(train, test)
    train, test = scale_train_test(train, test)
    return train, test


def split_wet_dry(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Split train and test into wet/dry subsets using precipitation/snow logic."""
    if "is_wet" in train.columns and "is_wet" in test.columns:
        wet_train_idx = train["is_wet"].astype(bool)
        wet_test_idx = test["is_wet"].astype(bool)
    else:
        prec_train = get_hourly_cols(train, PRECIPITATION_FEATURE)
        snow_train = get_hourly_cols(train, SNOW_FEATURE)
        prec_test = get_hourly_cols(test, PRECIPITATION_FEATURE)
        snow_test = get_hourly_cols(test, SNOW_FEATURE)
        wet_train_idx = train[prec_train + snow_train].fillna(0).gt(0).any(axis=1)
        wet_test_idx = test[prec_test + snow_test].fillna(0).gt(0).any(axis=1)

    dry_train_idx = ~wet_train_idx
    dry_test_idx = ~wet_test_idx

    wet_train = train[wet_train_idx].copy()
    dry_train = train[dry_train_idx].copy()

    X_wet = wet_train.drop(columns=[TARGET_COL, ID_COL])
    y_wet = wet_train[TARGET_COL]

    X_dry = dry_train.drop(columns=[TARGET_COL, ID_COL])
    y_dry = dry_train[TARGET_COL]

    X_test_wet = test.loc[wet_test_idx].drop(columns=[ID_COL])
    X_test_dry = test.loc[dry_test_idx].drop(columns=[ID_COL])

    print(f"wet train: {len(X_wet):,}, dry train: {len(X_dry):,}")
    print(f"wet test : {len(X_test_wet):,}, dry test : {len(X_test_dry):,}")

    return {
        "wet_train_idx": wet_train_idx,
        "dry_train_idx": dry_train_idx,
        "wet_test_idx": wet_test_idx,
        "dry_test_idx": dry_test_idx,
        "X_wet": X_wet,
        "y_wet": y_wet,
        "X_dry": X_dry,
        "y_dry": y_dry,
        "X_test_wet": X_test_wet,
        "X_test_dry": X_test_dry,
    }


def validate_splits(split_data: dict) -> None:
    """Run CV for wet and dry subsets."""
    run_cross_validation(split_data["X_wet"], split_data["y_wet"], name="wet")
    run_cross_validation(split_data["X_dry"], split_data["y_dry"], name="dry")


def train_final_models(split_data: dict):
    """Train final wet/dry stacking models on all available rows."""
    model_wet = make_stack().fit(split_data["X_wet"], split_data["y_wet"])
    model_dry = make_stack().fit(split_data["X_dry"], split_data["y_dry"])
    return model_wet, model_dry
