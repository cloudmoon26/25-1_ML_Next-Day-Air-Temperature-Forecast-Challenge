"""Configuration for Next-Day Air Temperature Residual Forecast."""

from __future__ import annotations

RANDOM_STATE = 42
CV_FOLDS = 5
STACKING_CV_FOLDS = 5

TARGET_COL = "target"
ID_COL = "id"

# -----------------------------------------------------------------------------
# Data cleaning configuration
# -----------------------------------------------------------------------------

MISSING_VALUE_CODE = -9999
CLOUD_FEATURE = "cloud_cover"
PRECIPITATION_FEATURE = "precipitation"
SNOW_FEATURE = "snow_depth"

# Variables whose missing hourly blocks are linearly interpolated.
INTERP_FEATURES = [
    "cloud_cover",
    "wind_speed",
    "wind_direction",
    "visibility",
    "vapor_pressure",
    "surface_temp",
    "sea_level_pressure",
    "local_pressure",
    "humidity",
    "dew_point",
]

# Variables where NaN can reasonably indicate no event / no observation.
ZERO_FILL_FEATURES = [
    "sunshine_duration",
    "snow_depth",
    "precipitation",
]

# Raw identifier/text/date columns removed after feature engineering.
DROP_COLS = [
    "date",
    "station",
    "station_name",
]

# Raw hourly blocks removed after safer derived representations are created.
DROP_HOURLY_BASE_FEATURES = [
    "min_cloud_height",  # extremely sparse
    "wind_direction",   # circular; use sin/cos features instead
]

# Variables used for hourly trend/late-day features.
TIME_DERIVED_FEATURES = [
    "surface_temp",
    "humidity",
    "dew_point",
    "wind_speed",
    "sea_level_pressure",
    "local_pressure",
    "vapor_pressure",
]

# Raw hourly variables standardized by train statistics only.
SCALE_BASE_FEATURES = [
    "dew_point",
    "humidity",
    "local_pressure",
    "precipitation",
    "sea_level_pressure",
    "snow_depth",
    "surface_temp",
    "vapor_pressure",
    "visibility",
    "wind_speed",
    "climatology_temp",
]

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------

LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "n_jobs": -1,
    "tree_method": "hist",
}

CAT_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.03,
    "verbose": 0,
    "random_seed": RANDOM_STATE,
    "loss_function": "RMSE",
}

RIDGE_ALPHA = 1.0


def is_warm(month: int) -> bool:
    """Return whether a month belongs to the warm season."""
    return 4 <= int(month) <= 9
