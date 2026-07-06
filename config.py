RANDOM_STATE = 42

AVG_FILL_FEATURES = [
    "wind_speed",
    "wind_direction",
    "visibility",
    "vapor_pressure",
    "surface_temp",
    "sea_level_pressure",
    "humidity",
    "dew_point",
]

ZERO_FILL_FEATURES = [
    "sunshine_duration",
    "snow_depth",
    "precipitation",
]

CLOUD_FEATURE = "cloud_cover"

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
    "wind_direction",
    "wind_speed",
    "climatology_temp",
]

DROP_COLS = [
    "date",
    "station",
    "station_name",
    "min_cloud_height",
    "wind_direction",
]

TIME_DERIVED_FEATURES = [
    "surface_temp",
    "humidity",
    "dew_point",
    "wind_speed",
]

PRECIPITATION_FEATURE = "precipitation"
SNOW_FEATURE = "snow_depth"

LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

CAT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "verbose": 0,
    "random_state": RANDOM_STATE,
}

RIDGE_ALPHA = 1.0
CV_FOLDS = 5


def is_warm(month: int) -> bool:
    """Return whether the month belongs to the warm season.

    This is equivalent to the original notebook's lambda:
    IS_WARM = lambda m: 4 <= m <= 9
    """
    return 4 <= month <= 9
