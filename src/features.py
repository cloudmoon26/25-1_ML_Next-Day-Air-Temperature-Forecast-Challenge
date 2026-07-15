"""Feature engineering utilities.

The functions in this module only use current-day observations and metadata
available in both train and test. They avoid target encoding and future values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CLOUD_FEATURE,
    PRECIPITATION_FEATURE,
    SNOW_FEATURE,
    TIME_DERIVED_FEATURES,
    is_warm,
)


def get_hourly_cols(df: pd.DataFrame, base_name: str) -> list[str]:
    """Return hourly columns in chronological order: base_0 ... base_23."""
    return [f"{base_name}_{h}" for h in range(24) if f"{base_name}_{h}" in df.columns]


def add_station_metadata(df: pd.DataFrame, station_info: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge station metadata such as latitude, longitude, and elevation.

    The function supports the original Korean column names in station_info.csv.
    If multiple rows exist for a station, the current row where 종료일 is missing
    is preferred.
    """
    if station_info is None or "station" not in df.columns:
        return df

    info = station_info.copy()
    rename_map = {
        "지점": "station",
        "위도": "station_latitude",
        "경도": "station_longitude",
        "노장해발고도(m)": "station_elevation",
        "기압계(관측장비지상높이(m))": "pressure_sensor_height",
        "기온계(관측장비지상높이(m))": "temp_sensor_height",
        "풍속계(관측장비지상높이(m))": "wind_sensor_height",
        "강우계(관측장비지상높이(m))": "rain_sensor_height",
    }
    info = info.rename(columns=rename_map)

    if "station" not in info.columns:
        return df

    if "종료일" in info.columns:
        info["_is_current_station_row"] = info["종료일"].isna().astype(int)
        info = info.sort_values(["station", "_is_current_station_row"], ascending=[True, False])

    info = info.drop_duplicates(subset=["station"], keep="first")

    keep_cols = [
        "station",
        "station_latitude",
        "station_longitude",
        "station_elevation",
        "pressure_sensor_height",
        "temp_sensor_height",
        "wind_sensor_height",
        "rain_sensor_height",
    ]
    keep_cols = [c for c in keep_cols if c in info.columns]

    return df.merge(info[keep_cols], on="station", how="left")


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and cyclical seasonality features from MM-DD date."""
    df = df.copy()
    if "date" not in df.columns:
        return df

    date_tmp = pd.to_datetime("2020-" + df["date"].astype(str), errors="coerce")
    month = date_tmp.dt.month
    dayofyear = date_tmp.dt.dayofyear

    df["month"] = month
    df["dayofyear"] = dayofyear
    df["is_warm"] = month.apply(lambda x: is_warm(x) if pd.notna(x) else False).astype(int)
    df["is_winter"] = month.isin([12, 1, 2]).astype(int)
    df["is_summer"] = month.isin([6, 7, 8]).astype(int)

    df["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 366)
    df["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 366)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def add_hourly_summary_features(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """Add mean/std/max/min summary features for one hourly feature group."""
    cols = get_hourly_cols(df, base_name)
    if not cols:
        return df

    df[f"{base_name}_mean"] = df[cols].mean(axis=1)
    df[f"{base_name}_std"] = df[cols].std(axis=1)
    df[f"{base_name}_max"] = df[cols].max(axis=1)
    df[f"{base_name}_min"] = df[cols].min(axis=1)
    return df


def add_time_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add intraday trends, ranges, AM/PM, and late-day weather features."""
    df = df.copy()

    for feat in TIME_DERIVED_FEATURES:
        cols = get_hourly_cols(df, feat)
        if len(cols) != 24:
            continue

        am_cols = [f"{feat}_{h}" for h in range(0, 12)]
        pm_cols = [f"{feat}_{h}" for h in range(12, 24)]
        first6_cols = [f"{feat}_{h}" for h in range(0, 6)]
        last6_cols = [f"{feat}_{h}" for h in range(18, 24)]
        daytime_cols = [f"{feat}_{h}" for h in range(10, 17)]
        night_cols = [f"{feat}_{h}" for h in list(range(0, 6)) + list(range(20, 24))]

        df[f"{feat}_diff"] = df[cols[-1]] - df[cols[0]]
        df[f"{feat}_am_mean"] = df[am_cols].mean(axis=1)
        df[f"{feat}_pm_mean"] = df[pm_cols].mean(axis=1)
        df[f"{feat}_pm_am_diff"] = df[f"{feat}_pm_mean"] - df[f"{feat}_am_mean"]
        df[f"{feat}_trend"] = df[cols].diff(axis=1).mean(axis=1)
        df[f"{feat}_range"] = df[cols].max(axis=1) - df[cols].min(axis=1)
        df[f"{feat}_first6h_mean"] = df[first6_cols].mean(axis=1)
        df[f"{feat}_last6h_mean"] = df[last6_cols].mean(axis=1)
        df[f"{feat}_daytime_mean"] = df[daytime_cols].mean(axis=1)
        df[f"{feat}_night_mean"] = df[night_cols].mean(axis=1)

    return df


def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode wind direction as circular sin/cos statistics."""
    df = df.copy()
    cols = get_hourly_cols(df, "wind_direction")
    if not cols:
        return df

    radians = np.deg2rad(df[cols])
    sin_values = np.sin(radians)
    cos_values = np.cos(radians)

    df["wind_dir_sin_mean"] = sin_values.mean(axis=1)
    df["wind_dir_cos_mean"] = cos_values.mean(axis=1)
    df["wind_dir_sin_std"] = sin_values.std(axis=1)
    df["wind_dir_cos_std"] = cos_values.std(axis=1)
    df["wind_dir_variability"] = np.sqrt(
        df["wind_dir_sin_std"].fillna(0) ** 2 + df["wind_dir_cos_std"].fillna(0) ** 2
    )
    return df


def add_advanced_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-event, interaction, and physics-inspired features."""
    df = df.copy()

    precip_cols = get_hourly_cols(df, PRECIPITATION_FEATURE)
    snow_cols = get_hourly_cols(df, SNOW_FEATURE)

    if precip_cols:
        precip = df[precip_cols].fillna(0)
        df["precip_total"] = precip.sum(axis=1)
        df["precip_max"] = precip.max(axis=1)
        df["precip_hours"] = precip.gt(0).sum(axis=1)
        df["log_precip_total"] = np.log1p(df["precip_total"])

    if snow_cols:
        snow = df[snow_cols].fillna(0)
        df["snow_total"] = snow.sum(axis=1)
        df["snow_max"] = snow.max(axis=1)
        df["snow_hours"] = snow.gt(0).sum(axis=1)
        df["log_snow_total"] = np.log1p(df["snow_total"])

    if precip_cols or snow_cols:
        precip_flag = df["precip_total"].gt(0) if "precip_total" in df.columns else False
        snow_flag = df["snow_total"].gt(0) if "snow_total" in df.columns else False
        df["is_wet"] = (precip_flag | snow_flag).astype(int)

    cloud_cols = get_hourly_cols(df, CLOUD_FEATURE)
    if cloud_cols:
        cloud = df[cloud_cols]
        df["cloud_cover_high_ratio"] = cloud.ge(7).mean(axis=1)
        df["cloud_cover_clear_ratio"] = cloud.le(2).mean(axis=1)
        df["cloud_cover_midday_mean"] = df[[f"{CLOUD_FEATURE}_{h}" for h in range(10, 17)]].mean(axis=1)
        df["cloud_cover_night_mean"] = df[
            [f"{CLOUD_FEATURE}_{h}" for h in list(range(0, 6)) + list(range(20, 24))]
        ].mean(axis=1)
        mode_df = cloud.mode(axis=1, dropna=True)
        df["cloud_cover_mode"] = mode_df[0] if not mode_df.empty else np.nan

    sun_cols = get_hourly_cols(df, "sunshine_duration")
    if sun_cols:
        sun = df[sun_cols].fillna(0)
        df["sunshine_total"] = sun.sum(axis=1)
        df["sunshine_max"] = sun.max(axis=1)
        df["sunshine_hours"] = sun.gt(0).sum(axis=1)

    if "climatology_temp" in df.columns:
        if "surface_temp_mean" in df.columns:
            df["surface_temp_anomaly_mean"] = df["surface_temp_mean"] - df["climatology_temp"]
        if "surface_temp_last6h_mean" in df.columns:
            df["surface_temp_anomaly_last6h"] = df["surface_temp_last6h_mean"] - df["climatology_temp"]
        if "dew_point_mean" in df.columns:
            df["dew_point_anomaly_mean"] = df["dew_point_mean"] - df["climatology_temp"]
        if "dew_point_last6h_mean" in df.columns:
            df["dew_point_anomaly_last6h"] = df["dew_point_last6h_mean"] - df["climatology_temp"]

    if "surface_temp_mean" in df.columns and "dew_point_mean" in df.columns:
        df["temp_dew_gap"] = df["surface_temp_mean"] - df["dew_point_mean"]

    if "surface_temp_mean" in df.columns and "vapor_pressure_mean" in df.columns:
        t = df["surface_temp_mean"]
        saturation_vapor_pressure = 6.112 * np.exp((17.67 * t) / (t + 243.5))
        df["vpd_mean"] = (saturation_vapor_pressure - df["vapor_pressure_mean"]).clip(lower=0)

    if "surface_temp_mean" in df.columns and "humidity_mean" in df.columns:
        df["heat_humidity_index"] = df["surface_temp_mean"] * df["humidity_mean"] / 100
        df["temp_humidity_interaction"] = df["surface_temp_mean"] * df["humidity_mean"]

    if "dew_point_mean" in df.columns and "humidity_mean" in df.columns:
        df["dew_humidity_interaction"] = df["dew_point_mean"] * df["humidity_mean"] / 100

    pressure_cols = get_hourly_cols(df, "sea_level_pressure")
    if len(pressure_cols) == 24:
        df["pressure_tendency"] = df[pressure_cols[-1]] - df[pressure_cols[0]]
        df["last6h_pressure_trend"] = df[pressure_cols[-1]] - df["sea_level_pressure_18"]

    if "pressure_tendency" in df.columns and "wind_speed_mean" in df.columns:
        df["pressure_wind_interaction"] = df["pressure_tendency"] * df["wind_speed_mean"]

    if "sea_level_pressure_range" in df.columns and "wind_speed_range" in df.columns:
        precip_part = df["log_precip_total"] if "log_precip_total" in df.columns else 0
        snow_part = df["log_snow_total"] if "log_snow_total" in df.columns else 0
        df["storminess_index"] = df["sea_level_pressure_range"] + df["wind_speed_range"] + precip_part + snow_part

    if "is_wet" in df.columns:
        if "surface_temp_mean" in df.columns:
            df["wet_temp_interaction"] = df["is_wet"] * df["surface_temp_mean"]
        if "sea_level_pressure_mean" in df.columns:
            df["wet_pressure_interaction"] = df["is_wet"] * df["sea_level_pressure_mean"]
        if "is_winter" in df.columns and "snow_total" in df.columns:
            df["cold_precip_interaction"] = df["is_wet"] * df["is_winter"] * df["snow_total"]

    wind_cols = get_hourly_cols(df, "wind_speed")
    if wind_cols and "cloud_cover_clear_ratio" in df.columns:
        low_wind_ratio = df[wind_cols].le(2).mean(axis=1)
        if "surface_temp_18" in df.columns and "surface_temp_23" in df.columns:
            night_temp_drop = df["surface_temp_18"] - df["surface_temp_23"]
        else:
            night_temp_drop = 0
        df["radiative_cooling_proxy"] = df["cloud_cover_clear_ratio"] * low_wind_ratio * night_temp_drop

    if "cloud_cover_mean" in df.columns and "sunshine_total" in df.columns:
        df["cloud_sunshine_interaction"] = df["cloud_cover_mean"] * df["sunshine_total"]

    return df
