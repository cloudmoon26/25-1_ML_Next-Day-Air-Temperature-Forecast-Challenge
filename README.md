# Next-Day Air Temperature Forecast Challenge

This repository presents a modular machine-learning pipeline developed for the **Next Day Air Temperature Forecast Challenge**, a Kaggle-based course project for **Machine Learning 1 at Seoul National University of Science and Technology in Spring 2025**.

<p align="center">
  <img src="image/kaggle_competition.png" width="300" alt="Kaggle competition">
</p>

[Competition Page](https://www.kaggle.com/competitions/next-day-air-temperature-forecast-challenge-2/data)

The goal of the competition was to predict the next day’s average temperature anomaly using hourly meteorological observations collected during the current day.

The final version achieved the following Kaggle leaderboard scores: 0.82459

The target is not the raw next-day temperature. It is the residual value below:

```text
target = next_day_average_temperature - climatology_temp
```

Therefore, the model learns whether the next day is expected to be warmer or colder than the historical average for the same calendar date.

---

## Dataset

The official files are organized as follows:

```text
train_dataset.csv       # Training data from six stations, 2019–2024
test_dataset.csv        # Test data from Paju and Suwon stations
submission_sample.csv   # Sample submission format
station_info.csv        # Station metadata, including location and sensor heights
```

| Split | Rows | Columns | Stations |
|---|---:|---:|---|
| Train | 13,132 | 342 | Dongducheon, Seoul, Ganghwa, Incheon, Icheon, Yangpyeong |
| Test | 3,004 | 341 | Paju, Suwon |

Each row corresponds to one station-day. Most meteorological variables are provided as 24 hourly features, such as:

```text
dew_point_0, dew_point_1, ..., dew_point_23
humidity_0, humidity_1, ..., humidity_23
surface_temp_0, ..., surface_temp_23
precipitation_0, ..., precipitation_23
```

The main hourly feature groups are:

```text
cloud_cover
dew_point
humidity
local_pressure
min_cloud_height
precipitation
sea_level_pressure
snow_depth
sunshine_duration
surface_temp
vapor_pressure
visibility
wind_direction
wind_speed
```

---

## Exploratory Data Analysis (EDA)

### 1. Climatology captures the dominant seasonal pattern

`climatology_temp` already contains the average temperature pattern for the same calendar date. As expected, it is lowest in winter and highest in summer.

<p align="center">
  <img src="image/eda_climatology_seasonality.png" width="850" alt="Climatology temperature seasonality">
</p>

Because the target subtracts this seasonal baseline, the remaining prediction problem is focused on short-term weather anomalies rather than the raw annual temperature cycle.

### 2. The residual target is still irregular

Even after subtracting climatology, the residual target remains volatile. The project report interpreted this irregular component as important and connected it to weather-event conditions such as rain and snow.

<p align="center">
  <img src="image/eda_target_residual_by_month.png" width="850" alt="Target residual distribution by month">
</p>

In the training set, the residual target has mean `0.222`°C and standard deviation `2.961`°C, with values ranging from `-12.86`°C to `11.78`°C.

### 3. Missing values require feature-specific interpretation

The raw data contains both `-9999` values and empty cells. The project treats `-9999` as a missing or abnormal sensor value. However, ordinary NaN values are not always the same type of error. For example, missing `snow_depth`, `precipitation`, or `sunshine_duration` can represent no snow, no rain, or nighttime/no sunshine.

<p align="center">
  <img src="image/eda_missing_ratio.png" width="780" alt="Raw missing ratio by feature">
</p>

The highest raw missing rates were observed in:

| Feature group | Raw missing rate |
|---|---:|
| `snow_depth` | 96.51% |
| `precipitation` | 90.85% |
| `min_cloud_height` | 53.15% |
| `sunshine_duration` | 45.36% |
| `cloud_cover` | 1.54% |

This motivated different preprocessing rules for different variable types.

### 4. Cloud-cover missingness differs by station

The report found that `cloud_cover` is an ordinal variable with values from 0 to 10, but its missingness is not evenly distributed across stations. Some stations have almost no missing cloud-cover values, while Yangpyeong, Ganghwa, and Icheon contain much larger missing blocks.

<p align="center">
  <img src="image/eda_cloud_cover_missing_by_station.png" width="760" alt="Cloud cover missing cells by station">
</p>

This supports using block interpolation for moderate missing sequences while filtering out rows with too many missing hourly values.

### 5. Feature correlation analysis

Because most meteorological variables are provided as 24 hourly columns, directly visualizing all hourly variables would make the correlation matrix too large and difficult to interpret. Therefore, each hourly feature group was first summarized into a daily mean feature, and the correlations among these daily-level variables were analyzed.

<p align="center">
  <img src="image/eda_variable_correlation_heatmap.png" width="900" alt="Correlation heatmap of daily mean weather variables">
</p>

The correlation heatmap shows that physically related weather variables are strongly connected. For example, temperature-related variables such as `surface_temp`, `dew_point`, and `vapor_pressure` tend to move together, while pressure-related variables also show strong relationships. This confirms that the dataset contains meaningful meteorological structure rather than independent tabular features.

However, the target variable is a residual after subtracting `climatology_temp`, so its relationship with individual daily-mean variables is weaker and less direct than the relationship among raw weather variables. This means that predicting the target requires capturing nonlinear interactions and conditional patterns rather than relying only on simple linear correlations.

<p align="center">
  <img src="image/eda_target_correlation_bar.png" width="800" alt="Top feature correlations with target">
</p>

The target-correlation plot was used to identify which weather variables were more directly related to next-day temperature anomalies. Although several variables showed meaningful correlation with the target, no single feature was dominant enough to explain the residual alone. This supported the use of a tree-based ensemble model, which can learn nonlinear relationships and interactions among multiple meteorological variables.

In addition, `cloud_cover` was examined separately because it is an ordinal variable with values from 0 to 10 and had station-dependent missing patterns.

<p align="center">
  <img src="image/eda_cloud_cover_hourly_correlation.png" width="750" alt="Hourly correlation of cloud cover">
</p>

The hourly correlation pattern of `cloud_cover` indicates that adjacent time points are related, but the variable still contains irregular fluctuations and station-specific missing blocks. Therefore, simple interpolation can be useful for short missing sequences, but rows with excessive missingness should be handled carefully.

### 6. Wet and dry days have different residual behavior

A row is defined as wet when at least one hourly `precipitation` or `snow_depth` value is greater than zero. Under this definition, the raw training set contains `4,133` wet rows (31.5%) and `8,999` dry rows (68.5%). The test set contains `765` wet rows (25.5%) and `2,239` dry rows (74.5%).

<p align="center">
  <img src="image/eda_wet_dry_target_distribution.png" width="780" alt="Wet and dry target residual distribution">
</p>

The target distribution also differs between the two groups. Wet rows have mean `-0.530`°C and standard deviation `3.064`°C, while dry rows have mean `0.567`°C and standard deviation `2.846`°C. This distribution shift motivated training separate models for wet and dry conditions.

---

## Pipeline Architecture

The final implementation is organized as a reproducible preprocessing, feature-engineering, validation, ensemble training, and inference pipeline.

<p align="center">
  <img src="image/temperature_pipeline.jpg" width="1000" alt="Temperature prediction pipeline architecture">
</p>

The overall workflow:

```text
Data
 |
EDA
 |
Data Cleaning
 |
Feature Engineering
 |
Cross Validation
 |
Base Models
 |
 +---- LightGBM
 |
 +---- XGBoost
 |
 +---- CatBoost
 |
OOF Prediction
 |
Ensemble
 |
Final Training
 |
Submission
```

---

## Repository Structure

```text
project/
│
├── data/
│   ├── train_dataset.csv
│   ├── test_dataset.csv
│   ├── submission_sample.csv
│   └── station_info.csv
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── features.py
│   ├── validation.py
│   ├── models.py
│   ├── train.py
│   └── inference.py
│
├── notebooks/
│   ├── EDA.ipynb
│
├── image/
│   ├── kaggle_competition.png
│   ├── temperature_pipeline.jpg
│   ├── eda_climatology_seasonality.png
│   ├── eda_target_residual_by_month.png
│   ├── eda_missing_ratio.png
│   ├── eda_cloud_cover_missing_by_station.png
│   ├── eda_variable_correlation_heatmap.png
│   ├── eda_target_correlation_bar.png
│   ├── eda_cloud_cover_hourly_correlation.png
│   └── eda_wet_dry_target_distribution.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Code Structure

| File | Description |
|---|---|
| `src/config.py` | Stores global settings, feature groups, model hyperparameters, and random seed. |
| `src/preprocess.py` | Handles abnormal values, missing values, block interpolation, zero-filling, row filtering, column alignment, and train-based scaling. |
| `src/features.py` | Generates time-based, weather-event, seasonality, interaction, circular wind-direction, station-metadata, and physics-inspired features. |
| `src/validation.py` | Runs cross-validation and evaluates RMSE for wet and dry subsets. |
| `src/models.py` | Defines LightGBM, XGBoost, CatBoost, and the Ridge-based stacking ensemble. |
| `src/train.py` | Loads data, preprocesses train/test, performs wet/dry splitting, runs validation, and trains final models. |
| `src/inference.py` | Applies wet/dry routing to the test set and creates the final submission file. |
| `main.py` | Executes the full pipeline from data loading to submission generation. |

---

## Preprocessing Strategy

### 1. Missing and abnormal values

The preprocessing logic follows the feature-specific interpretation from the EDA.

1. Convert all `-9999` values to `NaN`.
2. Apply block interpolation to selected continuous hourly variables.
3. Keep `cloud_cover` as an ordinal feature by flooring interpolated values.
4. Fill event-like variables with zero when missingness can indicate absence of the event.
5. Remove rows with excessive missingness in key interpolated variables.
6. Drop sparse or unsafe raw columns after useful derived features are created.

Variables filled by zero:

```text
sunshine_duration
snow_depth
precipitation
```

Variables interpolated by hourly block interpolation:

```text
cloud_cover
wind_speed
wind_direction
visibility
vapor_pressure
surface_temp
sea_level_pressure
local_pressure
humidity
dew_point
```

Dropped raw columns:

```text
date
station
station_name
min_cloud_height hourly columns
wind_direction hourly columns
```

`min_cloud_height` was removed because it had very high missingness: approximately `53.2%` of all hourly cells were missing on average, and the worst hourly column exceeded `85.2%` missingness.

Raw `wind_direction` columns were removed after converting them into circular `sin` and `cos` features, because wind direction is a circular variable where 0° and 360° should be interpreted as close values.

### 2. Train-based scaling

Selected raw hourly variables are standardized using only the training-set mean and standard deviation. The same training statistics are then applied to the test set to prevent test-set information leakage.

---

## Feature Engineering

The first version of the pipeline used basic daily summary features such as mean, standard deviation, maximum, minimum, daily difference, AM mean, PM mean, and average hourly trend. To improve leaderboard performance, the final version extended the original feature engineering strategy with additional weather-domain features.

### 1. Time aggregation features

For selected hourly variables, the pipeline creates:

```text
mean
standard deviation
maximum
minimum
range
first-last difference
AM mean
PM mean
PM-AM difference
average hourly trend
first 6-hour mean
last 6-hour mean
daytime mean
night mean
```

These features summarize the 24-hour weather trajectory of a station-day. In particular, `last 6-hour mean` features were added because the late-day weather state may be more directly connected to the following day’s average temperature.

### 2. Wet/dry and precipitation intensity features

The original model only separated wet and dry rows. The improved version keeps this routing strategy and adds intensity-based precipitation and snow features:

```text
is_wet
precip_total
precip_max
precip_hours
log_precip_total
snow_total
snow_max
snow_hours
log_snow_total
```

These features allow the model to distinguish between light precipitation and stronger weather-event conditions rather than treating every wet day equally.

### 3. Cloud and sunshine features

```text
cloud_cover_high_ratio
cloud_cover_clear_ratio
cloud_cover_midday_mean
cloud_cover_night_mean
cloud_cover_mode
sunshine_total
sunshine_max
sunshine_hours
cloud_sunshine_interaction
```

These features capture radiation, cloudiness, and daily heating/cooling conditions. They were added to reflect the physical relationship between cloud cover, sunshine duration, surface heating, and nighttime cooling.

### 4. Residual-style anomaly features

Because the target is a residual relative to `climatology_temp`, the improved pipeline adds residual-style predictors:

```text
surface_temp_anomaly_mean
surface_temp_anomaly_last6h
dew_point_anomaly_mean
dew_point_anomaly_last6h
```

These features compare current-day weather conditions against the climatological baseline, making them aligned with the structure of the target variable.

### 5. Pressure and storminess features

```text
pressure_tendency
last6h_pressure_trend
pressure_wind_interaction
storminess_index
```

These features were added to represent weather-system movement and instability. For example, pressure tendency and pressure range can reflect changes in synoptic weather conditions, while the storminess index combines pressure movement, wind variability, and precipitation intensity.

### 6. Moisture and physics-inspired features

```text
temp_dew_gap
vpd_mean
heat_humidity_index
temp_humidity_interaction
dew_humidity_interaction
radiative_cooling_proxy
```

These features represent atmospheric moisture, dryness, and radiative cooling conditions. For example, `temp_dew_gap` captures the difference between air temperature-related surface conditions and dew point, while `vpd_mean` approximates vapor pressure deficit.

### 7. Categorical and circular encoding

Raw `station` and `station_name` are not used as direct categorical encodings because the training and test stations are different. Instead, `station_info.csv` is used to add station-level physical metadata when available:

```text
station_latitude
station_longitude
station_elevation
pressure_sensor_height
temp_sensor_height
wind_sensor_height
rain_sensor_height
```

Raw `wind_direction` is not used directly. It is converted into circular features:

```text
wind_dir_sin_mean
wind_dir_cos_mean
wind_dir_sin_std
wind_dir_cos_std
wind_dir_variability
```

This encoding prevents the model from treating 0° and 360° as far apart.

### 8. Seasonality features

```text
month
dayofyear
is_warm
is_winter
is_summer
dayofyear_sin
dayofyear_cos
month_sin
month_cos
```

Cyclical encoding helps represent the continuity between the end and beginning of the year.

---

## Improvements from the Previous Version

The previous version used the following main strategy:

```text
basic missing-value handling
basic hourly aggregation
wet/dry split
LightGBM + XGBoost + CatBoost stacking ensemble
Ridge meta learner
```

The improved version keeps the same overall modeling logic but strengthens the feature engineering and project structure.

Key improvements include:

| Area | Previous version | Improved version |
|---|---|---|
| Project structure | Flat script structure | Modular `src/` structure |
| Feature engineering | Mean/std/max/min, AM/PM, trend | Added late-day, residual-style, pressure, moisture, cloud, radiation, and station metadata features |
| Wet/dry logic | Wet/dry split only | Wet/dry split + precipitation/snow intensity features |
| Wind direction | Dropped raw feature | Converted to circular `sin`/`cos` features before dropping raw degree columns |
| Station information | Dropped station identifiers | Used station physical metadata from `station_info.csv` |
| Seasonality | `month`, `is_warm` | Added cyclical day-of-year and month encodings |
| Leakage control | Train-based scaling | Maintained train-based scaling and avoided station target encoding or unsafe lag features |

The goal of these changes was to improve leaderboard performance while keeping the original modeling idea intact.

---

## Modeling

The model uses a split-specific stacking ensemble.

### Wet/dry routing

```text
wet row = any precipitation_hour > 0 or any snow_depth_hour > 0
dry row = otherwise
```

The model then trains two independent predictors:

```text
wet training rows -> wet stacking model
dry training rows -> dry stacking model
```

The same routing logic is applied to the test set before prediction, and the predictions are merged back into the original test order.

### Stacking ensemble

Base regressors:

```text
LightGBM Regressor
XGBoost Regressor
CatBoost Regressor
```

Meta learner:

```text
Ridge Regression
```

The final stacking structure is:

```text
[LGBM OOF prediction, XGBoost OOF prediction, CatBoost OOF prediction]
    -> Ridge Regression
    -> final residual prediction
```

The final submitted configuration used the following main parameters:

```text
LightGBM: n_estimators=5000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8
XGBoost : n_estimators=500,  learning_rate=0.03
CatBoost: n_estimators=500,  learning_rate=0.03
Ridge   : alpha=1.0
```

Manual hyperparameter experiments were performed because full hyperparameter optimization was computationally expensive.

---

## Validation and Experiments

The project used RMSE for local validation because the task is a regression problem.

### Earlier version

The earlier report version recorded the following wet/dry cross-validation scores:

| Split | Reported CV RMSE |
|---|---:|
| Wet | 1.5707 |
| Dry | 1.5263 |

The best recorded Kaggle score for the earlier stacking model was: 0.80942

### Improved version

After applying the improved feature engineering pipeline, the processed dataset had the following shape:

| Dataset | Processed rows | Processed columns |
|---|---:|---:|
| Train | 12,935 | 469 |
| Test | 3,004 | 468 |

The number of training rows decreased from `13,132` to `12,935` because rows with excessive missingness in key hourly variables were removed during preprocessing.

The final wet/dry split after preprocessing was:

| Split | Train rows | Test rows |
|---|---:|---:|
| Wet | 4,057 | 765 |
| Dry | 8,878 | 2,239 |

The improved version used 5-fold cross-validation for each subset.

| Subset | CV folds | Mean RMSE | Fold RMSE |
|---|---:|---:|---|
| Wet | 5 | 1.3913 | 1.4376, 1.4703, 1.3961, 1.3518, 1.3008 |
| Dry | 5 | 1.2937 | 1.2646, 1.2556, 1.2747, 1.3285, 1.3454 |

Compared with the earlier version, the improved pipeline reduced local CV RMSE for both wet and dry subsets.

### Kaggle leaderboard result

| Version | Public Score |
|---|---:|
| Earlier stacking model | 0.80942 |
| Improved advanced-feature stacking model | 0.82459 |

The final model improved the leaderboard score from `0.80942` to `0.82459`

This improvement suggests that the added weather-domain features, wet/dry intensity features, station metadata, circular wind-direction encoding, pressure tendency features, and residual-style anomaly features helped the model generalize better to the unseen test stations.

## Future Improvements

Potential directions for additional performance improvement include:

1. Group-based validation by station to better evaluate generalization to unseen stations.
2. More systematic hyperparameter optimization with Optuna or Bayesian search.
3. Separate feature selection for wet and dry models.
4. Model-specific feature importance analysis.
5. Additional station metadata engineering from latitude, longitude, elevation, and sensor heights.
6. More careful lag/rolling feature construction if year-level chronological order can be restored safely.
7. Ensemble blending between the earlier submission and the improved advanced-feature submission.
