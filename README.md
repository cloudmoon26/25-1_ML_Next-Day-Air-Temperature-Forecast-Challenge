# Next-Day Air Temperature Residual Forecast

This repository contains a modular machine-learning pipeline for a Kaggle weather forecasting challenge. 
The goal is to predict the next day’s average temperature anomaly using hourly meteorological observations from the current day.

The target is not the raw next-day temperature. It is the residual value below:

```text
target = next_day_average_temperature - climatology_temp
```

This means the model learns whether the next day is expected to be warmer or colder than the historical average for the same calendar date.

## Competition Context

The training set contains 2019–2024 observations from Dongducheon, Seoul, Ganghwa, Incheon, Icheon, and Yangpyeong stations. The test set contains shuffled observations from Paju and Suwon stations. The competition notice stated that the test dataset was updated while the rest of the project remained unchanged, so the trained modeling logic can be reused and inference should be rerun on the updated test file.

Only five test submissions are allowed per day, so local validation is used before generating the final submission file.

## Pipeline Architecture

The full workflow is organized as a reproducible preprocessing, feature-engineering, and split-specific ensemble pipeline.

<p align="center">
  <img src="image/temperature_pipeline.png" width="1000" alt="Temperature prediction pipeline architecture">
</p>

## Repository Structure

```text
.
├── config.py          # Global settings, feature groups, model parameters
├── preprocess.py      # Missing-value handling, interpolation, feature engineering, scaling
├── data.py            # CSV loading and wet/dry data splitting
├── models.py          # LightGBM, XGBoost, CatBoost, and stacking ensemble definitions
├── train.py           # Cross-validation, final training, prediction, and submission creation
├── main.py            # Command-line entry point
├── requirements.txt   # Required Python packages
└── README.md
```

## Data

The competition provides the following files:

```text
train_dataset.csv       # Training data from six weather stations, 2019-2024
test_dataset.csv        # Test data from Paju and Suwon stations
submission_sample.csv   # Sample submission format
station_info.csv        # Station metadata, provided as reference information
```

Each row contains one day of observations. Most weather variables are provided hourly from `0` to `23`, for example:

```text
dew_point_0, dew_point_1, ..., dew_point_23
humidity_0, humidity_1, ..., humidity_23
surface_temp_0, ..., surface_temp_23
precipitation_0, ..., precipitation_23
```

The provided `climatology_temp` feature represents the average temperature for the same calendar date based on historical data.

## Method

### Missing-value handling

The raw dataset contains two types of missing information. Values encoded as `-9999` are treated as missing or abnormal sensor readings and are converted to `NaN`. For selected hourly variables, consecutive missing blocks are filled by linear interpolation using neighboring valid values.

For variables where missingness can naturally indicate absence of an event, values are filled with zero:

```text
sunshine_duration
snow_depth
precipitation
```

Rows in the training set with too many missing values in key interpolated variables are removed before training.

### Feature engineering

The model uses both the original hourly variables and additional daily summary features. For selected weather variables, the pipeline creates:

```text
mean, standard deviation, maximum, minimum
```

For variables with strong within-day temporal behavior, the pipeline also adds:

```text
daily difference
AM mean
PM mean
average trend
```

Calendar information is added using the month extracted from the `date` column. A warm-season indicator is also created for April through September.

### Scaling

Selected hourly variables are standardized using statistics computed from the training set only. The same train-derived mean and standard deviation are then applied to both training and test data.

### Wet/dry split

The pipeline trains separate models for wet and dry conditions. A row is treated as wet when any hourly precipitation or snow-depth value is greater than zero. Otherwise, it is treated as dry.

This split allows the model to learn different temperature-residual patterns for precipitation/snow days and non-precipitation days.

### Model

The final predictor is a stacking ensemble trained separately for the wet and dry subsets.

Base models:

```text
LightGBM Regressor
XGBoost Regressor
CatBoost Regressor
```

Meta model:

```text
Ridge Regression
```

The same stacking structure is used for both subsets:

```text
wet data  -> wet stacking model
dry data  -> dry stacking model
```

## Validation

The code evaluates the wet and dry models separately using 5-fold cross-validation with RMSE as the metric.

```text
scoring = neg_root_mean_squared_error
```
After validation, the final wet and dry models are trained on all available rows from their corresponding subsets.

