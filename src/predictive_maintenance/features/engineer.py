from dataclasses import dataclass

import pandas as pd

from predictive_maintenance.data.load import SENSOR_COLUMNS


@dataclass(frozen=True)
class FeatureConfig:
    sensor_columns: tuple[str, ...] = tuple(SENSOR_COLUMNS)
    rolling_windows: tuple[int, ...] = (5,)
    lags: tuple[int, ...] = (1, 5)
    slope_window: int = 5


def fit_feature_config(
    train_df: pd.DataFrame,
    sensor_columns: tuple[str, ...] = tuple(SENSOR_COLUMNS),
    rolling_windows: tuple[int, ...] = (5,),
    lags: tuple[int, ...] = (1, 5),
    slope_window: int = 5,
) -> FeatureConfig:
    validate_feature_inputs(train_df, sensor_columns, rolling_windows, lags, slope_window)
    return FeatureConfig(
        sensor_columns=sensor_columns,
        rolling_windows=rolling_windows,
        lags=lags,
        slope_window=slope_window,
    )


def fit_transform_features(
    train_df: pd.DataFrame,
    sensor_columns: tuple[str, ...] = tuple(SENSOR_COLUMNS),
    rolling_windows: tuple[int, ...] = (5,),
    lags: tuple[int, ...] = (1, 5),
    slope_window: int = 5,
) -> tuple[pd.DataFrame, FeatureConfig]:
    config = fit_feature_config(
        train_df=train_df,
        sensor_columns=sensor_columns,
        rolling_windows=rolling_windows,
        lags=lags,
        slope_window=slope_window,
    )
    return transform_features(train_df, config), config


def transform_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    validate_feature_inputs(
        df,
        config.sensor_columns,
        config.rolling_windows,
        config.lags,
        config.slope_window,
    )

    featured_df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True).copy()
    new_features: dict[str, pd.Series] = {}

    for sensor_column in config.sensor_columns:
        # feature history must stay inside one engine to avoid leakage across units
        sensor_by_unit = featured_df.groupby("unit_id", sort=False)[sensor_column]

        for window in config.rolling_windows:
            rolling = sensor_by_unit.rolling(window=window, min_periods=1)
            new_features[f"{sensor_column}_roll_mean_{window}"] = rolling.mean().reset_index(
                level=0,
                drop=True,
            )
            new_features[f"{sensor_column}_roll_std_{window}"] = rolling.std(ddof=0).reset_index(
                level=0,
                drop=True,
            )

        for lag in config.lags:
            new_features[f"{sensor_column}_lag_{lag}"] = sensor_by_unit.shift(lag)

        baseline = sensor_by_unit.shift(config.slope_window - 1)
        new_features[f"{sensor_column}_slope_{config.slope_window}"] = (
            featured_df[sensor_column] - baseline
        ) / (config.slope_window - 1)

    # build all generated columns at once to avoid pandas fragmentation warnings
    return pd.concat([featured_df, pd.DataFrame(new_features)], axis=1)


def validate_feature_inputs(
    df: pd.DataFrame,
    sensor_columns: tuple[str, ...],
    rolling_windows: tuple[int, ...],
    lags: tuple[int, ...],
    slope_window: int,
) -> None:
    required_columns = {"unit_id", "cycle", *sensor_columns}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"missing feature input columns: {sorted(missing_columns)}")

    if not rolling_windows:
        raise ValueError("at least one rolling window is required")
    if not lags:
        raise ValueError("at least one lag is required")
    if any(window < 1 for window in rolling_windows):
        raise ValueError("rolling windows must be positive")
    if any(lag < 1 for lag in lags):
        raise ValueError("lags must be positive")
    if slope_window < 2:
        raise ValueError("slope_window must be at least 2")
