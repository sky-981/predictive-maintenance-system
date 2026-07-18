import math

import pandas as pd
import pytest

from predictive_maintenance.data.load import CMAPSS_COLUMNS
from predictive_maintenance.features.engineer import (
    FeatureConfig,
    fit_transform_features,
    transform_features,
)


def test_transform_features_adds_per_unit_rolling_lag_and_slope() -> None:
    df = pd.DataFrame(
        [
            make_row(unit_id=1, cycle=2, sensor_1=20),
            make_row(unit_id=2, cycle=1, sensor_1=100),
            make_row(unit_id=1, cycle=1, sensor_1=10),
            make_row(unit_id=1, cycle=3, sensor_1=30),
            make_row(unit_id=2, cycle=2, sensor_1=200),
        ],
        columns=CMAPSS_COLUMNS,
    )
    config = FeatureConfig(
        sensor_columns=("sensor_1",),
        rolling_windows=(2,),
        lags=(1,),
        slope_window=2,
    )

    featured_df = transform_features(df, config)

    assert featured_df[["unit_id", "cycle"]].values.tolist() == [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
    ]
    assert featured_df["sensor_1_roll_mean_2"].tolist() == [10, 15, 25, 100, 150]
    assert featured_df["sensor_1_roll_std_2"].tolist() == [0, 5, 5, 0, 50]
    assert math.isnan(featured_df.loc[0, "sensor_1_lag_1"])
    assert featured_df["sensor_1_lag_1"].iloc[1:3].tolist() == [10, 20]
    assert math.isnan(featured_df.loc[3, "sensor_1_lag_1"])
    assert featured_df.loc[4, "sensor_1_lag_1"] == 100
    assert featured_df["sensor_1_slope_2"].tolist()[1:3] == [10, 10]
    assert math.isnan(featured_df.loc[3, "sensor_1_slope_2"])
    assert featured_df.loc[4, "sensor_1_slope_2"] == 100


def test_fit_transform_features_returns_reusable_config() -> None:
    df = pd.DataFrame(
        [
            make_row(unit_id=1, cycle=1, sensor_1=10),
            make_row(unit_id=1, cycle=2, sensor_1=12),
        ],
        columns=CMAPSS_COLUMNS,
    )

    featured_df, config = fit_transform_features(
        df,
        sensor_columns=("sensor_1",),
        rolling_windows=(2,),
        lags=(1,),
        slope_window=2,
    )

    assert config.sensor_columns == ("sensor_1",)
    assert "sensor_1_roll_mean_2" in featured_df.columns
    assert "sensor_1_lag_1" in featured_df.columns
    assert "sensor_1_slope_2" in featured_df.columns


def test_transform_features_rejects_missing_sensor_column() -> None:
    df = pd.DataFrame({"unit_id": [1], "cycle": [1]})
    config = FeatureConfig(sensor_columns=("sensor_1",))

    with pytest.raises(ValueError, match="missing feature input columns"):
        transform_features(df, config)


def make_row(unit_id: int, cycle: int, sensor_1: float) -> list[float]:
    op_settings = [0.0, 0.0, 100.0]
    sensors = [sensor_1, *[float(sensor_idx) for sensor_idx in range(2, 22)]]
    return [unit_id, cycle, *op_settings, *sensors]
