from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance.data.load import (
    CMAPSS_COLUMNS,
    add_test_rul_labels,
    load_rul_truth,
    load_test_data,
    load_train_data,
)


def test_load_train_data_adds_raw_and_clipped_rul(tmp_path: Path) -> None:
    write_cmapss_file(
        tmp_path / "train_FD001.txt",
        [
            make_row(unit_id=1, cycle=1),
            make_row(unit_id=1, cycle=2),
            make_row(unit_id=1, cycle=3),
            make_row(unit_id=2, cycle=1),
            make_row(unit_id=2, cycle=2),
        ],
    )

    train_df = load_train_data(tmp_path, "FD001", rul_clip=1)

    assert list(train_df.columns) == [*CMAPSS_COLUMNS, "max_cycle", "rul_raw", "rul"]
    assert train_df["rul_raw"].tolist() == [2, 1, 0, 1, 0]
    assert train_df["rul"].tolist() == [1, 1, 0, 1, 0]


def test_load_test_data_and_rul_truth_use_official_last_cycle_labels(tmp_path: Path) -> None:
    write_cmapss_file(
        tmp_path / "test_FD001.txt",
        [
            make_row(unit_id=1, cycle=1),
            make_row(unit_id=1, cycle=2),
            make_row(unit_id=2, cycle=1),
        ],
    )
    (tmp_path / "RUL_FD001.txt").write_text("5\n7\n", encoding="utf-8")

    test_df = load_test_data(tmp_path, "FD001")
    truth_df = load_rul_truth(tmp_path, "FD001")
    labeled_df = add_test_rul_labels(test_df, truth_df, rul_clip=6)

    assert truth_df.to_dict("records") == [
        {"unit_id": 1, "rul_truth": 5},
        {"unit_id": 2, "rul_truth": 7},
    ]
    assert labeled_df["rul_raw"].tolist() == [6, 5, 7]
    assert labeled_df["rul"].tolist() == [6, 5, 6]


def test_add_test_rul_labels_fails_when_truth_is_missing() -> None:
    test_df = pd.DataFrame([make_row(unit_id=2, cycle=1)], columns=CMAPSS_COLUMNS)
    truth_df = pd.DataFrame([{"unit_id": 1, "rul_truth": 5}])

    with pytest.raises(ValueError, match="missing RUL truth"):
        add_test_rul_labels(test_df, truth_df)


def write_cmapss_file(path: Path, rows: list[list[float]]) -> None:
    text = "\n".join(" ".join(str(value) for value in row) for row in rows)
    path.write_text(f"{text}\n", encoding="utf-8")


def make_row(unit_id: int, cycle: int) -> list[float]:
    op_settings = [0.0, 0.0, 100.0]
    sensors = [float(sensor_idx) for sensor_idx in range(1, 22)]
    return [unit_id, cycle, *op_settings, *sensors]
