from pathlib import Path

import pandas as pd

OP_SETTING_COLUMNS = [f"op_setting_{idx}" for idx in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]
CMAPSS_COLUMNS = ["unit_id", "cycle", *OP_SETTING_COLUMNS, *SENSOR_COLUMNS]
DEFAULT_RUL_CLIP = 125


def load_train_data(
    data_dir: str | Path,
    subset: str = "FD001",
    rul_clip: int = DEFAULT_RUL_CLIP,
) -> pd.DataFrame:
    """Load a C-MAPSS training file and add row-level RUL labels."""
    train_df = _read_cmapss_file(Path(data_dir) / f"train_{subset}.txt")
    return add_train_rul_labels(train_df, rul_clip=rul_clip)


def load_test_data(data_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Load a C-MAPSS test file with standard column names."""
    return _read_cmapss_file(Path(data_dir) / f"test_{subset}.txt")


def load_rul_truth(data_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Load official RUL labels for the last observed cycle of each test unit."""
    truth_path = Path(data_dir) / f"RUL_{subset}.txt"
    if not truth_path.exists():
        raise FileNotFoundError(f"missing RUL truth file: {truth_path}")

    truth_df = pd.read_csv(truth_path, sep=r"\s+", header=None, names=["rul_truth"])
    truth_df.insert(0, "unit_id", range(1, len(truth_df) + 1))
    return truth_df


def add_train_rul_labels(
    train_df: pd.DataFrame,
    rul_clip: int = DEFAULT_RUL_CLIP,
) -> pd.DataFrame:
    """Add raw and clipped RUL labels to complete run-to-failure training rows."""
    labeled_df = train_df.copy()
    max_cycle = labeled_df.groupby("unit_id")["cycle"].transform("max")

    labeled_df["max_cycle"] = max_cycle
    labeled_df["rul_raw"] = max_cycle - labeled_df["cycle"]
    labeled_df["rul"] = labeled_df["rul_raw"].clip(upper=rul_clip)
    return labeled_df


def add_test_rul_labels(
    test_df: pd.DataFrame,
    rul_truth_df: pd.DataFrame,
    rul_clip: int = DEFAULT_RUL_CLIP,
) -> pd.DataFrame:
    """Add row-level test RUL labels from official last-cycle truth values."""
    labeled_df = test_df.copy()
    truth_by_unit = rul_truth_df.set_index("unit_id")["rul_truth"]
    missing_units = set(labeled_df["unit_id"].unique()) - set(truth_by_unit.index)
    if missing_units:
        raise ValueError(f"missing RUL truth for units: {sorted(missing_units)}")

    last_seen_cycle = labeled_df.groupby("unit_id")["cycle"].transform("max")
    final_rul = labeled_df["unit_id"].map(truth_by_unit)

    labeled_df["last_seen_cycle"] = last_seen_cycle
    labeled_df["rul_truth"] = final_rul
    labeled_df["rul_raw"] = last_seen_cycle + final_rul - labeled_df["cycle"]
    labeled_df["rul"] = labeled_df["rul_raw"].clip(upper=rul_clip)
    return labeled_df


def _read_cmapss_file(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"missing C-MAPSS file: {file_path}")

    return pd.read_csv(file_path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
