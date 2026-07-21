from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class UnitSplit:
    train_unit_ids: tuple[int, ...]
    val_unit_ids: tuple[int, ...]


def split_units(
    train_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 0,
) -> UnitSplit:
    """Split a training file's unit_ids into train/validation groups.

    Splitting by unit_id (not by row) keeps every cycle of one engine on the
    same side of the split, so the model is never validated on a machine it
    already saw during training.
    """
    unit_ids = train_df["unit_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(splitter.split(train_df, groups=unit_ids))

    train_units = tuple(sorted(train_df.iloc[train_idx]["unit_id"].unique()))
    val_units = tuple(sorted(train_df.iloc[val_idx]["unit_id"].unique()))
    return UnitSplit(train_unit_ids=train_units, val_unit_ids=val_units)


def apply_unit_split(df: pd.DataFrame, split: UnitSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a dataframe into (train_df, val_df) using a fitted UnitSplit."""
    train_df = df[df["unit_id"].isin(split.train_unit_ids)].copy()
    val_df = df[df["unit_id"].isin(split.val_unit_ids)].copy()
    return train_df, val_df


def assert_no_unit_overlap(split: UnitSplit) -> None:
    """Fail loudly if any unit_id ended up on both sides of the split."""
    overlap = set(split.train_unit_ids) & set(split.val_unit_ids)
    if overlap:
        raise ValueError(f"unit_id leakage between train and val: {sorted(overlap)}")
