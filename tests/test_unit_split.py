import pandas as pd
import pytest

from predictive_maintenance.data.split import (
    UnitSplit,
    apply_unit_split,
    assert_no_unit_overlap,
    split_units,
)


def test_split_units_has_no_overlap_and_covers_every_unit() -> None:
    train_df = make_units_df(unit_count=20, cycles_per_unit=3)

    split = split_units(train_df, val_fraction=0.25, random_state=0)
    assert_no_unit_overlap(split)

    all_units = set(train_df["unit_id"].unique())
    assert set(split.train_unit_ids) | set(split.val_unit_ids) == all_units
    assert set(split.train_unit_ids) & set(split.val_unit_ids) == set()


def test_split_units_is_reproducible_with_same_random_state() -> None:
    train_df = make_units_df(unit_count=20, cycles_per_unit=3)

    first_split = split_units(train_df, val_fraction=0.25, random_state=0)
    second_split = split_units(train_df, val_fraction=0.25, random_state=0)

    assert first_split == second_split


def test_apply_unit_split_keeps_every_row_on_exactly_one_side() -> None:
    train_df = make_units_df(unit_count=10, cycles_per_unit=4)
    split = split_units(train_df, val_fraction=0.3, random_state=0)

    train_rows, val_rows = apply_unit_split(train_df, split)

    assert len(train_rows) + len(val_rows) == len(train_df)
    assert set(train_rows["unit_id"].unique()) == set(split.train_unit_ids)
    assert set(val_rows["unit_id"].unique()) == set(split.val_unit_ids)


def test_assert_no_unit_overlap_raises_on_leakage() -> None:
    leaking_split = UnitSplit(train_unit_ids=(1, 2, 3), val_unit_ids=(3, 4))

    with pytest.raises(ValueError, match="unit_id leakage"):
        assert_no_unit_overlap(leaking_split)


def make_units_df(unit_count: int, cycles_per_unit: int) -> pd.DataFrame:
    rows = [
        {"unit_id": unit_id, "cycle": cycle}
        for unit_id in range(1, unit_count + 1)
        for cycle in range(1, cycles_per_unit + 1)
    ]
    return pd.DataFrame(rows)
