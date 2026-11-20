from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.evaluation.trial import trial_support


def record(days: int, direction: float = 10.0) -> pl.DataFrame:
    stamps = [
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=10 * k) for k in range(144 * days)
    ]
    return pl.DataFrame(
        {"timestamp": stamps, "direction": [direction] * len(stamps), "rows": [1] * len(stamps)}
    ).with_columns(day=pl.col("timestamp").dt.date())


def arm_rows_by_brute_force(days: int, toggle: int) -> tuple[int, int]:
    minutes = np.arange(0, days * 1440, 10)
    arm = (minutes // toggle) % 2
    return int((arm == 0).sum()), int((arm == 1).sum())


@pytest.mark.parametrize("days", [7, 10, 14])
def test_identical_weeks_give_the_continuous_toggle_counts(days):
    first, second = arm_rows_by_brute_force(days, 50)
    table = trial_support(
        record(14),
        date(2020, 1, 1),
        2,
        block_days=7,
        arc_count=12,
        toggle_minutes=50,
        durations_days=[days],
        min_rows=min(first, second),
        min_days=days,
        replicates=200,
        seed=1,
    )
    arc = table.filter(pl.col("arc") == 0).row(0, named=True)
    assert arc["probability"] == 1.0
    assert arc["median_rows"] == min(first, second)
    stricter = trial_support(
        record(14),
        date(2020, 1, 1),
        2,
        block_days=7,
        arc_count=12,
        toggle_minutes=50,
        durations_days=[days],
        min_rows=max(first, second) + 1,
        min_days=days,
        replicates=200,
        seed=1,
    )
    assert stricter.filter(pl.col("arc") == 0)["probability"].item() == 0.0
    assert table.filter(pl.col("arc") == 30)["probability"].item() == 0.0


def test_a_record_with_one_good_week_in_four_succeeds_about_a_quarter_of_the_time():
    good = record(7)
    gaps = good.filter(pl.col("day") == date(2020, 1, 1)).with_columns(
        timestamp=pl.col("timestamp") + timedelta(days=21), day=pl.col("day") + timedelta(days=21)
    )
    table = trial_support(
        pl.concat([good, gaps]),
        date(2020, 1, 1),
        4,
        block_days=7,
        arc_count=12,
        toggle_minutes=50,
        durations_days=[7],
        min_rows=400,
        min_days=7,
        replicates=4000,
        seed=2,
    )
    assert table.filter(pl.col("arc") == 0)["probability"].item() == pytest.approx(0.25, abs=0.03)
