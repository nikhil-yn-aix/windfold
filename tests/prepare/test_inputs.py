import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.prepare.inputs import (
    direction_buckets,
    farm_directions,
    reference_speeds,
    speed_bins,
    speed_factors,
    steady_flags,
)

TEN = timedelta(minutes=10)
T0 = datetime(2020, 1, 1, tzinfo=UTC)


def test_farm_direction_is_the_circular_mean_with_its_resultant():
    rows = pl.DataFrame(
        {"timestamp": [1, 1, 2, 2, 2], "direction": [350.0, 10.0, 90.0, 90.0, 270.0]}
    )
    farm = farm_directions(rows, "direction").sort("timestamp")
    first, second = farm.row(0, named=True), farm.row(1, named=True)
    assert min(first["farm_direction"], 360.0 - first["farm_direction"]) == pytest.approx(
        0.0, abs=1e-9
    )
    assert first["resultant"] == pytest.approx(math.cos(math.radians(10.0)))
    assert second["farm_direction"] == pytest.approx(90.0)
    assert second["resultant"] == pytest.approx(1.0 / 3.0)


def test_steady_needs_a_small_change_from_the_previous_interval_and_agreement():
    farm = pl.DataFrame(
        {
            "timestamp": [T0 + i * TEN for i in (0, 1, 2, 4, 5, 6, 7)],
            "farm_direction": [10.0, 15.0, 30.0, 30.0, 35.0, 355.0, 3.0],
            "resultant": [0.99, 0.99, 0.99, 0.99, 0.90, 0.99, 0.99],
        }
    )
    steady = steady_flags(farm, TEN, 10.0, 0.95).sort("timestamp").get_column("steady")
    assert steady.to_list() == [False, True, False, False, False, False, True]


def test_reference_speed_uses_free_stream_rows_only():
    rows = pl.DataFrame(
        {
            "timestamp": [1, 1, 1, 1, 2, 3, 3],
            "speed": [7.0, 8.0, 9.0, 3.0, 6.0, 6.0, 7.0],
            "speed_std": [0.7, 0.8, 1.8, 0.1, 0.6, None, None],
            "waked": [False, False, False, True, True, False, False],
        }
    )
    reference = reference_speeds(rows).sort("timestamp")
    assert reference.get_column("timestamp").to_list() == [1, 3]
    assert reference.get_column("reference_speed").to_list() == [8.0, 6.5]
    assert reference.get_column("turbulence_intensity").to_list() == [pytest.approx(0.1), None]


def test_speed_factor_is_the_median_ratio_of_the_other_free_stream_speeds():
    rows = pl.DataFrame(
        {
            "timestamp": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "turbine": ["A", "B", "C"] * 3,
            "speed": [8.0, 10.0, 9.0, 10.0, 10.0, 12.0, 5.0, 6.0, 7.0],
            "waked": [False, False, True, False, False, False, False, True, True],
        }
    )
    factors = speed_factors(rows, 2)
    assert factors.select("turbine", "timestamps", "fitted").rows() == [
        ("A", 2, True),
        ("B", 2, True),
        ("C", 1, False),
    ]
    assert factors.get_column("factor").to_list() == pytest.approx([1.175, 0.95, 1.0])


def test_speed_bins_are_closed_on_the_left():
    speeds = pl.DataFrame({"s": [4.99, 5.0, 6.49, 6.5, 7.99, 8.0, 9.99, 10.0, None]})
    bins = speeds.select(speed_bins(pl.col("s"), [5.0, 6.5, 8.0, 10.0])).to_series()
    assert bins.to_list() == [None, 0, 0, 1, 1, 2, 2, None, None]


def test_direction_buckets_are_five_degrees_from_north():
    directions = pl.DataFrame({"d": [0.0, 4.99, 5.0, 359.99, 360.0]})
    buckets = directions.select(direction_buckets(pl.col("d"), 5.0)).to_series()
    assert buckets.to_list() == [0, 0, 1, 71, 0]
    assert buckets.dtype == pl.UInt8
