from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.prepare.direction import (
    apply_corrections,
    causal_offsets,
    consensus_deviations,
    consensus_offsets,
    fit_dip,
    turbine_offsets,
    wake_dip_rotation,
)
from windfold.schemas import UTC as UTC_TYPE

TEN = timedelta(minutes=10)
T0 = datetime(2023, 1, 1, tzinfo=UTC)


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def test_provider_corrections_apply_from_their_start_and_before_the_first():
    corrections = pl.DataFrame(
        {
            "turbine": ["T01", "T01", "T02"],
            "valid_from": [utc(2023, 1, 1), utc(2024, 9, 1), utc(2023, 6, 1)],
            "offset_deg": [10.0, -5.0, 200.0],
        },
        schema_overrides={"valid_from": UTC_TYPE},
    )
    rows = pl.DataFrame(
        {
            "turbine": ["T01", "T01", "T01", "T02"],
            "timestamp": [utc(2023, 6, 1), utc(2024, 9, 1), utc(2022, 12, 31), utc(2023, 7, 1)],
            "wind_direction": [355.0, 100.0, 20.0, 170.0],
        },
        schema_overrides={"timestamp": UTC_TYPE},
    )
    corrected = apply_corrections(rows, corrections, ("wind_direction",))
    assert corrected.get_column("wind_direction").to_list() == pytest.approx(
        [5.0, 95.0, 30.0, 10.0]
    )
    assert corrected.columns == rows.columns
    with pytest.raises(ValueError, match="T03"):
        apply_corrections(rows.with_columns(pl.lit("T03").alias("turbine")), corrections, ())


def test_deviation_is_from_the_median_of_the_other_turbines():
    rows = pl.DataFrame(
        {
            "timestamp": [1, 1, 1, 1, 2, 2, 2, 3],
            "turbine": ["A", "B", "C", "D", "A", "B", "C", "A"],
            "direction": [10.0, 12.0, 8.0, 60.0, 359.0, 1.0, 3.0, 50.0],
        }
    )
    deviation = consensus_deviations(rows, "direction", 2)
    assert deviation.to_list()[:7] == pytest.approx([-2.0, 2.0, -4.0, 50.0, -3.0, 0.0, 3.0])
    assert deviation.to_list()[7] is None


def ten_minute_times(n: int) -> np.ndarray:
    return np.datetime64("2023-01-01T00:00", "us") + np.arange(n) * np.timedelta64(10, "m")


def test_offsets_are_measured_against_the_others_less_their_own_offsets():
    angles = {"A": 180.0, "B": 181.0, "C": 182.0, "D": 183.0, "E": 230.0}
    rows = pl.DataFrame(
        {
            "timestamp": [T0 + i * TEN for i in range(300) for _ in angles],
            "turbine": list(angles) * 300,
            "direction": list(angles.values()) * 300,
        }
    )
    found = consensus_offsets(rows, "direction", 2, timedelta(days=1), timedelta(hours=2), 10.0, 3)
    offsets = found.get_column("offset")
    assert offsets.head(15).null_count() == 15
    assert offsets.tail(5).to_list() == pytest.approx([-1.75, -0.75, 0.0, 0.75, 47.75])
    assert not found.get_column("jump").any()


def test_offsets_follow_a_jump_away_and_back():
    day = 144
    jump, back = 40 * day, 60 * day
    values = np.full(80 * day, -5.0)
    values[jump:back] = 45.0
    offsets, jumped = turbine_offsets(
        ten_minute_times(values.size), values, timedelta(days=30), timedelta(minutes=1450), 10.0, 72
    )
    expected = np.full(values.size, np.nan)
    expected[72 : jump - 72] = -5.0
    expected[jump + 145 : back - 72] = 45.0
    expected[back + 145 :] = -5.0
    flagged = np.zeros(values.size, dtype=bool)
    flagged[jump - 72 : jump + 73] = True
    flagged[back - 72 : back + 73] = True
    np.testing.assert_array_equal(offsets, expected)
    np.testing.assert_array_equal(jumped, flagged)


def test_a_week_long_shift_of_thirteen_degrees_is_found():
    day = 144
    values = np.full(50 * day, -5.0)
    values[35 * day : 42 * day] = 8.0
    offsets, jumped = turbine_offsets(
        ten_minute_times(values.size), values, timedelta(days=30), timedelta(minutes=1450), 10.0, 72
    )
    assert jumped[35 * day + 72] and jumped[42 * day + 72]
    assert offsets[40 * day] == 8.0
    assert offsets[49 * day] == -5.0


def test_offsets_line_up_with_their_rows_and_carry_over_unknown_deviations():
    times = [T0 + i * TEN for i in range(5)]
    rows = pl.DataFrame(
        {
            "turbine": ["B", "A", "A", "A", "A", "B", "A"],
            "timestamp": [times[0], *times[:4], times[1], times[4]],
        },
        schema_overrides={"timestamp": UTC_TYPE},
    )
    deviation = pl.Series([1.0, 2.0, None, 4.0, 6.0, 3.0, None])
    found = causal_offsets(rows, deviation, timedelta(days=1), timedelta(hours=1), 50.0, 2)
    assert found.get_column("offset").to_list() == [None, None, None, None, 3.0, None, 3.0]
    assert found.get_column("jump").to_list() == [False] * 7


def test_wake_dips_set_the_farm_rotation():
    layout = pl.DataFrame(
        {
            "turbine": ["A", "B"],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, -500.0],
            "rotor_diameter_m": [100.0, 100.0],
        }
    )
    true_direction = np.repeat(np.arange(-30.0, 30.0, 0.1), 2)
    measured = (true_direction + 4.0) % 360.0
    ratio = 1.0 - 0.4 * np.exp(-0.5 * (true_direction / 6.0) ** 2)
    n = true_direction.size
    rows = pl.DataFrame(
        {
            "timestamp": np.tile(np.arange(n // 2), 2),
            "turbine": ["A"] * (n // 2) + ["B"] * (n // 2),
            "power_kw": np.concatenate([np.full(n // 2, 1000.0), 1000.0 * ratio[::2]]),
            "speed": np.full(n, 8.0),
            "direction": np.concatenate([measured[::2], measured[::2]]),
        }
    )
    rotation, pairs = wake_dip_rotation(
        rows,
        layout,
        distance_d=(2.0, 10.0),
        speed_ms=(5.0, 10.0),
        half_width_deg=30.0,
        min_rows=5,
        min_depth=0.1,
    )
    assert rotation == pytest.approx(-4.0, abs=0.2)
    assert pairs.select("source", "target").rows() == [("A", "B")]


def test_a_pair_whose_ratio_sits_above_two_has_no_dip():
    offsets = np.arange(-14.5, 15.0, 1.0)
    with pytest.raises(RuntimeError, match="outside 0 to 2"):
        fit_dip(offsets, np.full(offsets.size, 2.02), 30.0)
