from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.prepare.faults import frozen_rows, linear_angle_mean, stuck_values

START = datetime(2021, 3, 1, tzinfo=UTC)


def series_rows(turbine: str, values: dict[str, list]) -> pl.DataFrame:
    n = len(next(iter(values.values())))
    return pl.DataFrame(
        {
            "turbine": [turbine] * n,
            "timestamp": [START + timedelta(minutes=10 * i) for i in range(n)],
            **values,
        }
    )


def test_three_equal_speeds_in_a_row_are_stuck_and_runs_stay_inside_a_turbine():
    a = series_rows("A", {"speed": [5.1, 5.1, 5.1, 6.0, 6.0, None, 6.0]})
    b = series_rows("B", {"speed": [6.0, 6.0]})
    rows = pl.concat([a, b]).with_row_index("i")
    shuffled = rows.sample(fraction=1.0, shuffle=True, seed=3)
    flags = shuffled.with_columns(stuck_values(shuffled, "speed", 3).alias("stuck"))
    expected = {
        0: True,
        1: True,
        2: True,
        3: False,
        4: False,
        5: False,
        6: False,
        7: False,
        8: False,
    }
    assert dict(zip(flags["i"], flags["stuck"], strict=True)) == expected


def test_frozen_rows_repeat_every_field_above_the_speed_threshold():
    a = series_rows(
        "A",
        {
            "power_kw": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "speed": [5.0, 5.0, 5.0, 5.0, 1.0, 1.0],
            "wind_direction": [200.0, 200.0, 201.0, 201.0, 201.0, 201.0],
        },
    )
    b = series_rows(
        "B",
        {
            "power_kw": [100.0, None, None],
            "speed": [1.0, 5.0, 5.0],
            "wind_direction": [201.0, 201.0, 201.0],
        },
    )
    rows = pl.concat([b, a])
    flags = frozen_rows(rows, ["power_kw", "speed", "wind_direction"], "speed", 1.5)
    assert flags.to_list() == [False, False, True, False, True, False, True, False, False]


@pytest.mark.parametrize(
    ("mean", "minimum", "maximum", "expected"),
    [
        (180.0, 0.5, 359.5, True),
        (5.0, 0.5, 359.5, False),
        (350.0, 0.2, 359.9, False),
        (180.0, 2.0, 359.5, False),
        (200.0, 0.0, 358.5, False),
        (30.0, 0.0, 360.0, True),
        (None, 0.0, 360.0, None),
        (180.0, -1.0, 359.5, None),
        (180.0, None, 359.5, None),
        (359.0, -1.0, -1.0, False),
        (200.0, -1.0, 300.0, False),
    ],
)
def test_linear_mean_across_north(mean, minimum, maximum, expected):
    frame = pl.DataFrame(
        {"mean": [mean], "minimum": [minimum], "maximum": [maximum]},
        schema={"mean": pl.Float64, "minimum": pl.Float64, "maximum": pl.Float64},
    )
    flag = frame.select(
        linear_angle_mean(pl.col("mean"), pl.col("minimum"), pl.col("maximum"), 1.0, 15.0)
    )
    assert flag.item() is expected
