import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.config import Window
from windfold.prepare.control import exclusion_reasons, interval_means, with_washout
from windfold.prepare.operating import interval_overlaps
from windfold.schemas import EVENTS
from windfold.schemas import UTC as UTC_TYPE

MINUTE = timedelta(minutes=1)
TEN = timedelta(minutes=10)


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def stamps(*times: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {"turbine": ["A"] * len(times), "timestamp": list(times)},
        schema={"turbine": pl.String, "timestamp": UTC_TYPE},
    )


def test_exclusion_windows_drop_overlapping_intervals_with_their_reason():
    windows = [
        Window(start=utc(2024, 3, 14, 7), end=utc(2024, 3, 14, 18), reason="tune-up"),
        Window(start=utc(2023, 7, 1), end=utc(2023, 7, 4), reason="curtailment"),
        Window(start=utc(2024, 3, 14, 17), end=utc(2024, 3, 15), reason="later"),
    ]
    rows = stamps(
        utc(2024, 3, 14, 6, 50),
        utc(2024, 3, 14, 7, 0),
        utc(2024, 3, 14, 17, 50),
        utc(2024, 3, 14, 18, 0),
        utc(2023, 7, 3, 23, 50),
        utc(2023, 7, 4, 0, 0),
    )
    reasons = exclusion_reasons(rows, windows, TEN).to_list()
    assert reasons == [None, "tune-up", "tune-up", "later", "curtailment", None]


def test_control_minutes_and_twenty_minute_washout_are_dropped_farm_wide():
    control = pl.DataFrame(
        [
            {"turbine": "SMV6", "start": utc(2020, 2, 18, 7, 0), "end": utc(2020, 2, 18, 8, 1)},
            {"turbine": "SMV6", "start": utc(2020, 2, 18, 9, 0), "end": None},
        ],
        schema={k: EVENTS[k] for k in ("turbine", "start", "end")},
    )
    minutes = pl.DataFrame(
        {
            "turbine": ["SMV1"] * 6,
            "timestamp": [
                utc(2020, 2, 18, 6, 59),
                utc(2020, 2, 18, 7, 0),
                utc(2020, 2, 18, 8, 20),
                utc(2020, 2, 18, 8, 21),
                utc(2020, 2, 18, 9, 19),
                utc(2020, 2, 18, 9, 20),
            ],
        },
        schema={"turbine": pl.String, "timestamp": UTC_TYPE},
    )
    dropped = interval_overlaps(
        minutes, with_washout(control, timedelta(minutes=20)), MINUTE, farm_wide=True
    )
    assert dropped.to_list() == [False, True, True, False, True, False]


def minute_rows(start: datetime, n: int, **columns: list) -> pl.DataFrame:
    return pl.DataFrame(
        {"turbine": ["A"] * n, "timestamp": [start + i * MINUTE for i in range(n)], **columns},
        schema_overrides={"timestamp": UTC_TYPE},
    )


def test_minutes_average_into_ten_minute_windows_with_seven_complete_minutes():
    full = minute_rows(
        utc(2020, 3, 1, 10, 0),
        10,
        power_kw=[100.0 + i for i in range(10)],
        speed=[4.0, 6.0] * 5,
        speed_std=[1.0] * 10,
        wind_direction=[350.0, 10.0] * 5,
        nacelle_position=[350.0, 10.0] * 5,
        ambient_temperature=[5.0] * 10,
    )
    short = minute_rows(
        utc(2020, 3, 1, 10, 10),
        10,
        power_kw=[100.0] * 6 + [None] * 4,
        speed=[5.0] * 10,
        speed_std=[0.5] * 10,
        wind_direction=[180.0] * 10,
        nacelle_position=[180.0] * 10,
        ambient_temperature=[5.0] * 10,
    )
    enough = minute_rows(
        utc(2020, 3, 1, 10, 20),
        7,
        power_kw=[200.0] * 7,
        speed=[6.0] * 7,
        speed_std=[None] * 7,
        wind_direction=[90.0] * 7,
        nacelle_position=[None] * 7,
        ambient_temperature=[None] * 7,
    )
    means = interval_means(pl.concat([full, short, enough]), TEN, 7)
    assert means.get_column("timestamp").to_list() == [
        utc(2020, 3, 1, 10, 0),
        utc(2020, 3, 1, 10, 20),
    ]
    first, last = means.row(0, named=True), means.row(1, named=True)
    assert first["power_kw"] == pytest.approx(104.5)
    assert first["speed"] == pytest.approx(5.0)
    assert first["speed_std"] == pytest.approx(math.sqrt(2.0))
    assert min(first["wind_direction"], 360.0 - first["wind_direction"]) == pytest.approx(
        0.0, abs=1e-9
    )
    assert last["wind_direction"] == pytest.approx(90.0)
    assert last["speed_std"] is None
    assert last["nacelle_position"] is None
