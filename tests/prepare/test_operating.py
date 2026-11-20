from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.prepare.operating import (
    below_power_floor,
    binding_setpoint,
    classify_events,
    interval_overlaps,
    lowest_known_setpoint,
    operating_masks,
)
from windfold.schemas import EVENTS
from windfold.schemas import UTC as UTC_TYPE

TEN = timedelta(minutes=10)


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2021, 3, 1, hour, minute, second, tzinfo=UTC)


def events(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "turbine": turbine,
                "start": start,
                "end": end,
                "source": "status_log",
                "status": status,
                "code": code,
                "message": message,
                "category": None,
            }
            for turbine, start, end, status, code, message in rows
        ],
        schema=EVENTS,
    )


def grid(*rows: tuple[str, datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {"turbine": [t for t, _ in rows], "timestamp": [s for _, s in rows]},
        schema={"turbine": pl.String, "timestamp": UTC_TYPE},
    )


def test_reused_code_is_classified_by_its_message():
    logged = events(
        ("A", datetime(2016, 5, 1, tzinfo=UTC), None, "Stop", 111, "Emergency stop nacelle"),
        (
            "A",
            datetime(2021, 5, 1, tzinfo=UTC),
            None,
            "Curtailment",
            111,
            "Grid constraint curtailment",
        ),
        ("A", at(1, 0), None, "Warning", 9000, "P output externally reduced"),
        ("A", at(1, 0), None, "Informational", 6410, "Manual yaw"),
        ("A", at(1, 0), None, "Informational", 0, "System OK"),
        ("A", at(1, 0), None, "Stop", 6690, "Icing (stop)"),
        ("A", at(1, 0), None, "Warning", 5000, "Breakdown obstacle light"),
    )
    kinds = classify_events(
        logged,
        statuses={"Stop": "stop", "Curtailment": "curtailment", "Communication": "unknown"},
        messages={
            ("Warning", "P output externally reduced"): "derate",
            ("Informational", "Manual yaw"): "manual_yaw",
            ("Stop", "Icing (stop)"): "icing",
        },
    ).get_column("kind")
    assert kinds.to_list() == ["stop", "curtailment", "derate", "manual_yaw", None, "icing", None]


def test_event_marks_exactly_the_intervals_it_overlaps():
    rows = grid(
        ("A", at(10, 20)),
        ("A", at(9, 50)),
        ("A", at(10, 0)),
        ("B", at(10, 0)),
        ("A", at(10, 10)),
    )
    logged = events(("A", at(10, 3), at(10, 17), "Stop", 1, "x"))
    assert interval_overlaps(rows, logged, TEN).to_list() == [False, False, True, False, True]


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (at(10, 0), at(10, 20), [False, True, True, False]),
        (at(10, 20), at(10, 20), [False, False, False, True]),
        (at(9, 55), None, [True, False, False, False]),
        (at(10, 10), at(10, 10, 1), [False, False, True, False]),
    ],
)
def test_event_boundaries(start, end, expected):
    rows = grid(("A", at(9, 50)), ("A", at(10, 0)), ("A", at(10, 10)), ("A", at(10, 20)))
    logged = events(("A", start, end, "Stop", 1, "x"))
    assert interval_overlaps(rows, logged, TEN).to_list() == expected


def test_farm_wide_events_mark_every_turbine():
    rows = grid(("A", at(10, 0)), ("B", at(10, 0)), ("B", at(10, 20)))
    logged = events(("A", at(10, 3), at(10, 9), "on", 1, "x"))
    assert interval_overlaps(rows, logged, TEN, farm_wide=True).to_list() == [True, True, False]


def test_lowest_known_setpoint_takes_this_and_previous_end_values():
    rows = grid(
        ("A", at(10, 10)),
        ("A", at(10, 0)),
        ("A", at(10, 20)),
        ("A", at(10, 40)),
        ("A", at(10, 50)),
    ).with_columns(pl.Series("end_setpoint", [1990.0, 2300.0, 2300.0, 2210.0, None]))
    lowest = lowest_known_setpoint(rows, "end_setpoint", TEN)
    assert lowest.to_list() == [1990.0, 2300.0, 1990.0, 2210.0, 2210.0]


@pytest.mark.parametrize(
    ("setpoint", "power", "expected"),
    [
        (1993.0, 1990.0, True),
        (1993.0, 1500.0, False),
        (2300.0, 2290.0, False),
        (2210.0, 2100.0, True),
        (2210.0, 2090.0, False),
        (None, 2000.0, False),
        (1000.0, 1400.0, True),
        (1150.0, 330.0, False),
    ],
)
def test_binding_setpoint_on_a_2300_kw_turbine(setpoint, power, expected):
    frame = pl.DataFrame(
        {"setpoint": [setpoint], "power": [power]},
        schema={"setpoint": pl.Float64, "power": pl.Float64},
    )
    flag = frame.select(binding_setpoint(pl.col("power"), pl.col("setpoint"), 2300.0, 0.98, 0.05))
    assert flag.item() is expected


def test_power_floor_is_one_percent_of_rated():
    frame = pl.DataFrame({"power": [20.0, 20.5, None, -3.0]})
    flags = frame.select(below_power_floor(pl.col("power"), 2050.0, 0.01)).to_series()
    assert flags.to_list() == [True, False, False, True]


def test_operating_mask_sets_one_bit_per_turbine():
    rows = pl.DataFrame(
        {"timestamp": [1, 1, 2], "turbine_index": [0, 3, 5]},
        schema={"timestamp": pl.Int64, "turbine_index": pl.UInt8},
    )
    masks = operating_masks(rows).sort("timestamp")
    assert masks.get_column("operating_mask").to_list() == [9, 32]
    assert masks.schema["operating_mask"] == pl.UInt32
