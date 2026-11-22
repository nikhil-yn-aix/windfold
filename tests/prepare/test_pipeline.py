import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.config import (
    FarmConfig,
    FarmRules,
    Northing,
    TurbineConfig,
    Windows,
    load_protocol,
)
from windfold.locations import INSPECT_FILES
from windfold.prepare.inspect import write_inspect
from windfold.prepare.pipeline import (
    Ledger,
    low_wind_stops,
    prepare,
    provider_northing,
    write_prepared,
)
from windfold.schemas import EVENTS, LEDGER, PREPARED_ROWS, RAW_ROWS, conform
from windfold.schemas import LAYOUT as SCHEMA_LAYOUT

T0 = datetime(2020, 1, 1, tzinfo=UTC)
TEN = timedelta(minutes=10)

PROCESSING = load_protocol().processing.model_copy(
    update={
        "speed_bin_edges_ms": (5.0, 6.5, 8.0, 10.0),
        "direction_bucket_deg": 5.0,
        "stuck_repeats": 3,
        "stopped_power_fraction": 0.01,
        "control_min_valid_minutes": 7,
        "control_washout_minutes": 20,
        "validity_distance_d": 30.0,
        "max_speed_ms": 40.0,
        "max_power_fraction": 1.1,
        "frozen_min_speed_ms": 1.5,
        "curve_speed_tolerance": 0.2,
        "curve_power_tolerance": 0.1,
        "min_valid_turbines": 2,
    }
)
TURBINE = TurbineConfig.model_validate(
    {
        "name": "hand_made",
        "rated_power_kw": 2000.0,
        "rotor_diameter_m": 100.0,
        "source": "hand-made",
        "power_curve": {
            "speed_ms": [3, 5, 7, 9, 11, 25],
            "value": [0, 200, 600, 1200, 2000, 2000],
            "source": "hand-made",
        },
        "thrust_curve": {"speed_ms": [3, 25], "value": [0.8, 0.1], "source": "hand-made"},
    }
)
FARM = FarmConfig.model_validate(
    {
        "name": "line",
        "reader": "greenbyte",
        "turbine_type": "hand_made",
        "dataset": {"doi": "none", "licence": "none", "citation": "none"},
        "raw_files": [],
        "windows": {"development": {"start": T0, "end": T0 + timedelta(days=1)}},
        "histories": {"starts": [T0], "test": {"start": T0, "end": T0 + timedelta(days=1)}},
        "northing": {"farm_rotation": "none"},
        "source": {"archives": "none", "turbines": "none"},
        "channels": {},
        "rules": {
            "fit_power_curve": False,
            "causal_offsets": False,
            "status_kinds": {"Stop": "stop"},
        },
    }
)
LAYOUT = pl.DataFrame(
    {
        "turbine": ["A", "B", "C"],
        "latitude": [52.0, 52.0, 52.0],
        "longitude": [0.0, 0.0073, 0.0146],
        "x_m": [0.0, 500.0, 1000.0],
        "y_m": [0.0, 0.0, 0.0],
        "hub_height_m": [80.0, 80.0, 80.0],
    },
    schema=SCHEMA_LAYOUT,
)
RAW = {
    0: [("A", 600.0, 7.0, 270.0), ("B", 400.0, 6.0, 270.2), ("C", 10.0, 5.5, 270.4)],
    1: [("A", 0.0, 7.1, 271.0), ("B", 410.0, 6.1, 271.2), ("C", 310.0, 5.6, 271.4)],
    2: [("A", 620.0, 7.2, 269.0), ("B", 420.0, 6.2, 269.2), ("C", 320.0, 5.7, 269.4)],
    3: [("A", 1200.0, 9.0, 0.5), ("B", 900.0, 8.0, 0.7), ("C", 600.0, 7.0, 0.9)],
    4: [("A", 2000.0, 12.0, 359.5), ("B", 2000.0, 12.1, 359.7), ("C", 2000.0, 12.2, 359.9)],
    5: [("A", 600.0, 7.0, 270.5), ("B", 400.0, 6.0, 270.7), ("C", 300.0, None, 270.9)],
}


def raw_rows() -> pl.DataFrame:
    rows = [
        {
            "timestamp": T0 + i * TEN,
            "turbine": turbine,
            "power_kw": power,
            "speed": speed,
            "speed_std": 0.5,
            "wind_direction": angle,
            "nacelle_position": angle,
            "ambient_temperature": 10.0,
        }
        for i, block in RAW.items()
        for turbine, power, speed, angle in block
    ]
    return conform(pl.DataFrame(rows, schema=RAW_ROWS), RAW_ROWS)


def stop_on_b() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "turbine": "B",
                "start": T0 + 2 * TEN + timedelta(minutes=3),
                "end": T0 + 2 * TEN + timedelta(minutes=5),
                "source": "status_log",
                "status": "Stop",
                "code": 20,
                "message": "Manual stop - on site",
                "category": "Scheduled Maintenance",
            }
        ],
        schema=EVENTS,
    )


@pytest.fixture(scope="module")
def prepared():
    return prepare(raw_rows(), stop_on_b(), LAYOUT, FARM, TURBINE, PROCESSING)


def test_ledger_explains_every_dropped_row(prepared):
    ledger = prepared.ledger.select("step", "rule", "rows_dropped", "rows_kept", "timestamps_kept")
    assert ledger.rows() == [
        (1, "read", 0, 18, 6),
        (2, "windows", 0, 18, 6),
        (3, "power_missing", 0, 18, 6),
        (3, "stop_event", 1, 17, 6),
        (3, "curtailment_event", 0, 17, 6),
        (3, "derate_event", 0, 17, 6),
        (3, "unknown_event", 0, 17, 6),
        (3, "power_floor", 2, 15, 6),
        (4, "speed_missing", 1, 14, 6),
        (4, "direction_missing", 0, 14, 6),
        (4, "out_of_range", 0, 14, 6),
        (4, "frozen_row", 0, 14, 6),
        (4, "stuck_speed", 0, 14, 6),
        (4, "stuck_direction", 0, 14, 6),
        (5, "power_curve", 0, 14, 6),
        (7, "manual_yaw", 0, 14, 6),
        (8, "icing_event", 0, 14, 6),
        (9, "wake_source", 3, 11, 5),
        (10, "reference", 0, 11, 5),
        (10, "speed_window", 3, 8, 4),
        (11, "valid_turbines", 1, 7, 3),
        (12, "waked_low_wind_stop", 0, 7, 3),
        (12, "waked_low_wind_stop", 1, 7, 3),
        (12, "waked_low_wind_stop", 0, 7, 3),
    ]
    assert prepared.ledger.schema == LEDGER


def test_prepared_rows_hold_the_hand_computed_inputs_and_target(prepared):
    rows = prepared.rows
    assert rows.schema == PREPARED_ROWS
    half = 0.5 * math.log(1.5)
    logs = [math.log(1200.0), math.log(900.0), math.log(600.0)]
    centred = [v - sum(logs) / 3 for v in logs]
    expected = [
        (0, "A", 1, 7.0, False, 3, half),
        (0, "B", 1, 7.0, True, 3, -half),
        (3, "A", 2, 8.0, False, 7, centred[0]),
        (3, "B", 2, 8.0, False, 7, centred[1]),
        (3, "C", 2, 8.0, False, 7, centred[2]),
        (5, "A", 1, 7.0, False, 7, half),
        (5, "B", 1, 7.0, True, 7, -half),
    ]
    got = rows.select(
        ((pl.col("timestamp") - T0) / TEN).cast(pl.Int64),
        "turbine",
        "speed_bin",
        "reference_speed",
        "waked",
        "operating_mask",
        "y",
    ).rows()
    assert [g[:6] for g in got] == [e[:6] for e in expected]
    assert [g[6] for g in got] == pytest.approx([e[6] for e in expected])
    assert rows.get_column("direction_bucket").to_list() == [54, 54, 0, 0, 0, 54, 54]
    assert rows.get_column("valid_turbines").to_list() == [2, 2, 3, 3, 3, 2, 2]
    assert rows.get_column("turbulence_intensity").to_list() == pytest.approx(
        [0.5 / 7, 0.5 / 7, 0.0625, 0.0625, 0.0625, 0.5 / 7, 0.5 / 7]
    )
    sums = rows.group_by("timestamp").agg(pl.col("y").sum()).get_column("y")
    assert max(abs(v) for v in sums.to_list()) < 1e-12


def test_speed_factors_reach_the_reference_speed_only():
    days = {"start": T0, "end": T0 + timedelta(days=1)}
    later = {"start": T0 + timedelta(days=1), "end": T0 + timedelta(days=2)}
    windows = Windows.model_validate({"calibration": days, "development": later})
    farm = FARM.model_copy(update={"windows": windows})
    processing = PROCESSING.model_copy(update={"speed_factor_min_timestamps": 1})
    out = prepare(raw_rows(), stop_on_b(), LAYOUT, farm, TURBINE, processing)
    factors = out.provenance["speed_factors"]["turbines"]
    assert [(f["turbine"], f["timestamps"], f["fitted"]) for f in factors] == [
        ("A", 1, True),
        ("B", 1, True),
        ("C", 1, True),
    ]
    assert [f["factor"] for f in factors] == pytest.approx([7.5 / 9.0, 1.0, 8.5 / 7.0])
    got = out.rows.select(
        ((pl.col("timestamp") - T0) / TEN).cast(pl.Int64),
        "turbine",
        "speed",
        "reference_speed",
        "speed_bin",
    ).rows()
    a = 7.5 / 9.0 * 7.0
    expected = [
        (0, "A", 7.0, a, 0),
        (0, "B", 6.0, a, 0),
        (3, "A", 9.0, 8.0, 2),
        (3, "B", 8.0, 8.0, 2),
        (3, "C", 7.0, 8.0, 2),
        (5, "A", 7.0, a, 0),
        (5, "B", 6.0, a, 0),
    ]
    assert [g[:2] + g[4:] for g in got] == [e[:2] + e[4:] for e in expected]
    assert [g[2:4] for g in got] == [pytest.approx(e[2:4]) for e in expected]
    assert out.rows.get_column("turbulence_intensity").to_list() == pytest.approx(
        [0.5 / 7, 0.5 / 7, 0.0625, 0.0625, 0.0625, 0.5 / 7, 0.5 / 7]
    )


def test_low_wind_stops_take_the_power_floor_and_the_waiting_states():
    before = pl.DataFrame(
        {
            "timestamp": [T0, T0, T0, T0, T0],
            "turbine_index": [0, 1, 2, 3, 4],
            "state_code_min": [3.0, 0.0, 3.0, 0.0, 0.0],
            "state_code_max": [3.0, 3.0, 9.0, 0.0, None],
        },
        schema_overrides={"turbine_index": pl.UInt8},
    )
    first = pl.Series(["state_code", "state_code", "state_code", "power_floor", "state_code"])
    rules = FarmRules(
        fit_power_curve=False,
        state_codes=("state_code_min", "state_code_max"),
        low_wind_states=("3",),
    )
    stopped = low_wind_stops(before, first, rules)
    assert stopped.get_column("turbine_index").to_list() == [0, 1, 3]
    without = rules.model_copy(update={"low_wind_states": ()})
    assert low_wind_stops(before, first, without).get_column("turbine_index").to_list() == [3]


def test_outputs_are_written_whole(prepared, tmp_path):
    write_prepared(prepared, tmp_path / "line")
    written = pl.read_parquet(tmp_path / "line" / "rows.parquet")
    assert written.equals(prepared.rows)
    assert pl.read_csv(tmp_path / "line" / "ledger.csv").height == prepared.ledger.height
    assert (tmp_path / "line" / "provenance.json").read_text().startswith("{")
    assert sorted(p.name for p in (tmp_path / "line").iterdir()) == [
        "layout.parquet",
        "ledger.csv",
        "provenance.json",
        "rows.parquet",
    ]
    assert pl.read_parquet(tmp_path / "line" / "layout.parquet").equals(LAYOUT)


def test_inspect_files_are_written_whole_and_the_same_every_time(prepared, tmp_path):
    names = write_inspect(raw_rows(), prepared, FARM, TURBINE, PROCESSING, tmp_path / "one")
    write_inspect(raw_rows(), prepared, FARM, TURBINE, PROCESSING, tmp_path / "two")
    assert names == INSPECT_FILES
    assert sorted(p.name for p in (tmp_path / "one").iterdir()) == sorted(INSPECT_FILES)
    for name in INSPECT_FILES:
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    kept = pl.read_csv(tmp_path / "one" / "rows_kept.csv")
    assert kept.rows() == [("A", 3), ("B", 3), ("C", 1)]


def test_rules_sharing_a_name_count_their_own_rows():
    rows = pl.DataFrame(
        {"timestamp": [T0, T0, T0 + TEN], "turbine": ["A", "B", "A"]},
        schema={"timestamp": RAW_ROWS["timestamp"], "turbine": pl.String},
    )
    ledger = Ledger(rows)
    kept, first = ledger.apply(
        rows,
        [
            (6, "exclusion", "first window", pl.Series([True, False, False])),
            (6, "exclusion", "second window", pl.Series([True, True, False])),
        ],
    )
    lines = [(line["reason"], line["rows_dropped"], line["rows_kept"]) for line in ledger.lines]
    assert lines[1:] == [("first window", 1, 2), ("second window", 1, 1)]
    assert first.to_list() == ["exclusion", "exclusion", None]
    assert kept.get_column("turbine").to_list() == ["A"]
    assert ledger.by_turbine().rows() == [(6, "exclusion", "A", 1), (6, "exclusion", "B", 1)]


def test_off_grid_or_repeated_timestamps_are_refused():
    shifted = raw_rows().with_columns(pl.col("timestamp") + timedelta(minutes=1))
    with pytest.raises(ValueError, match="grid"):
        prepare(shifted, stop_on_b(), LAYOUT, FARM, TURBINE, PROCESSING)
    doubled = pl.concat([raw_rows(), raw_rows().head(1)])
    with pytest.raises(ValueError, match="more than once"):
        prepare(doubled, stop_on_b(), LAYOUT, FARM, TURBINE, PROCESSING)


def test_northing_files_take_over_from_their_start(tmp_path):
    (tmp_path / "before.yaml").write_text(
        "    - ['T01', 2016-01-01 00:00:00, 10.0]\n    - ['T01', 2024-10-01 00:00:00, 11.0]\n"
    )
    (tmp_path / "after.yaml").write_text(
        "    - ['T01', 2019-10-14 23:50:00, -5.0]\n    - ['T01', 2025-02-09 17:40:00, -6.0]\n"
    )
    switch = datetime(2024, 9, 1, tzinfo=UTC)
    northing = {
        "farm_rotation": "none",
        "provider_corrections": [
            {"file": "before.yaml", "start": datetime(2016, 1, 1, tzinfo=UTC)},
            {"file": "after.yaml", "start": switch},
        ],
    }
    farm = FARM.model_copy(update={"northing": Northing.model_validate(northing)})
    combined = provider_northing(farm, tmp_path)
    assert combined is not None
    assert combined.rows() == [
        ("T01", datetime(2016, 1, 1, tzinfo=UTC), 10.0),
        ("T01", switch, -5.0),
        ("T01", datetime(2025, 2, 9, 17, 40, tzinfo=UTC), -6.0),
    ]
    assert provider_northing(FARM, tmp_path) is None


def test_minutes_lose_derated_and_controlled_minutes_before_averaging():
    start = datetime(2020, 2, 18, 8, tzinfo=UTC)
    minutes = [
        {
            "timestamp": start + timedelta(minutes=i),
            "turbine": turbine,
            "power_kw": base + i,
            "speed": speed + 0.01 * i,
            "speed_std": 0.5,
            "wind_direction": 1.0 + 0.1 * i,
            "nacelle_position": 1.0 + 0.1 * i,
            "ambient_temperature": 932.0 if (turbine, i) == ("A", 25) else 10.0,
            "derated": turbine == "A" and 31 <= i <= 34,
        }
        for turbine, base, speed in (("A", 600.0, 7.0), ("B", 400.0, 6.0))
        for i in range(40)
    ]
    raw = conform(
        pl.DataFrame(minutes, schema=RAW_ROWS | {"derated": pl.Boolean}), RAW_ROWS, extra=True
    )
    control = pl.DataFrame(
        [
            {
                "turbine": "B",
                "start": start,
                "end": start + timedelta(minutes=1),
                "source": "control_log",
                "status": "active",
                "code": None,
                "message": None,
                "category": None,
            }
        ],
        schema=EVENTS,
    )
    farm = FARM.model_copy(
        update={
            "windows": FARM.windows.model_copy(
                update={
                    "development": FARM.windows.development.model_copy(
                        update={"start": start, "end": start + timedelta(hours=1)}
                    )
                }
            )
        }
    )
    rules = FarmRules(
        fit_power_curve=False, causal_offsets=False, interval_minutes=1, derated="derated"
    )
    farm = farm.model_copy(update={"rules": rules})
    out = prepare(raw, control, LAYOUT.head(2), farm, TURBINE, PROCESSING)
    lines = {rule: (dropped, kept) for _, rule, _, dropped, kept, _ in out.ledger.rows()}
    assert lines["read"] == (0, 80)
    assert lines["derated"] == (4, 76)
    assert lines["control"] == (42, 34)
    assert lines["minutes_per_window"] == (1, 3)
    assert lines["valid_turbines"] == (1, 2)
    rows = out.rows
    assert rows.select("turbine", "timestamp").rows() == [
        ("A", start + timedelta(minutes=20)),
        ("B", start + timedelta(minutes=20)),
    ]
    assert rows.get_column("power_kw").to_list() == pytest.approx([625.0, 425.0])
    assert rows.get_column("temperature").to_list() == pytest.approx([10.0, 10.0])
    assert out.provenance["temperature"]["readings_made_missing"] == 1
    assert rows.get_column("y").to_list() == pytest.approx(
        [0.5 * math.log(625.0 / 425.0), -0.5 * math.log(625.0 / 425.0)]
    )
