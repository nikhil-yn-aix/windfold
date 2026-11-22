import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from windfold.readers.hill_of_towie import read_alarms, read_rows, read_shutdowns
from windfold.schemas import EVENTS, RAW_ROWS, conform

TEN_MINUTES = timedelta(minutes=10)
STATIONS = {"2304510": "T01", "2304511": "T02"}
CHANNELS = {
    "power_kw": "tblSCTurGrid.wtc_ActPower_mean",
    "speed": "tblSCTurbine.wtc_PrWindSp_mean",
    "speed_std": "tblSCTurbine.wtc_PrWindSp_stddev",
    "wind_direction": "tblSCTurbine.wtc_ActualWindDirection_mean",
    "nacelle_position": "tblSCTurbine.wtc_NacelPos_mean",
    "ambient_temperature": "tblSCTurTemp.wtc_AmbieTmp_mean",
    "state_code_min": "tblSCTurbine.wtc_WTOperSt_min",
    "state_code_max": "tblSCTurbine.wtc_WTOperSt_max",
    "running_s": "tblSCTurFlag.wtc_ScInOper_timeon",
    "speed_active": "tblSCTurbine.wtc_AcWindSp_mean",
    "speed_secondary": "tblSCTurbine.wtc_SeWindSp_mean",
}
TURBINE_COLUMNS = [
    "wtc_PrWindSp_mean",
    "wtc_PrWindSp_stddev",
    "wtc_ActualWindDirection_mean",
    "wtc_NacelPos_mean",
    "wtc_WTOperSt_min",
    "wtc_WTOperSt_max",
    "wtc_AcWindSp_mean",
    "wtc_SeWindSp_mean",
]


def table(columns: list[str], rows: list[list[str]]) -> bytes:
    lines = [",".join(["TimeStamp", "StationId", "wtc_Other_mean", *columns])]
    lines += [",".join([row[0], row[1], "1.0", *row[2:]]) for row in rows]
    return ("\n".join(lines) + "\n").encode()


def month(path: Path, year_month: str, rows: dict[str, list[list[str]]]) -> Path:
    columns = {
        "tblSCTurbine": TURBINE_COLUMNS,
        "tblSCTurGrid": ["wtc_ActPower_mean"],
        "tblSCTurTemp": ["wtc_AmbieTmp_mean"],
        "tblSCTurFlag": ["wtc_ScInOper_timeon"],
    }
    with zipfile.ZipFile(path, "a") as archive:
        for name, cols in columns.items():
            archive.writestr(f"{name}_{year_month}.csv", table(cols, rows.get(name, [])))
    return path


def turbine_row(
    stamp: str, station: str, state: str = "0", speed: str = "7.5", active: str = "7.5"
):
    return [stamp, station, speed, "0.9", "123.4", "120.0", state, state, active, "7.1"]


def one_interval(stamp: str, station: str = "2304510", **turbine) -> dict[str, list[list[str]]]:
    return {
        "tblSCTurbine": [turbine_row(stamp, station, **turbine)],
        "tblSCTurGrid": [[stamp, station, "1500.5"]],
        "tblSCTurTemp": [[stamp, station, "8.25"]],
        "tblSCTurFlag": [[stamp, station, "600.0"]],
    }


def test_end_labels_become_interval_starts_once_across_the_four_tables(tmp_path: Path):
    path = month(tmp_path / "2024.zip", "2024_06", one_interval("2024-06-01 00:10:00"))
    rows = read_rows([path], STATIONS, CHANNELS, TEN_MINUTES)
    conform(rows, RAW_ROWS, extra=True)
    assert rows.drop("timestamp").row(0, named=True) == {
        "turbine": "T01",
        "power_kw": 1500.5,
        "speed": 7.5,
        "speed_std": 0.9,
        "wind_direction": 123.4,
        "nacelle_position": 120.0,
        "ambient_temperature": 8.25,
        "state_code_min": 0.0,
        "state_code_max": 0.0,
        "running_s": 600.0,
        "speed_active": 7.5,
        "speed_secondary": 7.1,
    }
    assert rows["timestamp"].to_list() == [datetime(2024, 6, 1, 0, 0, tzinfo=UTC)]


def test_month_boundary_copies_written_as_0_and_0_0_and_a_date_alone_merge(tmp_path: Path):
    path = tmp_path / "2024.zip"
    month(path, "2024_07", one_interval("2024-08-01", state="0.0"))
    month(path, "2024_08", one_interval("2024-08-01 00:00:00", state="0"))
    rows = read_rows([path], STATIONS, CHANNELS, TEN_MINUTES)
    assert rows.select("timestamp", "state_code_min").rows() == [
        (datetime(2024, 7, 31, 23, 50, tzinfo=UTC), 0.0)
    ]


def test_copies_that_differ_stop_the_reader(tmp_path: Path):
    path = tmp_path / "2024.zip"
    month(path, "2024_07", one_interval("2024-08-01 00:00:00", state="0"))
    month(path, "2024_08", one_interval("2024-08-01 00:00:00", state="3"))
    with pytest.raises(ValueError, match="tblSCTurbine: 2 rows give different values"):
        read_rows([path], STATIONS, CHANNELS, TEN_MINUTES)


def test_a_station_missing_from_the_metadata_stops_the_reader(tmp_path: Path):
    path = month(tmp_path / "2024.zip", "2024_06", one_interval("2024-06-01 00:10:00", "2304599"))
    with pytest.raises(ValueError, match=r"stations \['2304599'\] are not in the turbine metadata"):
        read_rows([path], STATIONS, CHANNELS, TEN_MINUTES)


def test_a_row_where_the_controller_used_the_secondary_sensor_has_no_speed(tmp_path: Path):
    rows = one_interval("2024-06-01 00:10:00", speed="9.0", active="7.15")
    rows["tblSCTurbine"].append(
        turbine_row("2024-06-01 00:20:00", "2304510", speed="9.0", active="8.9")
    )
    path = month(tmp_path / "2024.zip", "2024_06", rows)
    read = read_rows([path], STATIONS, CHANNELS, TEN_MINUTES)
    assert read.select("speed", "speed_std").rows() == [(None, None), (9.0, 0.9)]
    assert read["wind_direction"].to_list() == [123.4, 123.4]


def alarm_log(records: list[str]) -> bytes:
    return ("TimeOn,TimeOff,StationNr,Alarmcode\n" + "\n".join(records) + "\n").encode()


def test_alarms_become_events_with_open_ends_and_without_other_stations(tmp_path: Path):
    path = tmp_path / "2024.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "tblAlarmLog_2024_06.csv",
            alarm_log(
                [
                    "2024-06-01 00:00:07,,2304514,25",
                    "2024-06-02 10:00:00,2024-06-02 11:30:00,2304510,3130",
                    "2024-06-02 10:00:00,2024-06-02 11:30:00,2304510,3130",
                    "2024-06-03 08:15:42,2024-06-03 08:15:44,91,50393",
                    "2024-06-04 07:00:00,,2304511,102",
                ]
            ),
        )
    events = read_alarms([path], STATIONS | {"2304514": "T05"})
    assert events.schema == EVENTS
    assert events.select("turbine", "start", "end", "source", "code").rows() == [
        ("T05", datetime(2024, 6, 1, 0, 0, 7, tzinfo=UTC), None, "alarm_log", 25),
        (
            "T01",
            datetime(2024, 6, 2, 10, tzinfo=UTC),
            datetime(2024, 6, 2, 11, 30, tzinfo=UTC),
            "alarm_log",
            3130,
        ),
        ("T02", datetime(2024, 6, 4, 7, tzinfo=UTC), None, "alarm_log", 102),
    ]


def test_shutdown_runs_cover_their_start_labelled_intervals(tmp_path: Path):
    lines = ["TimeStamp_StartFormat,TurbineName,ShutdownDuration"]
    for minute, seconds in [(0, 0), (10, 120), (20, 600), (30, 300), (40, 0), (50, 600)]:
        lines.append(f"2016-01-01 00:{minute:02d}:00+00:00,T01,{seconds}")
    lines.append("2016-01-01 00:20:00+00:00,T02,600")
    path = tmp_path / "shutdowns.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ShutdownDuration.csv", "\n".join(lines) + "\n")
    events = read_shutdowns([path], ["T01", "T02"], TEN_MINUTES)
    assert events.schema == EVENTS
    start = datetime(2016, 1, 1, tzinfo=UTC)
    assert events.select("turbine", "start", "end").rows() == [
        ("T01", start + timedelta(minutes=10), start + timedelta(minutes=40)),
        ("T01", start + timedelta(minutes=50), start + timedelta(minutes=60)),
        ("T02", start + timedelta(minutes=20), start + timedelta(minutes=30)),
    ]
    assert set(events["source"]) == {"shutdown_log"}


def test_a_shutdown_log_turbine_missing_from_the_metadata_stops_the_reader(tmp_path: Path):
    path = tmp_path / "shutdowns.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "ShutdownDuration.csv",
            "TimeStamp_StartFormat,TurbineName,ShutdownDuration\n2016-01-01 00:00:00+00:00,T99,0\n",
        )
    with pytest.raises(ValueError, match=r"\['T99'\]"):
        read_shutdowns([path], ["T01"], TEN_MINUTES)
