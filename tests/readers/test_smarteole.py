import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from windfold.readers.smarteole import read_control_log, read_rows
from windfold.schemas import EVENTS, RAW_ROWS, conform

FOLDER = "SMARTEOLE-WFC-open-dataset/"
CHANNELS = {
    "power_kw": "active_power_{i}_avg",
    "speed": "wind_speed_{i}_avg",
    "speed_std": "wind_speed_{i}_std",
    "wind_direction": "wind_direction_{i}_avg",
    "nacelle_position": "nacelle_position_{i}_avg",
    "ambient_temperature": "temperature_{i}_avg",
    "derated": "derate_{i}",
}
SIGNALS = ["active_power", "wind_speed", "wind_direction", "nacelle_position", "temperature"]


def scada(minutes: list[tuple[str, dict[int, list[str]]]]) -> str:
    header = ["time"]
    for i in (1, 2):
        for signal in SIGNALS:
            header += [f"{signal}_{i}_{stat}" for stat in ("avg", "min", "max", "std", "count")]
    header += ["derate_1", "derate_2"]
    lines = [",".join(header)]
    for stamp, turbines in minutes:
        values = []
        for i in (1, 2):
            power, speed, speed_std, direction, nacelle, temperature, _ = turbines[i]
            for avg, std in [
                (power, "5"),
                (speed, speed_std),
                (direction, "1"),
                (nacelle, "0"),
                (temperature, "0"),
            ]:
                values += [avg, "", "", std, "60"]
        values += [turbines[1][6], turbines[2][6]]
        lines.append(",".join([stamp, *values]))
    return "\n".join(lines) + "\n"


def package(path: Path, scada_csv: str = "", control_csv: str = "") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(FOLDER + "SMARTEOLE_WakeSteering_SCADA_1minData.csv", scada_csv)
        archive.writestr(FOLDER + "SMARTEOLE_WakeSteering_ControlLog_1minData.csv", control_csv)
    return path


def test_wide_minutes_become_one_row_per_turbine_and_utc_minute_start(tmp_path: Path):
    path = package(
        tmp_path / "p.zip",
        scada(
            [
                (
                    "2020-02-17 16:30:00",
                    {
                        1: ["2023.7", "13.1", "0.88", "239.9", "240.0", "8.5", "0"],
                        2: ["2045.4", "", "", "244.1", "244.0", "8.4", "1"],
                    },
                )
            ]
        ),
    )
    rows = read_rows(path, ["SMV1", "SMV2"], CHANNELS)
    conform(rows, RAW_ROWS, extra=True)
    assert rows.columns == [*RAW_ROWS.names(), "derated"]
    assert rows.rows() == [
        (
            datetime(2020, 2, 17, 16, 30, tzinfo=UTC),
            "SMV1",
            2023.7,
            13.1,
            0.88,
            239.9,
            240.0,
            8.5,
            0.0,
        ),
        (
            datetime(2020, 2, 17, 16, 30, tzinfo=UTC),
            "SMV2",
            2045.4,
            None,
            None,
            244.1,
            244.0,
            8.4,
            1.0,
        ),
    ]


def test_a_channel_missing_from_the_scada_is_named(tmp_path: Path):
    path = package(tmp_path / "p.zip", scada([]))
    with pytest.raises(ValueError, match=r"blade_pitch_3_avg"):
        read_rows(path, ["SMV3"], {"power_kw": "blade_pitch_{i}_avg"})


def test_control_runs_cover_active_unknown_and_unlogged_minutes(tmp_path: Path):
    activity = {0: "0", 1: "0", 2: "1", 3: "1", 4: "0.5", 5: "0", 6: "", 7: "0", 9: "0"}
    lines = ["time,control_log_offset_avg,control_log_offset_active_avg"]
    lines += [f"2020-02-17 17:{m:02d}:00,0,{a}" for m, a in activity.items()]
    path = package(tmp_path / "p.zip", control_csv="\n".join(lines) + "\n")
    events = read_control_log(path, timedelta(minutes=1))
    assert events.schema == EVENTS

    def at(minute: int) -> datetime:
        return datetime(2020, 2, 17, 17, minute, tzinfo=UTC)

    assert events.select("turbine", "start", "end", "source", "status").rows() == [
        ("SMV6", at(2), at(5), "control_log", "active"),
        ("SMV6", at(6), at(7), "control_log", "unknown"),
        ("SMV6", at(8), at(9), "control_log", "unknown"),
    ]
