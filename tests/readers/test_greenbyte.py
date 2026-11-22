import csv
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from windfold.config import load_farm
from windfold.readers.greenbyte import read_rows, read_turbine_names
from windfold.schemas import RAW_ROWS, conform
from windfold.schemas import UTC as UTC_TYPE

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = dict(load_farm("kelmarsh").channels)
KELMARSH_2023 = ROOT / "data/raw/kelmarsh/Kelmarsh_SCADA_2023_5961.zip"
KWF1_2023 = "Turbine_Data_Kelmarsh_1_2023-01-01_-_2024-01-01_228.csv"

COLUMNS = (
    "Wind speed (m/s)",
    "Wind speed, Standard deviation (m/s)",
    "Wind speed Sensor 1 (m/s)",
    "Wind speed Sensor 2 (m/s)",
    "Wind speed Sensor 2, Standard deviation (m/s)",
    "Wind speed Sensor 2, Minimum (m/s)",
    "Wind direction (°)",
    "Nacelle position (°)",
    "Power (kW)",
    "Nacelle ambient temperature (°C)",
    "Turbine Power setpoint (kW)",
    "Turbine Power setpoint, Min (kW)",
    "Time-based System Avail.",
    "Lost Production to Downtime (kWh)",
    "Production-based System Avail.",
)
SHORT = {
    "speed": "Wind speed Sensor 2 (m/s)",
    "speed_std": "Wind speed Sensor 2, Standard deviation (m/s)",
    "speed_min": "Wind speed Sensor 2, Minimum (m/s)",
    "power": "Power (kW)",
    "availability": "Time-based System Avail.",
}
NAMES = {"Kelmarsh 1": "KWF1", "Kelmarsh 2": "KWF2"}
CODES = (-15.0,)


def measured(timestamp: str, **values: float | None) -> dict[str, str]:
    row: dict[str, float | None] = {
        "Wind speed (m/s)": 7.0,
        "Wind speed, Standard deviation (m/s)": 0.75,
        "Wind speed Sensor 1 (m/s)": 7.5,
        "Wind speed Sensor 2 (m/s)": 6.9,
        "Wind speed Sensor 2, Standard deviation (m/s)": 0.8,
        "Wind speed Sensor 2, Minimum (m/s)": 4.2,
        "Wind direction (°)": 212.5,
        "Nacelle position (°)": 210.0,
        "Power (kW)": 905.25,
        "Nacelle ambient temperature (°C)": 11.5,
        "Turbine Power setpoint (kW)": 2060.0,
        "Turbine Power setpoint, Min (kW)": 1990.0,
        "Time-based System Avail.": 1.0,
        "Lost Production to Downtime (kWh)": 0.0,
        "Production-based System Avail.": 0.98,
    }
    row.update({SHORT[key]: value for key, value in values.items()})
    return {"Date and time": timestamp} | {
        column: "NaN" if value is None else repr(value) for column, value in row.items()
    }


def copy_of(timestamp: str) -> dict[str, str]:
    return {"Date and time": timestamp, "Production-based System Avail.": "0.5"}


def export(title: str, rows: list[dict[str, str]]) -> bytes:
    header = io.StringIO()
    csv.writer(header, lineterminator="").writerow(["Date and time", *COLUMNS])
    lines = [
        "# This file was exported by Greenbyte at 2025-08-11 12:35:00.",
        "#",
        f"# Turbine: {title}",
        "# Turbine type: Senvion MM92",
        "# Time zone: UTC",
        "# Time interval: 2023-01-01 00:00:00 - 2024-01-01 00:00:00 (365 days)",
        "#",
        '# Data that is missing or is erroneous has been marked with the value "NaN"',
        "#",
        f"# {header.getvalue()}",
    ]
    for row in rows:
        lines.append(",".join([row["Date and time"], *(row.get(c, "NaN") for c in COLUMNS)]))
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in members.items():
            target.writestr(name, data)
    return path


def read(tmp_path: Path, rows: list[dict[str, str]], channels=CHANNELS) -> pl.DataFrame:
    path = archive(
        tmp_path / "a.zip", {"Turbine_Data_Kelmarsh_1_x.csv": export("Kelmarsh 1", rows)}
    )
    return read_rows([path], NAMES, channels, ("NaN",), CODES)


def test_the_declared_anemometer_is_the_one_read(tmp_path: Path):
    rows = read(tmp_path, [measured("2023-01-01 00:00:00")])
    assert rows.select("speed", "speed_std").row(0) == (6.9, 0.8)
    sensor_1 = CHANNELS | {
        "speed": "Wind speed Sensor 1 (m/s)",
        "speed_std": "Wind speed, Standard deviation (m/s)",
    }
    assert read(tmp_path, [measured("2023-01-01 00:00:00")], sensor_1)["speed"].item() == 7.5


def test_keep_first_keeps_the_measured_copy_of_a_paginated_export(tmp_path: Path):
    page_1 = [measured("2023-01-01 00:00:00", power=905.25), measured("2023-01-01 00:10:00")]
    page_2 = [
        copy_of("2023-01-01 00:00:00"),
        copy_of("2023-01-01 00:10:00"),
        measured("2023-01-01 00:20:00", power=-3.5, availability=0.0),
    ]
    rows = read(tmp_path, page_1 + page_2)
    assert rows["timestamp"].to_list() == [
        datetime(2023, 1, 1, 0, minute, tzinfo=UTC) for minute in (0, 10, 20)
    ]
    assert rows["power_kw"].to_list() == [905.25, 905.25, -3.5]
    assert rows["time_based_availability"].to_list() == [1.0, 1.0, 0.0]


def test_a_timestamp_published_in_two_files_stops_the_reader(tmp_path: Path):
    paths = [
        archive(
            tmp_path / f"{year}.zip",
            {
                "Turbine_Data_Kelmarsh_1_x.csv": export(
                    "Kelmarsh 1", [measured("2023-01-01 00:00:00")]
                )
            },
        )
        for year in (2022, 2023)
    ]
    with pytest.raises(ValueError, match="2 rows share a turbine and timestamp across files"):
        read_rows(paths, NAMES, CHANNELS, ("NaN",), CODES)


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        ((-15.0, None, -15.0), (None, None)),
        ((-11.754394, 7.727219, -15.0), (None, None)),
        ((9.052089, 5.621066, -14.93025), (None, None)),
        ((0.0, 0.0, 0.0), (0.0, 0.0)),
        ((8.2, 0.9, 6.1), (8.2, 0.9)),
    ],
    ids=["code", "negative mean", "code inside the interval", "calm", "ordinary"],
)
def test_speeds_with_codes_or_negative_samples_become_missing_with_their_spread(
    tmp_path: Path, published, expected
):
    speed, spread, minimum = published
    rows = read(
        tmp_path,
        [measured("2023-07-07 12:40:00", speed=speed, speed_std=spread, speed_min=minimum)],
    )
    assert rows.select("speed", "speed_std").row(0) == expected
    assert rows["speed_min"].item() == minimum
    assert rows["power_kw"].item() == 905.25


def test_timestamps_are_the_published_interval_starts_in_utc(tmp_path: Path):
    rows = read(tmp_path, [measured("2016-01-03 00:00:00"), measured("2016-01-03 00:10:00")])
    assert rows.schema["timestamp"] == UTC_TYPE
    assert rows["timestamp"].to_list() == [
        datetime(2016, 1, 3, 0, 0, tzinfo=UTC),
        datetime(2016, 1, 3, 0, 10, tzinfo=UTC),
    ]


def test_rows_conform_with_provider_columns_after_the_core(tmp_path: Path):
    rows = read(tmp_path, [measured("2023-01-01 00:00:00")])
    provider = [name for name in CHANNELS if name not in RAW_ROWS]
    assert rows.columns == [*RAW_ROWS.names(), *provider]
    conform(rows, RAW_ROWS, extra=True)
    assert rows.drop("timestamp").row(0, named=True) == {
        "turbine": "KWF1",
        "power_kw": 905.25,
        "speed": 6.9,
        "speed_std": 0.8,
        "wind_direction": 212.5,
        "nacelle_position": 210.0,
        "ambient_temperature": 11.5,
        "speed_min": 4.2,
        "power_setpoint_kw": 2060.0,
        "power_setpoint_min_kw": 1990.0,
        "time_based_availability": 1.0,
        "downtime_loss_kwh": 0.0,
    }


def test_turbines_take_their_names_from_the_static_file(tmp_path: Path):
    static = tmp_path / "static.csv"
    static.write_bytes(
        "\ufeffWind Farm,Title,Alternative Title,Identity,\r\n"
        "Kelmarsh,Kelmarsh 2,KWF2,SEN 93421,\r\n"
        ",,,,\r\n".encode()
    )
    assert read_turbine_names(static) == {"Kelmarsh 2": "KWF2"}
    path = archive(
        tmp_path / "a.zip",
        {"Turbine_Data_Kelmarsh_2_x.csv": export("Kelmarsh 2", [measured("2023-01-01 00:00:00")])},
    )
    rows = read_rows([path], read_turbine_names(static), CHANNELS, ("NaN",), CODES)
    assert rows["turbine"].to_list() == ["KWF2"]


def test_a_declared_channel_missing_from_an_export_is_named(tmp_path: Path):
    channels = CHANNELS | {"speed": "Wind speed Sensor 3 (m/s)"}
    with pytest.raises(
        ValueError, match=r"Turbine_Data_Kelmarsh_1_x\.csv.*Wind speed Sensor 3 \(m/s\)"
    ):
        read(tmp_path, [measured("2023-01-01 00:00:00")], channels)


@pytest.mark.skipif(not KELMARSH_2023.exists(), reason="kelmarsh raw files are not on this machine")
def test_keep_first_on_a_real_2023_page_boundary_keeps_the_measured_copy(tmp_path: Path):
    with zipfile.ZipFile(KELMARSH_2023) as source, source.open(KWF1_2023) as stream:
        lines = [stream.readline() for _ in range(10 + 1965)]
    header = next(csv.reader([lines[9].decode()[2:].strip()]))
    published = [
        dict(zip(header, record, strict=True))
        for record in csv.reader(line.decode().strip() for line in lines[10:])
    ]
    first, copy = published[0], published[655]
    assert copy["Date and time"] == first["Date and time"] == "2023-01-01 00:00:00"
    assert all(first[column] != "NaN" for column in CHANNELS.values())
    assert all(copy[column] == "NaN" for column in CHANNELS.values())

    path = archive(tmp_path / "page.zip", {KWF1_2023: b"".join(lines)})
    rows = read_rows([path], {"Kelmarsh 1": "KWF1"}, CHANNELS, ("NaN",), CODES)
    assert rows.height == len({record["Date and time"] for record in published}) == 1310
    kept = rows.row(0, named=True)
    assert kept["timestamp"] == datetime(2023, 1, 1, tzinfo=UTC)
    for name, column in CHANNELS.items():
        assert kept[name] == float(first[column]), name
