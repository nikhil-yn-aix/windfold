import csv
import io
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from windfold.analyses.direction_spread import direction_spread
from windfold.config import load_farm
from windfold.readers import archives_of

NAMES = {"Kelmarsh 1": "KWF1", "Kelmarsh 2": "KWF2", "Kelmarsh 3": "KWF3", "Kelmarsh 4": "KWF4"}
WAKED = "KWF4"
TIMES = [datetime(2023, 1, 1, tzinfo=UTC) + timedelta(minutes=10 * k) for k in range(5)]
CALIBRATION = datetime(2017, 12, 31, 23, 50, tzinfo=UTC)
SPEEDS = (6.0, 7.5, 9.0, 12.0, 7.0)
TURBULENCE = (0.05, 0.10, 0.15, 0.20, 0.12)
DEVIATIONS = {
    "KWF1": (1.0, 2.0, 3.0, 0.0, None),
    "KWF2": (2.0, 3.0, 4.0, 1.0, None),
    "KWF3": (3.0, 10.0, 5.0, 2.0, None),
    "KWF4": (100.0, 0.0, 0.0, 50.0, 3.0),
}


def direction(name: str, time: datetime) -> float:
    return 200.0 + int(name[-1]) + time.minute / 8


def export(title: str, records: list[tuple[datetime, float | None]]) -> bytes:
    header = io.StringIO()
    csv.writer(header, lineterminator="").writerow(
        ["Date and time", "Wind direction (°)", "Wind direction, Standard deviation (°)"]
    )
    name = NAMES[title]
    lines = [f"# Turbine: {title}", f"# {header.getvalue()}"] + [
        f"{time:%Y-%m-%d %H:%M:%S},{direction(name, time)!r},{'NaN' if std is None else repr(std)}"
        for time, std in records
    ]
    return ("\r\n".join(lines) + "\r\n").encode()


def prepared() -> pl.DataFrame:
    development = [
        (time, name, "development", name == WAKED, direction(name, time), speed, turbulence)
        for time, speed, turbulence in zip(TIMES, SPEEDS, TURBULENCE, strict=True)
        for name in NAMES.values()
    ]
    calibration = [
        (CALIBRATION, name, "calibration", False, direction(name, CALIBRATION), 7.0, 0.5)
        for name in NAMES.values()
    ]
    columns = ["timestamp", "turbine", "period", "waked", "direction_raw"]
    return pl.DataFrame(
        development + calibration,
        schema=[*columns, "reference_speed", "turbulence_intensity"],
        orient="row",
    )


def write(data: Path, rows: pl.DataFrame, repeated: bool = False) -> Path:
    farm = load_farm("kelmarsh")
    raw = data / "raw" / "kelmarsh"
    raw.mkdir(parents=True)
    titles = "".join(f"{title},{name}\n" for title, name in NAMES.items())
    (raw / farm.source.turbines).write_text("Title,Alternative Title\n" + titles)
    members = {
        f"Turbine_Data_Kelmarsh_{name[-1]}.csv": export(
            title, [(CALIBRATION, 1.0), *zip(TIMES, DEVIATIONS[name], strict=True)]
        )
        for title, name in NAMES.items()
    }
    if repeated:
        members["Turbine_Data_Kelmarsh_1_again.csv"] = members["Turbine_Data_Kelmarsh_1.csv"]
    first, *others = archives_of(farm, raw)
    with zipfile.ZipFile(first, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    for path in others:
        zipfile.ZipFile(path, "w").close()
    (data / "prepared" / "kelmarsh").mkdir(parents=True)
    rows.write_parquet(data / "prepared" / "kelmarsh" / "rows.parquet")
    return data


def test_turbulence_is_ranked_against_the_scatter_of_the_development_turbines_not_waked(
    tmp_path: Path,
):
    result = direction_spread(write(tmp_path, prepared()), ("kelmarsh",))
    assert result == {
        "kelmarsh": {
            "rows_checked": 20,
            "speeds_6_to_9": {
                "median": {"spearman": pytest.approx(1.0), "timestamps": 3},
                "mean": {"spearman": pytest.approx(0.5), "timestamps": 3},
            },
            "every_speed": {
                "median": {"spearman": pytest.approx(-0.2), "timestamps": 4},
                "mean": {"spearman": pytest.approx(-0.4), "timestamps": 4},
            },
        }
    }


def test_rows_are_refused_unless_each_is_exported_once_with_its_direction(tmp_path: Path):
    rows = prepared()
    first = pl.int_range(pl.len()) == 0
    moved = rows.with_columns(
        direction_raw=pl.when(first).then(pl.col("direction_raw") + 0.5).otherwise("direction_raw")
    )
    with pytest.raises(ValueError, match="1 rows' re-read wind direction"):
        direction_spread(write(tmp_path / "moved", moved), ("kelmarsh",))
    with pytest.raises(pl.exceptions.ComputeError, match="1:1 validation"):
        direction_spread(write(tmp_path / "repeated", rows, repeated=True), ("kelmarsh",))
