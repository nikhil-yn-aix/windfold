import io
import zipfile
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.analyses import lidar_turbulence as lt

START = datetime(2020, 3, 1, 10, tzinfo=UTC)
SHARES = (1 / 3, 2 / 3, 14 / 15, 16 / 15, 4 / 3, 5 / 3)


def minutes(window: int, speeds: list, std: float, available: list[float]) -> list[tuple]:
    first = START + timedelta(minutes=10 * window)
    return [
        ((first + timedelta(minutes=k)).strftime("%Y-%m-%d %H:%M:%S"), speed, std, share)
        for k, (speed, share) in enumerate(zip(speeds, available, strict=True))
    ]


def write_lidar(path) -> None:
    """ten-minute windows 0, 2 and 4 hold eight usable minutes; window 0's speeds alternate 6
    and 8 m/s at a standard deviation of 2, and it drops one minute at 0.89 availability and one
    without a speed. window 1 keeps only seven minutes."""
    rows = minutes(0, [6, 8] * 4 + ["NA", 9], 2.0, [0.9] * 8 + [1.0, 0.89])
    rows += minutes(1, [8] * 10, 1.0, [1.0] * 7 + [0.5] * 3)
    rows += minutes(2, [9] * 8, 1.8, [1.0] * 8)
    rows += minutes(3, [9] * 8, 1.8, [1.0] * 8)
    rows += minutes(4, [8] * 8, 1.2, [1.0] * 8)
    rows += minutes(5, [8] * 8, 1.2, [1.0] * 8)
    rows += minutes(6, [8] * 8, 1.2, [1.0] * 8)
    cube = pl.DataFrame(
        [tuple(str(v) for v in row) for row in rows],
        schema=[lt.smarteole.TIME, lt.SPEED, lt.SPEED_STD, lt.AVAILABILITY],
        orient="row",
    )
    text = io.BytesIO()
    cube.write_csv(text)
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(lt.WINDCUBE, text.getvalue())


def turbines(window: int, ti: float, direction: float, period: str, count: int) -> list[dict]:
    """count unwaked turbines whose nacelle turbulence intensities are ti times SHARES, plus a
    waked and a stopped turbine with wild readings."""
    stamp = START + timedelta(minutes=10 * window)
    shared = {"timestamp": stamp, "period": period, "farm_direction": direction}
    rows = [
        shared | {"turbine": f"T{k}", "waked": False, "speed": 10.0, "speed_std": 10 * ti * share}
        for k, share in enumerate(SHARES[:count])
    ]
    rows.append(shared | {"turbine": "T8", "waked": True, "speed": 10.0, "speed_std": 9.0})
    rows.append(shared | {"turbine": "T9", "waked": False, "speed": 0.0, "speed_std": 9.0})
    return rows


def write_rows(path, lidar: dict[int, float]) -> None:
    """crosswind development timestamps at windows 0, 2 and 4 with six unwaked turbines whose
    median turbulence is 1.5 times the lidar's and a free-stream turbulence twice it; then a
    window without enough lidar minutes, an along-row wind, only five unwaked turbines and a
    calibration timestamp."""
    reference = {0: 7.0, 2: 8.0, 4: 9.0}
    rows = []
    for window, ti in lidar.items():
        part = turbines(window, 1.5 * ti, 100.0 if window else 275.0, "development", 6)
        extra = {"reference_speed": reference[window], "turbulence_intensity": 2 * ti}
        rows += [row | extra for row in part]
    for window, direction, period, count in (
        (1, 90.0, "development", 6),
        (3, 180.0, "development", 6),
        (5, 90.0, "development", 5),
        (6, 90.0, "calibration", 6),
    ):
        part = turbines(window, 0.5, direction, period, count)
        rows += [row | {"reference_speed": 8.0, "turbulence_intensity": 0.9} for row in part]
    path.parent.mkdir(parents=True)
    pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=1).write_parquet(path)


def test_lidar_turbulence_pairs_eligible_crosswind_timestamps_with_full_lidar_windows(tmp_path):
    lidar = {0: 5**0.5 / 7, 2: 0.2, 4: 0.15}
    write_lidar(tmp_path / "raw" / lt.FARM / lt.ARCHIVE)
    write_rows(tmp_path / "prepared" / lt.FARM / "rows.parquet", lidar)
    got = lt.lidar_turbulence(tmp_path)
    assert got["timestamps"] == 3
    assert got["speed_correlation"] == pytest.approx(0.5)
    assert got["farm_free_stream"] == pytest.approx(
        {"pearson": 1.0, "spearman": 1.0, "log_ratio_mean": np.log(2.0), "log_ratio_sd": 0.0}
    )
    assert got["random_nacelles"][6] == pytest.approx(
        {"pearson": 1.0, "spearman": 1.0, "log_ratio_mean": np.log(1.5), "log_ratio_sd": 0.0}
    )
    assert lt.lidar_turbulence(tmp_path) == got
    assert lt.lidar_turbulence(tmp_path, seed=1)["random_nacelles"][1] != got["random_nacelles"][1]
