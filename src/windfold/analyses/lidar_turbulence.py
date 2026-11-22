"""nacelle turbulence against the smarteole lidar at hub height.

the windcube's 80 m minutes with at least MIN_AVAILABILITY availability are pooled into ten
minutes holding at least MIN_MINUTES of them; the ten-minute turbulence intensity is the
standard deviation over those minutes, within and between them, over their mean speed. it is
compared on crosswind development timestamps with at least MIN_NACELLES unwaked operating
turbines with: the farm's free-stream turbulence intensity, and the median nacelle turbulence
intensity (speed_std over speed) of k of the unwaked turbines picked at random, for k from one to
MIN_NACELLES, timestamps in time order and turbines in name order.
"""

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.analyses.development import STUDY
from windfold.readers import smarteole

FARM = "smarteole"
ARCHIVE = "SMARTEOLE-WFC-open-dataset.zip"
WINDCUBE = smarteole.FOLDER + "SMARTEOLE_WakeSteering_WindCube_1minData.csv"
SPEED, SPEED_STD, AVAILABILITY = "ws_avg_80", "ws_std_80", "avail_80"
MIN_AVAILABILITY = 0.9
MIN_MINUTES = 8
CROSSWIND_DEG = (90.0, 270.0)
CROSSWIND_HALF_WIDTH_DEG = 30.0
MIN_NACELLES = 6
SEED = 20261117


def lidar_ten_minutes(archive: Path) -> pl.DataFrame:
    """timestamp, minutes, lidar_speed, lidar_std and lidar_ti of each ten minutes."""
    with zipfile.ZipFile(archive) as opened:
        cube = pl.read_csv(
            io.BytesIO(opened.read(WINDCUBE)),
            columns=[smarteole.TIME, SPEED, SPEED_STD, AVAILABILITY],
            null_values="NA",
            infer_schema=False,
        )
    minutes = cube.select(
        smarteole.minute_starts(smarteole.TIME).alias("timestamp"),
        *[pl.col(c).cast(pl.Float64) for c in (SPEED, SPEED_STD, AVAILABILITY)],
    ).filter(
        pl.col(SPEED).is_not_null(),
        pl.col(SPEED_STD).is_not_null(),
        pl.col(AVAILABILITY) >= MIN_AVAILABILITY,
    )
    return (
        minutes.with_columns(pl.col("timestamp").dt.truncate("10m"))
        .group_by("timestamp")
        .agg(
            pl.len().alias("minutes"),
            pl.col(SPEED).mean().alias("lidar_speed"),
            ((pl.col(SPEED_STD) ** 2).mean() + pl.col(SPEED).var(ddof=0)).sqrt().alias("lidar_std"),
        )
        .filter(pl.col("minutes") >= MIN_MINUTES)
        .with_columns(lidar_ti=pl.col("lidar_std") / pl.col("lidar_speed"))
    )


def crosswind() -> pl.Expr:
    return pl.any_horizontal(
        (pl.col("farm_direction") - centre).abs() <= CROSSWIND_HALF_WIDTH_DEG
        for centre in CROSSWIND_DEG
    )


def paired_timestamps(rows: pl.DataFrame, lidar: pl.DataFrame) -> pl.DataFrame:
    """per crosswind development timestamp seen by the lidar, in time order: each unwaked
    turbine's nacelle turbulence (turbine order), the lidar's, the farm's free-stream turbulence
    and the reference and lidar speeds."""
    free = rows.filter(
        pl.col("period") == STUDY, crosswind(), ~pl.col("waked"), pl.col("speed") > 0
    ).with_columns(own_ti=pl.col("speed_std") / pl.col("speed"))
    joined = free.join(lidar, on="timestamp", how="inner").sort("timestamp", "turbine")
    return (
        joined.group_by("timestamp", maintain_order=True)
        .agg(
            pl.col("own_ti"),
            pl.col("lidar_ti", "turbulence_intensity", "reference_speed", "lidar_speed").first(),
        )
        .filter(pl.col("own_ti").list.len() >= MIN_NACELLES)
    )


def agreement(nacelle: np.ndarray, lidar: np.ndarray) -> dict[str, float]:
    log_ratio = np.log(nacelle / lidar)
    ranks = pl.DataFrame({"a": nacelle, "b": lidar}).select(pl.corr("a", "b", method="spearman"))
    return {
        "pearson": float(np.corrcoef(nacelle, lidar)[0, 1]),
        "spearman": float(ranks.item()),
        "log_ratio_mean": float(log_ratio.mean()),
        "log_ratio_sd": float(log_ratio.std()),
    }


def lidar_turbulence(data: Path, seed: int = SEED) -> dict[str, Any]:
    rows = pl.read_parquet(
        data / "prepared" / FARM / "rows.parquet",
        columns=[
            "timestamp",
            "turbine",
            "period",
            "waked",
            "speed",
            "speed_std",
            "farm_direction",
            "reference_speed",
            "turbulence_intensity",
        ],
    )
    times = paired_timestamps(rows, lidar_ten_minutes(data / "raw" / FARM / ARCHIVE))
    lidar = times["lidar_ti"].to_numpy()
    generator = np.random.default_rng(seed)
    by_count = {}
    for k in range(1, MIN_NACELLES + 1):
        picked = [
            float(np.median(generator.choice(np.asarray(values), size=k, replace=False)))
            for values in times["own_ti"].to_list()
        ]
        by_count[k] = agreement(np.asarray(picked), lidar)
    speeds = times.select("reference_speed", "lidar_speed").to_numpy()
    return {
        "timestamps": times.height,
        "speed_correlation": float(np.corrcoef(speeds[:, 0], speeds[:, 1])[0, 1]),
        "farm_free_stream": agreement(times["turbulence_intensity"].to_numpy(), lidar),
        "random_nacelles": by_count,
    }
