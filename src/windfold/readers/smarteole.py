"""smarteole: the 1-minute scada of every turbine and the wake-steering control log of SMV6."""

import io
import re
import zipfile
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path

import polars as pl

from windfold.schemas import EVENTS, RAW_ROWS, conform

FOLDER = "SMARTEOLE-WFC-open-dataset/"
SCADA = FOLDER + "SMARTEOLE_WakeSteering_SCADA_1minData.csv"
CONTROL_LOG = FOLDER + "SMARTEOLE_WakeSteering_ControlLog_1minData.csv"
TIME = "time"
ACTIVE = "control_log_offset_active_avg"
CONTROLLED_TURBINE = "SMV6"
TURBINE_NUMBER = re.compile(r"SMV(\d+)")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def minute_starts(text: str) -> pl.Expr:
    """the naive minute labels read as utc minute starts, as wind-up reads them."""
    return pl.col(text).str.to_datetime(TIMESTAMP_FORMAT, time_unit="us", time_zone="UTC")


def read_rows(package: Path, turbines: Sequence[str], channels: Mapping[str, str]) -> pl.DataFrame:
    """raw rows: one row per turbine and published minute.

    channels maps each raw-row column to a scada column with {i} standing for the turbine number.
    """
    with zipfile.ZipFile(package) as archive:
        published = pl.read_csv(io.BytesIO(archive.read(SCADA)), infer_schema=False)
    frames = []
    for turbine in turbines:
        number = TURBINE_NUMBER.fullmatch(turbine)
        if number is None:
            raise ValueError(f"{turbine} is not a smarteole turbine name")
        columns = {name: column.format(i=number.group(1)) for name, column in channels.items()}
        missing = [column for column in columns.values() if column not in published.columns]
        if missing:
            raise ValueError(f"{SCADA} has no columns {missing}")
        frames.append(
            published.select(
                minute_starts(TIME).alias("timestamp"),
                pl.lit(turbine).alias("turbine"),
                *[pl.col(column).cast(pl.Float64).alias(name) for name, column in columns.items()],
            )
        )
    return conform(pl.concat(frames).sort("turbine", "timestamp"), RAW_ROWS, extra=True)


def read_control_log(package: Path, interval: timedelta) -> pl.DataFrame:
    """control_log events: one per run of minutes whose offset was active for part of the minute,
    or whose state is unknown (no value or no record), covering [first minute, last minute + 1)."""
    with zipfile.ZipFile(package) as archive:
        published = pl.read_csv(
            io.BytesIO(archive.read(CONTROL_LOG)), columns=[TIME, ACTIVE], infer_schema=False
        )
    logged = published.select(
        minute_starts(TIME).alias("start"), pl.col(ACTIVE).cast(pl.Float64).alias("active")
    )
    grid = logged.select(
        pl.datetime_range(pl.col("start").min(), pl.col("start").max(), interval).alias("start")
    )
    minutes = grid.join(logged, on="start", how="left")
    controlled = (pl.col("active") > 0) | pl.col("active").is_null()
    events = (
        minutes.with_columns(controlled.alias("controlled"))
        .with_columns(
            (pl.col("controlled") != pl.col("controlled").shift())
            .fill_null(True)
            .cum_sum()
            .alias("run")
        )
        .filter(pl.col("controlled"))
        .group_by("run", maintain_order=True)
        .agg(
            pl.col("start").first(),
            (pl.col("start").last() + interval).alias("end"),
            pl.when((pl.col("active") > 0).any())
            .then(pl.lit("active"))
            .otherwise(pl.lit("unknown"))
            .alias("status"),
        )
        .select(
            pl.lit(CONTROLLED_TURBINE).alias("turbine"),
            "start",
            "end",
            pl.lit("control_log").alias("source"),
            "status",
            pl.lit(None, dtype=pl.Int64).alias("code"),
            pl.lit(None, dtype=pl.String).alias("message"),
            pl.lit(None, dtype=pl.String).alias("category"),
        )
    )
    return conform(events, EVENTS)
