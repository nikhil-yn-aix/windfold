"""hill of towie: the 10-minute tables, the alarm log and the operator shutdown log."""

import csv
import io
import logging
import zipfile
from collections.abc import Collection, Iterator, Mapping, Sequence
from datetime import timedelta
from functools import reduce
from pathlib import Path

import polars as pl

from windfold.schemas import EVENTS, RAW_ROWS, conform

log = logging.getLogger(__name__)

TIMESTAMP = "TimeStamp"
STATION = "StationId"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ALARM_TABLE = "tblAlarmLog"
SHUTDOWN_FORMAT = "%Y-%m-%d %H:%M:%S%z"


def utc(text: pl.Expr) -> pl.Expr:
    """utc instants from text written as a date and time, or as a date alone meaning midnight."""
    full = pl.when(text.str.contains(" ", literal=True)).then(text).otherwise(text + " 00:00:00")
    return full.str.to_datetime(TIMESTAMP_FORMAT, time_unit="us", time_zone="UTC")


def read_stations(path: Path) -> dict[str, str]:
    """station id of each turbine (2304510) to its name (T01), from the turbine metadata."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["Station ID"]: row["Turbine Name"] for row in csv.DictReader(stream)}


def members(archives: Sequence[Path], table: str) -> Iterator[tuple[str, bytes]]:
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if Path(member).name.startswith(f"{table}_"):
                    yield f"{path.name} {member}", archive.read(member)


def read_table(
    archives: Sequence[Path],
    table: str,
    columns: Mapping[str, str],
    stations: Mapping[str, str],
    interval: timedelta,
) -> pl.DataFrame:
    """the named columns of one 10-minute table by turbine and interval start.

    the provider labels each row with the end of its interval. monthly files repeat the row at
    each month boundary; equal copies are kept once.
    """
    parts = []
    for where, data in members(archives, table):
        try:
            published = pl.read_csv(
                io.BytesIO(data),
                columns=[TIMESTAMP, STATION, *columns.values()],
                infer_schema=False,
            )
        except pl.exceptions.ColumnNotFoundError as error:
            raise ValueError(f"{where}: a declared column is missing: {error}") from error
        strangers = set(published.get_column(STATION).unique()) - set(stations)
        if strangers:
            raise ValueError(
                f"{where}: stations {sorted(strangers)} are not in the turbine metadata"
            )
        parts.append(
            published.select(
                (utc(pl.col(TIMESTAMP)) - interval).alias("timestamp"),
                pl.col(STATION)
                .replace_strict(dict(stations), return_dtype=pl.String)
                .alias("turbine"),
                *[pl.col(column).cast(pl.Float64).alias(name) for name, column in columns.items()],
            )
        )
    rows = pl.concat(parts)
    merged = rows.unique(maintain_order=True)
    differing = merged.filter(pl.struct("turbine", "timestamp").is_duplicated())
    if not differing.is_empty():
        raise ValueError(
            f"{table}: {differing.height} rows give different values for one turbine and "
            f"timestamp, such as {differing.select('turbine', 'timestamp').row(0)}"
        )
    log.info(
        "%s: %d rows, %d repeated copies merged", table, rows.height, rows.height - merged.height
    )
    return merged


def mark_fallback_speeds(rows: pl.DataFrame) -> pl.DataFrame:
    """speed and its spread become missing where the active sensor reads closer to the secondary
    than to the primary, so the controller had left the primary."""
    active = pl.col("speed_active")
    fallback = (active - pl.col("speed_secondary")).abs() < (active - pl.col("speed")).abs()
    return rows.with_columns(
        pl.when(fallback).then(None).otherwise(pl.col(name)).alias(name)
        for name in ("speed", "speed_std")
    )


def read_rows(
    archives: Sequence[Path],
    stations: Mapping[str, str],
    channels: Mapping[str, str],
    interval: timedelta,
) -> pl.DataFrame:
    """raw rows: one row per turbine and interval start, with the declared channels.

    channels maps each raw-row column to a provider column written as table.column.
    """
    tables: dict[str, dict[str, str]] = {}
    for name, declared in channels.items():
        table, column = declared.split(".", 1)
        tables.setdefault(table, {})[name] = column
    frames = [read_table(archives, t, columns, stations, interval) for t, columns in tables.items()]
    rows = reduce(
        lambda left, right: left.join(
            right, on=["timestamp", "turbine"], how="full", coalesce=True
        ),
        frames,
    )
    return conform(
        mark_fallback_speeds(rows)
        .select("timestamp", "turbine", *channels)
        .sort("turbine", "timestamp"),
        RAW_ROWS,
        extra=True,
    )


def no_text() -> pl.Expr:
    return pl.lit(None, dtype=pl.String)


def read_alarms(archives: Sequence[Path], stations: Mapping[str, str]) -> pl.DataFrame:
    """alarm-log events as published; alarms of stations that are not turbines, and records listed
    more than once, are dropped."""
    published = pl.concat(
        pl.read_csv(io.BytesIO(data), infer_schema=False)
        for _, data in members(archives, ALARM_TABLE)
    )
    listed = published.filter(pl.col("StationNr").is_in(list(stations)))
    events = listed.select(
        pl.col("StationNr").replace_strict(dict(stations), return_dtype=pl.String).alias("turbine"),
        utc(pl.col("TimeOn")).alias("start"),
        utc(pl.col("TimeOff")).alias("end"),
        pl.lit("alarm_log").alias("source"),
        no_text().alias("status"),
        pl.col("Alarmcode").cast(pl.Int64).alias("code"),
        no_text().alias("message"),
        no_text().alias("category"),
    )
    kept = events.unique(maintain_order=True)
    log.info(
        "%d alarms; %d of stations that are not turbines and %d repeated records dropped",
        kept.height,
        published.height - listed.height,
        events.height - kept.height,
    )
    return conform(kept, EVENTS)


def read_shutdowns(
    paths: Sequence[Path], turbines: Collection[str], interval: timedelta
) -> pl.DataFrame:
    """shutdown-log events: one per run of consecutive intervals with shutdown time, covering them.

    the log gives the shutdown seconds of every turbine and interval, labeled with its start.
    """
    parts = []
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                parts.append(pl.read_csv(io.BytesIO(archive.read(member)), infer_schema=False))
    published = pl.concat(parts).select(
        pl.col("TurbineName").alias("turbine"),
        pl.col("TimeStamp_StartFormat")
        .str.to_datetime(SHUTDOWN_FORMAT, time_unit="us")
        .dt.convert_time_zone("UTC")
        .alias("start"),
        pl.col("ShutdownDuration").cast(pl.Float64).alias("seconds"),
    )
    strangers = set(published.get_column("turbine").unique()) - set(turbines)
    if strangers:
        raise ValueError(
            f"the shutdown log names turbines {sorted(strangers)} the metadata does not"
        )
    shut = published.filter(pl.col("seconds") > 0).sort("turbine", "start")
    run = (
        (pl.col("turbine") != pl.col("turbine").shift()) | (pl.col("start").diff() != interval)
    ).fill_null(True)
    events = (
        shut.with_columns(run.cum_sum().alias("run"))
        .group_by("run", maintain_order=True)
        .agg(
            pl.col("turbine").first(),
            pl.col("start").first(),
            (pl.col("start").last() + interval).alias("end"),
        )
        .select(
            "turbine",
            "start",
            "end",
            pl.lit("shutdown_log").alias("source"),
            no_text().alias("status"),
            pl.lit(None, dtype=pl.Int64).alias("code"),
            no_text().alias("message"),
            no_text().alias("category"),
        )
    )
    log.info("%d shutdown runs from %d intervals with shutdown time", events.height, shut.height)
    return conform(events, EVENTS)
