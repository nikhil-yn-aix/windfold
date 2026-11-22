"""greenbyte status logs (kelmarsh, penmanshiel) as events, uninterpreted."""

import io
import logging
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

import polars as pl

from windfold.readers.greenbyte import TIMESTAMP_FORMAT, read_preamble
from windfold.schemas import EVENTS, conform

log = logging.getLogger(__name__)

STATUS_MEMBER = "Status_"
HEADER = "Timestamp start"
OPEN_END = "-"


def utc(text: pl.Expr) -> pl.Expr:
    return text.str.to_datetime(TIMESTAMP_FORMAT, time_unit="us", time_zone="UTC")


def read_status_log(stream: IO[bytes]) -> tuple[str, pl.DataFrame]:
    """the turbine title and events of one status csv; an end of '-' (no end logged) is null."""
    title, header = read_preamble(stream, HEADER)
    published = pl.read_csv(io.BytesIO(header.encode() + b"\n" + stream.read()), infer_schema=False)
    end = pl.col("Timestamp end")
    events = published.select(
        utc(pl.col("Timestamp start")).alias("start"),
        utc(pl.when(end != OPEN_END).then(end)).alias("end"),
        pl.lit("status_log").alias("source"),
        pl.col("Status").alias("status"),
        pl.col("Code").cast(pl.Int64).alias("code"),
        pl.col("Message").alias("message"),
        pl.col("IEC category").alias("category"),
    )
    return title, events


def read_status_logs(archives: Sequence[Path], names: Mapping[str, str]) -> pl.DataFrame:
    """events of every status csv in archives; a record listed more than once is kept once."""
    frames = []
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if not Path(member).name.startswith(STATUS_MEMBER):
                    continue
                with archive.open(member) as stream:
                    title, events = read_status_log(stream)
                if title not in names:
                    raise ValueError(f"{path.name} {member} is for {title}, not a listed turbine")
                frames.append(events.with_columns(turbine=pl.lit(names[title])))
    listed = pl.concat(frames)
    events = listed.unique(maintain_order=True)
    log.info(
        "%d status events, %d repeated records dropped",
        events.height,
        listed.height - events.height,
    )
    return conform(events.sort("turbine", "start", maintain_order=True), EVENTS)
