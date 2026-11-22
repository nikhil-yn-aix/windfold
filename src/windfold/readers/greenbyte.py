"""greenbyte exports (kelmarsh, penmanshiel): 10-minute turbine data and turbine names."""

import csv
import io
import logging
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, cast

import polars as pl

from windfold.schemas import RAW_ROWS, conform

log = logging.getLogger(__name__)

TIMESTAMP = "Date and time"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DATA_MEMBER = "Turbine_Data_"
TURBINE_LINE = "# Turbine:"
READ_BUFFER_BYTES = 1 << 20


def read_turbine_names(path: Path) -> dict[str, str]:
    """greenbyte title of each turbine (Kelmarsh 1) to its name (KWF1), from the static csv."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            row["Title"]: row["Alternative Title"] for row in csv.DictReader(stream) if row["Title"]
        }


def read_preamble(stream: IO[bytes], header: str) -> tuple[str, str]:
    """the turbine title and header line of a greenbyte csv; stream stops at the first record."""
    title = None
    while line := stream.readline():
        text = line.decode("utf-8").rstrip("\r\n")
        if text.startswith(TURBINE_LINE):
            title = text.removeprefix(TURBINE_LINE).strip()
        elif text.startswith(header) and title is not None:
            return title, text
    raise ValueError(f"no '{TURBINE_LINE}' line followed by a '{header}' header")


def first_copies(stream: IO[bytes]) -> tuple[bytes, int]:
    """the records whose timestamp has not appeared earlier in the file, and how many records
    the file held.

    paginated exports repeat every timestamp on each later page with only derived ratios filled,
    so the first copy is the measured one.
    """
    seen: set[bytes] = set()
    kept = []
    published = 0
    for record in stream:
        published += 1
        timestamp = record.split(b",", 1)[0]
        if timestamp not in seen:
            seen.add(timestamp)
            kept.append(record)
    return b"".join(kept), published


def read_turbine_data(
    stream: IO[bytes], channels: Mapping[str, str], null_values: Sequence[str]
) -> tuple[str, pl.DataFrame, int]:
    """the turbine title, the declared channels of the first copy of each timestamp (renamed, in
    file order) and the number of records in one turbine-data csv."""
    title, line = read_preamble(stream, f"# {TIMESTAMP}")
    header = line.removeprefix("# ")
    missing = [column for column in channels.values() if column not in next(csv.reader([header]))]
    if missing:
        raise ValueError(f"declared channels {missing} are not in the export")
    records, published = first_copies(stream)
    frame = pl.read_csv(
        io.BytesIO(header.encode() + b"\n" + records),
        columns=[TIMESTAMP, *channels.values()],
        infer_schema=False,
        null_values=list(null_values),
    ).select(
        pl.col(TIMESTAMP).alias("timestamp"),
        *[pl.col(column).cast(pl.Float64).alias(name) for name, column in channels.items()],
    )
    return title, frame, published


def mark_missing_speeds(frame: pl.DataFrame, codes: Sequence[float]) -> pl.DataFrame:
    """speed and its spread become missing where the speed is a provider code or negative, or the
    interval held a negative sample (its minimum)."""
    faulty = pl.col("speed").is_in(list(codes)) | (pl.col("speed") < 0) | (pl.col("speed_min") < 0)
    return frame.with_columns(
        pl.when(faulty).then(None).otherwise(pl.col(name)).alias(name)
        for name in ("speed", "speed_std")
    )


def read_rows(
    archives: Sequence[Path],
    names: Mapping[str, str],
    channels: Mapping[str, str],
    null_values: Sequence[str],
    speed_missing_codes: Sequence[float],
) -> pl.DataFrame:
    """raw rows of a greenbyte farm: one row per turbine and published 10-minute interval.

    channels maps each raw-row column to its greenbyte title; names maps turbine titles to names.
    """
    frames = []
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist() if Path(n).name.startswith(DATA_MEMBER))
            for member in members:
                raw = cast(zipfile.ZipExtFile, archive.open(member))
                with io.BufferedReader(raw, READ_BUFFER_BYTES) as stream:
                    try:
                        title, frame, published = read_turbine_data(stream, channels, null_values)
                    except (ValueError, pl.exceptions.PolarsError) as error:
                        raise ValueError(f"{path.name} {member}: {error}") from error
                if title not in names:
                    raise ValueError(f"{path.name} {member} is for {title}, not a listed turbine")
                log.info("%s %s: %d records, %d kept", path.name, member, published, frame.height)
                frames.append(frame.with_columns(turbine=pl.lit(names[title])))
    rows = pl.concat(frames)
    repeated = rows.filter(pl.struct("turbine", "timestamp").is_duplicated())
    if not repeated.is_empty():
        raise ValueError(
            f"{repeated.height} rows share a turbine and timestamp across files, such as "
            f"{repeated.select('turbine', 'timestamp').row(0)}; no rule says which file to keep"
        )
    rows = mark_missing_speeds(
        rows.with_columns(
            pl.col("timestamp").str.to_datetime(TIMESTAMP_FORMAT, time_unit="us", time_zone="UTC")
        ),
        speed_missing_codes,
    )
    return conform(
        rows.select("timestamp", "turbine", *channels).sort("turbine", "timestamp"),
        RAW_ROWS,
        extra=True,
    )
