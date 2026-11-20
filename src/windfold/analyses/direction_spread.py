"""free-stream turbulence against the 10-minute scatter of wind direction.

the prepared rows keep each turbine's wind direction but not its 10-minute standard deviation, so
both are read again from the greenbyte exports of kelmarsh and penmanshiel, within the development
window. every development row's re-read direction must be its direction_raw, which shows they are
the same rows. per development timestamp, the scatter is the median, or the mean, of the standard
deviation over the turbines that are not waked, and the result is its spearman correlation with
the timestamp's free-stream turbulence intensity, at reference speeds of 6 to 9 m/s and at every
speed.
"""

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import polars as pl

from windfold.analyses.development import STUDY
from windfold.config import load_farm
from windfold.readers import archives_of
from windfold.readers.greenbyte import (
    DATA_MEMBER,
    READ_BUFFER_BYTES,
    TIMESTAMP_FORMAT,
    read_turbine_data,
    read_turbine_names,
)

FARMS = ("kelmarsh", "penmanshiel")
DIRECTION_STD = "Wind direction, Standard deviation (°)"
SPEEDS = (6.0, 9.0)
STATISTICS = ("median", "mean")
COLUMNS = (
    "timestamp",
    "turbine",
    "period",
    "waked",
    "direction_raw",
    "reference_speed",
    "turbulence_intensity",
)


def raw_directions(data: Path, farm: str, configs: Path | None = None) -> pl.DataFrame:
    """timestamp, turbine, wind_direction and direction_std of each exported 10-minute row within
    the farm's development window."""
    config = load_farm(farm, configs)
    raw = data / "raw" / farm
    names = read_turbine_names(raw / config.source.turbines)
    channels = {"wind_direction": config.channels["wind_direction"], "direction_std": DIRECTION_STD}
    window = config.windows.development
    stamp = pl.col("timestamp").str.to_datetime(TIMESTAMP_FORMAT, time_unit="us", time_zone="UTC")
    frames = []
    for path in archives_of(config, raw):
        with zipfile.ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist() if Path(n).name.startswith(DATA_MEMBER))
            for member in members:
                opened = cast(zipfile.ZipExtFile, archive.open(member))
                with io.BufferedReader(opened, READ_BUFFER_BYTES) as stream:
                    title, frame, _ = read_turbine_data(stream, channels, config.source.null_values)
                frames.append(
                    frame.with_columns(stamp, turbine=pl.lit(names[title])).filter(
                        pl.col("timestamp").is_between(window.start, window.end, closed="left")
                    )
                )
    return pl.concat(frames)


def spread_by_timestamp(rows: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """per timestamp of rows, in their order: the free-stream turbulence intensity, the reference
    speed and the median and mean direction standard deviation of the turbines that are not
    waked. raises unless each row is exported once, with its direction_raw as its direction."""
    joined = rows.join(
        raw, on=["timestamp", "turbine"], how="left", validate="1:1", maintain_order="left"
    )
    unequal = joined.filter(pl.col("wind_direction").ne_missing(pl.col("direction_raw")))
    if not unequal.is_empty():
        raise ValueError(
            f"{unequal.height} rows' re-read wind direction is not their direction_raw, such as "
            f"{unequal.select('turbine', 'timestamp').row(0)}"
        )
    free = pl.col("direction_std").filter(~pl.col("waked"))
    return joined.group_by("timestamp", maintain_order=True).agg(
        turbulence=pl.col("turbulence_intensity").first(),
        speed=pl.col("reference_speed").first(),
        median=free.median(),
        mean=free.mean(),
    )


def correlation(times: pl.DataFrame, statistic: str) -> dict[str, Any]:
    both = times.filter(pl.col("turbulence").is_not_null(), pl.col(statistic).is_not_null())
    rho = both.select(pl.corr("turbulence", statistic, method="spearman")).item()
    return {"spearman": float(rho), "timestamps": both.height}


def direction_spread(
    data: Path, farms: Sequence[str] = FARMS, configs: Path | None = None
) -> dict[str, Any]:
    """per farm, the development rows checked against the exports and the correlation of each
    statistic at 6 to 9 m/s and at every speed."""
    out: dict[str, Any] = {}
    for farm in farms:
        path = data / "prepared" / farm / "rows.parquet"
        rows = pl.read_parquet(path, columns=list(COLUMNS)).filter(pl.col("period") == STUDY)
        times = spread_by_timestamp(rows, raw_directions(data, farm, configs))
        band = times.filter(pl.col("speed").is_between(*SPEEDS))
        out[farm] = {
            "rows_checked": rows.height,
            "speeds_6_to_9": {statistic: correlation(band, statistic) for statistic in STATISTICS},
            "every_speed": {statistic: correlation(times, statistic) for statistic in STATISTICS},
        }
    return out
