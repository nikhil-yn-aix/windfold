"""step 10: farm direction, reference speed, speed bins, direction buckets and steadiness, and
the anemometer factors that put every turbine's speed on one basis."""

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import polars as pl


def farm_directions(rows: pl.DataFrame, column: str) -> pl.DataFrame:
    """circular mean of column over each timestamp's rows, its resultant length and row count."""
    radians = pl.col(column).radians()
    sine, cosine = radians.sin().mean(), radians.cos().mean()
    return rows.group_by("timestamp").agg(
        (pl.arctan2(sine, cosine).degrees() % 360.0).alias("farm_direction"),
        (sine**2 + cosine**2).sqrt().alias("resultant"),
    )


def steady_flags(
    farm: pl.DataFrame, interval: timedelta, change_deg: float, min_resultant: float
) -> pl.DataFrame:
    """steady where the farm direction moved less than change_deg since the previous interval
    and the turbines agree with a resultant above min_resultant; unknown history is not steady.
    """
    previous = farm.select(
        (pl.col("timestamp") + interval).alias("timestamp"),
        pl.col("farm_direction").alias("previous"),
    )
    change = ((pl.col("farm_direction") - pl.col("previous") + 180.0) % 360.0 - 180.0).abs()
    return (
        farm.join(previous, on="timestamp", how="left", maintain_order="left")
        .with_columns(
            ((change < change_deg) & (pl.col("resultant") > min_resultant))
            .fill_null(False)
            .alias("steady")
        )
        .select("timestamp", "steady")
    )


def reference_speeds(rows: pl.DataFrame) -> pl.DataFrame:
    """median speed and turbulence intensity over each timestamp's free-stream rows.

    rows hold speed, speed_std and waked; waked rows never enter. turbulence intensity is null
    where no free-stream row carries a speed spread.
    """
    free = rows.filter(~pl.col("waked"))
    return free.group_by("timestamp").agg(
        pl.col("speed").median().alias("reference_speed"),
        (pl.col("speed_std") / pl.col("speed")).median().alias("turbulence_intensity"),
    )


def speed_factors(rows: pl.DataFrame, min_timestamps: int) -> pl.DataFrame:
    """each turbine's anemometer factor, from prepared rows holding timestamp, turbine, speed and
    waked.

    the factor is the median, over timestamps where the turbine and at least one other turbine
    are free-stream, of the other free-stream turbines' median speed over the turbine's own. a
    turbine with fewer than min_timestamps such timestamps keeps a factor of 1. returns turbine,
    factor, timestamps and fitted, one row per turbine in rows.
    """
    names = np.array(sorted(rows.get_column("turbine").unique().to_list()))
    free = rows.filter(~pl.col("waked") & (pl.col("speed") > 0.0))
    _, row_of = np.unique(
        free.get_column("timestamp").to_physical().to_numpy(), return_inverse=True
    )
    speeds = np.full((int(row_of.max()) + 1 if row_of.size else 0, names.size), np.nan)
    speeds[row_of, np.searchsorted(names, free.get_column("turbine").to_numpy())] = (
        free.get_column("speed").cast(pl.Float64).to_numpy()
    )
    found = []
    for i, name in enumerate(names):
        own = speeds[:, i]
        others = np.delete(speeds, i, axis=1)
        usable = ~np.isnan(own) & (~np.isnan(others)).any(axis=1)
        count = int(usable.sum())
        fitted = count > 0 and count >= min_timestamps
        factor = (
            float(np.median(np.nanmedian(others[usable], axis=1) / own[usable])) if fitted else 1.0
        )
        found.append(
            {"turbine": str(name), "factor": factor, "timestamps": count, "fitted": fitted}
        )
    schema = {
        "turbine": pl.String,
        "factor": pl.Float64,
        "timestamps": pl.Int64,
        "fitted": pl.Boolean,
    }
    return pl.DataFrame(found, schema=schema)


def speed_bins(speed: pl.Expr, edges: Sequence[float]) -> pl.Expr:
    """the index of the bin [edges[i], edges[i + 1]) holding speed, null outside every bin."""
    index = pl.lit(None, dtype=pl.UInt8)
    for i in reversed(range(len(edges) - 1)):
        inside = (speed >= edges[i]) & (speed < edges[i + 1])
        index = pl.when(inside).then(pl.lit(i, dtype=pl.UInt8)).otherwise(index)
    return index


def direction_buckets(direction: pl.Expr, width_deg: float) -> pl.Expr:
    """bucket i holds directions in [i * width_deg, (i + 1) * width_deg)."""
    count = round(360.0 / width_deg)
    return ((direction / width_deg).floor() % count).cast(pl.UInt8)
