"""step 6: exclusion periods, farm-control washout and the averaging of minute data."""

from collections.abc import Sequence
from datetime import timedelta

import polars as pl

from windfold.config import Window


def exclusion_reasons(
    rows: pl.DataFrame, windows: Sequence[Window], interval: timedelta
) -> pl.Series:
    """the reason of the first window each row's interval overlaps, null where none does."""
    reason = pl.lit(None, dtype=pl.String)
    for window in reversed(windows):
        overlaps = (pl.col("timestamp") < window.end) & (
            pl.col("timestamp") + interval > window.start
        )
        reason = (
            pl.when(overlaps).then(pl.lit(window.reason or "excluded period")).otherwise(reason)
        )
    return rows.select(reason.alias("reason")).get_column("reason")


def with_washout(events: pl.DataFrame, washout: timedelta) -> pl.DataFrame:
    """events extended by the washout after their end; an event without an end ends at its start."""
    end = pl.max_horizontal(pl.col("start"), pl.coalesce("end", "start"))
    return events.with_columns((end + washout).alias("end"))


def circular_mean(column: str) -> pl.Expr:
    radians = pl.col(column).radians()
    return (pl.arctan2(radians.sin().mean(), radians.cos().mean()).degrees() % 360.0).alias(column)


def interval_means(minutes: pl.DataFrame, interval: timedelta, min_minutes: int) -> pl.DataFrame:
    """average each turbine's minutes into intervals, kept where at least min_minutes are complete.

    a complete minute has power, speed and wind direction. angles are averaged on the circle and
    the speed spread pools the variance within and between minutes.
    """
    complete = minutes.filter(
        pl.col("power_kw").is_not_null()
        & pl.col("speed").is_not_null()
        & pl.col("wind_direction").is_not_null()
    )
    return (
        complete.with_columns(pl.col("timestamp").dt.truncate(interval))
        .group_by("turbine", "timestamp")
        .agg(
            pl.len().alias("minutes"),
            pl.col("power_kw").mean(),
            pl.col("speed").mean(),
            ((pl.col("speed_std") ** 2).mean() + pl.col("speed").var(ddof=0))
            .sqrt()
            .alias("speed_std"),
            circular_mean("wind_direction"),
            circular_mean("nacelle_position"),
            pl.col("ambient_temperature").mean(),
        )
        .filter(pl.col("minutes") >= min_minutes)
        .drop("minutes")
        .sort("turbine", "timestamp")
    )
