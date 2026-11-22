"""step 3: whether each turbine-interval is in normal operation."""

from collections.abc import Mapping
from datetime import timedelta

import polars as pl


def classify_events(
    events: pl.DataFrame,
    statuses: Mapping[str, str],
    messages: Mapping[tuple[str, str], str],
) -> pl.DataFrame:
    """add a kind to every event: the kind of its (status, message) pair, else of its status.

    events matched by neither get a null kind. codes are never used, because providers reuse
    them for different events across years.
    """
    pairs = pl.DataFrame(
        {
            "status": [status for status, _ in messages],
            "message": [message for _, message in messages],
            "pair_kind": list(messages.values()),
        },
        schema={"status": pl.String, "message": pl.String, "pair_kind": pl.String},
    )
    by_status = pl.DataFrame(
        {"status": list(statuses), "status_kind": list(statuses.values())},
        schema={"status": pl.String, "status_kind": pl.String},
    )
    return (
        events.join(pairs, on=["status", "message"], how="left", maintain_order="left")
        .join(by_status, on="status", how="left", maintain_order="left")
        .with_columns(pl.coalesce("pair_kind", "status_kind").alias("kind"))
        .drop("pair_kind", "status_kind")
    )


def interval_overlaps(
    rows: pl.DataFrame, events: pl.DataFrame, interval: timedelta, *, farm_wide: bool = False
) -> pl.Series:
    """true where a row's interval [timestamp, timestamp + interval) overlaps an event.

    an event covers [start, end); one without an end, or ending at its start, covers its start
    instant. events belong to their turbine unless farm_wide, when they mark every turbine.
    """
    if events.is_empty():
        return pl.Series("overlaps", [False] * rows.height, dtype=pl.Boolean)
    keys = ["timestamp"] if farm_wide else ["turbine", "timestamp"]
    last = (
        pl.when(pl.col("end") > pl.col("start"))
        .then(pl.col("end") - timedelta(microseconds=1))
        .otherwise(pl.col("start"))
    )
    marked = (
        events.select(
            *keys[:-1],
            pl.datetime_ranges(
                pl.col("start").dt.truncate(interval), last.dt.truncate(interval), interval
            ).alias("timestamp"),
        )
        .explode("timestamp")
        .unique()
        .with_columns(pl.lit(True).alias("overlaps"))
    )
    return (
        rows.select(keys)
        .join(marked, on=keys, how="left", maintain_order="left")
        .get_column("overlaps")
        .fill_null(False)
    )


def previous_interval_value(rows: pl.DataFrame, column: str, interval: timedelta) -> pl.Series:
    """column in the same turbine's previous interval, null where that row is absent."""
    previous = rows.select(
        "turbine",
        (pl.col("timestamp") + interval).alias("timestamp"),
        pl.col(column).alias("previous"),
    )
    return (
        rows.select("turbine", "timestamp")
        .join(previous, on=["turbine", "timestamp"], how="left", maintain_order="left")
        .get_column("previous")
    )


def lowest_known_setpoint(rows: pl.DataFrame, end_column: str, interval: timedelta) -> pl.Series:
    """the lower of the setpoint at the end of this interval and at the end of the previous one."""
    previous = previous_interval_value(rows, end_column, interval)
    return rows.select(
        pl.min_horizontal(pl.col(end_column), previous).alias("lowest_setpoint")
    ).get_column("lowest_setpoint")


def binding_setpoint(
    power: pl.Expr,
    lowest_setpoint: pl.Expr,
    rated_kw: float,
    binding_fraction: float,
    tolerance_fraction: float,
) -> pl.Expr:
    """true where a setpoint below binding_fraction of rated holds the power at or near it.

    near means at or above the setpoint minus tolerance_fraction of rated, so an interval whose
    cap changed within it also counts. an unknown setpoint never binds. only the turbine's own
    power and setpoint enter, so a waked turbine whose power sits below a cap is not flagged.
    """
    return (
        (lowest_setpoint < binding_fraction * rated_kw)
        & (power >= lowest_setpoint - tolerance_fraction * rated_kw)
    ).fill_null(False)


def below_power_floor(power: pl.Expr, rated_kw: float, fraction: float) -> pl.Expr:
    return (power < fraction * rated_kw).fill_null(False)


def turbine_bits(turbine_index: pl.Expr) -> pl.Expr:
    return pl.lit(2, dtype=pl.UInt32).pow(turbine_index.cast(pl.UInt32))


def operating_masks(rows: pl.DataFrame) -> pl.DataFrame:
    """one bit per turbine_index for the turbines with a row at each timestamp."""
    return rows.group_by("timestamp").agg(
        turbine_bits(pl.col("turbine_index")).bitwise_or().alias("operating_mask")
    )
