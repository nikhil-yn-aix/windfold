"""step 4: sensor faults that make a row's measurements untrustworthy."""

from collections.abc import Sequence

import polars as pl


def in_time_order(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.with_row_index("row").sort("turbine", "timestamp")


def stuck_values(rows: pl.DataFrame, column: str, repeats: int) -> pl.Series:
    """true for every row in a run of at least repeats equal consecutive values of column.

    runs follow each turbine's rows in time order; a null ends a run and is never stuck.
    """
    ordered = in_time_order(rows)
    value = pl.col(column)
    starts = (
        (pl.col("turbine") != pl.col("turbine").shift(1))
        | value.ne_missing(value.shift(1))
        | value.is_null()
    ).fill_null(True)
    flagged = (
        ordered.with_columns(starts.cum_sum().alias("run"))
        .with_columns(
            ((pl.len().over("run") >= repeats) & value.is_not_null()).alias("stuck"),
        )
        .sort("row")
    )
    return flagged.get_column("stuck")


def frozen_rows(
    rows: pl.DataFrame, columns: Sequence[str], speed: str, min_speed: float
) -> pl.Series:
    """true where every column equals the turbine's previous row and speed is at least min_speed.

    nulls compare equal to nulls, as a frozen logger repeats them too.
    """
    ordered = in_time_order(rows)
    same = [pl.col(c).eq_missing(pl.col(c).shift(1).over("turbine")) for c in columns]
    first = pl.col("turbine") != pl.col("turbine").shift(1)
    flagged = ordered.with_columns(
        (pl.all_horizontal(same) & ~first.fill_null(True) & (pl.col(speed) >= min_speed))
        .fill_null(False)
        .alias("frozen")
    ).sort("row")
    return flagged.get_column("frozen")


def linear_angle_mean(
    mean: pl.Expr, minimum: pl.Expr, maximum: pl.Expr, north_band: float, min_distance: float
) -> pl.Expr:
    """true where an angle crossed north within the interval but its mean sits far from north.

    such a mean was averaged linearly across the 0/360 wrap and is wrong. a minimum or maximum
    outside 0 to 360 degrees is a placeholder (Hill of Towie writes -1) and counts as missing;
    the flag is null where a missing value decides it.
    """

    def angle(value: pl.Expr) -> pl.Expr:
        return pl.when(value.is_between(0.0, 360.0)).then(value)

    from_north = 180.0 - ((mean % 360.0) - 180.0).abs()
    return (
        (angle(minimum) <= north_band)
        & (angle(maximum) >= 360.0 - north_band)
        & (from_north >= min_distance)
    )
