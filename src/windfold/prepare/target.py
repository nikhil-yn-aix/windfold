"""step 11: the centered log-power target."""

import polars as pl


def centred_log_power(rows: pl.DataFrame) -> pl.DataFrame:
    """y is log power minus its mean over the timestamp's valid turbines, so each timestamp's y
    sums to zero; valid_turbines is their count m_t. every power must be positive.
    """
    if rows.select((pl.col("power_kw") <= 0).any()).item():
        raise ValueError(
            "log power needs positive power; the power floor should have removed these"
        )
    log_power = pl.col("power_kw").log()
    return rows.with_columns(
        (log_power - log_power.mean().over("timestamp")).alias("y"),
        pl.len().over("timestamp").cast(pl.UInt8).alias("valid_turbines"),
    )
