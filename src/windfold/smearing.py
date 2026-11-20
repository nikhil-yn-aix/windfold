"""duan smearing: the factor that turns the median power a log-scale prediction implies into a
mean."""

import numpy as np
import polars as pl

SMEARING = pl.Schema(
    {"speed_bin": pl.UInt8, "rows": pl.Int64, "factor": pl.Float64, "pooled": pl.Boolean}
)


def smearing_factors(
    residuals: np.ndarray, speed_bins: np.ndarray, min_rows: int, bin_count: int
) -> pl.DataFrame:
    """duan smearing factors by speed bin from training residuals on the leave-one-out scale.

    a bin's factor is the mean of exp(residual) over its rows; a bin with fewer than min_rows
    rows takes the pooled factor over all rows.
    """
    values = np.asarray(residuals, dtype=float)
    bins = np.asarray(speed_bins, dtype=np.int64)
    if values.size == 0 or values.shape != bins.shape or not np.isfinite(values).all():
        raise ValueError("smearing needs finite residuals, one speed bin each")
    if bins.min() < 0 or bins.max() >= bin_count:
        raise ValueError(f"speed bins must lie in 0 to {bin_count - 1}")
    growth = np.exp(values)
    pooled = float(growth.mean())
    rows = np.bincount(bins, minlength=bin_count)
    sums = np.bincount(bins, weights=growth, minlength=bin_count)
    own = rows >= min_rows
    factor = np.where(own, sums / np.maximum(rows, 1), pooled)
    return pl.DataFrame(
        {"speed_bin": np.arange(bin_count), "rows": rows, "factor": factor, "pooled": ~own},
        schema=SMEARING,
    )
