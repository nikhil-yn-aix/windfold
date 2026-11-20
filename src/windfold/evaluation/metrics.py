"""metrics and their reductions: errors, kW, smearing, energy ratios, intervals and rank agreement.

reductions are sums keyed by day, so the bootstrap can form calendar blocks of any length. mse is on
the centered scale of y. crps, interval coverage and width are on the leave-one-out contrast scale,
m / (m - 1) times the centered one, where m is the timestamp's valid turbines.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.special import ndtr, ndtri

from windfold.evaluation.schemas import BREAKDOWN, ENERGY_RATIO, ERRORS, INTERVALS
from windfold.schemas import PREDICTIONS, conform
from windfold.smearing import SMEARING

PREPARED_COLUMNS = (
    "power_kw",
    "valid_turbines",
    "speed_bin",
    "direction_bucket",
    "farm_direction",
    "turbulence_intensity",
    "steady",
)


def scored(predictions: pl.DataFrame, prepared: pl.DataFrame) -> pl.DataFrame:
    """predictions joined to their prepared rows, with day, scale = m / (m - 1) and the centered
    error y - mean. raises when a prediction has no prepared row, no finite mean or a negative
    variance."""
    predictions = conform(predictions, PREDICTIONS)
    if predictions["mean"].null_count() or predictions["mean"].is_nan().any():
        raise ValueError("every prediction needs a finite mean")
    if (predictions["variance"] < 0).any():
        raise ValueError("a predictive variance cannot be negative")
    joined = predictions.join(
        prepared.select("timestamp", "turbine", *PREPARED_COLUMNS),
        on=["timestamp", "turbine"],
        how="left",
        validate="1:1",
    )
    if joined["power_kw"].null_count():
        raise ValueError(f"{joined['power_kw'].null_count()} predictions have no prepared row")
    if (joined["valid_turbines"] < 2).any():
        raise ValueError("a leave-one-out contrast needs at least two valid turbines")
    m = pl.col("valid_turbines").cast(pl.Float64)
    return joined.with_columns(
        day=pl.col("timestamp").dt.date(),
        scale=m / (m - 1),
        error=pl.col("y") - pl.col("mean"),
    )


def median_power(power_kw: np.ndarray, error: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """predicted power by the median conversion: the peers' geometric mean times exp(predicted
    leave-one-out contrast), which equals power times exp(-scale * error)."""
    return power_kw * np.exp(-scale * error)


def gaussian_crps(error: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """closed-form crps of a normal forecast; error is observed minus mean, sd may be zero."""
    error, sd = np.asarray(error, dtype=float), np.asarray(sd, dtype=float)
    if (sd < 0).any():
        raise ValueError("a standard deviation cannot be negative")
    safe = np.where(sd > 0, sd, 1.0)
    z = error / safe
    density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    spread = safe * (z * (2 * ndtr(z) - 1) + 2 * density - 1 / np.sqrt(np.pi))
    return np.where(sd > 0, spread, np.abs(error))


def keyed(frame: pl.DataFrame, farm: str, model: str, split: str, seed: int | None):
    return frame.with_columns(
        farm=pl.lit(farm), model=pl.lit(model), split=pl.lit(split), seed=pl.lit(seed, pl.Int64)
    )


def errors(rows: pl.DataFrame, farm: str, model: str, split: str, seed: int | None) -> pl.DataFrame:
    """squared error, absolute kW error, crps and predicted variance summed by day, waked and
    speed bin; crps and variance are null for a model without variance."""
    error = rows["error"].to_numpy()
    scale = rows["scale"].to_numpy()
    power = rows["power_kw"].to_numpy()
    variance = rows["variance"].to_numpy()
    sd = np.sqrt(scale**2 * np.where(np.isnan(variance), 0.0, variance))
    crps = np.where(np.isnan(variance), np.nan, gaussian_crps(scale * error, sd))
    measured = rows.with_columns(
        squared=pl.Series(error**2),
        abs_error_kw=pl.Series(np.abs(power - median_power(power, error, scale))),
        crps=pl.Series(crps).fill_nan(None),
    )
    totals = measured.group_by("day", "waked", "speed_bin").agg(
        rows=pl.len().cast(pl.Int64),
        sse=pl.col("squared").sum(),
        abs_error_kw=pl.col("abs_error_kw").sum(),
        crps=pl.when(pl.col("crps").null_count() == 0).then(pl.col("crps").sum()),
        variance=pl.when(pl.col("variance").null_count() == 0).then(pl.col("variance").sum()),
    )
    return conform(keyed(totals, farm, model, split, seed), ERRORS).sort(
        "day", "waked", "speed_bin"
    )


def intervals(
    rows: pl.DataFrame,
    levels: Sequence[float],
    farm: str,
    model: str,
    split: str,
    seed: int | None,
) -> pl.DataFrame:
    """central normal intervals at each level: rows covered and width, by day and waked.

    empty for a model without variance.
    """
    usable = rows.filter(pl.col("variance").is_not_null())
    if usable.is_empty():
        return pl.DataFrame(schema=INTERVALS)
    loo_error = (usable["scale"] * usable["error"]).abs().to_numpy()
    sd = (usable["scale"] * usable["variance"].sqrt()).to_numpy()
    parts = []
    for level in levels:
        z = float(ndtri((1 + level) / 2))
        parts.append(
            usable.select("day", "waked")
            .with_columns(
                level=pl.lit(level),
                covered=pl.Series(loo_error <= z * sd),
                width=pl.Series(2 * z * sd),
            )
            .group_by("day", "waked", "level")
            .agg(
                rows=pl.len().cast(pl.Int64),
                covered=pl.col("covered").sum().cast(pl.Int64),
                width=pl.col("width").sum(),
            )
        )
    totals = pl.concat(parts)
    return conform(keyed(totals, farm, model, split, seed), INTERVALS).sort("level", "day", "waked")


def breakdown(
    rows: pl.DataFrame,
    turbulence_edges: Sequence[float],
    arc_width: float,
    farm: str,
    model: str,
    split: str,
    seed: int | None,
) -> pl.DataFrame:
    """squared error by day and waked within each level of hour (utc), turbulence bin, steady
    (1 when the row is steady) and arc. speed bin is in the errors reduction, and month and
    season follow from the day.

    turbulence bins are [edge_k, edge_k+1); rows without turbulence intensity or outside the
    edges have no turbulence level. arc k holds farm directions in [k * arc_width, (k + 1) *
    arc_width).
    """
    edges = np.asarray(turbulence_edges, dtype=float)
    intensity = rows["turbulence_intensity"].fill_null(np.nan).to_numpy()
    turbulence = np.searchsorted(edges, intensity, side="right") - 1
    turbulence = np.where(
        np.isnan(intensity) | (turbulence < 0) | (turbulence >= edges.size - 1), -1, turbulence
    )
    levels = {
        "hour": rows["timestamp"].dt.hour().cast(pl.Int64).to_numpy(),
        "turbulence": turbulence,
        "steady": rows["steady"].cast(pl.Int64).to_numpy(),
        "arc": np.floor(rows["farm_direction"].to_numpy() % 360.0 / arc_width).astype(np.int64),
    }
    squared = (rows["error"] ** 2).to_numpy()
    parts = [
        rows.select("day", "waked")
        .with_columns(factor=pl.lit(factor), level=pl.Series(level), squared=pl.Series(squared))
        .filter(pl.col("level") >= 0)
        .group_by("day", "waked", "factor", "level")
        .agg(rows=pl.len().cast(pl.Int64), sse=pl.col("squared").sum())
        for factor, level in levels.items()
    ]
    totals = pl.concat(parts)
    return conform(keyed(totals, farm, model, split, seed), BREAKDOWN).sort(
        "factor", "level", "day", "waked"
    )


def free_stream_reference(prepared: pl.DataFrame) -> pl.DataFrame:
    """each row's reference: the mean power of the other free-stream (not waked) turbines at its
    timestamp, null when there is none."""
    free = pl.col("power_kw").filter(~pl.col("waked"))
    with_totals = prepared.select("timestamp", "turbine", "power_kw", "waked").with_columns(
        free_sum=free.sum().over("timestamp"),
        free_count=(~pl.col("waked")).sum().over("timestamp"),
    )
    own = pl.when(pl.col("waked")).then(0.0).otherwise(pl.col("power_kw"))
    others = pl.col("free_count") - (~pl.col("waked")).cast(pl.UInt32)
    return with_totals.select(
        "timestamp",
        "turbine",
        reference_kw=pl.when(others > 0).then((pl.col("free_sum") - own) / others),
    )


def energy_ratio_sums(
    rows: pl.DataFrame,
    prepared: pl.DataFrame,
    smearing: pl.DataFrame,
    farm: str,
    model: str,
    split: str,
    seed: int | None,
) -> pl.DataFrame:
    """observed, predicted and free-stream reference power summed by day, turbine and 5 degree
    direction bucket (the flasc energy ratio is sum of power over sum of reference).

    predicted power is the median conversion times the model's smearing factor for the row's
    speed bin. rows without a free-stream reference are left out.
    """
    reference = free_stream_reference(
        prepared.join(rows.select("timestamp").unique(), on="timestamp", how="semi")
    )
    joined = rows.join(reference, on=["timestamp", "turbine"], how="left").join(
        conform(smearing, SMEARING).select("speed_bin", "factor"), on="speed_bin", how="left"
    )
    if joined["factor"].null_count():
        raise ValueError("a speed bin has no smearing factor")
    predicted = median_power(
        joined["power_kw"].to_numpy(), joined["error"].to_numpy(), joined["scale"].to_numpy()
    )
    totals = (
        joined.with_columns(predicted_kw=pl.Series(predicted) * pl.col("factor"))
        .filter(pl.col("reference_kw").is_not_null())
        .group_by("day", "turbine", "direction_bucket")
        .agg(
            rows=pl.len().cast(pl.Int64),
            power_kw=pl.col("power_kw").sum(),
            predicted_kw=pl.col("predicted_kw").sum(),
            reference_kw=pl.col("reference_kw").sum(),
        )
    )
    return conform(keyed(totals, farm, model, split, seed), ENERGY_RATIO).sort(
        "turbine", "direction_bucket", "day"
    )


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """kendall's tau-b between the last axes of x and y, for any leading (replicate) axes."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    first, second = np.triu_indices(x.shape[-1], 1)
    dx = np.sign(x[..., first] - x[..., second])
    dy = np.sign(y[..., first] - y[..., second])
    untied = np.count_nonzero(dx, axis=-1) * np.count_nonzero(dy, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dx * dy).sum(axis=-1) / np.sqrt(untied)


def direction_weighted_mse(sse: np.ndarray, rows: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """mean of the per-arc mse weighted by each arc's share of timestamps (its direction
    frequency), over the last axis; arcs without rows are left out."""
    sse, rows, timestamps = (np.asarray(a, dtype=float) for a in (sse, rows, timestamps))
    present = rows > 0
    weight = np.where(present, timestamps, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = np.where(present, sse / np.where(present, rows, 1.0), 0.0)
        return (weight * mse).sum(axis=-1) / weight.sum(axis=-1)
