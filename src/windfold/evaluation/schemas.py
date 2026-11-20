"""column schemas of the evaluation tables."""

import polars as pl

REDUCTION_KEYS = {"farm": pl.String, "model": pl.String, "split": pl.String, "seed": pl.Int64}

ERRORS = pl.Schema(
    REDUCTION_KEYS
    | {
        "day": pl.Date,
        "waked": pl.Boolean,
        "speed_bin": pl.UInt8,
        "rows": pl.Int64,
        "sse": pl.Float64,
        "abs_error_kw": pl.Float64,
        "crps": pl.Float64,
        "variance": pl.Float64,
    }
)

INTERVALS = pl.Schema(
    REDUCTION_KEYS
    | {
        "day": pl.Date,
        "waked": pl.Boolean,
        "level": pl.Float64,
        "rows": pl.Int64,
        "covered": pl.Int64,
        "width": pl.Float64,
    }
)

BREAKDOWN = pl.Schema(
    REDUCTION_KEYS
    | {
        "day": pl.Date,
        "waked": pl.Boolean,
        "factor": pl.String,
        "level": pl.Int64,
        "rows": pl.Int64,
        "sse": pl.Float64,
    }
)

ENERGY_RATIO = pl.Schema(
    REDUCTION_KEYS
    | {
        "day": pl.Date,
        "turbine": pl.String,
        "direction_bucket": pl.UInt8,
        "rows": pl.Int64,
        "power_kw": pl.Float64,
        "predicted_kw": pl.Float64,
        "reference_kw": pl.Float64,
    }
)

SCALES = pl.Schema(
    {
        "turbine": pl.String,
        "rows": pl.Int64,
        "days": pl.Int64,
        "sigma": pl.Float64,
        "rho": pl.Float64,
        "eligible": pl.Boolean,
    }
)

EPISODES = pl.Schema(
    {
        "injection": pl.Int64,
        "turbine": pl.String,
        "start": pl.Date,
        "drop": pl.Float64,
        "first_alarm": pl.Date,
        "delay_days": pl.Int64,
        "censored": pl.Boolean,
    }
)

CONTRASTS = pl.Schema(
    {
        "component": pl.String,
        "status": pl.String,
        "estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "p": pl.Float64,
        "replicates": pl.Int64,
    }
)
HYPOTHESES = pl.Schema(
    {
        "farm": pl.String,
        "hypothesis": pl.String,
        "component": pl.String,
        "status": pl.String,
        "estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "p": pl.Float64,
        "replicates": pl.Int64,
        "hypothesis_p": pl.Float64,
        "holm_p": pl.Float64,
        "outcome": pl.String,
    }
)
