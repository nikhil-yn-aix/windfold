"""column schemas of the tables windfold reads and writes."""

import polars as pl

UTC = pl.Datetime("us", "UTC")

PERIODS = ("calibration", "development", "confirmatory")
EVENT_SOURCES = ("status_log", "alarm_log", "shutdown_log", "control_log")
MANIFEST_ROLES = ("train", "test", "buffer", "fit", "calibrate")
SPLIT_STATUSES = ("included", "ineligible", "unmatched", "insufficient data")
PHYSICS_FAMILIES = ("jensen", "niayifar", "turbopark")

RAW_ROWS = pl.Schema(
    {
        "timestamp": UTC,
        "turbine": pl.String,
        "power_kw": pl.Float64,
        "speed": pl.Float64,
        "speed_std": pl.Float64,
        "wind_direction": pl.Float64,
        "nacelle_position": pl.Float64,
        "ambient_temperature": pl.Float64,
    }
)

LAYOUT = pl.Schema(
    {
        "turbine": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "x_m": pl.Float64,
        "y_m": pl.Float64,
        "hub_height_m": pl.Float64,
    }
)

EVENTS = pl.Schema(
    {
        "turbine": pl.String,
        "start": UTC,
        "end": UTC,
        "source": pl.String,
        "status": pl.String,
        "code": pl.Int64,
        "message": pl.String,
        "category": pl.String,
    }
)

PREPARED_ROWS = pl.Schema(
    {
        "timestamp": UTC,
        "turbine": pl.String,
        "turbine_index": pl.UInt8,
        "period": pl.String,
        "power_kw": pl.Float64,
        "speed": pl.Float64,
        "speed_std": pl.Float64,
        "direction_raw": pl.Float64,
        "direction": pl.Float64,
        "nacelle_position": pl.Float64,
        "farm_direction": pl.Float64,
        "reference_speed": pl.Float64,
        "speed_bin": pl.UInt8,
        "direction_bucket": pl.UInt8,
        "steady": pl.Boolean,
        "operating_mask": pl.UInt32,
        "waked": pl.Boolean,
        "valid_turbines": pl.UInt8,
        "y": pl.Float64,
        "temperature": pl.Float64,
        "turbulence_intensity": pl.Float64,
    }
)

LEDGER = pl.Schema(
    {
        "step": pl.UInt8,
        "rule": pl.String,
        "reason": pl.String,
        "rows_dropped": pl.UInt64,
        "rows_kept": pl.UInt64,
        "timestamps_kept": pl.UInt64,
    }
)

PHYSICS = pl.Schema(
    {"timestamp": UTC, "turbine": pl.String}
    | {f"physics_{family}": pl.Float64 for family in PHYSICS_FAMILIES}
    | {f"physics_{family}_default": pl.Float64 for family in PHYSICS_FAMILIES}
)

MANIFEST = pl.Schema({"split": pl.String, "regime": pl.String, "timestamp": UTC, "role": pl.String})

SPLIT_SUMMARY = pl.Schema(
    {
        "split": pl.String,
        "regime": pl.String,
        "status": pl.String,
        "reason": pl.String,
        "train_timestamps": pl.Int64,
        "train_rows": pl.Int64,
        "test_timestamps": pl.Int64,
        "test_rows": pl.Int64,
        "test_waked_rows": pl.Int64,
        "test_days": pl.Int64,
        "removed_rows": pl.Int64,
    }
)

PREDICTIONS = pl.Schema(
    {
        "timestamp": UTC,
        "turbine": pl.String,
        "y": pl.Float64,
        "mean": pl.Float64,
        "variance": pl.Float64,
        "support": pl.Int64,
        "waked": pl.Boolean,
    }
)


class SchemaError(ValueError):
    pass


def conform(frame: pl.DataFrame, schema: pl.Schema, *, extra: bool = False) -> pl.DataFrame:
    """return frame with schema's columns first, in order; raise SchemaError naming every problem.

    extra allows further snake_case columns after the schema's own, such as provider columns
    a farm config declares.
    """
    problems = []
    for name, dtype in schema.items():
        if name not in frame.columns:
            problems.append(f"missing column {name} ({dtype})")
        elif frame.schema[name] != dtype:
            problems.append(f"column {name} is {frame.schema[name]}, expected {dtype}")
    others = [name for name in frame.columns if name not in schema]
    if others and not extra:
        problems.append(f"unexpected columns {others}")
    bad_names = [name for name in others if not name.isidentifier() or name != name.lower()]
    if bad_names:
        problems.append(f"column names must be snake_case: {bad_names}")
    if problems:
        raise SchemaError("; ".join(problems))
    return frame.select([*schema.names(), *others])
