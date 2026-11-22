"""measured inflow: each timestamp's free-stream turbulence ratio and the wake physics at it.

a row's ratio is its timestamp's free-stream turbulence intensity over the farm's reference, the
median over its calibration-window timestamps; a farm without a calibration window takes the
median of the calibrated farms' references. a timestamp without a measurement (a free-stream
turbulence intensity that is a positive finite number) takes ratio one, and ratios are clipped
to the grid. each wake family's ambient
turbulence parameter (k for jensen, ti for niayifar and turbopark) is its calibrated or default
value times the ratio, so ratio one gives the constant-inflow phi exactly. phi is evaluated on the
grid and interpolated linearly in the log of the ratio.

three controls replace a measured timestamp's ratio by its expectation over the other measured
timestamps of the same period: all of them (average), those in the same direction sector, speed
bin and calendar month (climatology) and those that also share the part of the day in utc
(diurnal). each averages phi over those timestamps' ratios rather than taking phi at an average
ratio, so it keeps everything about the wake that does not need the timestamp's own measurement.
"""

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from windfold import physics
from windfold.config import Inflow, TurbineConfig
from windfold.schemas import PHYSICS, PHYSICS_FAMILIES, conform

TABLES = {
    "measured": "inflow_physics.parquet",
    "average": "average_inflow_physics.parquet",
    "climatology": "climatology_inflow_physics.parquet",
    "diurnal": "diurnal_inflow_physics.parquet",
}
TABLE = TABLES["measured"]
CONDITIONS: dict[str, tuple[str, ...]] = {
    "average": (),
    "climatology": ("sector", "speed_bin", "month"),
    "diurnal": ("sector", "speed_bin", "month", "day_part"),
}
RECORD = "inflow_fit.json"
INFLOW_PHYSICS = pl.Schema(dict(PHYSICS) | {"ratio": pl.Float64})
TURBULENCE_PARAMETER = {"jensen": "k", "niayifar": "ti", "turbopark": "ti"}

Array = NDArray[np.float64]


def ratio_grid(settings: Inflow) -> Array:
    """the ratios two to the power low, low + step, ..., high, with exactly one at exponent 0."""
    count = round((settings.log2_ratio_high - settings.log2_ratio_low) / settings.log2_ratio_step)
    exponents = settings.log2_ratio_low + settings.log2_ratio_step * np.arange(count + 1)
    exponents[round(-settings.log2_ratio_low / settings.log2_ratio_step)] = 0.0
    return np.exp2(exponents)


def measured(rows: pl.DataFrame) -> NDArray[np.bool_]:
    """whether each row's free-stream turbulence intensity is a measurement: positive and finite."""
    turbulence = rows["turbulence_intensity"].to_numpy().astype(np.float64)
    return np.isfinite(turbulence) & (turbulence > 0)


def reference(rows: pl.DataFrame) -> float:
    """the median measured free-stream turbulence intensity over the calibration-window
    timestamps."""
    times = (
        rows.filter(pl.col("period") == "calibration")
        .group_by("timestamp")
        .agg(pl.col("turbulence_intensity").first())
    )
    values = times["turbulence_intensity"].to_numpy().astype(np.float64)[measured(times)]
    if not values.size:
        raise ValueError("no calibration-window timestamp has a free-stream turbulence intensity")
    return float(np.median(values))


def ratios(rows: pl.DataFrame, reference_ti: float, grid: Array) -> Array:
    if not (np.isfinite(reference_ti) and reference_ti > 0):
        raise ValueError(f"the reference turbulence intensity {reference_ti} is not positive")
    turbulence = rows["turbulence_intensity"].to_numpy().astype(np.float64)
    known = measured(rows)
    ratio = np.ones(rows.height)
    ratio[known] = turbulence[known] / reference_ti
    return np.clip(ratio, grid[0], grid[-1])


def interpolation(ratio: ArrayLike, grid: Array) -> tuple[NDArray[np.int64], Array]:
    """the grid position below each ratio and the weight of the one above, linear in log ratio."""
    position = np.interp(np.log(np.asarray(ratio)), np.log(grid), np.arange(grid.size))
    lower = np.minimum(np.floor(position).astype(np.int64), grid.size - 2)
    return lower, position - lower


def grid_weights(ratio: ArrayLike, grid: Array) -> Array:
    """each ratio as weights on the grid points, linear in the log of the ratio."""
    lower, weight = interpolation(ratio, grid)
    out = np.zeros((lower.size, grid.size))
    at = np.arange(lower.size)
    out[at, lower] = 1 - weight
    out[at, lower + 1] += weight
    return out


def codes(frame: pl.DataFrame, columns: Sequence[str]) -> NDArray[np.int64]:
    keys = frame.select(columns).unique(maintain_order=True).with_row_index("code")
    joined = frame.select(columns).join(keys, on=list(columns), how="left", maintain_order="left")
    return joined["code"].to_numpy().astype(np.int64)


def conditions(rows: pl.DataFrame, ratio: Array, settings: Inflow) -> pl.DataFrame:
    """one row per measured timestamp, in first-seen order: its period, ratio, direction sector,
    speed bin, calendar month and part of the day."""
    return (
        rows.select("timestamp", "period", "farm_direction", "speed_bin")
        .with_columns(ratio=ratio, measured=measured(rows))
        .unique("timestamp", keep="first", maintain_order=True)
        .filter(pl.col("measured"))
        .select(
            "timestamp",
            "period",
            "ratio",
            "speed_bin",
            sector=(pl.col("farm_direction") // settings.climatology_sector_deg).cast(pl.Int64),
            month=pl.col("timestamp").dt.month(),
            day_part=(pl.col("timestamp").dt.hour() // settings.diurnal_hours).cast(pl.Int64),
        )
    )


def expected_weights(times: pl.DataFrame, keys: Sequence[str], grid: Array) -> Array:
    """each timestamp's grid weights averaged over the other timestamps of its period that share
    its keys; a timestamp alone in its group takes all the others of its period, and one alone in
    its period takes ratio one."""
    own = grid_weights(times["ratio"].to_numpy(), grid)
    period = codes(times, ["period"])
    group = codes(times, ["period", *keys])
    by_period = np.zeros((int(period.max(initial=-1)) + 1, grid.size))
    by_group = np.zeros((int(group.max(initial=-1)) + 1, grid.size))
    np.add.at(by_period, period, own)
    np.add.at(by_group, group, own)
    alone = (np.bincount(group) == 1)[group]
    rest = np.where(alone[:, None], by_period[period], by_group[group]) - own
    total = rest.sum(axis=1, keepdims=True)
    one = np.broadcast_to((grid == 1.0).astype(np.float64), rest.shape)
    return np.where(total > 1e-9, rest / np.maximum(total, 1e-300), one)


def control_weights(rows: pl.DataFrame, ratio: Array, settings: Inflow) -> dict[str, Array]:
    """each row's grid weights under each control (CONDITIONS); a row without a measurement
    keeps ratio one."""
    grid = ratio_grid(settings)
    times = conditions(rows, ratio, settings)
    slot = rows.select("timestamp").join(
        times.select("timestamp").with_row_index("slot"),
        on="timestamp",
        how="left",
        maintain_order="left",
    )["slot"]
    known = slot.is_not_null().to_numpy()
    slot = slot.to_numpy()[known].astype(np.int64)
    one = (grid == 1.0).astype(np.float64)
    weights = {}
    for name, keys in CONDITIONS.items():
        out = np.tile(one, (rows.height, 1))
        if times.height:
            out[known] = expected_weights(times, keys, grid)[slot]
        weights[name] = out
    return weights


def tables(
    rows: pl.DataFrame,
    turbine: TurbineConfig,
    east: ArrayLike,
    north: ArrayLike,
    hub_height: ArrayLike,
    speeds_ms: ArrayLike,
    calibrated: Mapping[str, Mapping[str, float]],
    defaults: Mapping[str, Mapping[str, float]],
    truncate_sd: float,
    floor_fraction: float,
    reference_ti: float,
    settings: Inflow,
    workers: int = 1,
) -> dict[str, pl.DataFrame]:
    """the measured-inflow table and its controls (TABLES), each with every family's phi at
    calibrated and published default parameters, and the ratio it stands for (INFLOW_PHYSICS):
    a control's ratio is the geometric mean of the ratios it averages over."""
    physics.check_rows(rows, len(np.asarray(east)), len(np.asarray(speeds_ms)))
    grid = ratio_grid(settings)
    ratio = ratios(rows, reference_ti, grid)
    lower, weight = interpolation(ratio, grid)
    controls = control_weights(rows, ratio, settings)
    pywake = physics.pywake_turbine(turbine)
    masks, degrees, bins, inverse = physics.row_keys(rows)
    index = rows["turbine_index"].to_numpy().astype(np.int64)
    floor_w = floor_fraction * turbine.rated_power_kw * 1000.0
    at = np.arange(rows.height)
    columns: dict[str, dict[str, Array]] = {table: {} for table in TABLES}
    with physics.worker_pool(workers) as pool:
        for family in PHYSICS_FAMILIES:
            name = TURBULENCE_PARAMETER[family]
            for suffix, parameters in (("", calibrated[family]), ("_default", defaults[family])):
                on_grid = np.stack(
                    [
                        physics.phi_keys(
                            physics.wake_model(
                                family, {**parameters, name: parameters[name] * g}, pywake
                            ),
                            east,
                            north,
                            hub_height,
                            masks,
                            degrees,
                            bins,
                            speeds_ms,
                            parameters[physics.SPREAD],
                            truncate_sd,
                            floor_w,
                            pool,
                        )[inverse, index]
                        for g in grid
                    ]
                )
                column = f"physics_{family}{suffix}"
                columns["measured"][column] = (1 - weight) * on_grid[lower, at] + weight * (
                    on_grid[lower + 1, at]
                )
                for control, weights in controls.items():
                    columns[control][column] = np.einsum("rg,gr->r", weights, on_grid)
    standing = {"measured": ratio} | {
        control: np.exp(weights @ np.log(grid)) for control, weights in controls.items()
    }
    keys = rows.select("timestamp", "turbine")
    return {
        table: conform(keys.with_columns(**columns[table], ratio=standing[table]), INFLOW_PHYSICS)
        for table in TABLES
    }


def record(rows: pl.DataFrame, reference_ti: float, source: str, grid: Array) -> dict[str, Any]:
    """what inflow_fit.json holds: the reference, where it came from, the grid and by period the
    share of timestamps without a measurement and the shares clipped at each end."""
    times = rows.group_by("timestamp").agg(pl.col("period", "turbulence_intensity").first())
    shares = {}
    for (period,), group in times.group_by("period"):
        turbulence = group["turbulence_intensity"].to_numpy().astype(np.float64)
        known = turbulence[measured(group)] / reference_ti
        shares[str(period)] = {
            "timestamps": group.height,
            "missing": float(1 - known.size / group.height),
            "clipped_low": float(np.mean(known < grid[0])) if known.size else 0.0,
            "clipped_high": float(np.mean(known > grid[-1])) if known.size else 0.0,
        }
    return {"reference_ti": reference_ti, "source": source, "grid": grid.tolist(), "shares": shares}


@lru_cache(maxsize=4)
def read(folder: Path, stamp: tuple[int, int, int]) -> pl.DataFrame:
    return pl.read_parquet(folder / TABLE)


def read_reference(folder: Path) -> float:
    return float(json.loads((folder / RECORD).read_text())["reference_ti"])


def ratios_of(rows: pl.DataFrame, folder: Path) -> Array:
    """each row's ratio from the folder's inflow table, in row order."""
    path = folder / TABLE
    status = path.stat()
    joined = rows.select("timestamp", "turbine").join(
        read(folder, (status.st_ino, status.st_size, status.st_mtime_ns)).select(
            "timestamp", "turbine", "ratio"
        ),
        on=["timestamp", "turbine"],
        how="left",
        maintain_order="left",
    )
    if joined["ratio"].null_count():
        raise ValueError(f"{path} has no ratio for some rows; rebuild it for this farm")
    return joined["ratio"].to_numpy()
