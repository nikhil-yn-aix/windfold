"""what the measured turbulence carries, by the cells of the inflow controls (windfold.inflow).

over the development timestamps with a measured free-stream turbulence intensity: the share of
the variance of its log that lies between the cells of the direction sector, the speed bin,
both, and the diurnal control's full cell (sector, speed bin, calendar month, part of the day),
and its median by sector. then how the diurnal control is fed: every other timestamp of a cell
is a donor, so a timestamp has its cell's size minus one donors, at a median time from it.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import inflow
from windfold.analyses.development import FARMS, STUDY
from windfold.config import Inflow, load_protocol

CELLS = {
    "sector": ("sector",),
    "speed_bin": ("speed_bin",),
    "sector_and_speed_bin": ("sector", "speed_bin"),
    "diurnal_cell": inflow.CONDITIONS["diurnal"],
}
FEW_DONORS = 5
SECONDS_PER_DAY = 86400


def measured_times(rows: pl.DataFrame, settings: Inflow) -> pl.DataFrame:
    """one row per measured development timestamp with its cells; its ratio column holds the
    turbulence intensity itself, a ratio to one."""
    turbulence = rows["turbulence_intensity"].fill_null(np.nan).to_numpy()
    return inflow.conditions(rows, turbulence, settings).filter(pl.col("period") == STUDY)


def explained(times: pl.DataFrame, keys: Sequence[str]) -> float:
    """the share of the variance of log turbulence that lies between the cells of keys."""
    logs = np.log(times["ratio"].to_numpy())
    cells = inflow.codes(times, keys)
    counts = np.bincount(cells)
    means = np.bincount(cells, weights=logs) / counts
    between = np.sum(counts * (means - logs.mean()) ** 2) / logs.size
    return float(between / logs.var())


def donors(times: pl.DataFrame) -> dict[str, float]:
    """how many donors the diurnal control averages over and how far away in time they are."""
    seconds = times["timestamp"].dt.epoch("s").to_numpy()
    cells = inflow.codes(times, inflow.CONDITIONS["diurnal"])
    order = np.argsort(cells, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(cells[order]) != 0])
    ends = np.r_[starts[1:], order.size]
    counts, distances = [], []
    for start, end in zip(starts, ends, strict=True):
        members = seconds[order[start:end]]
        counts.extend([members.size - 1] * members.size)
        if members.size > 1:
            apart = np.abs(members[:, None] - members[None, :]) / SECONDS_PER_DAY
            np.fill_diagonal(apart, np.nan)
            distances.extend(np.nanmedian(apart, axis=1).tolist())
    return {
        "development_days": float((seconds.max() - seconds.min()) / SECONDS_PER_DAY),
        "cells": int(starts.size),
        "median_donors": float(np.median(counts)),
        "share_with_few_donors": float(np.mean(np.asarray(counts) <= FEW_DONORS)),
        "median_days_to_donors": float(np.median(distances)),
    }


def inflow_structure(
    data: Path, farms: Sequence[str] = FARMS, configs: Path | None = None
) -> dict[str, Any]:
    """per farm: the measured timestamps, the variance shares by cell, the median turbulence by
    sector and the diurnal control's donors; few donors means at most FEW_DONORS."""
    settings = load_protocol(configs).inflow
    out: dict[str, Any] = {}
    for farm in farms:
        rows = pl.read_parquet(
            data / "prepared" / farm / "rows.parquet",
            columns=["timestamp", "period", "farm_direction", "speed_bin", "turbulence_intensity"],
        )
        times = measured_times(rows, settings)
        medians = times.group_by("sector").agg(pl.col("ratio").median()).sort("sector")
        out[farm] = {
            "measured_timestamps": times.height,
            "variance_share": {name: explained(times, keys) for name, keys in CELLS.items()},
            "median_turbulence_by_sector": dict(medians.iter_rows()),
            "donors": donors(times),
        }
    return out
