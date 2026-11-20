"""measured against diurnal inflow by the number of free-stream sensors behind the reading.

a timestamp's free-stream turbulence intensity comes from its unwaked turbines, so each waked
single-arc test row is put in a stratum by how many turbines are unwaked at its timestamp. per
stratum with at least MIN_ROWS rows: the waked-row skill of windfold (measured inflow) minus
windfold_diurnal_inflow, with a bootstrap of the stratum's own blocks, BLOCK_DAYS days from the
first day of the farm's waked single-arc rows. the blocks are drawn in the order the rows first
reach them, splits in name order and rows in time order, and one generator, seeded with SEED,
draws every stratum's replicates in farm and stratum order.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.analyses.block_bootstrap import BLOCK_DAYS, interval, weights
from windfold.analyses.development import FARMS, STUDY, check_paired, single_arcs, waked_rows

MEASURED, DIURNAL, REFERENCE = "windfold", "windfold_diurnal_inflow", "no_wake"
STRATA = {"1": (1, 1), "2": (2, 2), "3-4": (3, 4), "5+": (5, None)}
MIN_ROWS = 200
REPLICATES = 1000
SEED = 20261117


def free_stream_counts(rows: pl.DataFrame) -> pl.DataFrame:
    """timestamp and free_stream, the number of unwaked turbines, per development timestamp."""
    return (
        rows.filter(pl.col("period") == STUDY)
        .group_by("timestamp")
        .agg(free_stream=(~pl.col("waked")).sum())
    )


def stratified_errors(runs: Path, farm: str, counts: pl.DataFrame) -> pl.DataFrame:
    """each waked single-arc row's squared error under each model, its block and stratum count."""
    parts = []
    for split in single_arcs(runs, farm):
        frames = {m: waked_rows(runs, farm, m, split) for m in (MEASURED, DIURNAL, REFERENCE)}
        check_paired(frames, f"{farm} {split}")
        parts.append(
            frames[REFERENCE]
            .select("timestamp", "y")
            .with_columns(
                **{name: frame["mean"] for name, frame in frames.items()},
            )
        )
    frame = pl.concat(parts).join(counts, on="timestamp", how="left", maintain_order="left")
    day = pl.col("timestamp").dt.date()
    return frame.select(
        "free_stream",
        block=(day - day.min()).dt.total_days() // BLOCK_DAYS,
        **{name: (pl.col("y") - pl.col(name)) ** 2 for name in (MEASURED, DIURNAL, REFERENCE)},
    )


def stratum_difference(errors: pl.DataFrame, generator: np.random.Generator) -> dict[str, Any]:
    """the stratum's measured minus diurnal skill over its blocks."""
    sums = errors.group_by("block", maintain_order=True).agg(
        pl.col(MEASURED, DIURNAL, REFERENCE).sum()
    )
    totals = (
        weights(sums.height, REPLICATES, generator)
        @ sums.select(MEASURED, DIURNAL, REFERENCE).to_numpy()
    )
    difference = 100 * (totals[:, 1] - totals[:, 0]) / totals[:, 2]
    return {
        "rows": errors.height,
        "blocks": sums.height,
        "measured_minus_diurnal": interval(difference),
    }


def sensor_count(data: Path, runs: Path, farms: Sequence[str] = FARMS) -> dict[str, Any]:
    generator = np.random.default_rng(SEED)
    out: dict[str, Any] = {}
    for farm in farms:
        rows = pl.read_parquet(
            data / "prepared" / farm / "rows.parquet", columns=["timestamp", "period", "waked"]
        )
        errors = stratified_errors(runs, farm, free_stream_counts(rows))
        out[farm] = {}
        for name, (low, high) in STRATA.items():
            upper = pl.col("free_stream") <= high if high is not None else pl.lit(True)
            part = errors.filter(pl.col("free_stream") >= low, upper)
            if part.height >= MIN_ROWS:
                out[farm][name] = stratum_difference(part, generator)
    return out
