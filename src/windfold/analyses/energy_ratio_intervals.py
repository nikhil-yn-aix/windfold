"""the energy-ratio errors of energy_ratio_error with 95% intervals from the farm's paired
bootstrap of 7-day blocks, and windfold's error minus each model's on the same replicates.

the cells are those with at least the report's minimum rows in the full sample. a replicate
resamples the blocks, and the seeds of seeded models as the study's bootstrap redraws them, for
observed, free-stream and predicted power alike, and weights each cell by its resampled rows."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import config
from windfold.config import folder_of, for_farm
from windfold.evaluation import bootstrap, paths
from windfold.evaluation.results import bootstrap_windows, study_windows
from windfold.evaluation.summaries import summarise
from windfold.report.style import load_report
from windfold.secondary.contrasts import confirmatory_ready

REPLICATES = 2000
MAIN = "windfold"
CELL = ["turbine", "direction_bucket"]


def blockwise(
    frame: pl.DataFrame,
    cells: pl.DataFrame,
    origin: date,
    block_days: int,
    blocks: int,
    columns: Sequence[str],
) -> np.ndarray:
    """each column summed by block and cell: columns by blocks by cells."""
    keyed = frame.with_columns(
        block=pl.Series(bootstrap.block_of(frame["day"], origin, block_days))
    ).join(cells, on=CELL)
    grouped = keyed.group_by("block", "cell").agg(pl.col(c).sum() for c in columns)
    out = np.zeros((len(columns), blocks, cells.height))
    b, c = grouped["block"].to_numpy(), grouped["cell"].to_numpy().astype(np.int64)
    for i, column in enumerate(columns):
        out[i, b, c] = grouped[column].to_numpy()
    return out


def replicate_errors(
    observed: np.ndarray, predicted: np.ndarray, counts: np.ndarray, seeds: np.ndarray | None
) -> np.ndarray:
    """the error on each replicate. observed holds power, free-stream power and rows by block and
    cell, predicted each seed's predicted and free-stream power; without seeds, or for a model
    with one, every seed is pooled."""
    power, reference, rows = (counts @ part for part in observed)
    each = np.stack([[counts @ part for part in seed] for seed in predicted], axis=1)
    if seeds is None or len(predicted) == 1:
        totals = each.sum(axis=1)
    else:
        totals = each[:, seeds.T, np.arange(counts.shape[0])].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.abs(totals[0] / totals[1] - power / reference)
    finite = np.isfinite(gap) & (rows > 0)
    weights = np.where(finite, rows, 0.0)
    return (np.where(finite, gap, 0.0) * weights).sum(axis=1) / weights.sum(axis=1)


def farm_intervals(
    runs: Path, study: str, farm: str, min_rows: int, folder: Path | None
) -> dict[str, Any]:
    protocol, farm_config, _ = for_farm(farm, None, folder)
    statistics = protocol.statistics
    windows = bootstrap_windows(
        study_windows(farm_config, config.load_study(study, folder).period),
        statistics.block_days,
        REPLICATES,
        len(protocol.seeds.model),
        statistics.interval,
        statistics.min_finite,
        bootstrap.farm_seed(protocol.seeds.study, farm),
    )
    draw = windows.draws["period"]
    span = (windows.origins["period"][0], statistics.block_days, windows.blocks["period"])
    sums = pl.read_parquet(runs / paths.reduction_file(study, farm, None, "energy_ratio"))
    first = sums.select("model", "seed").unique().sort("model", "seed", nulls_last=True).row(0)
    observed = sums.filter((pl.col("model") == first[0]) & pl.col("seed").eq_missing(first[1]))
    cells = (
        observed.group_by(CELL)
        .agg(pl.col("rows").sum())
        .filter(pl.col("rows") >= min_rows)
        .select(CELL)
        .sort(CELL)
        .with_row_index("cell")
    )
    seen = blockwise(observed, cells, *span, ["power_kw", "reference_kw", "rows"])
    order = {seed: i for i, seed in enumerate(protocol.seeds.model)}
    errors: dict[str, tuple[float, np.ndarray]] = {}
    for model in sorted(sums["model"].unique().to_list()):
        own = sums.filter(pl.col("model") == model)
        seeds = sorted(own["seed"].unique().to_list(), key=lambda s: -1 if s is None else order[s])
        predicted = np.stack(
            [
                blockwise(
                    own.filter(pl.col("seed").eq_missing(s)),
                    cells,
                    *span,
                    ["predicted_kw", "reference_kw"],
                )
                for s in seeds
            ]
        )
        point = float(replicate_errors(seen, predicted, draw.observed(), None)[0])
        errors[model] = (point, replicate_errors(seen, predicted, draw.counts, draw.seeds))
    found: dict[str, Any] = {}
    for model, (point, replicates) in errors.items():
        estimate, lower, upper = summarise(point, replicates, draw)
        found[model] = {"error": estimate, "lower": lower, "upper": upper}
        if model != MAIN:
            estimate, lower, upper = summarise(
                errors[MAIN][0] - point, errors[MAIN][1] - replicates, draw
            )
            found[model]["windfold_minus"] = {"estimate": estimate, "lower": lower, "upper": upper}
    return {"cells": cells.height, "replicates": REPLICATES, "models": found}


def energy_ratio_intervals(
    runs: Path, study: str, farms: Sequence[str], configs: str | None = None
) -> dict[str, Any]:
    confirmatory_ready(runs, study, configs)
    folder = folder_of(configs)
    min_rows = load_report(folder).failure.min_rows
    return {farm: farm_intervals(runs, study, farm, min_rows, folder) for farm in farms}
