"""each compared model's waked-row skill under time-blocked validation and on held-out single
sectors, both on the rows that held-out validation tests and both against the no-wake model's
time-blocked error on those rows, so the two skills differ only by the model's own error.

an error is y minus the mean prediction, as the reductions score it, and a model's mse is the mean
over its seeds of the per-seed mse pooled over a validation's splits. the compared models are
every model but windfold's variants whose time-blocked and single-sector fits are all done.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.analyses.development import FARMS, STUDY
from windfold.locations import prediction_files

REGIMES = ("time_blocked", "arc_single")
KEYS = ["timestamp", "turbine"]
REFERENCE = "no_wake"
VARIANTS = "windfold_"


def squared(
    runs: Path, farm: str, model: str, splits: Sequence[str], seed: int | None
) -> pl.DataFrame:
    """a model's squared errors on the waked test rows of the splits."""
    frames = [
        pl.read_parquet(
            runs / prediction_files(STUDY, None, farm, model, split, seed)[0],
            columns=[*KEYS, "y", "mean", "waked"],
        )
        for split in splits
    ]
    return (
        pl.concat(frames).filter("waked").select(*KEYS, squared=(pl.col("y") - pl.col("mean")) ** 2)
    )


def pooled(found: pl.DataFrame, rows: pl.DataFrame) -> float:
    on = found.join(rows, on=KEYS, how="semi")
    if on.height != rows.height:
        raise ValueError(f"{on.height} of {rows.height} held-out rows have a prediction")
    return float(on.select(pl.col("squared").mean()).item())


def complete(fits: pl.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[int | None]]]:
    """each regime's planned splits, and the seeds of every compared model whose fits of both
    regimes are all done."""
    planned = {r: sorted(set(fits.filter(pl.col("regime") == r)["split"])) for r in REGIMES}
    done = fits.filter(pl.col("status") == "done")
    models = {}
    for model in sorted(set(fits["model"])):
        if model.startswith(VARIANTS):
            continue
        seeds = sorted(set(fits.filter(pl.col("model") == model)["seed"]), key=lambda s: s or 0)
        mine = done.filter(pl.col("model") == model)
        finished = all(
            sorted(set(mine.filter(pl.col("seed").eq_missing(s), pl.col("regime") == r)["split"]))
            == planned[r]
            for s in seeds
            for r in REGIMES
        )
        if finished:
            models[model] = seeds
    return planned, models


def farm_skill(runs: Path, farm: str) -> dict[str, dict[str, float]]:
    fits = pl.read_parquet(runs / STUDY / "reductions" / farm / "fits.parquet")
    planned, models = complete(fits.filter(pl.col("regime").is_in(REGIMES)))
    rows = squared(runs, farm, REFERENCE, planned["arc_single"], models[REFERENCE][0]).select(KEYS)
    mse = {
        model: np.mean(
            [
                [
                    pooled(squared(runs, farm, model, planned[regime], seed), rows)
                    for regime in REGIMES
                ]
                for seed in seeds
            ],
            axis=0,
        )
        for model, seeds in models.items()
    }
    base = mse[REFERENCE][0]
    return {
        model: {"time_blocked": 100 * (1 - blocked / base), "held_out": 100 * (1 - held / base)}
        for model, (blocked, held) in mse.items()
    }


def same_row_skill(runs: Path, farms: Sequence[str] = FARMS) -> dict[str, Any]:
    """per farm, each compared model's time-blocked and held-out skill on the same rows."""
    return {farm: farm_skill(runs, farm) for farm in farms}
