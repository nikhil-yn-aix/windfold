"""windfold's local block and what it absorbs, on single arcs.

per farm, over the included single-arc splits with waked test rows (in name order): the physics
weight beta and the rms of the shared-kernel coefficients of windfold and windfold_no_local, each
split's waked-row skill of windfold minus windfold_no_local and minus pywake_inflow, the rank
correlation of beta with the gap to pywake_inflow and the mean waked residual, y minus the
prediction, of several models and of no_wake, whose residual is the wake itself.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from windfold.analyses.development import FARMS, check_paired, fit_record, single_arcs, waked_rows

COMPARED = ("windfold", "windfold_no_local", "pywake_inflow", "no_wake")
RESIDUALS = ("windfold", "windfold_unit_physics", "windfold_no_local", "pywake_inflow", "no_wake")


def coefficient_rms(record: dict[str, Any]) -> float:
    return float(np.sqrt(np.mean(np.square(record["facts"]["blocks"]["shared_kernel"]))))


def beta(record: dict[str, Any]) -> float:
    return float(record["facts"]["blocks"]["physics"][0])


def split_skills(runs: Path, farm: str, split: str) -> dict[str, float]:
    """each compared model's waked-row skill on one split, in percent against no_wake."""
    frames = {model: waked_rows(runs, farm, model, split) for model in COMPARED}
    check_paired(frames, f"{farm} {split}")
    sse = {m: float(((f["y"] - f["mean"]) ** 2).sum()) for m, f in frames.items()}
    return {m: 100 * (1 - sse[m] / sse["no_wake"]) for m in COMPARED if m != "no_wake"}


def local_block(runs: Path, farms: Sequence[str] = FARMS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for farm in farms:
        splits = single_arcs(runs, farm)
        skills = [split_skills(runs, farm, split) for split in splits]
        local = np.array([s["windfold"] - s["windfold_no_local"] for s in skills])
        physics = np.array([s["windfold"] - s["pywake_inflow"] for s in skills])
        records = [fit_record(runs, farm, "windfold", split) for split in splits]
        without = [fit_record(runs, farm, "windfold_no_local", split) for split in splits]
        betas = np.array([beta(r) for r in records])
        rho, p = cast(tuple[float, float], spearmanr(betas, physics))
        residuals = {}
        for model in RESIDUALS:
            joined = pl.concat([waked_rows(runs, farm, model, split) for split in splits])
            residuals[model] = float(joined.select((pl.col("y") - pl.col("mean")).mean()).item())
        out[farm] = {
            "splits": len(splits),
            "beta_median": float(np.median(betas)),
            "beta_median_without_local": float(np.median([beta(r) for r in without])),
            "kernel_rms_median": float(np.median([coefficient_rms(r) for r in records])),
            "kernel_rms_median_without_local": float(
                np.median([coefficient_rms(r) for r in without])
            ),
            "splits_windfold_beats_pywake_inflow": int((physics > 0).sum()),
            "beta_against_gap_to_pywake_inflow": {"spearman": float(rho), "p": float(p)},
            "splits_removing_local_helps": int((local < 0).sum()),
            "per_split": {
                "split": splits,
                "windfold_minus_no_local": local.tolist(),
                "windfold_minus_pywake_inflow": physics.tolist(),
                "beta": betas.tolist(),
            },
            "mean_waked_residual": residuals,
        }
    return out
