"""physics recalibrated without the held-out arc: does windfold lean on calibration data from the
directions it is tested on?

the shared calibration (physics_fit.json) uses every direction of the calibration window. for
each single-arc split of a calibrated farm, every wake family is calibrated again on the
calibration rows outside the held-out arc and its buffers, with the shared calibration's seed and
the protocol's trials; the inflow tables of the development rows are rebuilt with those
parameters and the farm's reference turbulence, and windfold and pywake_inflow are refitted on the
split. the result is each model's pooled single-arc skill with the shared and the strict
calibration and the change, strict minus shared, by paired block bootstrap.

a refit folder holds data/<split>/prepared/<farm>, the farm's prepared files linked with the
rebuilt tables, fits/<split>, the refits' predictions and records, and records/<farm>/<split>.json,
written last, which marks the split done and carries its provenance. a split whose record has
another provenance, or none, is refused rather than reused.
"""

import json
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import inflow, physics
from windfold.analyses.block_bootstrap import (
    REPLICATES,
    interval,
    replicate_sse,
    skill,
    weights,
    window_blocks,
)
from windfold.analyses.development import (
    FARMS,
    LARGE_FARMS,
    STUDY,
    check_paired,
    single_arcs,
    waked_rows,
)
from windfold.analyses.refits import link_prepared, write_atomic
from windfold.config import for_farm, load_models, load_study
from windfold.provenance import fit_inputs, provenance, reusable
from windfold.tasks.data import layout_arrays
from windfold.tasks.fitting import fit_engine

MODELS = ("windfold", "pywake_inflow")
REFERENCE = "no_wake"
SEED = 20261117


def arc_start(split: str) -> float:
    """the first direction of a single-arc split's arc, from a name such as arc_030_single."""
    parts = split.split("_")
    if len(parts) != 3 or parts[0] != "arc" or parts[2] != "single":
        raise ValueError(f"{split} is not a single-arc split")
    return float(parts[1])


def held_out(direction: np.ndarray, start: float, width: float, buffer: float) -> np.ndarray:
    """whether each direction lies in the arc [start, start + width), widened by buffer on each
    side, around the circle."""
    return np.mod(direction - start + buffer, 360.0) < width + 2 * buffer


def calibration_record(work: Path, farm: str, split: str) -> Path:
    return work / "records" / farm / f"{split}.json"


def split_provenance(
    data: Path, runs: Path, farm: str, split: str, configs: Path | None = None
) -> str:
    """the provenance of one split's strict calibration and refits."""
    protocol, farm_config, turbine = for_farm(farm, None, configs)
    specs = load_models(configs)
    limits = load_study(STUDY, configs).limits
    inputs = fit_inputs(data, runs, STUDY, farm, protocol)
    models = [specs[model] for model in MODELS]
    return provenance(recalibrate, inputs, protocol, farm_config, turbine, limits, *models, split)


def recalibrate(
    data: Path,
    runs: Path,
    work: Path,
    farm: str,
    split: str,
    workers: int = 1,
    configs: Path | None = None,
) -> dict[str, Any]:
    """one split's strict calibration, inflow tables and refits under work and its record."""
    began = time.time()
    made = split_provenance(data, runs, farm, split, configs)
    protocol, _, turbine = for_farm(farm, None, configs)
    source = data / "prepared" / farm
    rows = pl.read_parquet(source / "rows.parquet")
    east, north, hub = layout_arrays(rows, pl.read_parquet(source / "layout.parquet"))
    speeds = np.asarray(protocol.processing.speed_bin_centres_ms)
    calibration = rows.filter(pl.col("period") == "calibration")
    if calibration.is_empty():
        raise ValueError(f"{farm} has no calibration rows to calibrate on")
    away = ~held_out(
        calibration["farm_direction"].to_numpy(),
        arc_start(split),
        protocol.splits.arc_width_deg,
        protocol.splits.arc_buffer_deg,
    )
    shared = json.loads((source / "physics_fit.json").read_text())
    settings = protocol.physics
    fit = physics.fit_farm(
        calibration.filter(pl.Series(away)),
        turbine,
        east,
        north,
        hub,
        speeds,
        settings.search,
        settings.defaults,
        settings.trials,
        shared["seed"],
        settings.truncate_sd,
        settings.power_floor_fraction,
        workers,
    )
    tables = inflow.tables(
        rows.filter(pl.col("period") == STUDY),
        turbine,
        east,
        north,
        hub,
        speeds,
        fit["parameters"],
        settings.defaults,
        settings.truncate_sd,
        settings.power_floor_fraction,
        inflow.read_reference(source),
        protocol.inflow,
        workers,
    )
    prepared = work / "data" / split / "prepared" / farm
    link_prepared(source, prepared, set(inflow.TABLES.values()))
    for name, table in tables.items():
        write_atomic(prepared / inflow.TABLES[name], table.write_parquet)
    fit_engine(
        STUDY,
        farm,
        split,
        list(MODELS),
        None,
        str(work / "data" / split),
        str(runs),
        str(configs) if configs else None,
        work / "fits" / split,
    )
    record = {
        "farm": farm,
        "split": split,
        "kept_calibration_rows": int(fit["rows"]),
        "calibration_rows": calibration.height,
        "parameters": fit["parameters"],
        "shared_parameters": shared["parameters"],
        "teacher": fit["teacher"],
        "seconds": round(time.time() - began),
        "provenance": made,
    }
    text = json.dumps(record, indent=1)
    write_atomic(calibration_record(work, farm, split), lambda path: path.write_text(text))
    return record


def skill_change(runs: Path, work: Path, farm: str, configs: Path | None = None) -> dict[str, Any]:
    """one farm's single-arc skill of each model with the shared and the strict calibration, and
    strict minus shared, from its refits under work and the study's fits under runs."""
    splits = single_arcs(runs, farm)
    origin, blocks = window_blocks(farm, configs)
    draws = weights(blocks, REPLICATES, np.random.default_rng(SEED + FARMS.index(farm)))
    shared: dict[str, list[pl.DataFrame]] = {m: [] for m in (*MODELS, REFERENCE)}
    strict: dict[str, list[pl.DataFrame]] = {m: [] for m in MODELS}
    for split in splits:
        frames = {m: waked_rows(runs, farm, m, split) for m in (*MODELS, REFERENCE)}
        refits = {f"strict {m}": waked_rows(work / "fits" / split, farm, m, split) for m in MODELS}
        check_paired(frames | refits, f"{farm} {split}")
        for model in shared:
            shared[model].append(frames[model])
        for model in strict:
            strict[model].append(refits[f"strict {model}"])
    reference = replicate_sse(shared[REFERENCE], origin, draws)
    out: dict[str, Any] = {"splits": len(splits)}
    for model in MODELS:
        before = skill(replicate_sse(shared[model], origin, draws), reference)
        after = skill(replicate_sse(strict[model], origin, draws), reference)
        out[model] = {
            "shared": float(before[0]),
            "strict": float(after[0]),
            "strict_minus_shared": interval(after - before),
        }
    records = [json.loads(calibration_record(work, farm, split).read_text()) for split in splits]
    excluded = [1 - r["kept_calibration_rows"] / r["calibration_rows"] for r in records]
    out["excluded_calibration_share"] = float(np.mean(excluded))
    return out


def strict_calibration(
    data: Path,
    runs: Path,
    work: Path,
    farms: Sequence[str] = LARGE_FARMS,
    refit: bool = False,
    workers: int = 1,
    configs: Path | None = None,
) -> dict[str, Any]:
    """the skill change per farm; with refit, the missing splits are recalibrated and refitted
    first, one after another, else a missing split is an error. a split made otherwise is
    refused."""
    refit_elsewhere = f"refit it under another --work folder, or delete {work} first"
    missing = [
        (farm, split)
        for farm in farms
        for split in single_arcs(runs, farm)
        if not reusable(
            calibration_record(work, farm, split),
            partial(split_provenance, data, runs, farm, split, configs),
            refit_elsewhere,
        )
    ]
    if missing and not refit:
        raise ValueError(
            f"{len(missing)} splits are not recalibrated under {work}, such as {missing[0]}; "
            "refit makes them (windfold analyse --refit)"
        )
    for farm, split in missing:
        recalibrate(data, runs, work, farm, split, workers, configs)
    return {farm: skill_change(runs, work, farm, configs) for farm in farms}
