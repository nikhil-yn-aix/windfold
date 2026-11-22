"""the farm rotation fitted again without each held-out arc: does the direction alignment lean on
the directions it is tested on?

the shared rotation (step 7, provenance.json) centers the wake dips of every turbine pair in the
calibration window on the layout bearings. for each single-arc split of a farm with that
rotation, the rotation is fitted again on the prepared calibration rows outside the held-out arc
and its buffers, with the shared rotation taken off first, and compared with the same fit on all
of them. with refit, the measured-inflow physics of the development rows is rebuilt at farm
directions moved by that shift, windfold and pywake_inflow are refitted on the split, and the
result adds each model's pooled single-arc skill with the shared and the strict alignment and
the change, strict minus shared, by paired block bootstrap. the shared correction keeps the
shared alignment, and a split whose rotation does not move keeps the study's fits.

a refit folder holds data/<split>/prepared/<farm>, the farm's prepared files linked with the
rebuilt tables, fits/<split>, the refits, and records/<farm>/<split>.json, written last, which
carries the provenance.
"""

import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import inflow
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
from windfold.analyses.strict_calibration import arc_start, held_out
from windfold.config import for_farm, load_models
from windfold.prepare import direction
from windfold.provenance import fit_inputs, provenance, reusable
from windfold.tasks.data import layout_arrays
from windfold.tasks.fitting import fit_engine

MODELS = ("windfold", "pywake_inflow")
REFERENCE = "no_wake"
SEED = 20261122


def rotation(rows: pl.DataFrame, layout: pl.DataFrame, processing: Any) -> tuple[float, int]:
    fitted, pairs = direction.wake_dip_rotation(
        rows,
        layout,
        distance_d=processing.dip_distance_d,
        speed_ms=processing.dip_speed_ms,
        half_width_deg=processing.dip_half_width_deg,
        min_rows=processing.dip_min_rows,
        min_depth=processing.dip_min_depth,
    )
    return float(fitted), pairs.height


def rotations(data: Path, runs: Path, farm: str, configs: Path | None) -> dict[str, Any]:
    """the shared rotation, the rotation refitted on all prepared calibration rows and on those
    outside each held-out arc, and each arc's shift."""
    protocol, farm_config, turbine = for_farm(farm, None, configs)
    if farm_config.northing.farm_rotation != "wake_dips":
        raise ValueError(f"{farm} has no wake-dip rotation to refit")
    source = data / "prepared" / farm
    northing = json.loads((source / "provenance.json").read_text())["northing"]
    shared = northing["farm_rotation_deg"]
    rows = pl.read_parquet(source / "rows.parquet").filter(pl.col("period") == "calibration")
    unrotated = rows.with_columns(direction=(pl.col("direction") - shared) % 360.0)
    layout = pl.read_parquet(source / "layout.parquet").with_columns(
        rotor_diameter_m=pl.lit(turbine.rotor_diameter_m)
    )
    whole, pairs = rotation(unrotated, layout, protocol.processing)
    splits = {}
    for split in single_arcs(runs, farm, waked=False):
        away = ~held_out(
            rows["farm_direction"].to_numpy(),
            arc_start(split),
            protocol.splits.arc_width_deg,
            protocol.splits.arc_buffer_deg,
        )
        fitted, kept = rotation(unrotated.filter(pl.Series(away)), layout, protocol.processing)
        splits[split] = {"rotation": fitted, "shift": fitted - whole, "pairs": kept}
    shifts = np.abs([s["shift"] for s in splits.values()])
    return {
        "shared": shared,
        "all_rows": whole,
        "pairs": pairs,
        "largest_shift": float(shifts.max()),
        "median_shift": float(np.median(shifts)),
        "splits": splits,
    }


def record_path(work: Path, farm: str, split: str) -> Path:
    return work / "records" / farm / f"{split}.json"


def split_provenance(
    data: Path, runs: Path, farm: str, split: str, shift: float, configs: Path | None
) -> str:
    protocol, farm_config, turbine = for_farm(farm, None, configs)
    specs = load_models(configs)
    inputs = fit_inputs(data, runs, STUDY, farm, protocol)
    models = [specs[model] for model in MODELS]
    return provenance(realign, inputs, protocol, farm_config, turbine, *models, split, shift)


def realign(
    data: Path,
    runs: Path,
    work: Path,
    farm: str,
    split: str,
    shift: float,
    workers: int = 1,
    configs: Path | None = None,
) -> None:
    """one split's physics at the moved farm directions, its refits under work and its record."""
    made = split_provenance(data, runs, farm, split, shift, configs)
    protocol, _, turbine = for_farm(farm, None, configs)
    source = data / "prepared" / farm
    rows = pl.read_parquet(source / "rows.parquet")
    east, north, hub = layout_arrays(rows, pl.read_parquet(source / "layout.parquet"))
    shared = json.loads((source / "physics_fit.json").read_text())
    settings = protocol.physics
    moved = rows.filter(pl.col("period") == STUDY).with_columns(
        farm_direction=(pl.col("farm_direction") + shift) % 360.0
    )
    tables = inflow.tables(
        moved,
        turbine,
        east,
        north,
        hub,
        np.asarray(protocol.processing.speed_bin_centres_ms),
        shared["parameters"],
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
    text = json.dumps({"farm": farm, "split": split, "shift": shift, "provenance": made}, indent=1)
    write_atomic(record_path(work, farm, split), lambda path: path.write_text(text))


def skill_change(runs: Path, work: Path, farm: str, moved: set[str], configs: Path | None) -> dict:
    """each model's pooled single-arc skill with the shared and the strict alignment."""
    origin, blocks = window_blocks(farm, configs)
    draws = weights(blocks, REPLICATES, np.random.default_rng(SEED + FARMS.index(farm)))
    shared: dict[str, list[pl.DataFrame]] = {m: [] for m in (*MODELS, REFERENCE)}
    strict: dict[str, list[pl.DataFrame]] = {m: [] for m in MODELS}
    for split in single_arcs(runs, farm):
        frames = {m: waked_rows(runs, farm, m, split) for m in (*MODELS, REFERENCE)}
        root = work / "fits" / split if split in moved else runs
        refits = {f"strict {m}": waked_rows(root, farm, m, split) for m in MODELS}
        check_paired(frames | refits, f"{farm} {split}")
        for model in shared:
            shared[model].append(frames[model])
        for model in strict:
            strict[model].append(refits[f"strict {model}"])
    reference = replicate_sse(shared[REFERENCE], origin, draws)
    out: dict[str, Any] = {}
    for model in MODELS:
        before = skill(replicate_sse(shared[model], origin, draws), reference)
        after = skill(replicate_sse(strict[model], origin, draws), reference)
        out[model] = {
            "shared": float(before[0]),
            "strict": float(after[0]),
            "strict_minus_shared": interval(after - before),
        }
    return out


def strict_alignment(
    data: Path,
    runs: Path,
    work: Path,
    farms: Sequence[str] = LARGE_FARMS,
    refit: bool = False,
    workers: int = 1,
    configs: Path | None = None,
) -> dict[str, Any]:
    """per farm, the rotations and shifts and, once every moved split is refitted, the skill
    change; with refit, the missing splits are refitted first, one after another."""
    again = f"refit it under another --work folder, or delete {work} first"
    result: dict[str, Any] = {}
    for farm in farms:
        found = rotations(data, runs, farm, configs)
        result[farm] = found
        moved = {
            split: s["shift"]
            for split, s in found["splits"].items()
            if split in single_arcs(runs, farm) and s["shift"] != 0.0
        }
        missing = [
            split
            for split, shift in moved.items()
            if not reusable(
                record_path(work, farm, split),
                partial(split_provenance, data, runs, farm, split, shift, configs),
                again,
            )
        ]
        if missing and not refit:
            continue
        for split in missing:
            realign(data, runs, work, farm, split, moved[split], workers, configs)
        found["skill"] = skill_change(runs, work, farm, set(moved), configs)
    return result
