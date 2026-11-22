"""windfold, pywake at measured inflow and the no-wake model refitted on two variants of the
single-arc splits of the development window:

- time_buffer: training timestamps within an hour of a test timestamp of the split also leave
  training, so no training row is minutes away from a test row across an arc's edge.
- shifted_grid: the arcs start half an arc later, so every arc edge moves to where the study's
  arcs have their middle.

the result is windfold's pooled single-arc skill minus pywake's, by paired block bootstrap, for
each variant and for the study's own splits. a variant's folder under work holds its manifest at
runs/development/manifests/<farm>.parquet, the refits under fits/<split> and records/<farm>.json,
written last, which carries the provenance.
"""

import json
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import splits
from windfold.analyses.block_bootstrap import (
    interval,
    replicate_sse,
    skill,
    weights,
    window_blocks,
)
from windfold.analyses.development import FARMS, STUDY, check_paired, single_arcs, waked_rows
from windfold.analyses.refits import write_atomic
from windfold.config import for_farm, load_models
from windfold.locations import manifest_file
from windfold.provenance import fit_inputs, provenance, reusable
from windfold.tasks.fitting import fit_engine

VARIANTS = ("time_buffer", "shifted_grid")
TIME_BUFFER = timedelta(hours=1)
MODELS = ("windfold", "pywake_inflow")
REFERENCE = "no_wake"
SEED = 20261122


def time_buffered(manifest: pl.DataFrame, buffer: timedelta) -> pl.DataFrame:
    """each split's training timestamps within buffer of one of its test timestamps become
    buffer."""
    parts = []
    for _, chosen in manifest.group_by("split", maintain_order=True):
        tests = np.sort(chosen.filter(pl.col("role") == "test")["timestamp"].to_numpy())
        stamps = chosen["timestamp"].to_numpy()
        after = np.clip(np.searchsorted(tests, stamps), 0, len(tests) - 1)
        before = np.clip(after - 1, 0, len(tests) - 1)
        gap = np.minimum(np.abs(tests[after] - stamps), np.abs(stamps - tests[before]))
        near = pl.Series(gap <= np.timedelta64(buffer))
        parts.append(
            chosen.with_columns(
                role=pl.when((pl.col("role") == "train") & near)
                .then(pl.lit("buffer"))
                .otherwise(pl.col("role"))
            )
        )
    return pl.concat(parts)


def shifted_arcs(table: pl.DataFrame, offset: float, **settings: Any) -> pl.DataFrame:
    """the included single-arc splits of the arc grid started offset degrees later, named by the
    first direction of their arc."""
    turned = table.with_columns(direction=(pl.col("direction") - offset) % 360.0)
    manifest, summary = splits.arcs(turned, **settings)
    included = summary.filter((pl.col("regime") == "arc_single") & (pl.col("status") == "included"))
    names = {
        name: f"arc_{round(float(name.split('_')[1]) + offset) % 360:03d}_single"
        for name in included["split"].to_list()
    }
    return manifest.filter(pl.col("split").is_in(list(names))).with_columns(
        split=pl.col("split").replace_strict(names)
    )


def variant_manifest(
    data: Path, runs: Path, farm: str, variant: str, configs: Path | None
) -> pl.DataFrame:
    protocol, farm_config, _ = for_farm(farm, None, configs)
    if variant == "time_buffer":
        manifest = pl.read_parquet(runs / manifest_file(STUDY, farm, None, protocol))
        return time_buffered(
            manifest.filter(pl.col("split").is_in(single_arcs(runs, farm, waked=False))),
            TIME_BUFFER,
        )
    if variant == "shifted_grid":
        rows = pl.read_parquet(data / "prepared" / farm / "rows.parquet")
        window = farm_config.windows.development
        table = splits.timestamp_table(rows, STUDY, window.start, window.end)
        s = protocol.splits
        return shifted_arcs(
            table,
            s.arc_width_deg / 2,
            count=s.arc_count,
            width=s.arc_width_deg,
            buffer=s.arc_buffer_deg,
            min_rows=s.arc_min_rows,
            min_days=s.arc_min_days,
            tolerance=s.control_tolerance,
            seed=protocol.seeds.study,
        )
    raise ValueError(f"no variant {variant}; the variants are {', '.join(VARIANTS)}")


def record_path(work: Path, variant: str, farm: str) -> Path:
    return work / variant / "records" / f"{farm}.json"


def variant_provenance(
    data: Path, runs: Path, farm: str, variant: str, configs: Path | None
) -> str:
    protocol, farm_config, turbine = for_farm(farm, None, configs)
    specs = load_models(configs)
    models = [specs[m] for m in (REFERENCE, *MODELS)]
    inputs = fit_inputs(data, runs, STUDY, farm, protocol)
    return provenance(
        refit_variant, inputs, protocol, farm_config, turbine, *models, variant, str(TIME_BUFFER)
    )


def refit_variant(
    data: Path, runs: Path, work: Path, farm: str, variant: str, workers: int, configs: Path | None
) -> dict[str, Any]:
    """write one farm's variant manifest, refit the models on each of its splits and write the
    record."""
    made = variant_provenance(data, runs, farm, variant, configs)
    protocol, _, _ = for_farm(farm, None, configs)
    manifest = variant_manifest(data, runs, farm, variant, configs)
    root = work / variant / "runs"
    target = root / manifest_file(STUDY, farm, None, protocol)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(target, manifest.write_parquet)
    names = manifest["split"].unique().sort().to_list()
    folder = str(configs) if configs else None
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(
                fit_engine,
                STUDY,
                farm,
                split,
                [REFERENCE, *MODELS],
                None,
                str(data),
                str(root),
                folder,
                work / variant / "fits" / split,
            )
            for split in names
        ]
        for future in futures:
            future.result()
    roles = manifest.group_by("role").len().sort("role")
    record = {
        "farm": farm,
        "variant": variant,
        "splits": names,
        "roles": dict(zip(roles["role"].to_list(), roles["len"].to_list(), strict=True)),
        "provenance": made,
    }
    text = json.dumps(record, indent=1)
    write_atomic(record_path(work, variant, farm), lambda path: path.write_text(text))
    return record


def margin(
    frames: dict[str, list[pl.DataFrame]], farm: str, configs: Path | None, replicates: int
) -> dict[str, Any]:
    origin, blocks = window_blocks(farm, configs)
    draws = weights(blocks, replicates, np.random.default_rng(SEED + FARMS.index(farm)))
    reference = replicate_sse(frames[REFERENCE], origin, draws)
    skills = {m: skill(replicate_sse(frames[m], origin, draws), reference) for m in MODELS}
    return {
        "splits": len(frames[REFERENCE]),
        "skill": {m: float(skills[m][0]) for m in MODELS},
        "windfold_minus_pywake_inflow": interval(skills["windfold"] - skills["pywake_inflow"]),
    }


def collected(
    folder: Callable[[str], Path], farm: str, names: Sequence[str], where: str
) -> dict[str, list[pl.DataFrame]]:
    """the waked test rows of every split with any, per model, checked to be paired."""
    frames: dict[str, list[pl.DataFrame]] = {m: [] for m in (REFERENCE, *MODELS)}
    for split in names:
        found = {m: waked_rows(folder(split), farm, m, split) for m in frames}
        if found[REFERENCE].is_empty():
            continue
        check_paired(found, f"{where} {farm} {split}")
        for m, frame in found.items():
            frames[m].append(frame)
    return frames


def arc_variants(
    data: Path,
    runs: Path,
    work: Path,
    farms: Sequence[str] = FARMS,
    refit: bool = False,
    workers: int = 1,
    configs: Path | None = None,
) -> dict[str, Any]:
    """per farm, windfold minus pywake at measured inflow on the study's single arcs and on each
    variant; with refit, the missing variants are refitted first, else a missing one is an error."""
    again = f"refit it under another --work folder, or delete {work} first"
    result: dict[str, Any] = {}
    for farm in farms:
        protocol, _, _ = for_farm(farm, None, configs)
        replicates = protocol.statistics.replicates
        study_frames = collected(lambda _: runs, farm, single_arcs(runs, farm), "study")
        result[farm] = {"study": margin(study_frames, farm, configs, replicates)}
        for variant in VARIANTS:
            made = partial(variant_provenance, data, runs, farm, variant, configs)
            if not reusable(record_path(work, variant, farm), made, again):
                if not refit:
                    raise ValueError(f"{farm} {variant} is not refitted under {work}; --refit")
                refit_variant(data, runs, work, farm, variant, workers, configs)
            record = json.loads(record_path(work, variant, farm).read_text())
            fits = work / variant / "fits"
            frames = collected(fits.joinpath, farm, record["splits"], variant)
            result[farm][variant] = margin(frames, farm, configs, replicates) | {
                "roles": record["roles"]
            }
    return result
