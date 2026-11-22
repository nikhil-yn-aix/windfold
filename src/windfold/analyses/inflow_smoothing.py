"""smoothed inflow: does windfold use the measured turbulence of its ten minutes, or a level a
few hours' average of the reading would do as well?

a variant replaces each measured timestamp's free-stream turbulence intensity by the geometric
mean of the measured ones from before hours earlier to after hours later in its period, both ends
included; unmeasured timestamps stay unmeasured and RAW keeps the reading. each variant's
measured-inflow table is rebuilt from phi on the ratio grid, evaluated once per farm as
windfold.inflow evaluates it, and windfold is refitted on every single-arc split with waked test
rows. the result is each variant's pooled single-arc skill and its change against windfold on
the reading, the study's own fit, beside windfold_diurnal_inflow's, by paired block bootstrap,
and the means of those changes over the large farms and over all four.

a refit folder holds data/<variant>/prepared/<farm>, the farm's prepared files linked with the
variant's table, records/<farm>/<variant>.json, the provenance of that table, written before it,
and fits/<variant>, the refits' predictions and records, each record with its provenance. a table
or refit with another provenance, or none, is refused rather than reused.
"""

import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

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
    prediction_file,
    record_file,
    single_arcs,
    waked_rows,
)
from windfold.analyses.refits import link_prepared, write_atomic
from windfold.config import TurbineConfig, for_farm, load_models, load_study
from windfold.provenance import fit_inputs, provenance, reusable
from windfold.schemas import PHYSICS_FAMILIES, conform
from windfold.tasks.data import layout_arrays
from windfold.tasks.fitting import fit_engine

VARIANTS = {"h00": (0, 0), "h01": (1, 1), "h03": (3, 3), "h12": (12, 12), "t06": (6, 0)}
RAW = "h00"
MODEL, DIURNAL, REFERENCE = "windfold", "windfold_diurnal_inflow", "no_wake"
SEED = 20261118
SECONDS_PER_HOUR = 3600


def smoothed(rows: pl.DataFrame, before: float, after: float) -> pl.DataFrame:
    """rows with each measured timestamp's turbulence intensity replaced by the geometric mean of
    the measured ones from before hours earlier to after hours later in its period."""
    if before == 0 and after == 0:
        return rows
    times = (
        rows.select("timestamp", "period", "turbulence_intensity")
        .unique("timestamp", keep="first")
        .sort("timestamp")
    )
    parts = []
    for _, group in times.group_by("period", maintain_order=True):
        reading = group["turbulence_intensity"].fill_null(np.nan).to_numpy().astype(np.float64)
        known = inflow.measured(group)
        seconds = group["timestamp"].dt.epoch("s").to_numpy()[known]
        logs = np.r_[0.0, np.cumsum(np.log(reading[known]))]
        low = np.searchsorted(seconds, seconds - before * SECONDS_PER_HOUR, side="left")
        high = np.searchsorted(seconds, seconds + after * SECONDS_PER_HOUR, side="right")
        value = reading.copy()
        value[known] = np.exp((logs[high] - logs[low]) / (high - low))
        parts.append(group.select("timestamp").with_columns(smoothed=value))
    joined = rows.select("timestamp").join(
        pl.concat(parts), on="timestamp", how="left", maintain_order="left"
    )
    return rows.with_columns(turbulence_intensity=joined["smoothed"])


def phi_on_grid(
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
    grid: np.ndarray,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    """each physics column's phi for every row at every grid ratio (grid by rows), evaluated
    exactly as inflow.tables evaluates it."""
    physics.check_rows(rows, len(np.asarray(east)), len(np.asarray(speeds_ms)))
    pywake = physics.pywake_turbine(turbine)
    masks, degrees, bins, inverse = physics.row_keys(rows)
    index = rows["turbine_index"].to_numpy().astype(np.int64)
    floor_w = floor_fraction * turbine.rated_power_kw * 1000.0
    on_grid = {}
    with physics.worker_pool(workers) as pool:
        for family in PHYSICS_FAMILIES:
            name = inflow.TURBULENCE_PARAMETER[family]
            for suffix, parameters in (("", calibrated[family]), ("_default", defaults[family])):
                on_grid[f"physics_{family}{suffix}"] = np.stack(
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
    return on_grid


def measured_table(
    rows: pl.DataFrame, on_grid: Mapping[str, np.ndarray], reference_ti: float, grid: np.ndarray
) -> pl.DataFrame:
    """the measured-inflow table of rows, phi interpolated as inflow.tables interpolates it."""
    ratio = inflow.ratios(rows, reference_ti, grid)
    lower, weight = inflow.interpolation(ratio, grid)
    at = np.arange(rows.height)
    columns = {
        column: (1 - weight) * phi[lower, at] + weight * phi[lower + 1, at]
        for column, phi in on_grid.items()
    }
    keys = rows.select("timestamp", "turbine")
    return conform(keys.with_columns(**columns, ratio=ratio), inflow.INFLOW_PHYSICS)


def table_file(work: Path, farm: str, variant: str) -> Path:
    return work / "data" / variant / "prepared" / farm / inflow.TABLE


def table_record(work: Path, farm: str, variant: str) -> Path:
    return work / "records" / farm / f"{variant}.json"


def table_provenance(
    data: Path, runs: Path, farm: str, variant: str, configs: Path | None = None
) -> str:
    protocol, farm_config, turbine = for_farm(farm, None, configs)
    inputs = fit_inputs(data, runs, STUDY, farm, protocol)
    return provenance(build_tables, inputs, protocol, farm_config, turbine, variant)


def fit_provenance(
    data: Path, runs: Path, farm: str, split: str, variant: str, configs: Path | None = None
) -> str:
    """the provenance of windfold refitted on one split with a variant's table, whose own
    provenance it includes."""
    protocol, farm_config, turbine = for_farm(farm, None, configs)
    inputs = fit_inputs(data, runs, STUDY, farm, protocol)
    table = table_provenance(data, runs, farm, variant, configs)
    spec = load_models(configs)[MODEL]
    limits = load_study(STUDY, configs).limits
    return provenance(
        refit_split, inputs, protocol, farm_config, turbine, limits, spec, split, variant, table
    )


def build_tables(
    data: Path,
    runs: Path,
    work: Path,
    farm: str,
    variants: Sequence[str],
    workers: int = 1,
    configs: Path | None = None,
) -> None:
    """the farm's table of each variant under work, each after the record of its provenance;
    raises when the RAW variant's table is not the farm's own measured-inflow table, the check
    that both are built alike."""
    made = {variant: table_provenance(data, runs, farm, variant, configs) for variant in variants}
    protocol, _, turbine = for_farm(farm, None, configs)
    source = data / "prepared" / farm
    rows = pl.read_parquet(source / "rows.parquet")
    east, north, hub = layout_arrays(rows, pl.read_parquet(source / "layout.parquet"))
    grid = inflow.ratio_grid(protocol.inflow)
    settings = protocol.physics
    on_grid = phi_on_grid(
        rows,
        turbine,
        east,
        north,
        hub,
        np.asarray(protocol.processing.speed_bin_centres_ms),
        json.loads((source / "physics_fit.json").read_text())["parameters"],
        settings.defaults,
        settings.truncate_sd,
        settings.power_floor_fraction,
        grid,
        workers,
    )
    reference = inflow.read_reference(source)
    for variant in variants:
        table = measured_table(smoothed(rows, *VARIANTS[variant]), on_grid, reference, grid)
        if variant == RAW and not table.equals(pl.read_parquet(source / inflow.TABLE)):
            raise ValueError(f"{farm}: the {RAW} table differs from {source / inflow.TABLE}")
        link_prepared(source, table_file(work, farm, variant).parent, {inflow.TABLE})
        text = json.dumps({"farm": farm, "variant": variant, "provenance": made[variant]})
        write_atomic(
            table_record(work, farm, variant), lambda path, text=text: path.write_text(text)
        )
        write_atomic(table_file(work, farm, variant), table.write_parquet)


def refit_split(
    data: Path,
    runs: Path,
    work: Path,
    farm: str,
    split: str,
    variant: str,
    configs: Path | None = None,
) -> None:
    """windfold on one split with the variant's table, its record given its provenance."""
    made = fit_provenance(data, runs, farm, split, variant, configs)
    fit_engine(
        STUDY,
        farm,
        split,
        [MODEL],
        None,
        str(work / "data" / variant),
        str(runs),
        str(configs) if configs else None,
        work / "fits" / variant,
    )
    record = record_file(work / "fits" / variant, farm, MODEL, split)
    text = json.dumps(
        json.loads(record.read_text()) | {"provenance": made}, indent=1, sort_keys=True
    )
    write_atomic(record, lambda path: path.write_text(text + "\n"))


def farm_skills(
    runs: Path, work: Path, farm: str, variants: Sequence[str], configs: Path | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """one farm's summary and, per variant and for windfold_diurnal_inflow, the replicates of
    its skill minus windfold's."""
    splits = single_arcs(runs, farm)
    origin, blocks = window_blocks(farm, configs)
    draws = weights(blocks, REPLICATES, np.random.default_rng(SEED + FARMS.index(farm)))
    frames: dict[str, list[pl.DataFrame]] = {n: [] for n in (REFERENCE, MODEL, DIURNAL, *variants)}
    identical = 0
    for split in splits:
        here = {m: waked_rows(runs, farm, m, split) for m in (REFERENCE, MODEL, DIURNAL)}
        here |= {v: waked_rows(work / "fits" / v, farm, MODEL, split) for v in variants}
        check_paired(here, f"{farm} {split}")
        for name, frame in here.items():
            frames[name].append(frame)
        raw = prediction_file(work / "fits" / RAW, farm, MODEL, split)
        if raw.is_file():
            study = prediction_file(runs, farm, MODEL, split)
            if not pl.read_parquet(raw).equals(pl.read_parquet(study)):
                raise ValueError(f"{farm} {split}: the {RAW} refit differs from {study}")
            identical += 1
    reference = replicate_sse(frames[REFERENCE], origin, draws)
    skills = {
        name: skill(replicate_sse(parts, origin, draws), reference)
        for name, parts in frames.items()
        if name != REFERENCE
    }
    changes = {name: skills[name] - skills[MODEL] for name in (*variants, DIURNAL)}
    summary = {
        "splits": len(splits),
        "raw_refits_identical": identical,
        "skill": {name: float(values[0]) for name, values in skills.items()},
        "minus_raw": {name: interval(values) for name, values in changes.items()},
    }
    return summary, changes


def inflow_smoothing(
    data: Path,
    runs: Path,
    work: Path,
    farms: Sequence[str] = FARMS,
    refit: bool = False,
    workers: int = 1,
    configs: Path | None = None,
    variants: Sequence[str] = tuple(VARIANTS),
) -> dict[str, Any]:
    """per farm the summary of farm_skills, and the changes averaged over farms when every
    large farm, or every farm, is analyzed; with refit, missing tables and refits are
    made first, one after another, else they are an error. a table or refit made otherwise is
    refused."""
    refit_elsewhere = f"refit it under another --work folder, or delete {work} first"
    tables = [
        (farm, variant)
        for farm in farms
        for variant in variants
        if not reusable(
            table_record(work, farm, variant),
            partial(table_provenance, data, runs, farm, variant, configs),
            refit_elsewhere,
            table_file(work, farm, variant),
        )
    ]
    fits = [
        (farm, split, variant)
        for farm in farms
        for split in single_arcs(runs, farm)
        for variant in variants
        if not reusable(
            record_file(work / "fits" / variant, farm, MODEL, split),
            partial(fit_provenance, data, runs, farm, split, variant, configs),
            refit_elsewhere,
        )
    ]
    if (tables or fits) and not refit:
        raise ValueError(
            f"{len(tables)} tables and {len(fits)} refits are missing under {work}; "
            "refit makes them (windfold analyse --refit)"
        )
    for farm in farms:
        unbuilt = [variant for f, variant in tables if f == farm]
        if unbuilt:
            build_tables(data, runs, work, farm, unbuilt, workers, configs)
    for farm, split, variant in fits:
        refit_split(data, runs, work, farm, split, variant, configs)
    out: dict[str, Any] = {}
    changes = {}
    for farm in farms:
        out[farm], changes[farm] = farm_skills(runs, work, farm, variants, configs)
    for label, group in (("large_farms", LARGE_FARMS), ("equal_farm", FARMS)):
        if set(group) <= set(farms):
            out[f"{label}_minus_raw"] = {
                name: interval(np.mean([changes[f][name] for f in group], axis=0))
                for name in (*variants, DIURNAL)
            }
    return out
