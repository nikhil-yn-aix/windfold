"""pywake at measured inflow with pywake's published default wake parameters in place of the
calibrated ones, fitted on every included single arc of a study. calibrated pywake at measured
inflow and windfold are each scored against it, in mse and skill points, with the farm's planned
paired block bootstrap.

the fits sit under <runs>/<study>/secondary/pywake_inflow_default_physics and are reused while
their provenance matches.
"""

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from windfold import config
from windfold.config import folder_of, for_farm
from windfold.locations import kernel_folder, manifest_file, prepared_folder
from windfold.models.interface import FitContext
from windfold.models.registry import build_model
from windfold.provenance import fit_inputs, provenance, reusable
from windfold.secondary.contrasts import (
    confirmatory_ready,
    contrasts,
    output_folder,
    prediction_names,
    single_arcs,
)
from windfold.tasks.fitting import fit_one, split_rows, write_fit

MODEL = "pywake_inflow_default_physics"
BASE = "pywake_inflow"
TESTED = ("pywake_inflow", "windfold")


def default_spec(folder: Path | None) -> config.ModelSpec:
    return config.load_models(folder)[BASE].model_copy(update={"physics_parameters": "default"})


def fit(
    data: Path, runs: Path, study: str, farm: str, split: str, configs: str | None = None
) -> Path:
    """fit pywake at measured inflow with default physics on one single arc; a fit already written
    is kept while its provenance matches and refused when it differs."""
    confirmatory_ready(runs, study, configs)
    folder = folder_of(configs)
    protocol, farm_config, turbine = for_farm(farm, None, folder)
    out = output_folder(runs, study, MODEL)
    table, note = prediction_names(farm, split, None)
    spec = default_spec(folder)
    seed = protocol.seeds.model[0]
    inputs = fit_inputs(data, runs, study, farm, protocol)
    made = provenance(fit, inputs, protocol, farm_config, turbine, spec, split, seed)
    refit = f"delete {out / 'predictions' / farm} to fit it again"
    if reusable(out / note, lambda: made, refit) and (out / table).is_file():
        return out / table
    model = build_model(spec, farm_config, turbine, protocol)
    prepared = data / "prepared"
    rows = pl.read_parquet(prepared / prepared_folder(farm, None, protocol) / "rows.parquet")
    manifest = pl.read_parquet(runs / manifest_file(study, farm, None, protocol))
    train, test = split_rows(rows, manifest, split, None)
    context = FitContext(
        farm_config.name,
        prepared / prepared_folder(farm, None, protocol),
        seed,
        1,
        None,
        prepared / kernel_folder(farm, None, protocol),
    )
    predictions, record, _ = fit_one(model, train, test, context, protocol)
    record |= {"base": BASE, "physics_parameters": "default", "provenance": made}
    write_fit(out, (table, note), predictions, record)
    return out / table


def fit_all(
    data: Path, runs: Path, study: str, farm: str, workers: int, configs: str | None = None
) -> None:
    """fit every included single arc of the farm, workers at a time."""
    confirmatory_ready(runs, study, configs)
    protocol, _, _ = for_farm(farm, None, folder_of(configs))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(fit, data, runs, study, farm, split, configs)
            for split in single_arcs(runs, study, farm, protocol)
        ]
        for future in futures:
            future.result()


def default_physics(
    data: Path,
    runs: Path,
    study: str,
    farms: Sequence[str],
    refit: bool = False,
    workers: int = 1,
    configs: Path | None = None,
) -> dict[str, Any]:
    """per farm, calibrated pywake at measured inflow and windfold minus pywake at measured
    inflow with default physics; with refit, the missing fits are made first."""
    folder = str(configs) if configs else None
    result = {}
    for farm in farms:
        if refit:
            fit_all(data, runs, study, farm, workers, folder)
        table = contrasts(data, runs, study, farm, MODEL, [None], TESTED, (), folder)
        result[farm] = table.to_dicts()
    return result
