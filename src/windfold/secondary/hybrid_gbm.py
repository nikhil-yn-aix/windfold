"""a gbm on turbine identity, the wind-frame geometry features and reference speed, against each
side of the planned gbm pair on a study's single arcs (item 57 of the study plan).

the hybrid takes the pair's rows, projected objective and model seeds and the farm's tuned
geometry gbm configuration; it has no tuning of its own.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from windfold.config import folder_of, for_farm
from windfold.locations import chosen_file, kernel_folder, manifest_file, prepared_folder
from windfold.models.gbm import GeometrySettings, GradientBoosting
from windfold.models.interface import FitContext
from windfold.provenance import fit_inputs, provenance, reusable
from windfold.secondary.contrasts import (
    confirmatory_ready,
    contrasts,
    output_folder,
    prediction_names,
    single_arcs,
)
from windfold.tasks.fitting import fit_one, split_rows, write_fit

MODEL = "hybrid_gbm"
FEATURES = ("turbine", "wind_frame_geometry", "reference_speed")
TESTED = ("identity_gbm", "geometry_gbm")
DESCRIBED = ("windfold", "pywake_inflow")
TUNED_AS = "geometry_gbm"


def fit(
    data: Path,
    runs: Path,
    study: str,
    farm: str,
    split: str,
    seed: int,
    threads: int,
    configs: str | None = None,
) -> Path:
    """fit the hybrid on one single arc with one model seed and write its predictions and record.
    a fit already written is kept while its record's provenance matches, and refused when it
    differs or is missing."""
    confirmatory_ready(runs, study, configs)
    protocol, farm_config, turbine = for_farm(farm, None, folder_of(configs))
    out = output_folder(runs, study, MODEL)
    table, note = prediction_names(farm, split, seed)
    inputs = [*fit_inputs(data, runs, study, farm, protocol), runs / chosen_file(farm, TUNED_AS)]
    made = provenance(fit, inputs, protocol, farm_config, turbine, split, seed)
    refit = f"delete {out / 'predictions' / farm} to fit it again"
    if reusable(out / note, lambda: made, refit) and (out / table).is_file():
        return out / table
    prepared = data / "prepared"
    rows = pl.read_parquet(prepared / prepared_folder(farm, None, protocol) / "rows.parquet")
    manifest = pl.read_parquet(runs / manifest_file(study, farm, None, protocol))
    train, test = split_rows(rows, manifest, split, None)
    chosen = json.loads((runs / chosen_file(farm, TUNED_AS)).read_text())
    configuration: dict[str, Any] = dict(chosen["configuration"])
    model = GradientBoosting(
        MODEL,
        features=FEATURES,
        parameters=configuration,
        turbine=turbine,
        geometry=GeometrySettings(
            along_max_d=protocol.kernel.along_max_d,
            across_max_d=protocol.kernel.across_max_d,
            upstream_neighbours=protocol.gbm.upstream_neighbours,
            rotor_points=protocol.gbm.rotor_points,
        ),
    )
    context = FitContext(
        farm_config.name,
        prepared / prepared_folder(farm, None, protocol),
        seed,
        threads,
        None,
        prepared / kernel_folder(farm, None, protocol),
    )
    predictions, record, _ = fit_one(model, train, test, context, protocol)
    record |= {"configuration": configuration, "provenance": made}
    write_fit(out, (table, note), predictions, record)
    return out / table


def fit_all(
    data: Path,
    runs: Path,
    study: str,
    farm: str,
    workers: int,
    threads: int,
    configs: str | None = None,
) -> list[Path]:
    """fit the hybrid on every included single arc and model seed, workers at a time."""
    confirmatory_ready(runs, study, configs)
    protocol, _, _ = for_farm(farm, None, folder_of(configs))
    jobs = [
        (split, seed)
        for split in single_arcs(runs, study, farm, protocol)
        for seed in protocol.seeds.model
    ]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(fit, data, runs, study, farm, split, seed, threads, configs)
            for split, seed in jobs
        ]
        return [future.result() for future in futures]


def score(
    data: Path, runs: Path, study: str, farm: str, configs: str | None = None
) -> pl.DataFrame:
    """identity_gbm and geometry_gbm minus the hybrid, tested with holm over the two; windfold and
    pywake_inflow minus the hybrid, described; then each contrast by model seed and by arc."""
    protocol, _, _ = for_farm(farm, None, folder_of(configs))
    return contrasts(
        data,
        runs,
        study,
        farm,
        MODEL,
        protocol.seeds.model,
        TESTED,
        DESCRIBED,
        configs,
        breakdown=True,
    )
