"""windfold without its local block and with a tighter declared kernel scale, on a study's single
arcs (item 59 of the study plan).

on a held-out arc the local cells have no training rows, so only the shared terms predict there.
fitted together with them, the local block takes relation-shaped structure the kernel would
otherwise carry (hodges and reich 2010), and the declared kernel scale lets the kernel fit
structure that does not recur in other sectors.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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

MODEL = "windfold_restricted"
BASE = "windfold_no_local"
KERNEL_PRIOR_VARIANCE = 0.04 / 64
TESTED = ("windfold", "pywake_inflow")
DESCRIBED = ("windfold_no_local",)


def restricted(protocol: config.Protocol) -> config.Protocol:
    """the protocol with the declared prior variance of the shared kernel tightened."""
    engine = protocol.engine.model_copy(update={"fixed_prior_variance": KERNEL_PRIOR_VARIANCE})
    return protocol.model_copy(update={"engine": engine})


def fit(
    data: Path,
    runs: Path,
    study: str,
    farm: str,
    split: str,
    configs: str | None = None,
) -> Path:
    """fit the restricted windfold on one single arc and write its predictions and record. a fit
    already written is kept while its record's provenance matches, and refused when it differs or
    is missing."""
    confirmatory_ready(runs, study, configs)
    folder = folder_of(configs)
    protocol, farm_config, turbine = for_farm(farm, None, folder)
    out = output_folder(runs, study, MODEL)
    table, note = prediction_names(farm, split, None)
    changed = restricted(protocol)
    spec = config.load_models(folder)[BASE]
    seed = protocol.seeds.model[0]
    inputs = fit_inputs(data, runs, study, farm, protocol)
    made = provenance(fit, inputs, changed, farm_config, turbine, spec, split, seed)
    refit = f"delete {out / 'predictions' / farm} to fit it again"
    if reusable(out / note, lambda: made, refit) and (out / table).is_file():
        return out / table
    model = build_model(spec, farm_config, turbine, changed)
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
    predictions, record, _ = fit_one(model, train, test, context, changed)
    record |= {"base": BASE, "kernel_prior_variance": KERNEL_PRIOR_VARIANCE, "provenance": made}
    write_fit(out, (table, note), predictions, record)
    return out / table


def fit_all(
    data: Path, runs: Path, study: str, farm: str, workers: int, configs: str | None = None
) -> list[Path]:
    """fit the restricted windfold on every included single arc, workers at a time."""
    confirmatory_ready(runs, study, configs)
    protocol, _, _ = for_farm(farm, None, folder_of(configs))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(fit, data, runs, study, farm, split, configs)
            for split in single_arcs(runs, study, farm, protocol)
        ]
        return [future.result() for future in futures]


def score(
    data: Path, runs: Path, study: str, farm: str, configs: str | None = None
) -> pl.DataFrame:
    """windfold and pywake_inflow minus the restricted windfold, tested with holm over the two;
    windfold_no_local minus it, described."""
    return contrasts(data, runs, study, farm, MODEL, [None], TESTED, DESCRIBED, configs)
