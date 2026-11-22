"""the accuracy of the two models fitted outside the runner for the tests set before the
confirmation period was scored, the combined GBM (hybrid_gbm) and the tighter correction
(windfold_restricted): mse, skill against the no-wake model, mean absolute kW error and crps on
waked rows of the single arcs, from the farm's planned bootstrap, as the study's accuracy table
scores the runner's models."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from windfold import config
from windfold.config import folder_of, for_farm
from windfold.evaluation import bootstrap, summaries
from windfold.evaluation.results import (
    KEYS,
    QUANTITIES,
    REFERENCE,
    bootstrap_windows,
    study_windows,
)
from windfold.secondary.contrasts import confirmatory_ready, model_errors, single_arcs
from windfold.tasks.evaluation import read_reductions

SEEDED = {"hybrid_gbm": True, "windfold_restricted": False}


def scored(
    data: Path,
    runs: Path,
    study: str,
    farm: str,
    models: Sequence[str],
    configs: str | None = None,
) -> pl.DataFrame:
    """the accuracy rows of the named models on the farm's single arcs; a model the runner
    fitted is read from the reductions, the others from their own predictions."""
    folder = folder_of(configs)
    protocol, farm_config, _ = for_farm(farm, None, folder)
    statistics = protocol.statistics
    arcs = single_arcs(runs, study, farm, protocol)
    official = read_reductions(runs, study, farm, None)["errors"].filter(
        pl.col("model").is_in([*models, REFERENCE]) & pl.col("split").is_in(arcs)
    )
    own = [
        model_errors(
            data,
            runs,
            study,
            farm,
            m,
            protocol.seeds.model if SEEDED[m] else [None],
            arcs,
            protocol,
        )
        for m in models
        if m in SEEDED
    ]
    windows = bootstrap_windows(
        study_windows(farm_config, config.load_study(study, folder).period),
        statistics.block_days,
        statistics.replicates,
        len(protocol.seeds.model),
        statistics.interval,
        statistics.min_finite,
        bootstrap.farm_seed(protocol.seeds.study, farm),
    )
    totals = bootstrap.block_totals(
        pl.concat([official, *own]),
        KEYS,
        QUANTITIES,
        windows.origins["period"][0],
        statistics.block_days,
        windows.blocks["period"],
    )
    return summaries.accuracy(
        totals, windows.draws["period"], {("arc_single", None): arcs}, list(models), REFERENCE
    ).filter(pl.col("rows") == "waked")


def secondary_accuracy(
    data: Path, runs: Path, study: str, farms: Sequence[str], configs: str | None = None
) -> dict[str, Any]:
    confirmatory_ready(runs, study, configs)
    found: dict[str, Any] = {}
    for farm in farms:
        table = scored(data, runs, study, farm, list(SEEDED), configs)
        found[farm] = {
            model: {
                row["metric"]: {k: row[k] for k in ("estimate", "lower", "upper")}
                for row in table.filter(pl.col("model") == model).iter_rows(named=True)
            }
            for model in SEEDED
        }
    return found
