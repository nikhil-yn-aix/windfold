"""windfold's gain on held-out arcs at the scale a practitioner sees: per farm of a study, on the
waked rows of its single arcs, windfold against pywake_inflow and against the learned comparison
model of least mse, every model that learns from data but windfold's variants.

the levels, each model's mse and mean absolute power error in kw with their intervals, are the
farm's official accuracy table's, and per arc its arcs table's mse. the contrasts, the other model
minus windfold so that positive favors windfold, use the planned bootstrap of the secondary tests
(secondary.contrasts): the mse reduction, the relative mse reduction, the skill difference in points
of no_wake's mse (the units of the confirmation contrasts) and the mean absolute error reduction in
kw, over the arcs pooled and against pywake_inflow per arc. the cost: the runner's single-arc jobs
of every model family on the farm, their wall time and peak memory as jobs.jsonl records them
(logs/run.log prints the same lines) and the fit and predict seconds in the compared models' fit
records. one engine job fits every engine model of its split, windfold and pywake_inflow among them,
so its memory is theirs together. confirmatory data are read only after the second freeze.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import config
from windfold.config import for_farm
from windfold.evaluation import bootstrap, hypotheses, paths
from windfold.evaluation.hypotheses import pooled_mse
from windfold.evaluation.results import KEYS, bootstrap_windows, study_windows
from windfold.evaluation.summaries import Draw
from windfold.locations import prediction_files
from windfold.secondary.contrasts import confirmatory_ready, single_arcs

MODEL, PHYSICS, REFERENCE = "windfold", "pywake_inflow", "no_wake"
NOT_LEARNED = ("reference", "physics", "windfold")
FAMILIES = ("engine", "gbm", "graph", "automl", "pretrained_graph")
QUANTITIES = ["sse", "rows", "abs_error_kw"]

type Endpoint = Callable[[np.ndarray, np.ndarray | None], np.ndarray]


def learned(configs: Path | None = None) -> list[str]:
    """the comparison models that learn from data: all but the reference, the physics alone and
    windfold's variants."""
    models = config.load_models(configs)
    return [name for name, spec in models.items() if spec.representation not in NOT_LEARNED]


def official(runs: Path, study: str, farm: str, name: str) -> pl.DataFrame:
    return pl.read_parquet(runs / paths.results_folder(study, None) / farm / f"{name}.parquet")


def summary(row: dict[str, Any]) -> dict[str, float]:
    return {"estimate": row["estimate"], "low": row["lower"], "high": row["upper"]}


def endpoints(totals: bootstrap.Totals, arcs: Sequence[str], other: str) -> dict[str, Endpoint]:
    """the other model minus windfold over the arcs, replicate by replicate."""

    def mean(model: str, quantity: str = "sse") -> Endpoint:
        return lambda c, s: pooled_mse(totals, c, s, model, arcs, quantity=quantity)

    theirs, own, base = mean(other), mean(MODEL), mean(REFERENCE)
    theirs_kw, own_kw = mean(other, "abs_error_kw"), mean(MODEL, "abs_error_kw")
    return {
        "mse_reduction": lambda c, s: theirs(c, s) - own(c, s),
        "relative_mse_reduction_percent": lambda c, s: 100 * (1 - own(c, s) / theirs(c, s)),
        "skill_points": lambda c, s: 100 * (theirs(c, s) - own(c, s)) / base(c, s),
        "mae_reduction_kw": lambda c, s: theirs_kw(c, s) - own_kw(c, s),
    }


def contrasts(
    totals: bootstrap.Totals,
    draw: Draw,
    arcs: Sequence[str],
    other: str,
    statistics: config.Statistics,
) -> dict[str, dict[str, Any]]:
    """each endpoint's estimate, 95% interval and one-sided p, as the secondary tests give them."""
    components: dict[str, hypotheses.Component] = {
        name: hypotheses.attempt(lambda e=endpoint: hypotheses.estimate(e, draw.counts, draw.seeds))
        for name, endpoint in endpoints(totals, arcs, other).items()
    }
    rows = hypotheses.contrast_rows(components, statistics.interval, statistics.min_finite)
    return {row["component"]: summary(row) | {"p": row["p"]} for row in rows.iter_rows(named=True)}


def spread(values: pl.Series) -> dict[str, float]:
    return {"median": float(np.median(values.to_numpy())), "max": float(values.to_numpy().max())}


def costs(
    runs: Path, study: str, farm: str, arcs: Sequence[str], seeds: dict[str, list[int | None]]
) -> dict[str, Any]:
    """each model family's single-arc jobs on the farm, their wall time and peak memory and each
    compared model's fit and predict seconds and threads over its single-arc fits."""
    jobs = (
        pl.read_ndjson(runs / "jobs.jsonl", infer_schema_length=None)
        .filter(
            pl.col("status") == "done",
            pl.col("job").str.starts_with(f"{study}/"),
            ~pl.col("job").str.contains("/sensitivities/"),
            pl.col("farm") == farm,
            pl.col("family").is_in(FAMILIES),
            pl.col("split").is_in(list(arcs)),
        )
        .sort("time")
        .unique("job", keep="last")
    )
    families = {
        family: {
            "jobs": part.height,
            "gpu_jobs": int(part["device"].str.contains("gpu").sum()),
            "seconds": spread(part["seconds"]),
            "peak_memory_gb": spread(part["peak_memory_gb"]),
        }
        for (family,), part in jobs.sort("family").group_by("family", maintain_order=True)
    }
    models = {}
    for model, chosen in seeds.items():
        files = [prediction_files(study, None, farm, model, a, s)[1] for a in arcs for s in chosen]
        records = pl.DataFrame(
            [
                {key: record[key] for key in ("fit_seconds", "predict_seconds", "threads")}
                for record in (json.loads((runs / file).read_text()) for file in files)
            ]
        )
        models[model] = {
            "fits": records.height,
            "fit_seconds": spread(records["fit_seconds"]),
            "predict_seconds": spread(records["predict_seconds"]),
            "threads": sorted(records["threads"].unique().to_list()),
        }
    return {"families": families, "models": models}


def farm_scale(runs: Path, study: str, farm: str, configs: Path | None = None) -> dict[str, Any]:
    folder = str(configs) if configs else None
    confirmatory_ready(runs, study, folder)
    protocol, farm_config, _ = for_farm(farm, None, configs)
    statistics = protocol.statistics
    arcs = single_arcs(runs, study, farm, protocol)
    accuracy = official(runs, study, farm, "accuracy").filter(
        pl.col("regime") == "arc_single", pl.col("rows") == "waked"
    )
    mse = accuracy.filter(pl.col("metric") == "mse")
    strongest = mse.filter(pl.col("model").is_in(learned(configs))).sort("estimate")["model"][0]
    compared = (MODEL, PHYSICS, strongest)
    errors = (
        pl.scan_parquet(runs / paths.reduction_file(study, farm, None, "errors"))
        .filter(pl.col("model").is_in([*compared, REFERENCE]), pl.col("split").is_in(list(arcs)))
        .collect()
    )
    windows = bootstrap_windows(
        study_windows(farm_config, config.load_study(study, configs).period),
        statistics.block_days,
        statistics.replicates,
        len(protocol.seeds.model),
        statistics.interval,
        statistics.min_finite,
        bootstrap.farm_seed(protocol.seeds.study, farm),
    )
    totals = bootstrap.block_totals(
        errors,
        KEYS,
        QUANTITIES,
        windows.origins["period"][0],
        statistics.block_days,
        windows.blocks["period"],
    )
    draw = windows.draws["period"]
    waked = (
        errors.filter(pl.col("model") == MODEL, pl.col("waked"))
        .group_by("split")
        .agg(pl.col("rows").sum())
        .sort("split")
    )
    per_arc_mse = official(runs, study, farm, "arcs").filter(
        pl.col("model").is_in([MODEL, PHYSICS])
    )
    per_arc = []
    for split, rows in waked.iter_rows():
        levels = per_arc_mse.filter(pl.col("split") == split)
        per_arc.append(
            {
                "split": split,
                "waked_rows": rows,
                "mse": {r["model"]: summary(r) for r in levels.iter_rows(named=True)},
                f"against_{PHYSICS}": contrasts(totals, draw, [split], PHYSICS, statistics),
            }
        )
    seeds = {
        model: errors.filter(pl.col("model") == model)["seed"].unique().sort().to_list()
        for model in compared
    }
    return {
        "arcs": waked.height,
        "waked_rows": int(waked["rows"].sum()),
        "strongest_learned": strongest,
        "accuracy": {
            model: {
                r["metric"]: summary(r)
                for r in accuracy.filter(
                    pl.col("model") == model, pl.col("metric").is_in(["mse", "mae_kw"])
                ).iter_rows(named=True)
            }
            for model in compared
        },
        "against": {
            other: contrasts(totals, draw, arcs, other, statistics) for other in compared[1:]
        },
        "per_arc": per_arc,
        "cost": costs(runs, study, farm, arcs, seeds),
    }


def practical_scale(
    runs: Path, study: str, farms: Sequence[str] = (), configs: Path | None = None
) -> dict[str, Any]:
    """farm_scale over the farms given, or every farm of the study."""
    chosen = farms or config.load_study(study, configs).farms
    return {farm: farm_scale(runs, study, farm, configs) for farm in chosen}
