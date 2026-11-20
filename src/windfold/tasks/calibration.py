"""calibration jobs: tune the menus, choose each farm's configurations, pretrain the graph
and choose the AutoGluon budget, all on calibration windows, never on tested data.

each job is called as job(**args, out_dir=staging) and writes every output at
out_dir/<path relative to the runs folder>, at the paths windfold.locations names. a job imports
only the learner it runs, so a change to one learner reruns only that learner's jobs.
"""

import json
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import config
from windfold.config import folder_of, for_farm
from windfold.locations import (
    budget_file,
    chosen_file,
    prepared_folder,
    pretraining_files,
    search_file,
    tuning_file,
)
from windfold.models import menus
from windfold.models.interface import FitContext, Fitted, Model
from windfold.runner.worker import allotment

SEARCH_COLUMNS = ("index", "mse", "waked_mse", "fit_seconds", "fit_rows", "validation_rows")


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def calibration_rows(prepared_dir: Path) -> pl.DataFrame:
    rows = pl.read_parquet(prepared_dir / "rows.parquet")
    return rows.filter(pl.col("period") == "calibration").sort("timestamp", "turbine_index")


def tuning(
    learner: str,
    build: Callable[..., Model],
    farm: str,
    model: str,
    index: int,
    data_dir: str,
    configs: str | None,
) -> tuple[dict[str, Any], Fitted]:
    """one menu configuration of a learner's model, fitted on the first part of the farm's
    calibration window and scored on its last part (item 29 of the study plan)."""
    folder = folder_of(configs)
    protocol, farm_config, turbine = for_farm(farm, None, folder)
    window = farm_config.windows.calibration
    if window is None:
        raise ValueError(f"{farm} has no calibration window to tune on")
    spec = config.load_models(folder)[model]
    if spec.learner != learner or spec.pretrained:
        raise ValueError(f"{model} is not a {learner} model with a search of its own")
    configuration = config.load_menu(learner, folder)[index]
    prepared = Path(data_dir) / "prepared" / prepared_folder(farm, None, protocol)
    split = menus.calibration_split(
        calibration_rows(prepared),
        window.start,
        window.end,
        protocol.splits.tuning_validation_fraction,
        protocol.splits.tuning_embargo_days,
    )
    threads, device = allotment()
    context = FitContext(farm, prepared, protocol.seeds.model[0], threads, device)
    built = build(spec, farm_config, turbine, protocol, configuration=configuration)
    row, fitted = menus.score(built, split, context)
    record = {
        "farm": farm,
        "model": model,
        "index": index,
        "configuration": configuration,
        **row,
        "split": split.record,
        "seed": context.seed,
        "threads": threads,
        "device": device,
    }
    return record, fitted


def tune_gbm(
    farm: str,
    model: str,
    index: int,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """menu configuration index of a LightGBM model on the farm's calibration split."""
    from windfold.models import gbm

    record, _ = tuning("gbm", gbm.build_model, farm, model, index, data_dir, configs)
    write_json(out_dir / tuning_file(farm, model, index), record)


def tune_graph(
    farm: str,
    model: str,
    index: int,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """menu configuration index of a graph network on the farm's calibration split, with the
    training loss of each round."""
    from windfold.models import graph

    record, fitted = tuning("graph", graph.build_model, farm, model, index, data_dir, configs)
    if not isinstance(fitted, graph.FittedGraph):
        raise TypeError(f"{model} gave no graph fit")
    record["training_loss"] = fitted.curve["loss"].to_list()
    write_json(out_dir / tuning_file(farm, model, index), record)


def choose(
    farm: str,
    model: str,
    other_farms: list[str],
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """the configuration a farm uses: the lowest validation mse of its own search or, for a
    farm without a calibration window, the best mean rank over other_farms' searches; ties go
    to the earliest in the menu."""
    folder = folder_of(configs)
    _, farm_config, _ = for_farm(farm, None, folder)
    specs = config.load_models(folder)
    source = config.tuned_model(model, specs)
    if source != model:
        raise ValueError(f"{model} takes the choice of {source}; it has no search of its own")
    menu = config.load_menu(specs[model].learner or "", folder)
    runs = Path(runs_dir)
    if farm_config.windows.calibration is not None:
        scored = [read_json(runs / tuning_file(farm, model, i)) for i in range(len(menu))]
        search = pl.DataFrame([{key: row[key] for key in SEARCH_COLUMNS} for row in scored])
        index = menus.choose(search["mse"].to_list())
        evidence: dict[str, Any] = {
            "rule": "lowest validation mse on the calibration split, ties to the earliest",
            "mse": search["mse"][index],
            "split": scored[index]["split"],
        }
    else:
        searches = {
            other: pl.read_parquet(runs / search_file(other, model)) for other in other_farms
        }
        mse = {other: table.sort("index")["mse"].to_list() for other, table in searches.items()}
        index = menus.best_mean_rank(list(mse.values()))
        columns: dict[str, Any] = {"index": list(range(len(menu)))}
        for other, values in mse.items():
            columns[f"mse_{other}"] = values
            columns[f"rank_{other}"] = menus.ranks(values)
        search = pl.DataFrame(columns).with_columns(
            mean_rank=pl.mean_horizontal(pl.col(f"rank_{other}" for other in mse))
        )
        evidence = {
            "rule": "best mean rank over the other farms' searches, ties to the earliest",
            "farms": other_farms,
            "mean_rank": search["mean_rank"][index],
        }
    target = out_dir / search_file(farm, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    search.write_parquet(target)
    write_json(
        out_dir / chosen_file(farm, model),
        {
            "farm": farm,
            "model": model,
            "index": index,
            "configuration": menu[index],
            "evidence": evidence,
        },
    )


def pretrain(farm: str, data_dir: str, runs_dir: str, configs: str | None, out_dir: Path) -> None:
    """pretrained_graph's starting weights (item 25 of the study plan): the wind-frame graph with
    the architecture its tuning chose, trained from the first model seed for protocol.graph
    pretrain_rounds rounds on the teacher's PyWake contrasts over whole degrees, the speed-bin
    centers and the calibration window's operating sets plus all turbines operating."""
    import torch

    from windfold import physics
    from windfold.models import graph
    from windfold.models.farm import read_farm

    folder = folder_of(configs)
    protocol, farm_config, turbine = for_farm(farm, None, folder)
    specs = config.load_models(folder)
    (name,) = [n for n, spec in specs.items() if spec.pretrained]
    source = config.tuned_model(name, specs)
    chosen = read_json(Path(runs_dir) / chosen_file(farm, source))
    network = graph.build_model(
        specs[source], farm_config, turbine, protocol, configuration=chosen["configuration"]
    )
    prepared = Path(data_dir) / "prepared" / prepared_folder(farm, None, protocol)
    fit = read_json(prepared / "physics_fit.json")
    rows = pl.read_parquet(prepared / "rows.parquet")
    layout = read_farm(prepared, turbine.rotor_diameter_m, rows)
    masks = np.empty(0, dtype=np.int64)
    if farm_config.windows.calibration is not None:
        calibration = rows.filter(pl.col("period") == "calibration")
        masks = calibration["operating_mask"].unique().to_numpy().astype(np.int64)
    sets = graph.operating_sets(masks, layout.east.size)
    centres = np.asarray(protocol.processing.speed_bin_centres_ms)
    keys, degrees, bins, valid = graph.pretraining_cases(sets, np.arange(360), len(centres))
    threads, device = allotment()
    teacher, parameters = fit["teacher"], fit["teacher_parameters"]
    start = time.perf_counter()
    with physics.worker_pool(threads) as pool:
        phi = physics.phi_keys(
            physics.wake_model(teacher, parameters, physics.pywake_turbine(turbine)),
            layout.east,
            layout.north,
            layout.hub_height,
            keys,
            degrees,
            bins,
            centres,
            parameters[physics.SPREAD],
            protocol.physics.truncate_sd,
            protocol.physics.power_floor_fraction * turbine.rated_power_kw * 1000.0,
            pool,
        )
    physics_seconds = time.perf_counter() - start
    cases = (degrees.astype(np.float64), centres[bins], valid, graph.contrasts(phi, valid))
    context = FitContext(farm, prepared, protocol.seeds.model[0], threads, device)
    fitted = graph.pretrain(network, cases, protocol.graph.pretrain_rounds, context)
    weights, curve, note = (out_dir / path for path in pretraining_files(farm))
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted.state(), weights)
    fitted.curve.write_parquet(curve)
    write_json(
        note,
        {
            "farm": farm,
            "model": name,
            "configuration_of": source,
            "configuration_index": chosen["index"],
            "teacher": teacher,
            "teacher_parameters": parameters,
            "operating_sets": len(sets),
            "calibration_operating_sets": len(masks),
            "cases": len(keys),
            "speed_centres_ms": centres.tolist(),
            "rounds": protocol.graph.pretrain_rounds,
            "steps_per_round": protocol.graph.steps_per_round,
            "seed": context.seed,
            "device": device,
            "threads": threads,
            "physics_seconds": physics_seconds,
            "training_seconds": fitted.seconds,
            "final_loss": fitted.curve["loss"][-1],
        },
    )


def majority(minutes: Mapping[str, float]) -> float:
    """the budget most farms chose; a tie goes to the shorter, the default of the budget rule."""
    counts = Counter(minutes.values())
    return min(counts, key=lambda value: (-counts[value], value))


def budget(
    farm: str,
    other_farms: list[str],
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """the farm's AutoGluon budget from its calibration window (item 30 of the study plan), or, for
    a farm without one, the budget most of other_farms chose."""
    from windfold.models import automl

    folder = folder_of(configs)
    protocol, farm_config, _ = for_farm(farm, None, folder)
    window = farm_config.windows.calibration
    if window is None:
        chosen = {
            other: read_json(Path(runs_dir) / budget_file(other))["minutes"]
            for other in other_farms
        }
        result: dict[str, Any] = {
            "farm": farm,
            "minutes": majority(chosen),
            "rule": "the budget most other farms chose, ties to the shorter",
            "farms": chosen,
        }
    else:
        (spec,) = [s for s in config.load_models(folder).values() if s.learner == "automl"]
        prepared = Path(data_dir) / "prepared" / prepared_folder(farm, None, protocol)
        threads, device = allotment()
        result = automl.choose_budget(
            calibration_rows(prepared),
            window=(window.start, window.end),
            validation_fraction=protocol.splits.tuning_validation_fraction,
            embargo_days=protocol.splits.tuning_embargo_days,
            candidate_minutes=protocol.automl_budget.candidate_minutes,
            min_relative_gain=protocol.automl_budget.min_relative_gain,
            features=spec.features,
            settings=protocol.automl,
            workdir=out_dir / "work",
            context=FitContext(farm, prepared, protocol.seeds.model[0], threads, device),
        )
    write_json(out_dir / budget_file(farm), result)
