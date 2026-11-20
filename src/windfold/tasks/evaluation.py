"""evaluation jobs: reduce a farm's predictions, compute its results, decide across farms.

each job reads its inputs under data_dir and runs_dir, calls the evaluation modules and writes
its outputs at out_dir/<path under the runs folder>, at the paths windfold.evaluation.paths
gives the planner.
"""

import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import config
from windfold.config import folder_of, for_farm
from windfold.evaluation import bootstrap, kernel_transfer, paths, reductions, results, trial
from windfold.evaluation.hypotheses import CoverageConfirmation
from windfold.locations import kernel_folder, manifest_file, prepared_folder
from windfold.schemas import SPLIT_SUMMARY
from windfold.splits import timestamp_table


def write(out_dir: Path, path: str, frame: pl.DataFrame) -> None:
    target = out_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".csv"):
        frame.write_csv(target)
    else:
        frame.write_parquet(target, compression="zstd")


def second_freeze(runs: Path, study: str) -> dict[str, Any]:
    """the study's second freeze record; refuses when it is missing."""
    path = runs / paths.second_record_file(study)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; write the second freeze record before evaluating {study} data"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_bytes(runs: Path, path: str, hashes: Mapping[str, str] | None) -> bytes:
    """a file's bytes, checked against the second freeze record's sha256 when there is a
    record; a file the record does not list is refused."""
    data = (runs / path).read_bytes()
    if hashes is None:
        return data
    expected = hashes.get(path)
    if expected is None:
        raise ValueError(f"{path} is not in the second freeze record")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"{path} has sha256 {actual}; the second freeze record has {expected}")
    return data


def reduce_farm(
    study: str,
    farm: str,
    sensitivity: str | None,
    fits: list[list[Any]],
    failed: list[list[Any]],
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """every reduction a farm's results need, and the status of each planned fit.

    fits are the planned fits that ran, failed those that failed for good. for confirmatory
    data it refuses without the second freeze record and checks every prediction file's sha256
    against it before reading.
    """
    folder = folder_of(configs)
    protocol, _, turbine = for_farm(farm, sensitivity, folder)
    runs = Path(runs_dir)
    hashes = None
    study_config = config.load_study(study, folder)
    days = study_config.limits.days if study_config.limits else None
    if study_config.period == "confirmatory":
        hashes = second_freeze(runs, study)["predictions"]
    prepared_dir = Path(data_dir) / "prepared" / prepared_folder(farm, sensitivity, protocol)
    prepared = pl.read_parquet(prepared_dir / "rows.parquet")
    manifest_path = manifest_file(study, farm, sensitivity, protocol)
    manifest = pl.read_parquet(runs / manifest_path)
    summary = pl.read_csv(runs / paths.summary_file(manifest_path), schema=SPLIT_SUMMARY)
    regime_of = dict(summary.select("split", "regime").iter_rows())
    lost = {(model, split, seed) for model, split, seed in failed}
    members: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    for model, split, seed in [*fits, *failed]:
        members[split].append((model, seed))
    parts: dict[str, list[pl.DataFrame]] = defaultdict(list)
    status = []
    for split, chosen in members.items():
        regime = regime_of[split]
        here = reductions.window_rows(prepared, manifest, split, days)
        for model, seed in chosen:
            if (model, split, seed) in lost:
                status.append((model, split, seed, regime, "failed"))
                continue
            table, note = paths.fit_inputs(study, farm, sensitivity, model, split, seed, protocol)
            predictions = pl.read_parquet(io.BytesIO(frozen_bytes(runs, table, hashes)))
            record = json.loads(frozen_bytes(runs, note, hashes))
            found = reductions.fit_reductions(
                predictions,
                here,
                record,
                farm=farm,
                model=model,
                split=split,
                seed=seed,
                regime=regime,
                protocol=protocol,
            )
            for name, frame in found.items():
                parts[name].append(frame)
            status.append((model, split, seed, regime, "done"))
    reduced = reductions.finished(parts)
    reduced["fits"] = pl.DataFrame(status, schema=reductions.FITS, orient="row")
    reduced["splits"] = summary.filter(pl.col("split").is_in(list(members)))
    reduced["turbines"] = reductions.turbine_order(prepared)
    reduced["arcs"] = reductions.arc_table(
        reduced["splits"], manifest, prepared, prepared_dir, reduced["errors"], protocol, turbine
    )
    for name in paths.REDUCTIONS:
        write(out_dir, paths.reduction_file(study, farm, sensitivity, name), reduced[name])


def read_reductions(
    runs: Path, study: str, farm: str, sensitivity: str | None
) -> dict[str, pl.DataFrame]:
    return {
        name: pl.read_parquet(runs / paths.reduction_file(study, farm, sensitivity, name))
        for name in paths.REDUCTIONS
    }


def farm_results(
    study: str,
    farm: str,
    sensitivity: str | None,
    other_farms: list[str],
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """one farm's hypotheses and every result table, from its reductions and one bootstrap.

    the coverage map comes from the other farms' arc tables, or for confirmatory data from the
    second freeze record. a split sensitivity also reads the main study's reductions of the
    farm for the regimes it does not refit.
    """
    folder = folder_of(configs)
    protocol, farm_config, _ = for_farm(farm, sensitivity, folder)
    study_config = config.load_study(study, folder)
    runs = Path(runs_dir)
    source = paths.reduction_source(protocol, sensitivity)
    reduced = read_reductions(runs, study, farm, source)
    scope = results.scope_of(
        reduced["fits"], sensitivity, paths.statistics_only(protocol, sensitivity)
    )
    borrowed = paths.borrowed_regimes(protocol, sensitivity)
    if borrowed:
        main_protocol, _, _ = for_farm(farm, None, folder)
        reduced = results.combined(
            reduced,
            read_reductions(runs, study, farm, None),
            borrowed,
            scope,
            main_protocol.splits.arc_width_deg,
            protocol.splits.arc_width_deg,
        )
    if study_config.period == "confirmatory":
        frozen = second_freeze(runs, study)["coverage_confirmation"]
        confirmation = CoverageConfirmation(**frozen) if frozen else None
    else:
        arcs = [
            pl.read_parquet(runs / paths.reduction_file(study, other, source, "arcs"))
            for other in other_farms
        ]
        confirmation = results.confirmation_from_arcs(arcs, protocol)
    tables = results.farm_tables(
        reduced,
        farm=farm,
        protocol=protocol,
        farm_config=farm_config,
        study=study_config,
        scope=scope,
        confirmation=confirmation,
    )
    for name, path in zip(
        paths.RESULTS, paths.result_outputs(study, farm, sensitivity), strict=True
    ):
        write(out_dir, path, tables[name])


def read_farm_tables(
    runs: Path, study: str, farms: list[str], sensitivity: str | None, name: str
) -> pl.DataFrame:
    """one result table of every farm, each with its farm."""
    position = paths.RESULTS.index(name)
    frames = []
    for farm in farms:
        path = runs / paths.result_outputs(study, farm, sensitivity)[position]
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; compute the {study} farm results first")
        frame = pl.read_parquet(path)
        frames.append(frame if "farm" in frame.columns else frame.with_columns(farm=pl.lit(farm)))
    return pl.concat(frames, how="diagonal")


def study_decision(
    study: str,
    sensitivity: str | None,
    farms: list[str],
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """the decision rule and the cross-farm tables of one study.

    a development study (or sensitivity) gives each hypothesis's development part only. the
    confirmatory study adds its confirmatory part and the verdict, reading the development
    study's farm results, and refuses while any is missing.
    """
    folder = folder_of(configs)
    protocol = config.load_protocol(folder)
    period = config.load_study(study, folder).period
    runs = Path(runs_dir)
    combined = {
        name: read_farm_tables(runs, study, farms, sensitivity, name) for name in paths.DECISION[1:]
    }
    own = results.outcomes_of(combined["hypotheses"])
    if period == "confirmatory":
        development = config.load_study(paths.DEVELOPMENT_STUDY, folder)
        earlier = read_farm_tables(
            runs, development.name, list(development.farms), sensitivity, "hypotheses"
        )
        verdicts = results.decision_table(
            protocol.hypotheses, results.outcomes_of(earlier), own, protocol
        )
    else:
        verdicts = results.decision_table(protocol.hypotheses, own, None, protocol)
    outputs = dict(
        zip(
            (*paths.DECISION, "decision_csv", "hypotheses_csv"),
            paths.decision_outputs(study, sensitivity),
            strict=True,
        )
    )
    write(out_dir, outputs["decision"], verdicts)
    write(out_dir, outputs["decision_csv"], verdicts)
    write(out_dir, outputs["hypotheses_csv"], combined["hypotheses"])
    for name in paths.DECISION[1:]:
        write(out_dir, outputs[name], combined[name])


def trial_planning(
    study: str,
    farm: str,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """the label-free trial-planning illustration on one farm's direction record, with the
    toggle, durations and replicates of protocol.trial."""
    folder = folder_of(configs)
    protocol, farm_config, _ = for_farm(farm, None, folder)
    period = config.load_study(study, folder).period
    window = getattr(farm_config.windows, period)
    rows = pl.read_parquet(
        Path(data_dir) / "prepared" / prepared_folder(farm, None, protocol) / "rows.parquet"
    )
    table = timestamp_table(rows, period, window.start, window.end)
    first, days = bootstrap.window_days(window.start, window.end)
    block_days = protocol.statistics.block_days
    settings = protocol.trial
    result = trial.trial_support(
        table,
        first,
        bootstrap.block_count(first, first + timedelta(days=days), block_days),
        block_days=block_days,
        arc_count=protocol.splits.arc_count,
        toggle_minutes=settings.toggle_minutes,
        durations_days=list(settings.durations_days),
        min_rows=protocol.splits.arc_min_rows,
        min_days=protocol.splits.arc_min_days,
        replicates=settings.replicates,
        seed=bootstrap.farm_seed(protocol.seeds.study, farm),
    )
    write(out_dir, paths.trial_outputs(study, farm)[0], result)


def read_fit(
    runs: Path,
    study: str,
    farm: str,
    sensitivity: str | None,
    model: str,
    split: str,
    protocol: config.Protocol,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """one engine fit's predictions and record."""
    table, note = paths.fit_inputs(study, farm, sensitivity, model, split, None, protocol)
    record = json.loads((runs / note).read_text(encoding="utf-8"))
    return pl.read_parquet(runs / table), record


def kernel_transfer_table(
    study: str,
    farm: str,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """how windfold's shared kernel carries over to one farm's single- and opposite-arc splits,
    in the main analysis and in each sensitivity that refits windfold on the main data."""
    from windfold.models.registry import engine_settings, read_inputs
    from windfold.tasks.fitting import split_rows

    folder = folder_of(configs)
    protocol, farm_config, _ = for_farm(farm, None, folder)
    names = paths.transfer_sensitivities(protocol, config.load_study(study, folder).sensitivities)
    versions = [(None, protocol)] + [
        (name, config.with_sensitivity(name, protocol, {farm: farm_config})[0]) for name in names
    ]
    windfold = config.load_models(folder)["windfold"]
    runs, prepared = Path(runs_dir), Path(data_dir) / "prepared"
    rows = pl.read_parquet(prepared / prepared_folder(farm, None, protocol) / "rows.parquet")
    manifest_path = manifest_file(study, farm, None, protocol)
    manifest = pl.read_parquet(runs / manifest_path)
    summary = pl.read_csv(runs / paths.summary_file(manifest_path), schema=SPLIT_SUMMARY)
    arcs = summary.filter(
        pl.col("status") == "included", pl.col("regime").is_in(kernel_transfer.REGIMES)
    )
    records = []
    for split, regime in arcs.select("split", "regime").iter_rows():
        train, test = split_rows(rows, manifest, split, None)
        turbulence = {
            "test_turbulence": kernel_transfer.free_stream_turbulence(test),
            "train_turbulence": kernel_transfer.free_stream_turbulence(train),
        }
        reference, _ = read_fit(runs, study, farm, None, "no_wake", split, protocol)
        keys = reference.select("timestamp", "turbine", "y")
        y, waked = reference["y"].to_numpy(), reference["waked"].to_numpy()
        for sensitivity, changed in versions:
            fits = {
                model: read_fit(runs, study, farm, sensitivity, model, split, changed)
                for model in ("windfold", "windfold_no_kernel")
            }
            for model, (frame, _) in fits.items():
                if not frame.select("timestamp", "turbine", "y").equals(keys):
                    raise ValueError(
                        f"{farm} {split}: {model} of {sensitivity or 'the main analysis'} "
                        "is not on the no-wake model's rows"
                    )
            (with_kernel, record), (without_kernel, without_record) = fits.values()
            facts = record["facts"]
            design = read_inputs(
                with_kernel,
                prepared / prepared_folder(farm, sensitivity, changed),
                prepared / kernel_folder(farm, sensitivity, changed),
                windfold,
                engine_settings(changed).kernel,
            ).kernel
            assert design is not None
            part = kernel_transfer.kernel_part(
                design,
                np.asarray(facts["blocks"]["shared_kernel"], dtype=np.float64),
                with_kernel["timestamp"].to_physical().to_numpy(),
            )
            fixed = engine_settings(changed, windfold.scales).fixed_prior_variance
            physics = (
                facts["blocks"].get("physics"),
                without_record["facts"]["blocks"].get("physics"),
            )
            records.append(
                {
                    "split": split,
                    "regime": regime,
                    "sensitivity": sensitivity,
                    "kernel_scale": facts["scales"].get("shared_kernel", fixed),
                    "physics_with_kernel": physics[0][0] if physics[0] else None,
                    "physics_without_kernel": physics[1][0] if physics[1] else None,
                    **turbulence,
                    "waked_rows": int(waked.sum()),
                    "reference_sse": float(np.sum((y - reference["mean"].to_numpy())[waked] ** 2)),
                    **kernel_transfer.decompose(
                        y,
                        waked,
                        with_kernel["mean"].to_numpy(),
                        without_kernel["mean"].to_numpy(),
                        reference["mean"].to_numpy(),
                        part,
                    ),
                }
            )
    table = pl.DataFrame(records, schema=kernel_transfer.KERNEL_TRANSFER)
    write(out_dir, paths.kernel_transfer_outputs(study, farm)[0], table)


def second_record(runs_dir: str, configs: str | None) -> dict[str, Any]:
    """what the second freeze record stores from evaluation: the coverage map and the
    simple-predictor choice, fitted on the development arc tables of the confirmation farms.
    the map is none when those arcs cannot fit one (a single class)."""
    protocol = config.load_protocol(folder_of(configs))
    farms = list(protocol.endpoints.coverage_predicts_failure.confirmation_farms)
    runs = Path(runs_dir)
    arcs = [
        pl.read_parquet(runs / paths.reduction_file(paths.DEVELOPMENT_STUDY, farm, None, "arcs"))
        for farm in farms
    ]
    confirmation = results.confirmation_from_arcs(arcs, protocol)
    choice = (
        None
        if confirmation is None
        else {"name": confirmation.predictor, "direction": confirmation.direction}
    )
    return {
        "coverage_confirmation": None if confirmation is None else asdict(confirmation),
        "simple_predictor": choice,
        "confirmation_farms": farms,
    }
