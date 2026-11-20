"""fitting jobs: fit models on one split of one farm, predict its test rows, record the fit."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import config
from windfold.config import folder_of, for_farm
from windfold.locations import (
    automl_file,
    budget_file,
    chosen_file,
    curve_file,
    kernel_folder,
    manifest_file,
    prediction_files,
    prepared_folder,
    pretraining_files,
)
from windfold.models.interface import FitContext, Fitted, Model
from windfold.runner.worker import allotment
from windfold.schemas import PREDICTIONS, conform
from windfold.smearing import smearing_factors
from windfold.splits import first_days

TRAIN_ROLES = ("train", "fit")
TEST_ROLES = ("test", "calibrate")


def split_rows(
    rows: pl.DataFrame, manifest: pl.DataFrame, split: str, days: int | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """the prepared rows a split trains and tests on, each timestamp's rows together, in order."""
    chosen = manifest.filter(pl.col("split") == split)
    if chosen.is_empty():
        raise ValueError(f"split {split} is not in the manifest")
    parts = []
    for roles in (TRAIN_ROLES, TEST_ROLES):
        stamps = chosen.filter(pl.col("role").is_in(roles))["timestamp"].sort()
        stamps = first_days(stamps, days)
        part = rows.join(pl.DataFrame({"timestamp": stamps}), on="timestamp", how="semi")
        if part.is_empty():
            raise ValueError(f"split {split} has no prepared rows for roles {roles}")
        parts.append(part.sort("timestamp", "turbine_index"))
    return parts[0], parts[1]


def predicted(fitted: Fitted, rows: pl.DataFrame) -> pl.DataFrame:
    frame = fitted.predict(rows)
    if frame.height != rows.height or not {"mean", "variance"} <= set(frame.columns):
        raise ValueError(f"predict returned {frame.shape} with {frame.columns} for {rows.height}")
    support = pl.col("support") if "support" in frame.columns else pl.lit(None)
    frame = frame.select(
        pl.col("mean").cast(pl.Float64),
        pl.col("variance").cast(pl.Float64),
        support.cast(pl.Int64).alias("support"),
    )
    if frame["mean"].null_count() or not np.isfinite(frame["mean"].to_numpy()).all():
        raise ValueError("predict returned a mean that is missing or not finite")
    return frame


def smearing(fitted: Fitted, train: pl.DataFrame, protocol: config.Protocol) -> pl.DataFrame:
    mean_only = getattr(fitted, "predict_mean", None)
    if callable(mean_only):
        mean = np.asarray(mean_only(train), dtype=np.float64)
    else:
        mean = predicted(fitted, train)["mean"].to_numpy()
    if mean.shape != (train.height,) or not np.isfinite(mean).all():
        raise ValueError("the training means must be finite, one per training row")
    m = train["valid_turbines"].cast(pl.Float64).to_numpy()
    residual = m / (m - 1) * (train["y"].to_numpy() - mean)
    return smearing_factors(
        residual,
        train["speed_bin"].to_numpy(),
        protocol.smearing.min_rows_per_bin,
        len(protocol.processing.speed_bin_edges_ms) - 1,
    )


def fit_one(
    model: Model,
    train: pl.DataFrame,
    test: pl.DataFrame,
    context: FitContext,
    protocol: config.Protocol,
) -> tuple[pl.DataFrame, dict[str, Any], Fitted]:
    start = time.perf_counter()
    fitted = model.fit(train, context)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    frame = predicted(fitted, test)
    predict_seconds = time.perf_counter() - start
    predictions = conform(
        pl.concat([test.select("timestamp", "turbine", "y", "waked"), frame], how="horizontal"),
        PREDICTIONS,
    )
    facts = getattr(fitted, "facts", None)
    record = {
        "model": model.name,
        "seed": context.seed,
        "threads": context.threads,
        "device": context.device,
        "train_rows": train.height,
        "train_timestamps": train["timestamp"].n_unique(),
        "test_rows": test.height,
        "test_timestamps": test["timestamp"].n_unique(),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "smearing": smearing(fitted, train, protocol).to_dicts(),
        "facts": facts() if callable(facts) else {},
    }
    return predictions, record, fitted


def write_fit(
    out_dir: Path, outputs: tuple[str, str], predictions: pl.DataFrame, record: dict[str, Any]
) -> None:
    table, note = (out_dir / path for path in outputs)
    table.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(table, compression="zstd")
    note.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")


@dataclass(frozen=True)
class SplitData:
    protocol: config.Protocol
    farm: config.FarmConfig
    turbine: config.TurbineConfig
    train: pl.DataFrame
    test: pl.DataFrame
    prepared_dir: Path
    kernel_dir: Path


def load_split(
    study: str,
    farm: str,
    split: str,
    sensitivity: str | None,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
) -> SplitData:
    folder = folder_of(configs)
    protocol, farm_config, turbine = for_farm(farm, sensitivity, folder)
    limits = config.load_study(study, folder).limits
    prepared = Path(data_dir) / "prepared"
    rows = pl.read_parquet(prepared / prepared_folder(farm, sensitivity, protocol) / "rows.parquet")
    manifest = pl.read_parquet(Path(runs_dir) / manifest_file(study, farm, sensitivity, protocol))
    train, test = split_rows(rows, manifest, split, limits.days if limits else None)
    return SplitData(
        protocol,
        farm_config,
        turbine,
        train,
        test,
        prepared / prepared_folder(farm, sensitivity, protocol),
        prepared / kernel_folder(farm, sensitivity, protocol),
    )


def fit_engine(
    study: str,
    farm: str,
    split: str,
    models: list[str],
    sensitivity: str | None,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """every engine model in models on one split, one after another; engine fits are
    deterministic, so each uses the first model seed."""
    from windfold.models.registry import build_model

    data = load_split(study, farm, split, sensitivity, data_dir, runs_dir, configs)
    specs = config.load_models(folder_of(configs))
    threads, device = allotment()
    context = FitContext(
        farm, data.prepared_dir, data.protocol.seeds.model[0], threads, device, data.kernel_dir
    )
    for name in models:
        model = build_model(specs[name], data.farm, data.turbine, data.protocol)
        predictions, record, _ = fit_one(model, data.train, data.test, context, data.protocol)
        files = prediction_files(study, sensitivity, farm, name, split, None)
        write_fit(out_dir, files, predictions, record)


@dataclass(frozen=True)
class LearnerFit:
    data: SplitData
    spec: config.ModelSpec
    smoke: config.Limits | None
    context: FitContext
    runs: Path


def learner_fit(
    study: str,
    farm: str,
    model: str,
    split: str,
    seed: int | None,
    sensitivity: str | None,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
) -> LearnerFit:
    folder = folder_of(configs)
    data = load_split(study, farm, split, sensitivity, data_dir, runs_dir, configs)
    spec = config.load_models(folder)[model]
    smoke = config.load_study(study, folder).limits
    threads, device = allotment()
    used = data.protocol.seeds.model[0] if seed is None else seed
    context = FitContext(farm, data.prepared_dir, used, threads, device, data.kernel_dir)
    return LearnerFit(data, spec, smoke, context, Path(runs_dir))


def configuration_of(fit: LearnerFit, configs: str | None) -> dict[str, Any]:
    """the configuration the farm's tuning chose, or the menu's first in the smoke study."""
    folder = folder_of(configs)
    learner = fit.spec.learner
    if learner is None:
        raise ValueError(f"{fit.spec.name} has no learner")
    if fit.smoke is not None:
        return dict(config.load_menu(learner, folder)[0])
    source = config.tuned_model(fit.spec.name, config.load_models(folder))
    chosen = json.loads((fit.runs / chosen_file(fit.context.farm, source)).read_text())
    return dict(chosen["configuration"])


def finish_fit(
    fit: LearnerFit,
    model: Model,
    out_dir: Path,
    names: tuple[str, str, str, int | None, str | None],
    configuration: dict[str, Any] | None,
) -> Fitted:
    study, farm, split, seed, sensitivity = names
    predictions, record, fitted = fit_one(
        model, fit.data.train, fit.data.test, fit.context, fit.data.protocol
    )
    record["configuration"] = configuration
    files = prediction_files(study, sensitivity, farm, fit.spec.name, split, seed)
    write_fit(out_dir, files, predictions, record)
    return fitted


def fit_gbm(
    study: str,
    farm: str,
    model: str,
    split: str,
    seed: int | None,
    sensitivity: str | None,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    from windfold.models import gbm

    fit = learner_fit(study, farm, model, split, seed, sensitivity, data_dir, runs_dir, configs)
    configuration = configuration_of(fit, configs)
    built = gbm.build_model(
        fit.spec, fit.data.farm, fit.data.turbine, fit.data.protocol, configuration=configuration
    )
    finish_fit(fit, built, out_dir, (study, farm, split, seed, sensitivity), configuration)


def fit_graph(
    study: str,
    farm: str,
    model: str,
    split: str,
    seed: int | None,
    sensitivity: str | None,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """a graph network; a pretrained one starts from its farm's pretrained weights, or in the
    smoke study from an untrained network of the same architecture."""
    import torch

    from windfold.models import graph

    fit = learner_fit(study, farm, model, split, seed, sensitivity, data_dir, runs_dir, configs)
    configuration = configuration_of(fit, configs)
    protocol, turbine = fit.data.protocol, fit.data.turbine
    initial = None
    if fit.spec.pretrained and fit.smoke is not None:
        plain = fit.spec.model_copy(update={"pretrained": False})
        untrained = graph.build_model(
            plain, fit.data.farm, turbine, protocol, configuration=configuration
        )
        initial = {k: v.detach().cpu() for k, v in untrained.network().state_dict().items()}
    elif fit.spec.pretrained:
        weights = fit.runs / pretraining_files(farm)[0]
        initial = torch.load(weights, map_location="cpu", weights_only=True)
    built = graph.build_model(
        fit.spec, fit.data.farm, turbine, protocol, configuration=configuration, initial=initial
    )
    fitted = finish_fit(fit, built, out_dir, (study, farm, split, seed, sensitivity), configuration)
    curve = getattr(fitted, "curve", None)
    if not isinstance(curve, pl.DataFrame):
        raise ValueError(f"{model} gave no training curve")
    target = out_dir / curve_file(study, sensitivity, farm, model, split, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    curve.write_parquet(target)


def fit_automl(
    study: str,
    farm: str,
    model: str,
    split: str,
    seed: int | None,
    sensitivity: str | None,
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """peer_automl with its farm's budget, or the smoke study's; its work folder sits beside
    out_dir, inside the job's staging folder, and is removed once the record is written."""
    from windfold.models import automl

    fit = learner_fit(study, farm, model, split, seed, sensitivity, data_dir, runs_dir, configs)
    if fit.smoke is not None and fit.smoke.automl_minutes is not None:
        minutes = fit.smoke.automl_minutes
    else:
        minutes = float(json.loads((fit.runs / budget_file(farm)).read_text())["minutes"])
    built = automl.build_model(
        fit.spec,
        fit.data.farm,
        fit.data.turbine,
        fit.data.protocol,
        minutes=minutes,
        workdir=out_dir.parent / "automl",
    )
    fitted = finish_fit(fit, built, out_dir, (study, farm, split, seed, sensitivity), None)
    finish = getattr(fitted, "finish", None)
    if not callable(finish):
        raise ValueError(f"{model} cannot write its automl record")
    finish(out_dir / automl_file(study, sensitivity, farm, split))
