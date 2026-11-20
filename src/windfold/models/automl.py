"""peer_automl: AutoGluon on same-time peer features, as the Hill of Towie challenge winner used.

it learns each row's leave-one-out peer contrast from the other valid turbines' log power and
scores on the centered scale by multiplying by (m - 1)/m. predictions are never centered across
turbines: each peer's prediction used the target's power, so centering would bring it back.
"""

import json
import os
import shutil
import time
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

from windfold.config import AutoML, FarmConfig, ModelSpec, Protocol, TurbineConfig
from windfold.models.interface import FitContext, check_whole_timestamps
from windfold.models.menus import calibration_split, score

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor

FEATURE_SETS = ("peer_power", "direction", "reference_speed", "turbine")
LABEL = "contrast"
FOLD = "fold"
EVAL_METRIC = "mean_squared_error"
FOLD_FITTING = "sequential_local"
LEADERBOARD = (
    "model",
    "model_type",
    "stack_level",
    "num_models",
    "score_val",
    "fit_time",
    "pred_time_val",
    "fit_time_marginal",
    "pred_time_val_marginal",
    "fit_order",
    "can_infer",
)


def peer_features(
    rows: pl.DataFrame, turbines: Sequence[str], features: Sequence[str]
) -> pl.DataFrame:
    """peer_automl's features, one row per input row, in input order.

    peer_power_<turbine> is that turbine's log power at the same timestamp when it is valid and
    is not the row's own turbine. the row's own power and anemometer never enter.
    """
    unknown = sorted(set(features) - set(FEATURE_SETS))
    if unknown:
        raise ValueError(f"unknown peer_automl features {unknown}; known: {list(FEATURE_SETS)}")
    check_whole_timestamps(rows)
    columns: list[pl.Expr] = []
    if "peer_power" in features:
        if rows.filter(pl.col("power_kw").is_null() | (pl.col("power_kw") <= 0)).height:
            raise ValueError("peer_power needs positive power_kw in every row")
        log_power = pl.col("power_kw").log()
        columns += [
            pl.when(pl.col("turbine") != turbine)
            .then(log_power.filter(pl.col("turbine") == turbine).first().over("timestamp"))
            .alias(f"peer_power_{turbine}")
            for turbine in turbines
        ]
    if "direction" in features:
        radians = pl.col("farm_direction").radians()
        columns += [radians.sin().alias("direction_sin"), radians.cos().alias("direction_cos")]
    if "reference_speed" in features:
        columns.append(pl.col("reference_speed"))
    if "turbine" in features:
        columns.append(pl.col("turbine").cast(pl.Enum(turbines), strict=False))
    return rows.select(columns)


def peer_contrast(rows: pl.DataFrame) -> pl.Series:
    """each row's leave-one-out peer contrast m/(m - 1) y: its log power minus its peers' mean."""
    m = pl.col("valid_turbines").cast(pl.Float64)
    return rows.select((m / (m - 1) * pl.col("y")).alias(LABEL)).to_series()


def centred(contrast: np.ndarray, valid_turbines: np.ndarray) -> np.ndarray:
    m = valid_turbines.astype(np.float64)
    return (m - 1) / m * contrast


def day_folds(timestamps: pl.Series, folds: int, seed: int) -> pl.Series:
    """a bagging fold per row: every UTC calendar day lies whole in one fold, days dealt evenly."""
    days = timestamps.dt.date()
    calendar = days.unique().sort()
    if calendar.len() < folds:
        raise ValueError(
            f"{folds} day-grouped folds need at least {folds} calendar days, got {calendar.len()}"
        )
    deal = pl.Series(np.random.default_rng(seed).permutation(calendar.len()) % folds)
    return days.replace_strict(calendar, deal, return_dtype=pl.Int64).alias(FOLD)


def portfolio(
    hyperparameters: str, models: Sequence[str], gpu_models: Sequence[str], gpus: int
) -> dict[str, list[dict[str, Any]]]:
    """the named AutoGluon portfolio cut to models, with the GPU given only to gpu_models."""
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    configs = get_hyperparameter_config(hyperparameters)
    missing = sorted(set(models) - set(configs))
    if missing:
        raise ValueError(f"portfolio {hyperparameters} has no {missing}; it has {sorted(configs)}")
    strays = sorted(set(gpu_models) - set(models))
    if strays:
        raise ValueError(f"gpu_models {strays} are not among models {list(models)}")
    chosen = {}
    for model in models:
        entries = configs[model] if isinstance(configs[model], list) else [configs[model]]
        share = gpus if model in gpu_models else 0
        chosen[model] = [
            {**entry, "ag_args_fit": {**entry.get("ag_args_fit", {}), "num_gpus": share}}
            for entry in entries
        ]
    return chosen


def gpu_count(device: str | None) -> int:
    if device in (None, "cpu"):
        return 0
    if device in ("cuda", "cuda:0"):
        return 1
    raise ValueError(
        f"AutoGluon takes GPUs by count, not index: start the job with CUDA_VISIBLE_DEVICES "
        f"set to the device and pass cuda:0, not {device}"
    )


def plain(frame: Any) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


class FittedPeerAutoml:
    def __init__(
        self,
        predictor: "TabularPredictor",
        turbines: tuple[str, ...],
        features: tuple[str, ...],
        record: dict[str, Any],
        folder: Path,
    ):
        self.predictor = predictor
        self.turbines = turbines
        self.features = features
        self.record = record
        self.folder = folder

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        start = time.perf_counter()
        table = peer_features(rows, self.turbines, self.features).to_pandas()
        contrast = np.asarray(self.predictor.predict(table), dtype=np.float64)
        mean = centred(contrast, rows["valid_turbines"].to_numpy())
        self.record["predict_rows"] += rows.height
        self.record["predict_seconds"] += time.perf_counter() - start
        return pl.DataFrame({"mean": mean}).with_columns(variance=pl.lit(None, dtype=pl.Float64))

    def finish(self, path: Path) -> None:
        """write the record to path, then remove this fit's work folder; predict stops working."""
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_text(json.dumps(self.record, indent=1), encoding="utf-8")
        os.replace(part, path)
        self.remove()

    def remove(self) -> None:
        shutil.rmtree(self.folder)
        if not any(self.folder.parent.iterdir()):
            self.folder.parent.rmdir()


class PeerAutoml:
    """AutoGluon on peer features with bagging folds grouped by calendar day.

    settings and minutes come from config. each fit works in a fresh folder under workdir;
    finish writes the record and removes it.
    """

    def __init__(
        self,
        name: str,
        *,
        features: Sequence[str],
        settings: AutoML,
        minutes: float,
        workdir: Path,
    ):
        self.name = name
        self.features = tuple(features)
        self.settings = settings
        self.minutes = minutes
        self.workdir = workdir

    def fit(self, train: pl.DataFrame, context: FitContext) -> FittedPeerAutoml:
        from autogluon.tabular import TabularPredictor

        gpus = gpu_count(context.device)
        turbines = tuple(
            train.select("turbine_index", "turbine").unique().sort("turbine_index")["turbine"]
        )
        usable = train.filter(pl.col("valid_turbines") >= 2)
        settings = self.settings
        folds = day_folds(usable["timestamp"], settings.bag_folds, context.seed)
        table = peer_features(usable, turbines, self.features)
        features = table.columns
        table = table.with_columns(peer_contrast(usable), folds)
        self.workdir.mkdir(parents=True, exist_ok=True)
        folder = Path(mkdtemp(prefix=f"{context.farm}-", dir=self.workdir))
        path = folder / "predictor"
        predictor = TabularPredictor(
            label=LABEL,
            problem_type="regression",
            eval_metric=EVAL_METRIC,
            path=str(path),
            groups=FOLD,
            learner_kwargs={"random_state": context.seed},
        )
        start = time.perf_counter()
        predictor.fit(
            table.to_pandas(),
            presets=settings.preset,
            hyperparameters=portfolio(
                settings.hyperparameters, settings.models, settings.gpu_models, gpus
            ),
            time_limit=self.minutes * 60,
            num_bag_folds=settings.bag_folds,
            num_stack_levels=settings.stack_levels,
            dynamic_stacking=False,
            use_bag_holdout=False,
            num_cpus=context.threads,
            num_gpus=gpus,
            ag_args_ensemble={
                "fold_fitting_strategy": FOLD_FITTING,
                "model_random_seed": context.seed,
            },
        )
        fit_seconds = time.perf_counter() - start
        failures = predictor.model_failures()
        record = {
            "autogluon": version("autogluon.tabular"),
            "preset": settings.preset,
            "hyperparameters": settings.hyperparameters,
            "models": list(settings.models),
            "gpu_models": list(settings.gpu_models),
            "eval_metric": EVAL_METRIC,
            "bag_folds": settings.bag_folds,
            "stack_levels": settings.stack_levels,
            "fold_fitting": FOLD_FITTING,
            "time_limit_seconds": self.minutes * 60,
            "seed": context.seed,
            "threads": context.threads,
            "gpus": gpus,
            "turbines": list(turbines),
            "features": features,
            "train_rows": usable.height,
            "train_days": usable["timestamp"].dt.date().n_unique(),
            "fit_seconds": fit_seconds,
            "best_model": predictor.model_best,
            "completed_models": predictor.model_names(),
            "leaderboard": plain(predictor.leaderboard(extra_info=True)[list(LEADERBOARD)]),
            "failures": plain(failures[["model", "exc_type", "total_time"]])
            if len(failures)
            else [],
            "predict_rows": 0,
            "predict_seconds": 0.0,
        }
        return FittedPeerAutoml(predictor, turbines, self.features, record, folder)


def choose_budget(
    calibration: pl.DataFrame,
    *,
    window: tuple[datetime, datetime],
    validation_fraction: float,
    embargo_days: float,
    candidate_minutes: Sequence[float],
    min_relative_gain: float,
    features: Sequence[str],
    settings: AutoML,
    workdir: Path,
    context: FitContext,
) -> dict[str, Any]:
    """the AutoGluon budget for one farm, from its calibration window and the tuning split.

    each candidate budget is fitted on the fit part and scored on the validation part; going
    from the shortest upward, a longer budget is chosen only if its validation mse is more
    than min_relative_gain below the current choice's. returns the choice and its evidence.
    """
    split = calibration_split(calibration, *window, validation_fraction, embargo_days)
    candidates = []
    for minutes in sorted(candidate_minutes):
        model = PeerAutoml(
            "peer_automl", features=features, settings=settings, minutes=minutes, workdir=workdir
        )
        row, fitted = score(model, split, context)
        fitted = cast(FittedPeerAutoml, fitted)
        candidates.append({"minutes": minutes, **row, "record": fitted.record})
        fitted.remove()
    chosen = candidates[0]
    for candidate in candidates[1:]:
        if candidate["mse"] < (1.0 - min_relative_gain) * chosen["mse"]:
            chosen = candidate
    return {
        "farm": context.farm,
        "minutes": chosen["minutes"],
        "min_relative_gain": min_relative_gain,
        "split": split.record,
        "candidates": candidates,
    }


def build_model(
    spec: ModelSpec,
    farm: FarmConfig,
    turbine: TurbineConfig,
    protocol: Protocol,
    *,
    minutes: float,
    workdir: Path,
) -> PeerAutoml:
    """peer_automl with the budget its farm's check chose and the job's work folder."""
    return PeerAutoml(
        spec.name,
        features=spec.features,
        settings=protocol.automl,
        minutes=minutes,
        workdir=workdir,
    )
