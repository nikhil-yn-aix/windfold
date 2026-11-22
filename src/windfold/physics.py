"""engineering wake physics: phi per operating set, direction spread, calibration and kernels.

phi of turbine i is the log of its pywake power with the operating turbines of its timestamp,
minus the log of its free-stream power, both floored, at the speed-bin center and the whole
degree nearest the farm direction. the direction spread is a gaussian average of power over
neighboring whole degrees, truncated at truncate_sd standard deviations.
"""

import json
import logging
import multiprocessing
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
from numpy.typing import ArrayLike, NDArray
from py_wake.literature.gaussian_models import Niayifar_PorteAgel_2016
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.site import UniformSite
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular

from windfold.config import TurbineConfig
from windfold.schemas import PHYSICS, PHYSICS_FAMILIES, conform

log = logging.getLogger(__name__)

WAKE_PARAMETERS = {"jensen": ("k",), "niayifar": ("ti",), "turbopark": ("a", "ti")}
SPREAD = "spread_deg"
STOPPED, OPERATING, RECEIVING = 0, 1, 2
CASES_PER_CALL = 2048

Array = NDArray[np.float64]


def parameter_names(family: str) -> tuple[str, ...]:
    if family not in WAKE_PARAMETERS:
        raise ValueError(f"unknown physics family {family}; known: {list(WAKE_PARAMETERS)}")
    return (*WAKE_PARAMETERS[family], SPREAD)


def pywake_turbine(turbine: TurbineConfig) -> WindTurbines:
    """the configured curves as one pywake type with three operating codes.

    stopped turbines give no power and no thrust; receiving turbines give power but no thrust, so
    they feel wakes without making any. power and thrust share the union of both speed grids,
    which leaves each piecewise-linear curve unchanged.
    """
    speed = np.union1d(turbine.power_curve.speed_ms, turbine.thrust_curve.speed_ms)
    power = np.interp(speed, turbine.power_curve.speed_ms, turbine.power_curve.value)
    thrust = np.interp(speed, turbine.thrust_curve.speed_ms, turbine.thrust_curve.value)
    zero = np.zeros_like(speed)
    curves = [
        PowerCtTabular(speed, zero, "kW", zero),
        PowerCtTabular(speed, power, "kW", thrust),
        PowerCtTabular(speed, power, "kW", zero),
    ]
    return WindTurbine(
        turbine.name,
        turbine.rotor_diameter_m,
        0.0,
        powerCtFunction=PowerCtFunctionList("operating", curves, default_value=OPERATING),
    )


def wake_model(
    family: str, parameters: Mapping[str, float], turbine: WindTurbines
) -> PropagateDownwind:
    """the pywake literature model of a family, with its calibrated parameters set.

    jensen sets the expansion k; niayifar keeps the published expansion law and sets the ambient
    turbulence intensity ti; turbopark is pywake's Nygaard_2022 with its expansion a and ti set.
    """
    parameter_names(family)
    if family == "jensen":
        return Jensen_1983(UniformSite(), turbine, k=parameters["k"])
    if family == "niayifar":
        return Niayifar_PorteAgel_2016(UniformSite(ti=parameters["ti"]), turbine)
    model = Nygaard_2022(UniformSite(ti=parameters["ti"]), turbine)
    deficit: Any = model.wake_deficitModel
    deficit.A = parameters["a"]
    return model


def operating_sets(masks: ArrayLike, turbines: int) -> NDArray[np.bool_]:
    """one row per mask, true for each turbine whose bit is set."""
    masks = np.asarray(masks, dtype=np.int64)
    return ((masks[:, None] >> np.arange(turbines)) & 1).astype(bool)


def worker_pool(workers: int) -> AbstractContextManager[Executor | None]:
    """a process pool for pywake runs, or none for one worker; forkserver children start clean."""
    if workers <= 1:
        return nullcontext()
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver"))


def case_power(
    model: PropagateDownwind,
    east: Array,
    north: Array,
    hub_height: Array,
    operating: NDArray,
    direction: Array,
    speed: Array,
) -> Array:
    result: Any = model(
        east,
        north,
        h=hub_height,
        wd=direction,
        ws=speed,
        time=True,
        operating=operating.T,
        return_simulationResult=False,
    )
    return result[2][:, :, 0].T


def power(
    model: PropagateDownwind,
    east: ArrayLike,
    north: ArrayLike,
    hub_height: ArrayLike,
    operating: ArrayLike,
    direction_deg: ArrayLike,
    speed_ms: ArrayLike,
    pool: Executor | None = None,
) -> Array:
    """power in watts of every turbine in each flow case, given an operating code per turbine.

    each flow case is computed on its own, so splitting the cases over a pool changes nothing.
    """
    east_m, north_m, hub_m = (np.asarray(a, dtype=np.float64) for a in (east, north, hub_height))
    codes = np.asarray(operating)
    direction = np.asarray(direction_deg, dtype=np.float64)
    speed = np.asarray(speed_ms, dtype=np.float64)
    parts = [slice(s, s + CASES_PER_CALL) for s in range(0, len(direction), CASES_PER_CALL)]
    jobs = [(model, east_m, north_m, hub_m, codes[p], direction[p], speed[p]) for p in parts]
    if not jobs:
        return np.empty((0, len(east_m)))
    results = (
        pool.map(case_power, *zip(*jobs, strict=True)) if pool else (case_power(*j) for j in jobs)
    )
    return np.concatenate(list(results))


def spread_weights(spread_deg: float, truncate_sd: float) -> tuple[NDArray[np.int64], Array]:
    """whole-degree offsets and normalized gaussian weights of the direction spread."""
    radius = int(truncate_sd * spread_deg + 0.5)
    offsets = np.arange(-radius, radius + 1)
    if radius == 0:
        return offsets, np.ones(1)
    weights = np.exp(-0.5 * (offsets / spread_deg) ** 2)
    return offsets, weights / weights.sum()


def free_power(model: PropagateDownwind, speeds_ms: ArrayLike) -> Array:
    return np.asarray(
        model.windTurbines.power(np.asarray(speeds_ms, dtype=np.float64)), dtype=np.float64
    )


def phi_keys(
    model: PropagateDownwind,
    east: ArrayLike,
    north: ArrayLike,
    hub_height: ArrayLike,
    masks: ArrayLike,
    degrees: ArrayLike,
    bins: ArrayLike,
    speeds_ms: ArrayLike,
    spread_deg: float,
    truncate_sd: float,
    floor_w: float,
    pool: Executor | None = None,
) -> Array:
    """phi of every turbine at each key (operating mask, whole degree, speed bin).

    pywake runs once per distinct (mask, degree, bin) that a key or its spread window needs,
    which equals full 1 degree rings wherever the keys cover the ring. turbines outside a key's
    mask get nan.
    """
    turbines = len(np.asarray(east))
    speeds = np.asarray(speeds_ms, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.int64)
    degrees = np.asarray(degrees, dtype=np.int64)
    bins = np.asarray(bins, dtype=np.int64)
    distinct_masks, mask_index = np.unique(masks, return_inverse=True)
    offsets, weights = spread_weights(spread_deg, truncate_sd)
    around = np.mod(degrees[:, None] + offsets[None, :], 360)
    needed = (mask_index[:, None] * 360 + around) * len(speeds) + bins[:, None]
    cases = np.unique(needed)
    case_bin = cases % len(speeds)
    case_degree = (cases // len(speeds)) % 360
    case_set = operating_sets(distinct_masks[cases // (360 * len(speeds))], turbines)
    case_power = power(
        model,
        east,
        north,
        hub_height,
        np.where(case_set, OPERATING, STOPPED),
        case_degree.astype(np.float64),
        speeds[case_bin],
        pool,
    )
    positions = np.searchsorted(cases, needed)
    smoothed = np.zeros((len(degrees), turbines))
    for column, weight in enumerate(weights):
        smoothed += weight * case_power[positions[:, column]]
    reference = free_power(model, speeds)[bins]
    phi = np.log(np.maximum(smoothed, floor_w)) - np.log(np.maximum(reference, floor_w))[:, None]
    phi[~operating_sets(masks, turbines)] = np.nan
    return phi


def row_keys(
    rows: pl.DataFrame,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """the distinct (mask, degree, bin) keys of prepared rows and each row's key index."""
    degree = np.mod(np.rint(rows["farm_direction"].to_numpy()), 360).astype(np.int64)
    packed = np.stack(
        [
            rows["operating_mask"].to_numpy().astype(np.int64),
            degree,
            rows["speed_bin"].to_numpy().astype(np.int64),
        ],
        axis=1,
    )
    keys, inverse = np.unique(packed, axis=0, return_inverse=True)
    return keys[:, 0], keys[:, 1], keys[:, 2], inverse.ravel()


def timestamp_groups(rows: pl.DataFrame) -> NDArray[np.int64]:
    return np.unique(rows["timestamp"].to_physical().to_numpy(), return_inverse=True)[1].ravel()


def offset_gram(group: NDArray[np.int64], turbine: NDArray[np.int64], turbines: int) -> Array:
    """normal matrix of turbine offsets after centering each timestamp over its rows."""
    size = np.bincount(group)
    member = np.zeros((len(size), turbines))
    member[group, turbine] = 1.0
    return np.diag(np.bincount(turbine, minlength=turbines).astype(np.float64)) - member.T @ (
        member / size[:, None]
    )


def physics_mse(
    phi: Array, y: Array, group: NDArray[np.int64], turbine: NDArray[np.int64], gram: Array
) -> float:
    """training mse of turbine offsets plus phi at unit scale, on the centered scale of y."""
    size = np.bincount(group)
    residual = y - (phi - (np.bincount(group, phi) / size)[group])
    target = np.bincount(turbine, residual, minlength=gram.shape[0])
    offsets = np.linalg.lstsq(gram, target, rcond=1e-10)[0][turbine]
    fitted = offsets - (np.bincount(group, offsets) / size)[group]
    return float(np.mean((residual - fitted) ** 2))


def choose_family(rows: pl.DataFrame, phi: Mapping[str, Array], turbines: int) -> str:
    """the family whose unit-scale phi with turbine offsets has the lowest training mse.

    ties go to the first in PHYSICS_FAMILIES; this is the per-split choice every model uses.
    """
    group = timestamp_groups(rows)
    index = rows["turbine_index"].to_numpy().astype(np.int64)
    gram = offset_gram(group, index, turbines)
    y = rows["y"].to_numpy()
    mse = {f: physics_mse(phi[f], y, group, index, gram) for f in PHYSICS_FAMILIES}
    return min(PHYSICS_FAMILIES, key=lambda f: (mse[f], PHYSICS_FAMILIES.index(f)))


def check_rows(rows: pl.DataFrame, turbines: int, speed_bins: int) -> None:
    index = rows["turbine_index"].to_numpy().astype(np.int64)
    if index.max(initial=0) >= turbines:
        raise ValueError(
            f"rows name turbine_index {index.max()}, the layout has {turbines} turbines"
        )
    outside = ((rows["operating_mask"].to_numpy().astype(np.int64) >> index) & 1) == 0
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} rows belong to turbines outside their operating mask"
        )
    if not np.isfinite(rows["farm_direction"].to_numpy()).all():
        raise ValueError("rows need a finite farm_direction")
    bins = rows["speed_bin"].to_numpy().astype(np.int64)
    if bins.max(initial=0) >= speed_bins:
        raise ValueError(f"rows name speed_bin {bins.max()}, there are {speed_bins} speed bins")


def calibrate(
    family: str,
    rows: pl.DataFrame,
    turbine: TurbineConfig,
    east: ArrayLike,
    north: ArrayLike,
    hub_height: ArrayLike,
    speeds_ms: ArrayLike,
    search: Mapping[str, Sequence[float]],
    start: Mapping[str, float],
    trials: int,
    seed: int,
    truncate_sd: float,
    floor_fraction: float,
    pool: Executor | None = None,
) -> tuple[dict[str, float], float, pl.DataFrame]:
    """optuna tpe search for the parameters that minimize the physics-only training mse.

    the first trial is start, so the result is never worse on these rows than start.
    returns the best parameters, their mse and every trial.
    """
    names = parameter_names(family)
    turbines = len(np.asarray(east))
    check_rows(rows, turbines, len(np.asarray(speeds_ms)))
    pywake = pywake_turbine(turbine)
    masks, degrees, bins, inverse = row_keys(rows)
    group = timestamp_groups(rows)
    index = rows["turbine_index"].to_numpy().astype(np.int64)
    gram = offset_gram(group, index, turbines)
    y = rows["y"].to_numpy()
    floor_w = floor_fraction * turbine.rated_power_kw * 1000.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({name: start[name] for name in names})

    def objective(trial: optuna.Trial) -> float:
        parameters = {name: trial.suggest_float(name, *search[name]) for name in names}
        phi = phi_keys(
            wake_model(family, parameters, pywake),
            east,
            north,
            hub_height,
            masks,
            degrees,
            bins,
            speeds_ms,
            parameters[SPREAD],
            truncate_sd,
            floor_w,
            pool,
        )
        return physics_mse(phi[inverse, index], y, group, index, gram)

    study.optimize(objective, n_trials=trials)
    record = pl.DataFrame(
        [
            {"trial": t.number, **t.params, "mse": t.value, "seconds": t.duration.total_seconds()}
            for t in study.trials
            if t.duration is not None
        ]
    )
    log.info("%s calibrated in %d trials, mse %.6g", family, trials, study.best_value)
    return dict(study.best_params), float(study.best_value), record


def no_wake_mse(rows: pl.DataFrame, turbines: int) -> float:
    group = timestamp_groups(rows)
    index = rows["turbine_index"].to_numpy().astype(np.int64)
    return physics_mse(
        np.zeros(len(rows)), rows["y"].to_numpy(), group, index, offset_gram(group, index, turbines)
    )


def teacher(mse: Mapping[str, float]) -> str:
    """the family with the lowest calibration mse, ties to the first in PHYSICS_FAMILIES."""
    return min(PHYSICS_FAMILIES, key=lambda family: (mse[family], PHYSICS_FAMILIES.index(family)))


def median_parameters(fits: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """per family, the median of each fitted parameter over the given farms' physics fits."""
    return {
        family: {
            name: float(np.median([fit["parameters"][family][name] for fit in fits]))
            for name in parameter_names(family)
        }
        for family in PHYSICS_FAMILIES
    }


def consensus_teacher(fits: Sequence[Mapping[str, Any]]) -> str:
    """the teacher most farms chose; a tie goes to the lowest mean mse relative to no_wake."""
    votes = {family: sum(fit["teacher"] == family for fit in fits) for family in PHYSICS_FAMILIES}
    best = max(votes.values())
    tied = [family for family in PHYSICS_FAMILIES if votes[family] == best]
    relative = {
        family: float(np.mean([fit["mse"][family] / fit["no_wake_mse"] for fit in fits]))
        for family in tied
    }
    return min(tied, key=lambda family: (relative[family], PHYSICS_FAMILIES.index(family)))


def single_wake_kernel(
    family: str,
    parameters: Mapping[str, float],
    turbine: TurbineConfig,
    east: ArrayLike,
    north: ArrayLike,
    hub_height: ArrayLike,
    speeds_ms: ArrayLike,
    truncate_sd: float,
    floor_fraction: float,
    workers: int = 1,
) -> Array:
    """phi of each receiver in the wake of each single source, on the whole-degree ring.

    shape (360, speed bins, receiver, source): each pair as if the two turbines stood alone,
    with the direction spread of parameters applied.
    """
    pywake = pywake_turbine(turbine)
    model = wake_model(family, parameters, pywake)
    speeds = np.asarray(speeds_ms, dtype=np.float64)
    turbines = len(np.asarray(east))
    source, degree, speed_bin = (
        a.ravel()
        for a in np.meshgrid(
            np.arange(turbines), np.arange(360), np.arange(len(speeds)), indexing="ij"
        )
    )
    codes = np.full((len(source), turbines), RECEIVING)
    codes[np.arange(len(source)), source] = OPERATING
    with worker_pool(workers) as pool:
        rings = power(model, east, north, hub_height, codes, degree, speeds[speed_bin], pool)
    rings = rings.reshape(turbines, 360, len(speeds), turbines)
    offsets, weights = spread_weights(parameters[SPREAD], truncate_sd)
    smoothed = sum(
        weight * np.roll(rings, -offset, axis=1)
        for offset, weight in zip(offsets, weights, strict=True)
    )
    floor_w = floor_fraction * turbine.rated_power_kw * 1000.0
    reference = np.log(np.maximum(free_power(model, speeds), floor_w))
    kernel = np.log(np.maximum(smoothed, floor_w)) - reference[None, None, :, None]
    return kernel.transpose(1, 2, 3, 0)


def physics_table(
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
    workers: int = 1,
) -> pl.DataFrame:
    """phi of every prepared row for each family, calibrated and published defaults (PHYSICS)."""
    check_rows(rows, len(np.asarray(east)), len(np.asarray(speeds_ms)))
    pywake = pywake_turbine(turbine)
    masks, degrees, bins, inverse = row_keys(rows)
    index = rows["turbine_index"].to_numpy().astype(np.int64)
    floor_w = floor_fraction * turbine.rated_power_kw * 1000.0
    columns = {}
    with worker_pool(workers) as pool:
        for family in PHYSICS_FAMILIES:
            for suffix, parameters in (("", calibrated[family]), ("_default", defaults[family])):
                phi = phi_keys(
                    wake_model(family, parameters, pywake),
                    east,
                    north,
                    hub_height,
                    masks,
                    degrees,
                    bins,
                    speeds_ms,
                    parameters[SPREAD],
                    truncate_sd,
                    floor_w,
                    pool,
                )
                columns[f"physics_{family}{suffix}"] = phi[inverse, index]
    return conform(rows.select("timestamp", "turbine").with_columns(**columns), PHYSICS)


def fit_farm(
    rows: pl.DataFrame,
    turbine: TurbineConfig,
    east: ArrayLike,
    north: ArrayLike,
    hub_height: ArrayLike,
    speeds_ms: ArrayLike,
    search: Mapping[str, Mapping[str, Sequence[float]]],
    defaults: Mapping[str, Mapping[str, float]],
    trials: int,
    seed: int,
    truncate_sd: float,
    floor_fraction: float,
    workers: int = 1,
) -> dict[str, Any]:
    """calibrate every family on a farm's calibration rows and choose its pretraining teacher.

    each family starts its search from the published defaults, with its own seed drawn from seed.
    with workers above 1 the three searches run side by side over one shared process pool; each
    search stays sequential, so the result does not depend on workers. the record is what
    physics_fit.json holds.
    """
    for family in PHYSICS_FAMILIES:
        for name in parameter_names(family):
            low, high = search[family][name]
            if not low <= defaults[family][name] <= high:
                raise ValueError(f"{family} default {name} lies outside its search range")
    parameters, mse, default_mse, history = {}, {}, {}, {}
    with (
        worker_pool(workers) as pool,
        ThreadPoolExecutor(len(PHYSICS_FAMILIES) if pool else 1) as searches,
    ):
        running = {
            family: searches.submit(
                calibrate,
                family,
                rows,
                turbine,
                east,
                north,
                hub_height,
                speeds_ms,
                search[family],
                defaults[family],
                trials,
                int(np.random.SeedSequence([seed, number]).generate_state(1)[0]),
                truncate_sd,
                floor_fraction,
                pool,
            )
            for number, family in enumerate(PHYSICS_FAMILIES)
        }
        for family, future in running.items():
            best, best_mse, record = future.result()
            parameters[family], mse[family] = best, best_mse
            default_mse[family] = float(record["mse"][0])
            history[family] = record.to_dicts()
    chosen = teacher(mse)
    return {
        "source": "calibration window",
        "parameters": parameters,
        "defaults": {family: dict(defaults[family]) for family in PHYSICS_FAMILIES},
        "mse": mse,
        "default_mse": default_mse,
        "no_wake_mse": no_wake_mse(rows, len(np.asarray(east))),
        "teacher": chosen,
        "teacher_parameters": parameters[chosen],
        "seed": seed,
        "trials": history,
        "rows": rows.height,
        "timestamps": rows["timestamp"].n_unique(),
        "operating_sets": int(rows["operating_mask"].n_unique()),
    }


def fit_from_farms(
    fits: Mapping[str, Mapping[str, Any]], defaults: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    """physics for a farm without a calibration window, from the other farms' physics fits."""
    others = list(fits.values())
    parameters = median_parameters(others)
    chosen = consensus_teacher(others)
    return {
        "source": "median of " + ", ".join(fits),
        "parameters": parameters,
        "defaults": {family: dict(defaults[family]) for family in PHYSICS_FAMILIES},
        "teacher": chosen,
        "teacher_parameters": parameters[chosen],
    }


def write_atomic(path: Path, write: Callable[[Path], object]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    write(partial)
    os.replace(partial, path)


def write_physics(prepared_dir: Path, table: pl.DataFrame, fit: Mapping[str, Any]) -> None:
    """physics.parquet and physics_fit.json, each written whole or not at all."""
    table = conform(table, PHYSICS)
    write_atomic(
        prepared_dir / "physics.parquet", lambda p: table.write_parquet(p, compression="zstd")
    )
    write_atomic(
        prepared_dir / "physics_fit.json",
        lambda p: p.write_text(json.dumps(dict(fit), indent=2, sort_keys=True, default=str)),
    )
