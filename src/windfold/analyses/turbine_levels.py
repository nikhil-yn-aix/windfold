"""the turbine levels behind the gradient-boosting pair on held-out sectors.

geometry_gbm sees the upstream turbines in the wind frame and the reference speed but not the
turbine, so on an unseen arc it cannot transfer a turbine's level that holds in every direction,
which identity_gbm keeps. per farm, on the waked rows of the single arcs, each seed's value then
their mean: the pair's mse, identity minus geometry, before and after removing each model's mean
residual per turbine; each model's per-turbine part of the mse, the row mean of its squared mean
residual per turbine; v_alpha, the same square of no_wake's mean prediction per turbine, the
offset no_wake fits on the training sectors; how each model's mean residual per turbine follows
that offset; the difference on the rows with an operating upstream source within EDGE_D
diameters of the window's across edge, the per-turbine means still taken over every waked row;
and the per-turbine parts in time-blocked validation. beside them: the tuning menu's
configurations under which geometry_gbm validates worse than identity_gbm on the calibration
split, the turbines' speed levels behind their power levels and each seeded pair seed by seed.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.analyses.development import FARMS, LARGE_FARMS, STUDY, check_paired
from windfold.config import Kernel, for_farm
from windfold.geometry import SIDE_BY_SIDE_D
from windfold.locations import prediction_files, search_file
from windfold.models.farm import read_farm
from windfold.models.gbm import operating, receiver_relations

IDENTITY, GEOMETRY, REFERENCE = "identity_gbm", "geometry_gbm", "no_wake"
SEEDED_PAIRS = ((IDENTITY, GEOMETRY), ("absolute_graph", "wind_frame_graph"))
CELL_ROWS = 100
EDGE_D = 0.5


def regime_splits(runs: Path, farm: str, regime: str) -> list[str]:
    summary = pl.read_csv(runs / STUDY / "manifests" / f"{farm}_summary.csv")
    chosen = summary.filter(pl.col("regime") == regime, pl.col("status") == "included")
    return sorted(chosen["split"].to_list())


def predictions(runs: Path, farm: str, model: str, split: str, seed: int | None) -> pl.DataFrame:
    path = runs / prediction_files(STUDY, None, farm, model, split, seed)[0]
    return pl.read_parquet(path, columns=["timestamp", "turbine", "y", "mean", "waked"])


def waked_fits(
    runs: Path, farm: str, splits: Sequence[str], models: Sequence[str], seeds: Sequence[int]
) -> pl.DataFrame:
    """the waked test rows of splits: timestamp, turbine, split, y, no_wake's prediction and each
    model's per seed, as <model>_<seed>, after checking that every fit predicts the same rows."""
    parts = []
    for split in splits:
        frames = {f"{m}_{s}": predictions(runs, farm, m, split, s) for m in models for s in seeds}
        frames[REFERENCE] = predictions(runs, farm, REFERENCE, split, None)
        check_paired(frames, f"{farm} {split}")
        means = {name: frame["mean"] for name, frame in frames.items()}
        rows = frames[REFERENCE].select("timestamp", "turbine", "y", "waked")
        parts.append(rows.with_columns(split=pl.lit(split), **means))
    return pl.concat(parts).filter(pl.col("waked")).drop("waked")


def residuals(frame: pl.DataFrame, model: str, seeds: Sequence[int]) -> np.ndarray:
    """rows by seeds: y minus the model's prediction."""
    y = frame["y"].to_numpy()
    return np.column_stack([y - frame[f"{model}_{s}"].to_numpy() for s in seeds])


def turbine_means(turbine: np.ndarray, values: np.ndarray) -> np.ndarray:
    """each row's mean of every column of values over its turbine's rows."""
    _, index = np.unique(turbine, return_inverse=True)
    n = np.bincount(index)
    columns = [(np.bincount(index, weights=v) / n)[index] for v in values.T]
    return np.column_stack(columns)


def mse_parts(turbine: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    """the mse and its per-turbine part, each seed's then their mean."""
    _, index = np.unique(turbine, return_inverse=True)
    n = np.bincount(index)
    parts = [
        ((r**2).sum() / r.size, (n * (np.bincount(index, weights=r) / n) ** 2).sum() / r.size)
        for r in error.T
    ]
    total, level = np.mean(parts, axis=0)
    return float(total), float(level)


def difference(identity: np.ndarray, geometry: np.ndarray) -> float:
    """identity's mse minus geometry's, each the mean of its seeds'."""
    return float((identity**2).mean(axis=0).mean() - (geometry**2).mean(axis=0).mean())


def level_correlation(frame: pl.DataFrame, error: np.ndarray) -> float:
    """the correlation over turbines of a model's mean residual, seeds averaged and pooled over
    the turbine's arcs with at least CELL_ROWS rows, with no_wake's mean prediction there."""
    names, turbine = np.unique(frame["turbine"].to_numpy(), return_inverse=True)
    splits, split = np.unique(frame["split"].to_numpy(), return_inverse=True)
    cell = turbine * splits.size + split
    rows = np.bincount(cell, minlength=names.size * splits.size)
    kept = rows >= CELL_ROWS
    residual = np.bincount(cell, weights=error.mean(axis=1), minlength=rows.size)
    owner = np.repeat(np.arange(names.size), splits.size)
    pooled = np.bincount(owner, weights=np.where(kept, rows, 0), minlength=names.size)
    level = np.bincount(owner, weights=np.where(kept, residual, 0.0), minlength=names.size)
    offset = np.bincount(turbine, weights=frame[REFERENCE].to_numpy()) / np.bincount(turbine)
    has = pooled > 0
    return float(np.corrcoef(level[has] / pooled[has], offset[has])[0, 1])


def edge_rows(
    data: Path, farm: str, frame: pl.DataFrame, kernel: Kernel, diameter_m: float
) -> np.ndarray:
    """whether each row's turbine has an operating source upstream within the along window whose
    across distance lies within EDGE_D of the window's across edge."""
    columns = ["timestamp", "turbine", "turbine_index", "farm_direction", "operating_mask"]
    prepared = pl.read_parquet(data / "prepared" / farm / "rows.parquet", columns=columns)
    rows = frame.select("timestamp", "turbine").join(
        prepared, on=["timestamp", "turbine"], how="left", maintain_order="left"
    )
    layout = read_farm(data / "prepared" / farm, diameter_m, rows)
    on = operating(rows, layout.east.size)
    edge = np.zeros(rows.height, dtype=bool)
    for picked, along, across in receiver_relations(rows, layout):
        near = (
            (along > SIDE_BY_SIDE_D)
            & (along <= kernel.along_max_d)
            & (np.abs(np.abs(across) - kernel.across_max_d) <= EDGE_D)
            & on[picked]
        )
        edge[picked] = near.any(axis=1)
    return edge


def pair_levels(
    data: Path, runs: Path, farm: str, seeds: Sequence[int], kernel: Kernel, diameter_m: float
) -> dict[str, Any]:
    """the gbm pair's held-out difference and the turbine levels in it, on the single arcs."""
    frame = waked_fits(runs, farm, regime_splits(runs, farm, "arc_single"), SEEDED_PAIRS[0], seeds)
    turbine = frame["turbine"].to_numpy()
    error = {model: residuals(frame, model, seeds) for model in SEEDED_PAIRS[0]}
    parts = {model: mse_parts(turbine, e) for model, e in error.items()}
    level = {model: part[1] for model, part in parts.items()}
    removed = {model: e - turbine_means(turbine, e) for model, e in error.items()}
    offset = turbine_means(turbine, frame[REFERENCE].to_numpy()[:, None])[:, 0]
    slope = {
        model: float(
            np.mean([np.sum(b * offset) / np.sum(offset**2) for b in turbine_means(turbine, e).T])
        )
        for model, e in error.items()
    }
    edge = edge_rows(data, farm, frame, kernel, diameter_m)
    tested = waked_fits(runs, farm, regime_splits(runs, farm, "time_blocked"), [*error], seeds)
    blocked = {
        model: mse_parts(tested["turbine"].to_numpy(), residuals(tested, model, seeds))[1]
        for model in error
    }
    return {
        "waked_rows": frame.height,
        "mse": {model: part[0] for model, part in parts.items()},
        "identity_minus_geometry": difference(error[IDENTITY], error[GEOMETRY]),
        "identity_minus_geometry_levels_removed": (parts[IDENTITY][0] - level[IDENTITY])
        - (parts[GEOMETRY][0] - level[GEOMETRY]),
        "turbine_level_part": level,
        "turbine_level_loss": level[GEOMETRY] - level[IDENTITY],
        "v_alpha": float((offset**2).mean()),
        "level_slope_on_offset": slope,
        "level_correlation_with_offset": {
            model: level_correlation(frame, e) for model, e in error.items()
        },
        "edge_rows": {
            "share": float(edge.mean()),
            "identity_minus_geometry": difference(error[IDENTITY][edge], error[GEOMETRY][edge]),
            "identity_minus_geometry_levels_removed": difference(
                removed[IDENTITY][edge], removed[GEOMETRY][edge]
            ),
        },
        "time_blocked_turbine_level_part": blocked,
    }


def tuning_menu(runs: Path, farm: str) -> dict[str, Any]:
    """over the menu's configurations: how often geometry_gbm's validation mse on the calibration
    split exceeds identity_gbm's, and the median of identity's minus geometry's."""
    identity, geometry = (pl.read_parquet(runs / search_file(farm, m)) for m in SEEDED_PAIRS[0])
    both = identity.join(geometry, on="index", suffix="_geometry")
    gap = (both["mse"] - both["mse_geometry"]).to_numpy()
    return {
        "configurations": both.height,
        "geometry_worse": int((gap < 0).sum()),
        "median_identity_minus_geometry": float(np.median(gap)),
    }


def speed_levels(data: Path, farm: str) -> dict[str, float]:
    """each turbine's mean log speed and mean y over the development rows, both centered on the
    timestamp's turbines: the spread of the speed levels (sd over turbines) and how the power
    levels follow them (correlation and least-squares slope)."""
    rows = (
        pl.scan_parquet(data / "prepared" / farm / "rows.parquet")
        .filter(pl.col("period") == STUDY)
        .select("timestamp", "turbine", "y", "speed")
        .collect()
        .sort("turbine", "timestamp")
    )
    _, stamp = np.unique(rows["timestamp"].to_numpy(), return_inverse=True)
    _, turbine = np.unique(rows["turbine"].to_numpy(), return_inverse=True)
    log_speed = np.log(rows["speed"].to_numpy())
    centred = log_speed - (np.bincount(stamp, weights=log_speed) / np.bincount(stamp))[stamp]
    count = np.bincount(turbine)
    speed = np.bincount(turbine, weights=centred) / count
    power = np.bincount(turbine, weights=rows["y"].to_numpy()) / count
    return {
        "speed_level_sd": float(speed.std()),
        "power_on_speed_correlation": float(np.corrcoef(speed, power)[0, 1]),
        "power_on_speed_slope": float(np.polyfit(speed, power, 1)[0]),
    }


def seed_pairs(runs: Path, farm: str, seeds: Sequence[int]) -> dict[str, Any]:
    """each seeded pair's waked-row mse over the single arcs, side by side and seed by seed."""
    arcs = regime_splits(runs, farm, "arc_single")
    out = {}
    for identity, relative in SEEDED_PAIRS:
        frame = waked_fits(runs, farm, arcs, (identity, relative), seeds)
        mse = {m: (residuals(frame, m, seeds) ** 2).mean(axis=0) for m in (identity, relative)}
        out[f"{identity}_minus_{relative}"] = {
            "seeds": list(seeds),
            identity: mse[identity].tolist(),
            relative: mse[relative].tolist(),
            "difference": (mse[identity] - mse[relative]).tolist(),
        }
    return out


def turbine_levels(
    data: Path, runs: Path, farms: Sequence[str] = FARMS, configs: Path | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for farm in farms:
        protocol, _, turbine = for_farm(farm, None, configs)
        seeds = protocol.seeds.model
        levels = pair_levels(data, runs, farm, seeds, protocol.kernel, turbine.rotor_diameter_m)
        out[farm] = levels | {
            "speed_levels": speed_levels(data, farm),
            "seed_pairs": seed_pairs(runs, farm, seeds),
        }
        if farm in LARGE_FARMS:
            out[farm]["tuning_menu"] = tuning_menu(runs, farm)
    return out
