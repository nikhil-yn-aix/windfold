"""the three LightGBM baselines, on the timestamp-grouped projected loss the engine also fits.

identity_gbm sees the turbine, the farm direction and the reference speed. geometry_gbm sees the
nearest upstream turbines in the wind frame and the reference speed. physics_guided_gbm sees
blockage ratio, inverse blocking distance and physics efficiency (Zehtabiyan-Rezaie et al. 2023)
and the reference speed. every feature comes from farm direction, reference speed, operating set
and layout, never from the target turbine's own sensors. a model with measured inflow also sees
the log of its timestamp's turbulence ratio, a median over unwaked turbines like the reference
speed, and takes physics efficiency at the measured inflow (windfold.inflow).
"""

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from windfold import inflow
from windfold.config import FarmConfig, ModelSpec, Protocol, TurbineConfig
from windfold.geometry import relations, upstream
from windfold.models.farm import Farm, read_farm, timestamp_groups
from windfold.models.interface import FitContext, check_whole_timestamps
from windfold.physics import choose_family
from windfold.schemas import PHYSICS, PHYSICS_FAMILIES, conform

FEATURE_SETS = (
    "turbine",
    "direction",
    "reference_speed",
    "wind_frame_geometry",
    "blockage_ratio",
    "inverse_blocking_distance",
    "physics_efficiency",
)
CHUNK_TIMESTAMPS = 4096


def projected_objective(
    group: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray, lgb.Dataset], tuple[np.ndarray, np.ndarray]]:
    """LightGBM objective for the sum over timestamps of ||H_t f_t - y_t||^2, H_t = I - 11'/m_t.

    group numbers each row's timestamp from 0 to T - 1. the gradient is
    2 (f_ti - mean_t f - y_ti + mean_t y), the plan's 2 (f_ti - mean_t f - y_ti) for centered y,
    and the diagonal hessian is 2 (1 - 1/m_t). a common shift of one timestamp's f leaves both
    unchanged, so the loss ignores the level that centering removes.
    """
    counts = np.bincount(group).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("group must number the timestamps 0 to T - 1 without gaps")
    target = y - (np.bincount(group, weights=y) / counts)[group]
    hessian = 2.0 * (1.0 - 1.0 / counts[group])

    def objective(predictions: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        means = np.bincount(group, weights=predictions, minlength=counts.size) / counts
        return 2.0 * (predictions - means[group] - target), hessian

    return objective


def centre(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    counts = np.bincount(group)
    return values - (np.bincount(group, weights=values) / counts)[group]


def operating(rows: pl.DataFrame, turbines: int) -> np.ndarray:
    masks = rows["operating_mask"].to_numpy().astype(np.int64)
    return ((masks[:, None] >> np.arange(turbines)) & 1).astype(bool)


def receiver_relations(
    rows: pl.DataFrame, farm: Farm
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """row indices with the along and across of every source seen from each row's turbine."""
    group = timestamp_groups(rows)
    direction = np.empty(group.max(initial=-1) + 1)
    direction[group] = rows["farm_direction"].to_numpy()
    receiver = rows["turbine_index"].to_numpy().astype(np.int64)
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    for start in range(0, direction.size, CHUNK_TIMESTAMPS):
        stop = start + CHUNK_TIMESTAMPS
        along, across = relations(farm.east, farm.north, direction[start:stop], farm.diameter)
        picked = order[np.searchsorted(sorted_group, start) : np.searchsorted(sorted_group, stop)]
        local = group[picked] - start
        yield picked, along[local, receiver[picked]], across[local, receiver[picked]]


def wind_frame_geometry(
    rows: pl.DataFrame, farm: Farm, along_max_d: float, across_max_d: float, neighbours: int
) -> pl.DataFrame:
    """along and across of the nearest operating upstream turbines in the kernel window, nearest
    first by along distance (null where fewer exist), and how many lie in the window."""
    on = operating(rows, farm.east.size)
    along_k = np.full((rows.height, neighbours), np.nan)
    across_k = np.full((rows.height, neighbours), np.nan)
    count = np.zeros(rows.height, dtype=np.int64)
    keep = min(neighbours, farm.east.size)
    for picked, along, across in receiver_relations(rows, farm):
        window = upstream(along, across, along_max_d, across_max_d) & on[picked]
        key = np.where(window, along, np.inf)
        nearest = np.argsort(key, axis=1, kind="stable")[:, :keep]
        found = np.take_along_axis(window, nearest, axis=1)
        along_k[picked, :keep] = np.where(found, np.take_along_axis(along, nearest, axis=1), np.nan)
        across_k[picked, :keep] = np.where(
            found, np.take_along_axis(across, nearest, axis=1), np.nan
        )
        count[picked] = window.sum(axis=1)
    columns: dict[str, Any] = {}
    for k in range(neighbours):
        columns[f"upstream_{k + 1}_along"] = along_k[:, k]
        columns[f"upstream_{k + 1}_across"] = across_k[:, k]
    columns["upstream_count"] = count.astype(np.float64)
    return pl.DataFrame(columns).fill_nan(None)


def rotor_points(count: int) -> tuple[np.ndarray, np.ndarray]:
    """equal-area sunflower points on a rotor disc of diameter 1: lateral (left, facing
    downwind) and vertical offsets from the hub."""
    k = np.arange(count)
    radius = 0.5 * np.sqrt((k + 0.5) / count)
    angle = k * math.pi * (3.0 - math.sqrt(5.0))
    return radius * np.cos(angle), radius * np.sin(angle)


def blockage(rows: pl.DataFrame, farm: Farm, points: int) -> tuple[np.ndarray, np.ndarray]:
    """blockage ratio and inverse blocking distance of each row's turbine (Zehtabiyan-Rezaie et
    al. 2023, equations 3 and 5).

    every operating turbine upstream casts a cylindrical wake one rotor diameter wide. the ratio
    is the share of the rotor disc inside any wake; the inverse distance sums, over upstream
    turbines, the mean over disc points inside that turbine's wake of one over the point's
    distance to its hub, in rotor diameters. both are integrated over points disc points.
    """
    on = operating(rows, farm.east.size)
    lateral, vertical = rotor_points(points)
    receiver = rows["turbine_index"].to_numpy().astype(np.int64)
    ratio = np.zeros(rows.height)
    inverse = np.zeros(rows.height)
    for picked, along, across in receiver_relations(rows, farm):
        offset = -across
        rise = (farm.hub_height[None, :] - farm.hub_height[receiver[picked]][:, None]) / (
            farm.diameter
        )
        reach = on[picked] & (along > 0.0) & (offset**2 + rise**2 < 1.0)
        row, source = np.nonzero(reach)
        if row.size == 0:
            continue
        dy = lateral[None, :] - offset[row, source][:, None]
        dz = vertical[None, :] - rise[row, source][:, None]
        blocked = dy**2 + dz**2 <= 0.25
        distance = np.sqrt(along[row, source][:, None] ** 2 + dy**2 + dz**2)
        inverse[picked] += np.bincount(
            row, weights=(blocked / distance).mean(axis=1), minlength=picked.size
        )
        starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
        ratio[picked[row[starts]]] = np.logical_or.reduceat(blocked, starts, axis=0).mean(axis=1)
    return ratio, inverse


def physics_columns(rows: pl.DataFrame, physics: pl.DataFrame) -> dict[str, np.ndarray]:
    """each calibrated family's phi for every row, matched on timestamp and turbine."""
    families = [f"physics_{family}" for family in PHYSICS_FAMILIES]
    joined = rows.select("timestamp", "turbine").join(
        conform(physics, PHYSICS, extra=True).select("timestamp", "turbine", *families),
        on=["timestamp", "turbine"],
        how="left",
        maintain_order="left",
    )
    missing = joined.filter(pl.any_horizontal(pl.col(families).is_null())).height
    if missing:
        raise ValueError(f"{missing} rows have no phi in physics.parquet")
    return {family: joined[f"physics_{family}"].to_numpy() for family in PHYSICS_FAMILIES}


@dataclass(frozen=True)
class GeometrySettings:
    along_max_d: float
    across_max_d: float
    upstream_neighbours: int
    rotor_points: int


def gbm_features(
    rows: pl.DataFrame,
    features: Sequence[str],
    farm: Farm,
    settings: GeometrySettings,
    efficiency: np.ndarray | None,
) -> pl.DataFrame:
    """the feature table of a GBM, one row per input row, in input order."""
    unknown = sorted(set(features) - set(FEATURE_SETS))
    if unknown:
        raise ValueError(f"unknown GBM features {unknown}; known: {list(FEATURE_SETS)}")
    parts: list[pl.DataFrame] = []
    if "turbine" in features:
        parts.append(rows.select(turbine=pl.col("turbine_index").cast(pl.Float64)))
    if "direction" in features:
        parts.append(rows.select(direction=pl.col("farm_direction")))
    if "wind_frame_geometry" in features:
        parts.append(
            wind_frame_geometry(
                rows,
                farm,
                settings.along_max_d,
                settings.across_max_d,
                settings.upstream_neighbours,
            )
        )
    if "blockage_ratio" in features or "inverse_blocking_distance" in features:
        ratio, inverse = blockage(rows, farm, settings.rotor_points)
        if "blockage_ratio" in features:
            parts.append(pl.DataFrame({"blockage_ratio": ratio}))
        if "inverse_blocking_distance" in features:
            parts.append(pl.DataFrame({"inverse_blocking_distance": inverse}))
    if "physics_efficiency" in features:
        if efficiency is None:
            raise ValueError("physics_efficiency needs phi from physics.parquet")
        parts.append(pl.DataFrame({"physics_efficiency": efficiency}))
    if "reference_speed" in features:
        parts.append(rows.select("reference_speed"))
    return pl.concat(parts, how="horizontal")


class FittedGradientBoosting:
    def __init__(
        self,
        booster: lgb.Booster,
        model: "GradientBoosting",
        farm: Farm,
        family: str | None,
        folder: Path,
        threads: int,
    ):
        self.booster = booster
        self.model = model
        self.farm = farm
        self.family = family
        self.folder = folder
        self.threads = threads

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        check_whole_timestamps(rows)
        efficiency = None
        if self.family is not None:
            phi = physics_columns(rows, pl.read_parquet(self.model.physics_path(self.folder)))
            efficiency = np.exp(phi[self.family])
        table = self.model.table(rows, self.farm, efficiency, self.folder)
        raw = np.asarray(self.booster.predict(table.to_numpy(), num_threads=self.threads))
        mean = centre(raw.astype(np.float64), timestamp_groups(rows))
        return pl.DataFrame({"mean": mean}).with_columns(variance=pl.lit(None, dtype=pl.Float64))


class GradientBoosting:
    """LightGBM with the projected objective, deterministic row-wise histograms and no early
    stopping. parameters is one menu configuration: LightGBM parameters plus num_boost_round."""

    def __init__(
        self,
        name: str,
        *,
        features: Sequence[str],
        parameters: Mapping[str, Any],
        turbine: TurbineConfig,
        geometry: GeometrySettings,
        measured_inflow: bool = False,
    ):
        self.name = name
        self.features = tuple(features)
        self.parameters = dict(parameters)
        self.diameter_m = turbine.rotor_diameter_m
        self.geometry = geometry
        self.measured_inflow = measured_inflow

    def physics_path(self, folder: Path) -> Path:
        return folder / (inflow.TABLE if self.measured_inflow else "physics.parquet")

    def table(
        self, rows: pl.DataFrame, farm: Farm, efficiency: np.ndarray | None, folder: Path
    ) -> pl.DataFrame:
        frame = gbm_features(rows, self.features, farm, self.geometry, efficiency)
        if self.measured_inflow:
            frame = frame.with_columns(inflow=np.log(inflow.ratios_of(rows, folder)))
        return frame

    def fit(self, train: pl.DataFrame, context: FitContext) -> FittedGradientBoosting:
        """reads layout.parquet and, for physics_efficiency, the physics table in prepared_dir;
        with measured inflow it reads the inflow table there."""
        check_whole_timestamps(train)
        farm = read_farm(context.prepared_dir, self.diameter_m, train)
        family = None
        efficiency = None
        if "physics_efficiency" in self.features:
            phi = physics_columns(train, pl.read_parquet(self.physics_path(context.prepared_dir)))
            family = choose_family(train, phi, farm.east.size)
            efficiency = np.exp(phi[family])
        table = self.table(train, farm, efficiency, context.prepared_dir)
        parameters = dict(self.parameters)
        rounds = int(parameters.pop("num_boost_round"))
        parameters |= {
            "objective": projected_objective(timestamp_groups(train), train["y"].to_numpy()),
            "num_threads": context.threads,
            "deterministic": True,
            "force_row_wise": True,
            "seed": context.seed,
            "verbosity": -1,
        }
        data = lgb.Dataset(
            table.to_numpy(),
            feature_name=table.columns,
            categorical_feature=["turbine"] if "turbine" in table.columns else "auto",
            free_raw_data=True,
        )
        booster = lgb.train(parameters, data, num_boost_round=rounds)
        return FittedGradientBoosting(
            booster, self, farm, family, context.prepared_dir, context.threads
        )


def build_model(
    spec: ModelSpec,
    farm: FarmConfig,
    turbine: TurbineConfig,
    protocol: Protocol,
    *,
    configuration: Mapping[str, Any],
) -> GradientBoosting:
    """a GBM from its spec and the menu configuration its farm's tuning chose."""
    geometry = GeometrySettings(
        along_max_d=protocol.kernel.along_max_d,
        across_max_d=protocol.kernel.across_max_d,
        upstream_neighbours=protocol.gbm.upstream_neighbours,
        rotor_points=protocol.gbm.rotor_points,
    )
    return GradientBoosting(
        spec.name,
        features=spec.features,
        parameters=configuration,
        turbine=turbine,
        geometry=geometry,
        measured_inflow=spec.inflow == "measured",
    )
