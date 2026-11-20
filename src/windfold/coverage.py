"""where the data can support a learned correction: coverage scores and simple predictors.

the design-only score of a test row is the sum over its upstream neighbors j of
k_phys(r_ij)^2 var_post[g(r_ij)] / var_prior[g(r_ij)], where g is the shared kernel alone with
fixed declared scales and k_phys is the default single-wake kernel. it reads directions, speeds
and operating masks, never power. the fitted map uses a fitted model's posterior instead.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import KDTree

from windfold import geometry, physics
from windfold.config import Protocol, TurbineConfig
from windfold.models import blocks, engine
from windfold.models.registry import FittedEngine, engine_settings, modified

Array = NDArray[np.float64]
Index = NDArray[np.int64]


def pairs(
    rows: pl.DataFrame,
    east: Array,
    north: Array,
    diameter_m: float,
    along_max_d: float,
    across_max_d: float,
) -> tuple[Index, Index, Array, Array]:
    """every (row, upstream operating neighbor) inside the kernel window and its relation."""
    receiver = rows["turbine_index"].to_numpy().astype(np.int64)
    masks = rows["operating_mask"].to_numpy().astype(np.int64)
    to_east, to_north = geometry.downwind(rows["farm_direction"].to_numpy())
    d_east = east[receiver][:, None] - east[None, :]
    d_north = north[receiver][:, None] - north[None, :]
    along = (d_east * to_east[:, None] + d_north * to_north[:, None]) / diameter_m
    across = (d_north * to_east[:, None] - d_east * to_north[:, None]) / diameter_m
    operating = ((masks[:, None] >> np.arange(len(east))) & 1).astype(bool)
    near = geometry.upstream(along, across, along_max_d, across_max_d) & operating
    row, source = np.nonzero(near)
    return row.astype(np.int64), source.astype(np.int64), along[row, source], across[row, source]


def pair_basis(
    basis: blocks.KernelBasis, along: Array, across: Array, speed_bin: Index
) -> tuple[Index, Array]:
    """the nonzero kernel columns and values of each relation, in its speed bin's columns."""
    values = basis.values(along, across)
    width = (basis.degree + 1) ** 2
    columns = values.indices.reshape(-1, width) + speed_bin[:, None] * basis.per_bin
    return columns.astype(np.int64), values.data.reshape(-1, width)


def quadratic(matrix: Array, columns: Index, values: Array, chunk: int = 50_000) -> Array:
    """b' M b for each sparse b given by its columns and values."""
    out = np.zeros(len(columns))
    for start in range(0, len(columns), chunk):
        c, v = columns[start : start + chunk], values[start : start + chunk]
        block = matrix[c[:, :, None], c[:, None, :]]
        out[start : start + chunk] = np.einsum("pi,pij,pj->p", v, block, v)
    return out


def kernel_weights(rows: pl.DataFrame, row: Index, source: Index, kernel: Array) -> Array:
    """k_phys(r_ij)^2 from the single-wake kernel rings (degree, speed bin, receiver, source)."""
    degree = np.mod(np.rint(rows["farm_direction"].to_numpy()), 360).astype(np.int64)[row]
    speed_bin = rows["speed_bin"].to_numpy().astype(np.int64)[row]
    receiver = rows["turbine_index"].to_numpy().astype(np.int64)[row]
    return kernel[degree, speed_bin, receiver, source] ** 2


def kernel_precision(
    train: pl.DataFrame,
    east: Array,
    north: Array,
    diameter_m: float,
    basis: blocks.KernelBasis,
    along_max_d: float,
    across_max_d: float,
    noise_variance: float,
    prior_variance: float,
    ridge: float,
) -> tuple[Array, Array]:
    """posterior and prior precision of the kernel alone at fixed scales; reads no power."""
    design = blocks.kernel_design(train, east, north, diameter_m, basis, along_max_d, across_max_d)
    group = engine.groups(train["timestamp"].to_physical().to_numpy())
    speed_bin = train["speed_bin"].to_numpy().astype(np.int64)
    stats = engine.statistics(design, np.zeros(train.height), group, speed_bin, basis.speed_bins)
    prior = basis.penalty(ridge) / prior_variance
    return stats.gram.sum(0) / noise_variance + prior, prior


def ratio_score(
    rows: pl.DataFrame,
    posterior_covariance: Array,
    prior_covariance: Array,
    east: Array,
    north: Array,
    diameter_m: float,
    basis: blocks.KernelBasis,
    along_max_d: float,
    across_max_d: float,
    kernel: Array,
) -> Array:
    row, source, along, across = pairs(rows, east, north, diameter_m, along_max_d, across_max_d)
    speed_bin = rows["speed_bin"].to_numpy().astype(np.int64)[row]
    columns, values = pair_basis(basis, along, across, speed_bin)
    ratio = quadratic(posterior_covariance, columns, values) / quadratic(
        prior_covariance, columns, values
    )
    weight = kernel_weights(rows, row, source, kernel)
    return np.bincount(row, weight * ratio, minlength=rows.height)


def design_only_score(
    train: pl.DataFrame,
    test: pl.DataFrame,
    east: Array,
    north: Array,
    diameter_m: float,
    basis: blocks.KernelBasis,
    along_max_d: float,
    across_max_d: float,
    kernel: Array,
    noise_variance: float,
    prior_variance: float,
    ridge: float,
) -> Array:
    """the design-only coverage score of each test row: kernel block only, fixed scales."""
    posterior, prior = kernel_precision(
        train,
        east,
        north,
        diameter_m,
        basis,
        along_max_d,
        across_max_d,
        noise_variance,
        prior_variance,
        ridge,
    )
    eye = np.eye(len(prior))
    return ratio_score(
        test,
        cho_solve(cho_factor(posterior), eye),
        cho_solve(cho_factor(prior), eye),
        east,
        north,
        diameter_m,
        basis,
        along_max_d,
        across_max_d,
        kernel,
    )


def fitted_score(
    fitted: FittedEngine,
    test: pl.DataFrame,
    east: Array,
    north: Array,
    diameter_m: float,
    along_max_d: float,
    across_max_d: float,
    kernel: Array,
) -> Array:
    """the fitted coverage map: the same ratio from a fitted model, all blocks, fitted scales."""
    priors = {p.name: p for p in fitted.layout.priors()}
    if "shared_kernel" not in priors:
        raise ValueError(f"{fitted.layout.spec.name} has no shared kernel to map")
    block = priors["shared_kernel"].columns
    factor = fitted.posterior.factor
    inverse = cho_solve((factor, True), np.eye(len(factor)), check_finite=False)
    variance = fitted.posterior.variances.get(
        "shared_kernel", fitted.layout.settings.fixed_prior_variance
    )
    assert variance is not None
    prior = priors["shared_kernel"].penalty / variance
    return ratio_score(
        test,
        inverse[block, block],
        np.linalg.inv(prior),
        east,
        north,
        diameter_m,
        fitted.layout.settings.kernel,
        along_max_d,
        across_max_d,
        kernel,
    )


def angular_distance(train: pl.DataFrame, test: pl.DataFrame) -> Array:
    """degrees from each test row's farm direction to the nearest whole degree seen in training."""
    seen = np.unique(np.mod(np.rint(train["farm_direction"].to_numpy()), 360))
    direction = test["farm_direction"].to_numpy()
    if len(seen) == 0:
        return np.full(test.height, 180.0)
    gap = np.abs(((direction[:, None] - seen[None, :]) + 180.0) % 360.0 - 180.0)
    return gap.min(axis=1)


def wake_strength(
    rows: pl.DataFrame,
    east: Array,
    north: Array,
    diameter_m: float,
    along_max_d: float,
    across_max_d: float,
    kernel: Array,
) -> Array:
    """the physics-expected wake energy of each row: the sum of k_phys(r_ij)^2 over neighbors."""
    row, source, _, _ = pairs(rows, east, north, diameter_m, along_max_d, across_max_d)
    return np.bincount(row, kernel_weights(rows, row, source, kernel), minlength=rows.height)


def relational_features(
    rows: pl.DataFrame,
    east: Array,
    north: Array,
    diameter_m: float,
    along_max_d: float,
    across_max_d: float,
) -> Array:
    """along and across of the two nearest upstream neighbors, their count and reference speed.

    a missing neighbor sits at the far edge of the window (along_max_d, across_max_d).
    """
    row, _, along, across = pairs(rows, east, north, diameter_m, along_max_d, across_max_d)
    order = np.lexsort((along, row))
    row, along, across = row[order], along[order], across[order]
    count = np.bincount(row, minlength=rows.height)
    first = np.searchsorted(row, np.arange(rows.height))
    features = np.zeros((rows.height, 6))
    for k in range(2):
        has = count > k
        position = np.minimum(first + k, max(len(row) - 1, 0))
        features[:, 2 * k] = np.where(has, along[position] if len(row) else 0.0, along_max_d)
        features[:, 2 * k + 1] = np.where(has, across[position] if len(row) else 0.0, across_max_d)
    features[:, 4] = count
    features[:, 5] = rows["reference_speed"].to_numpy()
    return features


def dissimilarity(train_features: Array, test_features: Array, sample: int, seed: int) -> Array:
    """meyer and pebesma's dissimilarity index of each test row.

    features are standardized on the training rows; the index is the distance to the nearest
    training row divided by the mean distance between training rows, estimated on a seeded
    sample of them. where that mean is zero or undefined, the index is too, and comes back nan.
    """
    centre, spread = train_features.mean(0), train_features.std(0)
    spread = np.where(spread > 0, spread, 1.0)
    train = (train_features - centre) / spread
    test = (test_features - centre) / spread
    rng = np.random.default_rng(seed)
    chosen = train[rng.choice(len(train), min(sample, len(train)), replace=False)]
    gaps = np.sqrt(((chosen[:, None, :] - chosen[None, :, :]) ** 2).sum(-1))
    upper = gaps[np.triu_indices(len(chosen), 1)]
    mean_gap = float(upper.mean()) if upper.size else 0.0
    if not mean_gap > 0:
        return np.full(len(test_features), np.nan)
    nearest, _ = KDTree(train).query(test, k=1)
    return nearest / mean_gap


@lru_cache(maxsize=4)
def default_wake_rings(
    layout_path: Path, stamp: tuple[int, int, int], turbine_json: str, physics_json: str
) -> Array:
    turbine = TurbineConfig.model_validate_json(turbine_json)
    settings: dict[str, Any] = json.loads(physics_json)
    layout = pl.read_parquet(layout_path)
    family = settings["kernel_family"]
    return physics.single_wake_kernel(
        family,
        settings["defaults"][family],
        turbine,
        layout["x_m"].to_numpy(),
        layout["y_m"].to_numpy(),
        layout["hub_height_m"].to_numpy(),
        settings["centres"],
        settings["truncate_sd"],
        settings["power_floor_fraction"],
    )


def split_scores(
    train: pl.DataFrame,
    test: pl.DataFrame,
    prepared_dir: Path,
    protocol: Protocol,
    turbine: TurbineConfig,
    order: Index | None = None,
) -> Array:
    """the design-only coverage score of every test row of one split, as coverage predicts
    failure defines it.

    the kernel basis, window, ridge and fixed scales come from the protocol; the weight is the
    single-wake kernel of the default physics family at its published parameters. reads only
    layout.parquet, and no power values. order moves each turbine to another's position in the
    kernel's relations alone, as the permuted-layout ablation does; the weights keep the real
    layout.
    """
    settings = engine_settings(protocol)
    scales: dict[str, Any] = protocol.model_dump()["coverage"]
    edges = np.asarray(protocol.processing.speed_bin_edges_ms)
    layout_path = prepared_dir / "layout.parquet"
    p = protocol.physics
    physics_json = json.dumps(
        {
            "kernel_family": p.kernel_family,
            "defaults": p.defaults,
            "centres": ((edges[:-1] + edges[1:]) / 2).tolist(),
            "truncate_sd": p.truncate_sd,
            "power_floor_fraction": p.power_floor_fraction,
        },
        sort_keys=True,
    )
    rings = default_wake_rings(
        layout_path, modified(layout_path), turbine.model_dump_json(), physics_json
    )
    layout = pl.read_parquet(layout_path)
    east, north = layout["x_m"].to_numpy(), layout["y_m"].to_numpy()
    if order is not None:
        east, north = east[order], north[order]
    return design_only_score(
        train,
        test,
        east,
        north,
        turbine.rotor_diameter_m,
        settings.kernel,
        protocol.kernel.along_max_d,
        protocol.kernel.across_max_d,
        rings,
        scales["noise_variance"],
        scales["kernel_variance"],
        settings.kernel_ridge,
    )
