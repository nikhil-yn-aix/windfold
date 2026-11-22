"""how much of the error pywake_inflow leaves on held-out arcs windfold's shared terms can learn
from the other sectors, and how much windfold learns.

the shared terms are turbine offsets, the physics term phi of one wake family and the shared
kernel. on every single arc with waked test rows, pywake_inflow's residual, y minus its
prediction, is fitted by penalized least squares on the centered design of the arc's rows:

- on the arc's own rows, cross-fitted over FOLDS interleaved week folds: the offsets, the shared
  terms and the shared terms with turbine by direction-bucket by speed-bin cells, each at the
  penalties of least cross-fitted waked-row error;
- on the training buckets, outside the arc and its buffers: the shared terms and the kernel
  alone at each of PENALTIES, and the kernel alone and the physics weight alone at windfold's
  declared priors and pywake_inflow's noise, each carried to the arc;
- windfold's own changes to pywake_inflow, one at a time: its physics weight, kernel, offsets.

windfold is rebuilt from its record on the arc's design and checked against its predictions.
at windfold's noise, the coefficients without the local block are windfold's plus the shift the
local block's fitted training contribution causes (hodges and reich 2010). windfold without its
local block is also solved on the training buckets at windfold's fitted noise, with kernel
prior variances from the declared one down to a 4096th of it; windfold_restricted, the
planned model with the same tighter variance as no_local_tighter_kernel, refits its noise.

a farm's errors are pooled over its arcs in points: 100 times a squared-error reduction over
no_wake's waked-row squared error. pywake_inflow's error splits into what the shared terms learn
on the other sectors at the penalty best over the farm's arcs (transferable), what they learn
only on the arc's own rows (sector_specific), what the cells add there (turbine_by_direction)
and what none of these explains (unexplained). an arc's penalty, or kernel variance, is also
chosen from the other arcs' errors: of their fits on their own training buckets, which hold the
arc's rows (score_left_out), or of fits on those buckets without the arc's held-out ones
(nested); the optimism is the first's points minus the second's. either way the other arcs'
wake family, pywake_inflow offsets and windfold noise come from their study fits. intervals come
from the paired block bootstrap of block_bootstrap with every fit and choice held fixed. the
per-arc checks keep the arcs with at least MIN_WAKED_ROWS waked rows, each in points of its own
no_wake error.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from scipy.stats import spearmanr

from windfold import inflow
from windfold.analyses.block_bootstrap import (
    BLOCK_DAYS,
    REPLICATES,
    interval,
    skill,
    weights,
    window_blocks,
)
from windfold.analyses.development import (
    FARMS,
    STUDY,
    check_paired,
    fit_record,
    prediction_file,
    single_arcs,
)
from windfold.analyses.kernel_geometry import protocol_basis
from windfold.analyses.local_block import beta
from windfold.analyses.strict_calibration import arc_start, held_out
from windfold.config import Protocol, load_protocol
from windfold.evaluation.bootstrap import block_of
from windfold.models.blocks import read_kernel_design
from windfold.schemas import PHYSICS_FAMILIES
from windfold.secondary.contrasts import output_folder, prediction_names
from windfold.secondary.restricted_windfold import KERNEL_PRIOR_VARIANCE
from windfold.secondary.restricted_windfold import MODEL as RESTRICTED

MODELS = ("no_wake", "pywake_inflow", "windfold", "windfold_no_local", RESTRICTED)
FOLDS = 5
RIDGE = 1e-3
CELL_PENALTIES = (1.0, 10.0, 100.0, 1e3, 1e4)
OWN_KERNEL_PENALTIES = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4)
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
TIGHTENING = tuple(4.0**j for j in range(7))
HARMONICS = 2
MIN_WAKED_ROWS = 300
SEED = 20261119
FAMILIES = len(PHYSICS_FAMILIES)
GAINS = (
    "own_rows_offsets",
    "own_rows_shared_terms",
    "own_rows_shared_terms_and_cells",
    "transferable",
    "shared_terms_score_left_out",
    "shared_terms_nested",
    "kernel_from_other_sectors",
    "kernel_from_other_sectors_declared",
    "physics_weight_from_other_sectors_declared",
    "windfold",
    "windfold_no_local",
    "windfold_no_local_rebuilt",
    "windfold_physics_weight_alone",
    "windfold_kernel_alone",
    "windfold_offsets_alone",
    "windfold_physics_weight_and_kernel_alone",
    "no_local_tighter_kernel",
    "no_local_score_left_out",
    "no_local_nested",
)
RANGED = (
    "no_local_correlation",
    "local_share_in_shared_terms",
    "local_share_smooth_in_direction",
    "kernel_posterior_over_prior_variance",
)
NOISE = {
    "no_local_tighter_kernel": "windfold's fitted noise on each arc, held fixed",
    RESTRICTED: "refitted by the evidence on each arc",
}


@dataclass(frozen=True)
class Farm:
    """a farm's development rows with phi and kernel design rows, the protocol's priors and per
    (direction bucket, speed bin) cell the centered gram of [offsets, phi of each family, the
    bin's kernel columns] and its cross with y."""

    name: str
    rows: pl.DataFrame
    kernel: sp.csr_array
    grams: np.ndarray
    cross: np.ndarray
    protocol: Protocol
    penalty: np.ndarray

    @property
    def turbines(self) -> int:
        return self.grams.shape[2] - FAMILIES - self.per_bin

    @property
    def bins(self) -> int:
        return self.grams.shape[1]

    @property
    def per_bin(self) -> int:
        return self.penalty.shape[0] // self.bins

    @property
    def width(self) -> int:
        """the shared terms: offsets, one phi and every bin's kernel columns."""
        return self.turbines + 1 + self.penalty.shape[0]

    def columns(self, family: int) -> np.ndarray:
        """the shared terms with phi of family within [offsets, phi of each family, kernel]."""
        n = self.turbines
        return np.r_[np.arange(n), n + family, n + FAMILIES + np.arange(self.penalty.shape[0])]

    def buckets(self, split: str, buffer: bool = True) -> np.ndarray:
        """whether each direction bucket lies in the split's arc, with its buffers or without."""
        splits = self.protocol.splits
        starts = np.arange(self.grams.shape[0]) * self.protocol.processing.direction_bucket_deg
        margin = splits.arc_buffer_deg if buffer else 0.0
        return held_out(starts, arc_start(split), splits.arc_width_deg, margin)


def centred(x: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """each column minus its mean over the rows of the same timestamp."""
    group = np.unique(timestamps, return_inverse=True)[1].ravel()
    members = sp.csr_array((np.ones(group.size), (group, np.arange(group.size))))
    return x - ((members @ x) / np.bincount(group)[:, None])[group]


def diagonal(*blocks: ArrayLike) -> np.ndarray:
    """the blocks along the diagonal of a matrix that is zero elsewhere."""
    return np.asarray(block_diag(*blocks))


def load_farm(data: Path, farm: str, configs: Path | None = None) -> Farm:
    protocol = load_protocol(configs)
    basis = protocol_basis(protocol)
    prepared = data / "prepared" / farm
    phi = [f"physics_{family}" for family in PHYSICS_FAMILIES]
    columns = ["timestamp", "turbine", "turbine_index", "period", "direction_bucket", "speed_bin"]
    rows = (
        pl.read_parquet(prepared / "rows.parquet", columns=[*columns, "y"])
        .with_row_index("position")
        .join(
            pl.read_parquet(prepared / inflow.TABLE, columns=["timestamp", "turbine", *phi]),
            on=["timestamp", "turbine"],
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        .filter(pl.col("period") == STUDY)
    )
    stored = read_kernel_design(prepared / "kernel_design", basis_key=basis.key())
    kernel = sp.csr_array(stored[rows["position"].to_numpy()])
    turbines = int(rows["turbine_index"].to_numpy().max()) + 1
    buckets = round(360 / protocol.processing.direction_bucket_deg)
    width = turbines + FAMILIES + basis.per_bin
    grams = np.zeros((buckets, basis.speed_bins, width, width))
    cross = np.zeros((buckets, basis.speed_bins, width))
    turbine, values = rows["turbine_index"].to_numpy(), rows.select(phi).to_numpy()
    y, stamps = rows["y"].to_numpy(), rows["timestamp"].to_physical().to_numpy()
    keyed = rows.select("direction_bucket", "speed_bin").with_row_index("at")
    for (bucket, speed_bin), part in keyed.group_by("direction_bucket", "speed_bin"):
        b, k, at = cast(int, bucket), cast(int, speed_bin), part["at"].to_numpy()
        x = np.zeros((at.size, width))
        x[np.arange(at.size), turbine[at]] = 1.0
        x[:, turbines : turbines + FAMILIES] = values[at]
        bin_columns = slice(k * basis.per_bin, (k + 1) * basis.per_bin)
        x[:, turbines + FAMILIES :] = kernel[at][:, bin_columns].toarray()
        x = centred(x, stamps[at])
        grams[b, k] = x.T @ x
        cross[b, k] = x.T @ y[at]
    return Farm(farm, rows, kernel, grams, cross, protocol, basis.penalty(protocol.kernel.ridge))


def placed(farm: Farm, family: int, speed_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """a cell's [offsets, phi of family, kernel] within its gram, and where they sit in the
    shared terms."""
    n, kernel = farm.turbines, np.arange(farm.per_bin)
    take = np.r_[np.arange(n), n + family, n + FAMILIES + kernel]
    return take, np.r_[np.arange(n + 1), n + 1 + speed_bin * farm.per_bin + kernel]


def training_sums(
    farm: Farm, train: np.ndarray, family: int, noise: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """the gram of the shared terms over the training buckets and its cross with y, each speed
    bin weighted by one over its noise variance, or unweighted."""
    gram, cross = np.zeros((farm.width, farm.width)), np.zeros(farm.width)
    for k in range(farm.bins):
        weight = 1.0 if noise is None else 1.0 / noise[k]
        take, place = placed(farm, family, k)
        gram[np.ix_(place, place)] += weight * farm.grams[train, k].sum(axis=0)[np.ix_(take, take)]
        cross[place] += weight * farm.cross[train, k].sum(axis=0)[take]
    return gram, cross


def without_local(
    farm: Farm, train: np.ndarray, family: int, noise: np.ndarray, variance: float
) -> tuple[np.ndarray, np.ndarray]:
    """training_sums with each training cell's local turbine offsets, of prior variance
    variance, integrated out as windfold's engine eliminates them."""
    gram, cross = training_sums(farm, train, family, noise)
    n = farm.turbines
    for b in np.flatnonzero(train):
        for k in range(farm.bins):
            cell = farm.grams[b, k] / noise[k]
            if not cell[:n, :n].trace():
                continue
            take, place = placed(farm, family, k)
            coupling = cell[np.ix_(take, np.arange(n))]
            local = cell[:n, :n] + np.eye(n) / variance
            solved = np.linalg.solve(local, np.c_[coupling.T, farm.cross[b, k][:n] / noise[k]])
            gram[np.ix_(place, place)] -= coupling @ solved[:, :-1]
            cross[place] -= coupling @ solved[:, -1]
    return gram, cross


def prior(farm: Farm, kernel_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """windfold's prior precision of the shared terms, with the kernel's variance, and its mean."""
    engine, n = farm.protocol.engine, farm.turbines
    precision = diagonal(
        np.eye(n) / engine.offset_variance,
        [[1.0 / engine.physics_variance]],
        farm.penalty / kernel_variance,
    )
    mean = np.zeros(farm.width)
    mean[n] = 1.0
    return precision, mean


def posterior(
    farm: Farm, sums: tuple[np.ndarray, np.ndarray], kernel_variance: float
) -> tuple[np.ndarray, np.ndarray]:
    """the posterior mean of the shared terms under windfold's priors, and its precision."""
    precision, mean = prior(farm, kernel_variance)
    return np.linalg.solve(sums[0] + precision, sums[1] + precision @ mean), sums[0] + precision


def correction(
    sums: tuple[np.ndarray, np.ndarray], shift: np.ndarray, columns: np.ndarray, penalty: np.ndarray
) -> np.ndarray:
    """shared terms, zero outside columns, fitted to y minus the prediction of the shared terms
    shift."""
    gram, cross = sums
    target = cross - gram @ shift
    theta = np.zeros(len(cross))
    theta[columns] = np.linalg.solve(gram[np.ix_(columns, columns)] + penalty, target[columns])
    return theta


@dataclass(frozen=True)
class Base:
    """what refitting an arc's corrections takes from its study fits: its held-out buckets,
    pywake_inflow's wake family, offsets as shared terms and noise, windfold's family and noise."""

    held_out: np.ndarray
    physics_family: int
    physics_shift: np.ndarray
    physics_noise: np.ndarray
    windfold_family: int
    windfold_noise: np.ndarray


@dataclass(frozen=True)
class Waked:
    """an arc's waked-row sums over [offsets, phi of each family, kernel]: the gram, and the
    cross with and square of pywake_inflow's residual and of y."""

    gram: np.ndarray
    residual: tuple[np.ndarray, float]
    y: tuple[np.ndarray, float]

    def error(self, columns: np.ndarray, theta: np.ndarray, target: str) -> float:
        cross, square = self.residual if target == "residual" else self.y
        gram = self.gram[np.ix_(columns, columns)]
        return float(square - 2 * theta @ cross[columns] + theta @ gram @ theta)


type Refit = Callable[[Farm, Base, np.ndarray], tuple[list[np.ndarray], int, str]]


def shared_terms(farm: Farm, base: Base, train: np.ndarray) -> tuple[list[np.ndarray], int, str]:
    """the shared terms fitted to pywake_inflow's residual on the training buckets at each of
    PENALTIES, the family of their phi and their target."""
    sums = training_sums(farm, train, base.physics_family)
    penalty = diagonal(RIDGE * np.eye(farm.turbines + 1), farm.penalty)
    every = np.arange(farm.width)
    fits = [correction(sums, base.physics_shift, every, lam * penalty) for lam in PENALTIES]
    return fits, base.physics_family, "residual"


def no_local(farm: Farm, base: Base, train: np.ndarray) -> tuple[list[np.ndarray], int, str]:
    """windfold without its local block on the training buckets at windfold's noise, with the
    declared kernel variance over each of TIGHTENING, the family of their phi and their target."""
    sums = training_sums(farm, train, base.windfold_family, base.windfold_noise)
    declared = farm.protocol.engine.fixed_prior_variance
    fits = [posterior(farm, sums, declared / t)[0] for t in TIGHTENING]
    return fits, base.windfold_family, "y"


REFITS: dict[str, Refit] = {"shared_terms": shared_terms, "no_local": no_local}


@dataclass(frozen=True)
class Arc:
    """an arc's test rows: the centered [offsets, phi of each family, kernel, cells], y,
    pywake_inflow's prediction, which rows are waked, each row's week fold and each waked row's
    block."""

    x: np.ndarray
    y: np.ndarray
    physics: np.ndarray
    waked: np.ndarray
    fold: np.ndarray
    block: np.ndarray
    blocks: int

    def errors(self, prediction: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
        """the waked-row squared error of prediction, against y or target, summed by block."""
        miss = ((self.y if target is None else target) - prediction)[self.waked]
        return np.bincount(self.block, weights=miss**2, minlength=self.blocks)


def predictions(runs: Path, farm: str, model: str, split: str) -> pl.DataFrame:
    path = (
        output_folder(runs, STUDY, model) / prediction_names(farm, split, None)[0]
        if model == RESTRICTED
        else prediction_file(runs, farm, model, split)
    )
    return pl.read_parquet(path, columns=["timestamp", "turbine", "y", "mean", "waked"])


def load_arc(
    farm: Farm, runs: Path, split: str, configs: Path | None
) -> tuple[Arc, dict[str, pl.DataFrame]]:
    frames = {model: predictions(runs, farm.name, model, split) for model in MODELS}
    check_paired(frames, f"{farm.name} {split}")
    keys = frames["pywake_inflow"].join(
        farm.rows.select("timestamp", "turbine").with_row_index("at"),
        on=["timestamp", "turbine"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    at = keys["at"].to_numpy()
    rows = farm.rows[at]
    if not np.allclose(rows["y"].to_numpy(), keys["y"].to_numpy()):
        raise ValueError(f"{farm.name} {split}: the predictions' y are not the prepared rows' y")
    n, kernel = farm.turbines, farm.penalty.shape[0]
    turbine = rows["turbine_index"].to_numpy().astype(np.int64)
    bucket = rows["direction_bucket"].to_numpy().astype(np.int64)
    cell = (bucket * farm.bins + rows["speed_bin"].to_numpy()) * n + turbine
    cells = np.unique(cell, return_inverse=True)[1].ravel()
    x = np.zeros((rows.height, n + FAMILIES + kernel + cells.max() + 1))
    x[np.arange(rows.height), turbine] = 1.0
    x[:, n : n + FAMILIES] = rows.select(f"physics_{f}" for f in PHYSICS_FAMILIES).to_numpy()
    x[:, n + FAMILIES : n + FAMILIES + kernel] = farm.kernel[at].toarray()
    x[np.arange(rows.height), n + FAMILIES + kernel + cells] = 1.0
    days = keys["timestamp"].dt.date()
    waked = keys["waked"].to_numpy()
    origin, blocks = window_blocks(farm.name, configs)
    block = block_of(days.filter(pl.Series(waked)), origin, BLOCK_DAYS)
    if block.size and (block.min() < 0 or block.max() >= blocks):
        raise ValueError(f"{farm.name} {split}: rows fall outside the window's {blocks} blocks")
    arc = Arc(
        centred(x, keys["timestamp"].to_physical().to_numpy()),
        keys["y"].to_numpy(),
        keys["mean"].to_numpy(),
        waked,
        days.to_physical().to_numpy() // 7 % FOLDS,
        block,
        blocks,
    )
    return arc, frames


def cross_fitted(
    arc: Arc, residual: np.ndarray, columns: np.ndarray, penalties: Sequence[np.ndarray]
) -> np.ndarray:
    """residual predicted on each fold's rows by a fit on the other folds' rows, at the penalty
    of least cross-fitted waked-row squared error, the first of equals."""
    x = arc.x[:, columns]
    sums = []
    for k in range(FOLDS):
        rows = arc.fold == k
        hit = rows & arc.waked
        sums.append(
            (x[rows].T @ x[rows], x[rows].T @ residual[rows])
            + (x[hit].T @ x[hit], x[hit].T @ residual[hit], residual[hit] @ residual[hit])
        )
    gram, cross = sum(s[0] for s in sums), sum(s[1] for s in sums)
    best: tuple[float, list[np.ndarray]] | None = None
    for penalty in penalties:
        fits = [np.linalg.solve(gram - s[0] + penalty, cross - s[1]) for s in sums]
        error = sum(s[4] - 2 * t @ s[3] + t @ s[2] @ t for s, t in zip(sums, fits, strict=True))
        if best is None or error < best[0]:
            best = (float(error), fits)
    assert best is not None
    prediction = np.zeros(len(residual))
    for k, theta in enumerate(best[1]):
        prediction[arc.fold == k] = x[arc.fold == k] @ theta
    return prediction


def own_rows(farm: Farm, arc: Arc, family: int) -> dict[str, np.ndarray]:
    """errors by block of pywake_inflow's residual cross-fitted on the arc's own rows by the
    offsets, the shared terms with phi of family and those with the cells."""
    n, residual = farm.turbines, arc.y - arc.physics
    shared = farm.columns(family)
    cells = np.arange(n + FAMILIES + farm.penalty.shape[0], arc.x.shape[1])
    head = RIDGE * np.eye(n + 1)
    kernel = [diagonal(head, lk * farm.penalty) for lk in OWN_KERNEL_PENALTIES]
    with_cells = [
        diagonal(head, lk * farm.penalty, lc * np.eye(cells.size))
        for lc in CELL_PENALTIES
        for lk in OWN_KERNEL_PENALTIES
    ]
    fits = {
        "offsets": cross_fitted(arc, residual, np.arange(n), [RIDGE * np.eye(n)]),
        "shared_terms": cross_fitted(arc, residual, shared, kernel),
        "shared_terms_and_cells": cross_fitted(arc, residual, np.r_[shared, cells], with_cells),
    }
    return {f"own_rows_{name}": arc.errors(fit, residual) for name, fit in fits.items()}


def local_contribution(farm: Farm, base: Base, record: dict[str, Any]) -> tuple[np.ndarray, float]:
    """the cross of the shared terms with windfold's fitted local block over its training cells,
    at its noise, and that block's squared size in the same metric."""
    out, size, n = np.zeros(farm.width), 0.0, farm.turbines
    for cell, means in zip(record["local_cells"], record["local_means"], strict=True):
        b, k = divmod(int(cell), farm.bins)
        if base.held_out[b]:
            continue
        gram, u = farm.grams[b, k] / base.windfold_noise[k], np.asarray(means)
        take, place = placed(farm, base.windfold_family, k)
        out[place] += gram[np.ix_(take, np.arange(n))] @ u
        size += float(u @ gram[:n, :n] @ u)
    return out, size


def smooth_share(farm: Farm, base: Base, record: dict[str, Any]) -> float:
    """the share of windfold's fitted local block, in its cells' metric, that a curve of
    HARMONICS harmonics in direction per turbine and speed bin explains."""
    n = farm.turbines
    size = n * farm.bins * (1 + 2 * HARMONICS)
    normal, target, total = np.zeros((size, size)), np.zeros(size), 0.0
    for cell, means in zip(record["local_cells"], record["local_means"], strict=True):
        b, k = divmod(int(cell), farm.bins)
        if base.held_out[b]:
            continue
        angle = np.deg2rad((b + 0.5) * farm.protocol.processing.direction_bucket_deg)
        orders = np.arange(1, HARMONICS + 1) * angle
        curve = np.r_[1.0, np.column_stack([np.cos(orders), np.sin(orders)]).ravel()]
        basis = np.kron(np.eye(n), np.kron(np.eye(farm.bins)[k], curve))
        gram, u = farm.grams[b, k][:n, :n] / base.windfold_noise[k], np.asarray(means)
        normal += basis.T @ gram @ basis
        target += basis.T @ gram @ u
        total += float(u @ gram @ u)
    solution = np.linalg.lstsq(normal + 1e-9 * np.eye(size), target, rcond=None)[0]
    return float(solution @ target) / total


def least_squares_weight(farm: Farm, buckets: np.ndarray, family: int) -> float:
    """the weight of phi of family in a least-squares fit of y on it and the turbine offsets over
    the rows of the buckets."""
    gram, cross = training_sums(farm, buckets, family)
    head = np.arange(farm.turbines + 1)
    solution = np.linalg.lstsq(gram[np.ix_(head, head)], cross[head], rcond=None)[0]
    return float(solution[farm.turbines])


def windfold_parts(
    farm: Farm, arc: Arc, base: Base, record: dict[str, Any], frames: dict[str, pl.DataFrame]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """windfold rebuilt, checked and taken apart on one arc: errors by block of its changes to
    pywake_inflow one at a time and of the coefficients its local block shifts, and its values
    on the arc."""
    n, declared = farm.turbines, farm.protocol.engine.fixed_prior_variance
    train, noise = ~base.held_out, base.windfold_noise
    blocks = record["blocks"]
    theta = np.r_[blocks["turbine_offset"], blocks["physics"], blocks["shared_kernel"]]
    x = arc.x[:, farm.columns(base.windfold_family)]
    rebuilt = x @ theta
    if np.abs(rebuilt - frames["windfold"]["mean"].to_numpy()).max() > 1e-8:
        raise ValueError(f"{farm.name}: windfold's record does not rebuild its predictions")
    integrated = without_local(farm, train, base.windfold_family, noise, declared)
    again, precision = posterior(farm, integrated, declared)
    if np.abs(again - theta).max() > 1e-6:
        raise ValueError(f"{farm.name}: windfold's coefficients do not solve its training sums")
    local, size = local_contribution(farm, base, record)
    unlocal = training_sums(farm, train, base.windfold_family, noise)[0] + prior(farm, declared)[0]
    shift = np.linalg.solve(unlocal, local)
    shifted = x @ (theta + shift)
    actual = frames["windfold_no_local"]["mean"].to_numpy()
    residual = arc.y - arc.physics
    changes = {
        "physics_weight": (theta[n] - 1) * x[:, n],
        "kernel": x[:, n + 1 :] @ theta[n + 1 :],
        "offsets": x[:, :n] @ (theta[:n] - base.physics_shift[:n]),
    }
    changes["physics_weight_and_kernel"] = changes["physics_weight"] + changes["kernel"]
    errors = {f"windfold_{name}_alone": arc.errors(c, residual) for name, c in changes.items()}
    errors["windfold_no_local_rebuilt"] = arc.errors(shifted)
    kernel = x[arc.waked][:, n + 1 :]
    after = ((kernel @ np.linalg.inv(precision)[n + 1 :, n + 1 :]) * kernel).sum(axis=1)
    before = ((kernel @ (declared * np.linalg.inv(farm.penalty))) * kernel).sum(axis=1)
    kept = before > 1e-12
    values = {
        "no_local_correlation": float(np.corrcoef(shifted - rebuilt, actual - rebuilt)[0, 1]),
        "local_share_in_shared_terms": float(shift @ local) / size,
        "local_share_smooth_in_direction": smooth_share(farm, base, record),
        "kernel_posterior_over_prior_variance": float(after[kept].mean() / before[kept].mean()),
    }
    return errors, values


@dataclass(frozen=True)
class ArcFits:
    """an arc's errors by block of each quantity, its refits' errors (setting by block), its
    values and what refitting and scoring its corrections takes."""

    split: str
    errors: dict[str, np.ndarray]
    grids: dict[str, np.ndarray]
    values: dict[str, Any]
    base: Base
    waked: Waked


def fit_arc(farm: Farm, runs: Path, split: str, configs: Path | None = None) -> ArcFits:
    arc, frames = load_arc(farm, runs, split, configs)
    physics, windfold = (fit_record(runs, farm.name, m, split)["facts"] for m in MODELS[1:3])
    n, engine = farm.turbines, farm.protocol.engine
    base = Base(
        farm.buckets(split),
        PHYSICS_FAMILIES.index(physics["family"]),
        np.r_[physics["blocks"]["turbine_offset"], 1.0, np.zeros(farm.penalty.shape[0])],
        np.asarray(physics["noise"]),
        PHYSICS_FAMILIES.index(windfold["family"]),
        np.asarray(windfold["noise"]),
    )
    train, residual = ~base.held_out, arc.y - arc.physics
    x = arc.x[:, farm.columns(base.physics_family)]
    errors = {model: arc.errors(frame["mean"].to_numpy()) for model, frame in frames.items()}
    errors |= own_rows(farm, arc, base.physics_family)
    weighted = training_sums(farm, train, base.physics_family, base.physics_noise)
    kernel = np.arange(n + 1, farm.width)
    alone = {
        "kernel": (kernel, farm.penalty / engine.fixed_prior_variance),
        "physics_weight": (np.array([n]), np.array([[1.0 / engine.physics_variance]])),
    }
    for name, (columns, penalty) in alone.items():
        theta = correction(weighted, base.physics_shift, columns, penalty)
        errors[f"{name}_from_other_sectors_declared"] = arc.errors(x @ theta, residual)
    parts, values = windfold_parts(farm, arc, base, windfold, frames)
    errors |= parts
    unweighted = training_sums(farm, train, base.physics_family)
    kernel_fits = [
        correction(unweighted, base.physics_shift, kernel, lam * farm.penalty) for lam in PENALTIES
    ]
    grids = {"kernel": np.stack([arc.errors(x @ theta, residual) for theta in kernel_fits])}
    for name, refit in REFITS.items():
        fits, family, target = refit(farm, base, train)
        on = arc.x[:, farm.columns(family)]
        aim = residual if target == "residual" else None
        grids[name] = np.stack([arc.errors(on @ theta, aim) for theta in fits])
    values |= {
        "waked_rows": int(arc.waked.sum()),
        "physics_weight": beta({"facts": windfold}),
        "own_rows_physics_weight": least_squares_weight(
            farm, farm.buckets(split, buffer=False), base.windfold_family
        ),
    }
    hit = arc.waked
    design = arc.x[hit][:, : n + FAMILIES + farm.penalty.shape[0]]
    waked = Waked(
        design.T @ design,
        (design.T @ residual[hit], float(residual[hit] @ residual[hit])),
        (design.T @ arc.y[hit], float(arc.y[hit] @ arc.y[hit])),
    )
    return ArcFits(split, errors, grids, values, base, waked)


def score_left_out(table: np.ndarray) -> list[int]:
    """each arc's setting of least error summed over the other arcs' rows of table (arcs by
    settings), the first of equals: the arc's own score is left out, its rows are not."""
    total = table.sum(axis=0)
    return [int(np.argmin(total - row)) for row in table]


def nested(farm: Farm, arcs: Sequence[ArcFits], refit: Refit) -> list[int]:
    """each arc's setting of least error summed over the other arcs, each refitted on its
    training buckets without the arc's held-out ones, the first of equals."""
    chosen = []
    for outer in arcs:
        loss: list[float] = []
        for inner in arcs:
            if inner is outer:
                continue
            train = ~(inner.base.held_out | outer.base.held_out)
            fits, family, target = refit(farm, inner.base, train)
            errors = [inner.waked.error(farm.columns(family), theta, target) for theta in fits]
            loss = [a + b for a, b in zip(loss, errors, strict=True)] if loss else errors
        chosen.append(int(np.argmin(loss)))
    return chosen


def per_arc(arcs: Sequence[ArcFits], transferable: int) -> dict[str, list[Any]]:
    """each arc's values, and windfold's, the transferable part's and the own-rows fits' points
    in the arc's own no_wake error."""

    def points(arc: ArcFits, errors: np.ndarray) -> float:
        no_wake, physics = (float(arc.errors[m].sum()) for m in ("no_wake", "pywake_inflow"))
        return 100 * (physics - float(errors.sum())) / no_wake

    out: dict[str, list[Any]] = {"split": [a.split for a in arcs]}
    out |= {key: [a.values[key] for a in arcs] for key in ("waked_rows", "physics_weight")}
    out["own_rows_physics_weight"] = [a.values["own_rows_physics_weight"] for a in arcs]
    out["windfold"] = [points(a, a.errors["windfold"]) for a in arcs]
    out["transferable"] = [points(a, a.grids["shared_terms"][transferable]) for a in arcs]
    for name in ("own_rows_shared_terms", "own_rows_shared_terms_and_cells"):
        out[name] = [points(a, a.errors[name]) for a in arcs]
    return out


def summarise(
    farm: Farm, arcs: Sequence[ArcFits], configs: Path | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """one farm's result, and the bootstrap replicates of every quantity it gives in points."""
    _, blocks = window_blocks(farm.name, configs)
    draws = weights(blocks, REPLICATES, np.random.default_rng(SEED + FARMS.index(farm.name)))
    engine = farm.protocol.engine
    tables = {grid: np.array([a.grids[grid].sum(axis=1) for a in arcs]) for grid in arcs[0].grids}

    def everywhere(grid: str) -> list[int]:
        return [int(np.argmin(tables[grid].sum(axis=0)))] * len(arcs)

    restricted = TIGHTENING.index(engine.fixed_prior_variance / KERNEL_PRIOR_VARIANCE)
    choices = {
        "transferable": ("shared_terms", everywhere("shared_terms")),
        "shared_terms_score_left_out": ("shared_terms", score_left_out(tables["shared_terms"])),
        "shared_terms_nested": ("shared_terms", nested(farm, arcs, shared_terms)),
        "kernel_from_other_sectors": ("kernel", everywhere("kernel")),
        "no_local_tighter_kernel": ("no_local", [restricted] * len(arcs)),
        "no_local_score_left_out": ("no_local", score_left_out(tables["no_local"])),
        "no_local_nested": ("no_local", nested(farm, arcs, no_local)),
    }
    errors = {name: sum(a.errors[name] for a in arcs) for name in arcs[0].errors}
    errors |= {
        name: sum(a.grids[grid][s] for a, s in zip(arcs, settings, strict=True))
        for name, (grid, settings) in choices.items()
    }
    points = {name: skill(draws @ e, draws @ errors["no_wake"]) for name, e in errors.items()}
    physics, windfold = points["pywake_inflow"], points["windfold"]
    own, cells = points["own_rows_shared_terms"], points["own_rows_shared_terms_and_cells"]
    replicates = {name: points[name] - physics for name in GAINS} | {
        "pywake_inflow_error": 100 - physics,
        "unexplained": 100 - cells,
        "turbine_by_direction": cells - own,
        "sector_specific": own - points["transferable"],
        "shared_terms_optimism": points["shared_terms_score_left_out"]
        - points["shared_terms_nested"],
        "no_local_optimism": points["no_local_score_left_out"] - points["no_local_nested"],
        "no_local_tighter_kernel_minus_windfold": points["no_local_tighter_kernel"] - windfold,
        f"{RESTRICTED}_minus_windfold": points[RESTRICTED] - windfold,
    }
    family = Counter(a.base.windfold_family for a in arcs).most_common(1)[0][0]
    every = np.ones(farm.grams.shape[0], bool)
    result = {
        "arcs": len(arcs),
        "waked_rows": sum(a.values["waked_rows"] for a in arcs),
        "points": {name: interval(value) for name, value in replicates.items()},
        "penalties": {
            name: [PENALTIES[s] for s in settings]
            for name, (grid, settings) in choices.items()
            if grid != "no_local"
        },
        "kernel_prior_variances": {
            name: [engine.fixed_prior_variance / TIGHTENING[s] for s in settings]
            for name, (grid, settings) in choices.items()
            if grid == "no_local"
        },
        "noise": NOISE,
        "declared_kernel_penalty_by_speed_bin": np.median(
            [a.base.physics_noise / engine.fixed_prior_variance for a in arcs], axis=0
        ).tolist(),
        "physics_weight": {
            "windfold_median": float(np.median([a.values["physics_weight"] for a in arcs])),
            "least_squares_on_all_rows": least_squares_weight(farm, every, family),
        },
        "ranges": {
            key: [min(a.values[key] for a in arcs), max(a.values[key] for a in arcs)]
            for key in RANGED
        },
        "per_arc": per_arc(arcs, choices["transferable"][1][0]),
    }
    return result, replicates


def per_arc_checks(farms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """over the arcs with at least MIN_WAKED_ROWS waked rows: how closely windfold's points
    follow the transferable part's and the own-rows fit's with cells (spearman), how often the
    transferable part stays at or below the own-rows fit of the shared terms and how often
    windfold stays at or below the transferable part."""
    table = pl.concat([pl.DataFrame(result["per_arc"]) for result in farms.values()]).filter(
        pl.col("waked_rows") >= MIN_WAKED_ROWS
    )
    windfold, transferable = table["windfold"].to_numpy(), table["transferable"].to_numpy()

    def ranks(other: str) -> dict[str, float]:
        rho, p = cast(tuple[float, float], spearmanr(windfold, table[other].to_numpy()))
        return {"spearman": float(rho), "p": float(p)}

    own = table["own_rows_shared_terms"].to_numpy()
    return {
        "arcs": table.height,
        "min_waked_rows": MIN_WAKED_ROWS,
        "windfold_against_transferable": ranks("transferable"),
        "windfold_against_own_rows_shared_terms_and_cells": ranks(
            "own_rows_shared_terms_and_cells"
        ),
        "transferable_at_most_own_rows_shared_terms": int((transferable <= own).sum()),
        "windfold_at_most_transferable": int((windfold <= transferable + 1e-12).sum()),
    }


def transfer_ceiling(
    data: Path, runs: Path, farms: Sequence[str] = FARMS, configs: Path | None = None
) -> dict[str, Any]:
    """per farm the summary over its single arcs with waked test rows, the means over the four
    farms when every farm is analyzed, and the per-arc checks."""
    out: dict[str, Any] = {"folds": FOLDS, "replicates": REPLICATES, "farms": {}}
    replicates = {}
    for name in farms:
        farm = load_farm(data, name, configs)
        arcs = [fit_arc(farm, runs, split, configs) for split in single_arcs(runs, name)]
        out["farms"][name], replicates[name] = summarise(farm, arcs, configs)
    if set(FARMS) <= set(farms):
        out["equal_farm"] = {
            "arcs": sum(out["farms"][f]["arcs"] for f in FARMS),
            "points": {
                key: interval(np.mean([replicates[f][key] for f in FARMS], axis=0))
                for key in replicates[FARMS[0]]
            },
        }
    out["per_arc_checks"] = per_arc_checks(out["farms"])
    return out
