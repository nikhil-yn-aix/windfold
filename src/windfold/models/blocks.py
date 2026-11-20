"""named blocks of the gaussian engine: design columns, priors and the stored kernel design.

a block turns prepared rows into raw design columns, before any centering; the engine centers
them within each timestamp. a penalized block has prior precision penalty / variance, where the
variance is a fitted scale or a fixed number.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from windfold import geometry

Array = NDArray[np.float64]


@dataclass(frozen=True)
class KernelBasis:
    """tensor b-splines in along and across, one copy per speed bin.

    knots are fixed: 0 and a geometric grid up to the window along, and a symmetric geometric grid
    across. degree 0 gives the binned kernel on the same knots.
    """

    along_knots: Array
    across_knots: Array
    degree: int
    speed_bins: int

    def knot_vector(self, knots: Array) -> Array:
        return np.r_[[knots[0]] * self.degree, knots, [knots[-1]] * self.degree]

    @property
    def along_size(self) -> int:
        return len(self.along_knots) + self.degree - 1

    @property
    def across_size(self) -> int:
        return len(self.across_knots) + self.degree - 1

    @property
    def per_bin(self) -> int:
        return self.along_size * self.across_size

    @property
    def columns(self) -> int:
        return self.per_bin * self.speed_bins

    def key(self) -> str:
        """a hash of the knots, degree and speed bins; the window is the outermost knots."""
        layout = {
            "along": self.along_knots.tolist(),
            "across": self.across_knots.tolist(),
            "degree": self.degree,
            "speed_bins": self.speed_bins,
        }
        return hashlib.sha256(json.dumps(layout).encode()).hexdigest()

    def values(self, along: Array, across: Array) -> sp.csr_array:
        """basis values at each relation, without the speed-bin offset.

        scipy stores exactly degree + 1 values per point in each direction, so the tensor
        product is their outer product.
        """
        width = self.degree + 1
        if len(along) == 0:
            return sp.csr_array((0, self.per_bin))
        a = BSpline.design_matrix(along, self.knot_vector(self.along_knots), self.degree)
        c = BSpline.design_matrix(across, self.knot_vector(self.across_knots), self.degree)
        a_value, a_column = a.data.reshape(-1, width), a.indices.reshape(-1, width)
        c_value, c_column = c.data.reshape(-1, width), c.indices.reshape(-1, width)
        data = (a_value[:, :, None] * c_value[:, None, :]).reshape(len(along), -1)
        column = (a_column[:, :, None] * self.across_size + c_column[:, None, :]).reshape(
            len(along), -1
        )
        indptr = np.arange(0, data.size + 1, width * width)
        return sp.csr_array(
            (data.ravel(), column.ravel(), indptr), shape=(len(along), self.per_bin)
        )

    def penalty(self, ridge: float) -> Array:
        """second differences along and across within each bin, first differences between bins."""
        along = second_difference(self.along_size)
        across = second_difference(self.across_size)
        spatial = np.kron(along, np.eye(self.across_size)) + np.kron(
            np.eye(self.along_size), across
        )
        speed = first_difference(self.speed_bins)
        return (
            np.kron(np.eye(self.speed_bins), spatial)
            + np.kron(speed, np.eye(self.per_bin))
            + ridge * np.eye(self.columns)
        )


def second_difference(size: int) -> Array:
    if size < 3:
        return np.zeros((size, size))
    d = np.diff(np.eye(size), n=2, axis=0)
    return d.T @ d


def first_difference(size: int) -> Array:
    if size < 2:
        return np.zeros((size, size))
    d = np.diff(np.eye(size), axis=0)
    return d.T @ d


def kernel_basis(
    along_max_d: float,
    across_max_d: float,
    along_first_knot_d: float,
    along_knots: int,
    across_first_knot_d: float,
    across_knots: int,
    degree: int,
    speed_bins: int,
) -> KernelBasis:
    across = np.geomspace(across_first_knot_d, across_max_d, across_knots)
    return KernelBasis(
        along_knots=np.r_[0.0, np.geomspace(along_first_knot_d, along_max_d, along_knots)],
        across_knots=np.r_[-across[::-1], 0.0, across],
        degree=degree,
        speed_bins=speed_bins,
    )


def kernel_design(
    rows: pl.DataFrame,
    east: Array,
    north: Array,
    diameter_m: float,
    basis: KernelBasis,
    along_max_d: float,
    across_max_d: float,
    chunk: int = 200_000,
) -> sp.csr_array:
    """the shared-kernel design of prepared rows, summed over each row's upstream neighbors.

    a neighbor is a turbine in the row's operating mask, upstream of the row's turbine inside the
    window at the farm direction. its basis values go to the columns of the row's speed bin.
    """
    parts = []
    for start in range(0, rows.height, chunk):
        part = rows.slice(start, chunk)
        receiver = part["turbine_index"].to_numpy().astype(np.int64)
        masks = part["operating_mask"].to_numpy().astype(np.int64)
        speed_bin = part["speed_bin"].to_numpy().astype(np.int64)
        to_east, to_north = geometry.downwind(part["farm_direction"].to_numpy())
        d_east = east[receiver][:, None] - east[None, :]
        d_north = north[receiver][:, None] - north[None, :]
        along = (d_east * to_east[:, None] + d_north * to_north[:, None]) / diameter_m
        across = (d_north * to_east[:, None] - d_east * to_north[:, None]) / diameter_m
        operating = ((masks[:, None] >> np.arange(len(east))) & 1).astype(bool)
        near = geometry.upstream(along, across, along_max_d, across_max_d) & operating
        row, source = np.nonzero(near)
        values = basis.values(along[row, source], across[row, source]).tocoo()
        column = values.col + speed_bin[row[values.row]] * basis.per_bin
        design = sp.csr_array(
            (values.data, (row[values.row], column)), shape=(part.height, basis.columns)
        )
        design.sum_duplicates()
        parts.append(design)
    if not parts:
        return sp.csr_array((0, basis.columns))
    return sp.csr_array(sp.vstack(parts, format="csr"))


def build_kernel_designs(
    prepared_dir: Path,
    out_dir: Path,
    diameter_m: float,
    basis: KernelBasis,
    along_max_d: float,
    across_max_d: float,
    seed: int,
) -> None:
    """the real design in out_dir and a seeded derangement's in out_dir / "permuted".

    rows come from rows.parquet in file order and positions from layout.parquet in turbine_index
    order; the permuted design moves positions only, never masks, rows or physics.
    """
    rows = pl.read_parquet(prepared_dir / "rows.parquet", columns=list(DESIGN_COLUMNS))
    key = rows_key(rows)
    layout = pl.read_parquet(prepared_dir / "layout.parquet")
    east, north = layout["x_m"].to_numpy(), layout["y_m"].to_numpy()
    order = derangement(len(east), seed)
    for folder, (e, n) in (
        (out_dir, (east, north)),
        (out_dir / "permuted", (east[order], north[order])),
    ):
        design = kernel_design(rows, e, n, diameter_m, basis, along_max_d, across_max_d)
        write_kernel_design(folder, design, basis, key)


def derangement(size: int, seed: int) -> NDArray[np.int64]:
    """a seeded permutation that moves every turbine to another turbine's position."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(size)
        if size < 2 or not np.any(order == np.arange(size)):
            return order


DESIGN_COLUMNS = ("timestamp", "turbine_index", "operating_mask", "farm_direction", "speed_bin")


def rows_key(rows: pl.DataFrame) -> str:
    """a hash of everything a kernel design row depends on, in row order."""
    digest = hashlib.sha256()
    for name in DESIGN_COLUMNS:
        digest.update(np.ascontiguousarray(rows[name].to_physical().to_numpy()).tobytes())
    return digest.hexdigest()


def write_kernel_design(folder: Path, design: sp.csr_array, basis: KernelBasis, key: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("data", design.data),
        ("indices", design.indices),
        ("indptr", design.indptr),
    ):
        partial = folder / f".{name}.partial.npy"
        np.save(partial, values)
        partial.replace(folder / f"{name}.npy")
    shape = {
        "rows": len(design.indptr) - 1,
        "columns": basis.columns,
        "degree": basis.degree,
        "rows_key": key,
        "basis_key": basis.key(),
    }
    (folder / "shape.json").write_text(json.dumps(shape))


def read_kernel_design(
    folder: Path, key: str | None = None, basis_key: str | None = None
) -> sp.csr_array:
    """the stored design, memory-mapped; selecting rows copies only those rows.

    with key, a design built for other rows is refused instead of silently misaligned; with
    basis_key, so is a design built with other knots or another kernel window.
    """
    shape = json.loads((folder / "shape.json").read_text())
    if key is not None and shape.get("rows_key") != key:
        raise ValueError(
            f"the kernel design in {folder} was built for other rows; rebuild it with "
            "write_kernel_designs"
        )
    if basis_key is not None and shape.get("basis_key") != basis_key:
        raise ValueError(
            f"the kernel design in {folder} was built with another kernel basis or window; "
            "rebuild it with write_kernel_designs and the protocol this fit uses"
        )
    arrays = [
        np.load(folder / f"{name}.npy", mmap_mode="r") for name in ("data", "indices", "indptr")
    ]
    return sp.csr_array(tuple(arrays), shape=(shape["rows"], shape["columns"]), copy=False)


def one_hot(index: NDArray[np.int64], columns: int) -> sp.csr_array:
    return sp.csr_array(
        (np.ones(len(index)), (np.arange(len(index)), index)), shape=(len(index), columns)
    )


def harmonics(
    turbine: NDArray[np.int64],
    speed_bin: NDArray[np.int64],
    direction_deg: Array,
    turbines: int,
    speed_bins: int,
    order: int,
) -> sp.csr_array:
    """sine and cosine of each order of the farm direction, per turbine and speed bin."""
    radians = np.deg2rad(direction_deg)[:, None] * np.arange(1, order + 1)
    values = np.concatenate([np.sin(radians), np.cos(radians)], axis=1)
    cell = (turbine * speed_bins + speed_bin) * 2 * order
    columns = cell[:, None] + np.arange(2 * order)
    rows = np.repeat(np.arange(len(turbine)), 2 * order)
    return sp.csr_array(
        (values.ravel(), (rows, columns.ravel())),
        shape=(len(turbine), turbines * speed_bins * 2 * order),
    )


def harmonics_penalty(turbines: int, speed_bins: int, order: int) -> Array:
    """the integrated squared second derivative of each turbine's direction curve."""
    weights = np.tile(np.arange(1, order + 1, dtype=np.float64) ** 4, 2)
    return np.diag(np.tile(weights, turbines * speed_bins))


def periodic_spline(
    turbine: NDArray[np.int64],
    speed_bin: NDArray[np.int64],
    direction_deg: Array,
    turbines: int,
    speed_bins: int,
    knots: int,
) -> sp.csr_array:
    """a periodic cubic b-spline over direction with evenly spaced knots, per turbine and bin."""
    width = 360.0 / knots
    position = np.mod(direction_deg, 360.0) / width
    start = np.floor(position).astype(np.int64)
    u = position - start
    weights = np.stack(
        [
            (1 - u) ** 3 / 6,
            (3 * u**3 - 6 * u**2 + 4) / 6,
            (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6,
            u**3 / 6,
        ],
        axis=1,
    )
    basis = np.mod(start[:, None] + np.arange(-3, 1) + 2, knots)
    cell = (turbine * speed_bins + speed_bin) * knots
    rows = np.repeat(np.arange(len(turbine)), 4)
    design = sp.csr_array(
        (weights.ravel(), (rows, (cell[:, None] + basis).ravel())),
        shape=(len(turbine), turbines * speed_bins * knots),
    )
    design.sum_duplicates()
    return design


def periodic_penalty(turbines: int, speed_bins: int, knots: int, ridge: float) -> Array:
    """cyclic second differences of each turbine's spline coefficients, plus a ridge."""
    d = np.roll(np.eye(knots), -1, axis=1) - 2 * np.eye(knots) + np.roll(np.eye(knots), 1, axis=1)
    single = d.T @ d + ridge * np.eye(knots)
    return np.asarray(np.kron(np.eye(turbines * speed_bins), single), dtype=np.float64)
