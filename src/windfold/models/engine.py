"""the gaussian engine: centered sufficient statistics, evidence-fitted scales, the posterior.

the target at timestamp t is centered over its m_t rows, so the likelihood lives in m_t - 1
contrasts. projecting with H = I - 11'/m gives the same likelihood as orthonormal contrasts,
which lets every statistic be a centered sum. the noise variance has one value per speed bin.
local coefficients (turbine by direction bucket by speed bin) form small blocks; each block is
rotated onto the eigenvectors of its data precision and eliminated by a schur complement, so
every evidence step works on the global coefficients only. everything is float64.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import cho_solve, get_lapack_funcs, solve_triangular
from scipy.optimize import minimize

Array = NDArray[np.float64]
Index = NDArray[np.int64]


@dataclass(frozen=True)
class Prior:
    """a gaussian prior on a slice of the global coefficients: precision penalty / variance.

    variance None means the variance is a scale fitted by the evidence.
    """

    name: str
    columns: slice
    penalty: Array
    mean: Array
    variance: float | None
    log_det: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_det", float(np.linalg.slogdet(self.penalty)[1]))

    @property
    def size(self) -> int:
        return self.penalty.shape[0]


@dataclass(frozen=True)
class Bounds:
    noise_floor: float
    variance_floor: float
    variance_ceiling: float


@dataclass(frozen=True)
class Statistics:
    """centered sufficient statistics of one training set, by speed bin.

    local coordinates are rotated: block b holds turbines 0..n-1 of one (bucket, speed bin) cell,
    vectors[b] its eigenvectors and values its eigenvalues.
    """

    gram: Array
    data: Array
    square: Array
    dof: Array
    turbines: int
    local_cells: Index = field(default_factory=lambda: np.zeros(0, np.int64))
    local_bin: Index = field(default_factory=lambda: np.zeros(0, np.int64))
    vectors: Array = field(default_factory=lambda: np.zeros((0, 0, 0)))
    values: Array = field(default_factory=lambda: np.zeros(0))
    cross: Array = field(default_factory=lambda: np.zeros((0, 0)))
    local_data: Array = field(default_factory=lambda: np.zeros(0))

    @property
    def bins(self) -> int:
        return len(self.square)

    @property
    def local_size(self) -> int:
        return len(self.values)

    @property
    def coordinate_bin(self) -> Index:
        return np.repeat(self.local_bin, self.turbines)


def inverse_from(factor: Array) -> Array:
    """the inverse of l l' from its lower cholesky factor l."""
    (potri,) = get_lapack_funcs(("potri",), (factor,))
    lower, info = potri(factor, lower=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"dpotri failed with info {info}")
    return np.tril(lower) + np.tril(lower, -1).T


def shape_of(matrix: Any) -> tuple[int, int]:
    return int(matrix.shape[0]), int(matrix.shape[1])


def groups(timestamps: NDArray) -> Index:
    return np.unique(timestamps, return_inverse=True)[1].ravel().astype(np.int64)


def centre(values: Array, group: Index) -> Array:
    """each value minus the mean of its timestamp."""
    return values - (np.bincount(group, values) / np.bincount(group))[group]


def cross_product(design: sp.csr_array, columns: int, chunk: int = 50_000) -> Array:
    """x'x by dense products over the columns the rows actually use."""
    design = sp.csr_array(design)
    design.sum_duplicates()
    active = np.unique(design.indices)
    out = np.zeros((columns, columns))
    if len(active) == 0:
        return out
    position = np.zeros(columns, np.int64)
    position[active] = np.arange(len(active))
    block = np.zeros((len(active), len(active)))
    rows = shape_of(design)[0]
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        begin, end = design.indptr[start], design.indptr[stop]
        part = np.zeros((stop - start, len(active)))
        row = np.repeat(np.arange(stop - start), np.diff(design.indptr[start : stop + 1]))
        part[row, position[design.indices[begin:end]]] = design.data[begin:end]
        block += part.T @ part
    out[np.ix_(active, active)] = block
    return out


def statistics(
    design: sp.csr_array,
    y: Array,
    group: Index,
    speed_bin: Index,
    speed_bins: int,
    turbine: Index | None = None,
    local_cell: Index | None = None,
    turbines: int = 0,
) -> Statistics:
    """centered statistics: X'HX, X'Hy, y'Hy and contrast counts per speed bin.

    y must already be centered within timestamps. local_cell names each row's (bucket, bin) cell
    when the model has a local block; turbine then names its column inside the cell.
    """
    rows, columns = shape_of(design)
    size = np.bincount(group).astype(np.float64)
    timestamp_bin = np.zeros(len(size), np.int64)
    timestamp_bin[group] = speed_bin
    member = sp.csr_array((np.ones(rows), (group, np.arange(rows))), shape=(len(size), rows))
    sums = dense(member @ design)
    gram = np.zeros((speed_bins, columns, columns))
    data = np.zeros((speed_bins, columns))
    square = np.zeros(speed_bins)
    dof = np.zeros(speed_bins)
    for k in range(speed_bins):
        rows_k = np.flatnonzero(speed_bin == k)
        times_k = np.flatnonzero(timestamp_bin == k)
        x = design[rows_k]
        s = sums[times_k]
        gram[k] = cross_product(x, columns) - s.T @ (s / size[times_k, None])
        data[k] = x.T @ y[rows_k]
        square[k] = float(y[rows_k] @ y[rows_k])
        dof[k] = float(np.sum(size[times_k] - 1.0))
    result = Statistics(gram, data, square, dof, turbines)
    if local_cell is None or turbine is None:
        return result
    cells, cell = np.unique(local_cell, return_inverse=True)
    column = cell * turbines + turbine
    local = sp.csr_array(
        (np.ones(rows), (np.arange(rows), column)), shape=(rows, len(cells) * turbines)
    )
    local_sums = member @ local
    weight = sp.diags_array(1.0 / size)
    centring = (local_sums.T @ weight @ local_sums).tocoo()
    blocks = np.zeros((len(cells), turbines, turbines))
    blocks[
        centring.row // turbines, centring.row % turbines, centring.col % turbines
    ] = -centring.data
    diagonal = np.bincount(column, minlength=len(cells) * turbines).reshape(len(cells), turbines)
    blocks[:, np.arange(turbines), np.arange(turbines)] += diagonal
    cross = (design.T @ local).toarray() - (local_sums.T @ (sums / size[:, None])).T
    local_data = local.T @ y
    values, vectors = np.linalg.eigh(blocks)
    values = np.clip(values, 0.0, None)
    cross_blocks = cross.reshape(columns, len(cells), turbines)
    rotated = np.einsum("gct,ctr->gcr", cross_blocks, vectors).reshape(columns, -1)
    rotated_data = np.einsum("ct,ctr->cr", local_data.reshape(len(cells), turbines), vectors)
    cell_bin = np.zeros(len(cells), np.int64)
    cell_bin[cell] = speed_bin
    return Statistics(
        gram,
        data,
        square,
        dof,
        turbines,
        local_cells=cells.astype(np.int64),
        local_bin=cell_bin,
        vectors=vectors,
        values=values.ravel(),
        cross=rotated,
        local_data=rotated_data.ravel(),
    )


@dataclass(frozen=True)
class Posterior:
    """the posterior at fixed scales, in the rotated local coordinates."""

    mean: Array
    local: Array
    factor: Array
    local_precision: Array
    local_cross: Array
    noise: Array
    variances: dict[str, float]
    local_variance: float | None
    log_evidence: float
    residual_variance: Array


class Engine:
    """the evidence and posterior of one model on one training set."""

    def __init__(
        self,
        stats: Statistics,
        priors: list[Prior],
        has_local: bool,
        bounds: Bounds,
        local_variance: float | None = None,
    ):
        self.stats = stats
        self.priors = priors
        self.has_local = has_local and stats.local_size > 0
        self.local_fixed = local_variance
        self.local_fitted = self.has_local and local_variance is None
        self.bounds = bounds
        self.fitted = [p for p in priors if p.variance is None]
        self.columns = stats.gram.shape[1]
        self.bin_coordinates = [
            np.flatnonzero(stats.coordinate_bin == k) for k in range(stats.bins)
        ]

    def unpack(self, theta: Array) -> tuple[Array, dict[str, float], float | None]:
        bins = self.stats.bins
        noise = np.exp(theta[:bins])
        variances = {
            p.name: float(np.exp(t)) for p, t in zip(self.fitted, theta[bins:], strict=False)
        }
        local = None
        if self.has_local:
            local = self.local_fixed if self.local_fixed is not None else float(np.exp(theta[-1]))
        return noise, variances, local

    def prior_precision(self, variances: dict[str, float]) -> tuple[Array, Array, float, float]:
        """the global prior precision, precision times mean, log det and the mean's square."""
        precision = np.zeros((self.columns, self.columns))
        weighted = np.zeros(self.columns)
        log_det = 0.0
        square = 0.0
        for p in self.priors:
            variance = p.variance if p.variance is not None else variances[p.name]
            block = p.penalty / variance
            precision[p.columns, p.columns] += block
            weighted[p.columns] += block @ p.mean
            log_det += p.log_det - p.size * np.log(variance)
            square += float(p.mean @ block @ p.mean)
        return precision, weighted, log_det, square

    def evaluate(self, theta: Array, gradient: bool = True) -> tuple[float, Array]:
        """minus the log evidence and its gradient in the log scales."""
        s = self.stats
        noise, variances, local_variance = self.unpack(theta)
        prior, weighted, prior_log_det, prior_square = self.prior_precision(variances)
        precision = np.tensordot(1.0 / noise, s.gram, axes=1) + prior
        h = (s.data.T @ (1.0 / noise)) + weighted
        square = float(s.square @ (1.0 / noise)) + prior_square
        local_log_det = 0.0
        d, b, local_h = np.zeros(0), np.zeros((self.columns, 0)), np.zeros(0)
        if self.has_local:
            assert local_variance is not None
            scale = noise[s.coordinate_bin]
            d = s.values / scale + 1.0 / local_variance
            b = s.cross / scale
            local_h = s.local_data / scale
            precision = precision - (b / d) @ b.T
            h = h - b @ (local_h / d)
            local_log_det = float(np.sum(np.log(d)))
            prior_log_det += s.local_size * -np.log(local_variance)
        factor = np.linalg.cholesky(precision)
        mean = cho_solve((factor, True), h, check_finite=False)
        fit = float(h @ mean)
        if self.has_local:
            fit += float(local_h @ (local_h / d))
        log_det = 2.0 * float(np.sum(np.log(np.diag(factor)))) + local_log_det
        evidence = 0.5 * (
            -float(s.dof.sum()) * np.log(2 * np.pi)
            - float(s.dof @ np.log(noise))
            + prior_log_det
            - log_det
            - (square - fit)
        )
        if not gradient:
            return -evidence, np.zeros_like(theta)
        inverse = inverse_from(factor)
        grad = np.zeros_like(theta)
        local_coordinates = np.zeros(0)
        q = np.zeros(0)
        if self.has_local:
            assert local_variance is not None
            local_coordinates = (local_h - b.T @ mean) / d
            whitened = solve_triangular(factor, b, lower=True, check_finite=False)
            q = np.einsum("gj,gj->j", whitened, whitened)
        for k in range(s.bins):
            trace = float(np.sum(inverse * s.gram[k]))
            residual = s.square[k] - 2 * float(s.data[k] @ mean) + float(mean @ s.gram[k] @ mean)
            if self.has_local:
                j = self.bin_coordinates[k]
                lam, dj, qj, cj = s.values[j], d[j], q[j], local_coordinates[j]
                trace += float(np.sum(lam * (1 / dj + qj / dj**2)) - 2 * np.sum(noise[k] * qj / dj))
                residual += float(
                    -2 * s.local_data[j] @ cj
                    + 2 * mean @ (s.cross[:, j] @ cj)
                    + np.sum(lam * cj**2)
                )
            grad[k] = -0.5 * s.dof[k] + 0.5 * (trace + residual) / noise[k]
        for i, p in enumerate(self.fitted):
            v = variances[p.name]
            shift = mean[p.columns] - p.mean
            trace = float(np.sum(inverse[p.columns, p.columns] * p.penalty))
            grad[s.bins + i] = -0.5 * p.size + 0.5 * (trace + float(shift @ p.penalty @ shift)) / v
        if self.local_fitted:
            assert local_variance is not None
            trace = float(np.sum(1 / d + q / d**2))
            grad[-1] = (
                -0.5 * s.local_size
                + 0.5 * (trace + float(local_coordinates @ local_coordinates)) / local_variance
            )
        return -evidence, -grad

    def scale_bounds(self) -> list[tuple[float, float]]:
        ceiling = np.log(self.bounds.variance_ceiling)
        noise = [(np.log(self.bounds.noise_floor), ceiling)] * self.stats.bins
        others = [(np.log(self.bounds.variance_floor), ceiling)] * (
            len(self.fitted) + int(self.local_fitted)
        )
        return noise + others

    def starts(self, prior_variances: tuple[float, ...]) -> list[Array]:
        s = self.stats
        noise = np.clip(
            s.square / np.maximum(s.dof, 1.0),
            self.bounds.noise_floor,
            self.bounds.variance_ceiling,
        )
        count = len(self.fitted) + int(self.local_fitted)
        return [np.log(np.r_[noise, np.full(count, v)]) for v in prior_variances]

    def fit(self, prior_variances: tuple[float, ...]) -> Posterior:
        """maximize the evidence over log scales by l-bfgs-b from each start; keep the best."""
        bounds = self.scale_bounds()
        best = None
        for start in self.starts(prior_variances):
            if len(start) == 0:
                best = start
                break
            result = minimize(self.evaluate, start, jac=True, method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < self.evaluate(best, gradient=False)[0]:
                best = result.x
        assert best is not None
        return self.posterior(best)

    def posterior(self, theta: Array) -> Posterior:
        s = self.stats
        noise, variances, local_variance = self.unpack(theta)
        prior, weighted, _, _ = self.prior_precision(variances)
        precision = np.tensordot(1.0 / noise, s.gram, axes=1) + prior
        h = s.data.T @ (1.0 / noise) + weighted
        d = np.zeros(0)
        b = np.zeros((self.columns, 0))
        local_h = np.zeros(0)
        if self.has_local:
            assert local_variance is not None
            scale = noise[s.coordinate_bin]
            d = s.values / scale + 1.0 / local_variance
            b = s.cross / scale
            local_h = s.local_data / scale
            precision = precision - (b / d) @ b.T
            h = h - b @ (local_h / d)
        factor = np.linalg.cholesky(precision)
        mean = cho_solve((factor, True), h, check_finite=False)
        local = (local_h - b.T @ mean) / d if self.has_local else np.zeros(0)
        residual = np.zeros(s.bins)
        for k in range(s.bins):
            r = s.square[k] - 2 * float(s.data[k] @ mean) + float(mean @ s.gram[k] @ mean)
            if self.has_local:
                j = s.coordinate_bin == k
                cj = local[j]
                r += float(
                    -2 * s.local_data[j] @ cj
                    + 2 * mean @ (s.cross[:, j] @ cj)
                    + np.sum(s.values[j] * cj**2)
                )
            residual[k] = r / max(s.dof[k], 1.0)
        return Posterior(
            mean=mean,
            local=local,
            factor=factor,
            local_precision=d,
            local_cross=b,
            noise=noise,
            variances=variances,
            local_variance=local_variance,
            log_evidence=-self.evaluate(theta, gradient=False)[0],
            residual_variance=residual,
        )


def dense(matrix: Any) -> Array:
    return np.asarray(matrix.toarray() if sp.issparse(matrix) else matrix, dtype=np.float64)


def rotate(stats: Statistics, local_rows: Any) -> sp.csr_array:
    """local functionals in (cell, turbine) coordinates, moved onto each cell's eigenvectors."""
    entries = sp.coo_array(local_rows)
    turbines = stats.turbines
    cell, turbine = entries.col // turbines, entries.col % turbines
    values = entries.data[:, None] * stats.vectors[cell, turbine, :]
    columns = cell[:, None] * turbines + np.arange(turbines)
    rotated = sp.csr_array(
        (values.ravel(), (np.repeat(entries.row, turbines), columns.ravel())),
        shape=(shape_of(local_rows)[0], stats.local_size),
    )
    rotated.sum_duplicates()
    return rotated


def functional_variance(
    posterior: Posterior,
    stats: Statistics,
    global_rows: Any,
    local_rows: Any = None,
    chunk: int = 20000,
) -> Array:
    """var(f'beta) for each row f of the functionals, from the factors alone.

    global_rows hold the global part of each functional and local_rows the local part in
    (cell, turbine) coordinates of the training cells, sparse. no observation-by-observation
    matrix is formed: var = |L^-1 (f_g - B D^-1 f_l)|^2 + f_l' D^-1 f_l, with f_l rotated.
    """
    rows = shape_of(global_rows)[0]
    out = np.zeros(rows)
    rotated = None
    if local_rows is not None and stats.local_size:
        rotated = rotate(stats, local_rows)
    for start in range(0, rows, chunk):
        part = slice(start, min(start + chunk, rows))
        g = dense(global_rows[part])
        if rotated is not None:
            f = rotated[part]
            scaled = f @ sp.diags_array(1.0 / posterior.local_precision)
            g = g - scaled @ posterior.local_cross.T
            out[part] += np.asarray((f.multiply(scaled)).sum(axis=1)).ravel()
        solved = solve_triangular(posterior.factor, g.T, lower=True, check_finite=False)
        out[part] += np.einsum("gr,gr->r", solved, solved)
    return out
