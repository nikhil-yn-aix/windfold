"""models by name: engine models from their blocks, the wind-up lookup and the learners.

a fit reads only what sits under context.prepared_dir: rows.parquet (to align rows with the
stored designs), physics.parquet, kernel_design/ and layout.parquet.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from windfold import inflow, physics
from windfold.config import FarmConfig, ModelSpec, Protocol, TurbineConfig
from windfold.models import blocks, engine
from windfold.models.interface import FitContext, Model, check_whole_timestamps
from windfold.schemas import PHYSICS_FAMILIES

Array = NDArray[np.float64]
Index = NDArray[np.int64]


@dataclass(frozen=True)
class EngineSettings:
    """the engine's numbers, read from the protocol by engine_settings."""

    speed_bins: int
    buckets: int
    harmonics_order: int
    spline_knots: int
    spline_ridge: float
    offset_variance: float
    physics_variance: float
    starts: tuple[float, ...]
    bounds: engine.Bounds
    fixed_prior_variance: float | None
    lookup_window_deg: int
    kernel: blocks.KernelBasis
    kernel_ridge: float


def engine_settings(protocol: Protocol, scales: str | None = None) -> EngineSettings:
    """the engine's numbers; scales, a spec's own choice, overrides the protocol's engine.scales."""
    values: dict[str, Any] = protocol.model_dump()
    settings, kernel = values["engine"], values["kernel"]
    speed_bins = len(protocol.processing.speed_bin_edges_ms) - 1
    rule = scales or settings["scales"]
    return EngineSettings(
        speed_bins=speed_bins,
        buckets=round(360.0 / protocol.processing.direction_bucket_deg),
        harmonics_order=settings["harmonics_order"],
        spline_knots=settings["spline_knots"],
        spline_ridge=settings["spline_ridge"],
        offset_variance=settings["offset_variance"],
        physics_variance=settings["physics_variance"],
        starts=tuple(settings["starts"]),
        bounds=engine.Bounds(
            settings["noise_floor"], settings["variance_floor"], settings["variance_ceiling"]
        ),
        fixed_prior_variance=settings["fixed_prior_variance"] if rule == "fixed" else None,
        lookup_window_deg=settings["lookup_window_deg"],
        kernel=blocks.kernel_basis(
            kernel["along_max_d"],
            kernel["across_max_d"],
            kernel["along_first_knot_d"],
            kernel["along_knots"],
            kernel["across_first_knot_d"],
            kernel["across_knots"],
            kernel["degree"],
            speed_bins,
        ),
        kernel_ridge=kernel["ridge"],
    )


@lru_cache(maxsize=2)
def prepared_rows(path: Path, stamp: tuple[int, int, int]) -> tuple[pl.DataFrame, str]:
    """row positions of rows.parquet and the key its kernel designs must carry."""
    rows = pl.read_parquet(path, columns=["turbine", *blocks.DESIGN_COLUMNS])
    positions = rows.select("timestamp", "turbine").with_row_index("position")
    return positions.with_columns(pl.col("position").cast(pl.Int64)), blocks.rows_key(rows)


@lru_cache(maxsize=2)
def physics_columns(path: Path, stamp: tuple[int, int, int]) -> pl.DataFrame:
    return pl.read_parquet(path)


@lru_cache(maxsize=4)
def kernel_rows(
    folder: Path, stamp: tuple[int, int, int], key: str, basis_key: str
) -> sp.csr_array:
    return blocks.read_kernel_design(folder, key, basis_key)


def modified(path: Path) -> tuple[int, int, int]:
    """identifies a file version even when it was written earlier and renamed into place."""
    status = path.stat()
    return status.st_ino, status.st_size, status.st_mtime_ns


def aligned(rows: pl.DataFrame, table: pl.DataFrame, what: str) -> pl.DataFrame:
    """table's rows for each prepared row, in the rows' order."""
    joined = rows.select("timestamp", "turbine").join(
        table, on=["timestamp", "turbine"], how="left", maintain_order="left"
    )
    missing = joined.drop("timestamp", "turbine").null_count().sum_horizontal().item()
    if missing:
        raise ValueError(f"{what} has no entry for some prepared rows; rebuild it for this farm")
    return joined


@dataclass
class Inputs:
    """what a design needs from the prepared folder for a set of rows."""

    positions: Index
    phi: dict[str, Array]
    kernel: sp.csr_array | None


def read_inputs(
    rows: pl.DataFrame,
    folder: Path,
    kernel_folder: Path,
    spec: ModelSpec,
    basis: blocks.KernelBasis,
) -> Inputs:
    keys = folder / "rows.parquet"
    table, key = prepared_rows(keys, modified(keys))
    positions = aligned(rows, table, "rows.parquet")
    phi: dict[str, Array] = {}
    if "physics" in spec.blocks:
        path = folder / inflow.TABLES.get(spec.inflow, "physics.parquet")
        table = aligned(rows, physics_columns(path, modified(path)), path.name)
        suffix = "" if spec.physics_parameters == "calibrated" else "_default"
        phi = {f: table[f"physics_{f}{suffix}"].to_numpy() for f in PHYSICS_FAMILIES}
    kernel = None
    if "shared_kernel" in spec.blocks:
        design = kernel_folder / "permuted" if spec.kernel_layout == "permuted" else kernel_folder
        stored = kernel_rows(design, modified(design / "shape.json"), key, basis.key())
        kernel = sp.csr_array(stored[positions["position"].to_numpy()])
    return Inputs(positions["position"].to_numpy(), phi, kernel)


@dataclass(frozen=True)
class Layout:
    """the column layout of one engine model: global blocks in order and their priors."""

    spec: ModelSpec
    settings: EngineSettings
    turbines: int
    family: str | None

    def global_blocks(self) -> list[str]:
        names = [b for b in self.spec.blocks if b != "local"]
        if self.spec.physics_scale == "fixed":
            names = [b for b in names if b != "physics"]
        return names

    def width(self, block: str) -> int:
        s, n = self.settings, self.turbines
        return {
            "turbine_offset": n,
            "turbine_speed_cells": n * s.speed_bins,
            "turbine_direction_harmonics": n * s.speed_bins * 2 * s.harmonics_order,
            "turbine_direction_spline": n * s.speed_bins * s.spline_knots,
            "physics": 1,
            "shared_kernel": s.kernel.columns,
        }[block]

    def design(self, rows: pl.DataFrame, inputs: Inputs) -> tuple[sp.csr_array, Array]:
        """raw global design columns and the fixed offset (unit-scale physics) of each row."""
        s, n = self.settings, self.turbines
        turbine = rows["turbine_index"].to_numpy().astype(np.int64)
        speed_bin = rows["speed_bin"].to_numpy().astype(np.int64)
        direction = rows["farm_direction"].to_numpy()
        parts = []
        for block in self.global_blocks():
            if block == "turbine_offset":
                parts.append(blocks.one_hot(turbine, n))
            elif block == "turbine_speed_cells":
                parts.append(blocks.one_hot(turbine * s.speed_bins + speed_bin, n * s.speed_bins))
            elif block == "turbine_direction_harmonics":
                parts.append(
                    blocks.harmonics(
                        turbine, speed_bin, direction, n, s.speed_bins, s.harmonics_order
                    )
                )
            elif block == "turbine_direction_spline":
                parts.append(
                    blocks.periodic_spline(
                        turbine, speed_bin, direction, n, s.speed_bins, s.spline_knots
                    )
                )
            elif block == "physics":
                assert self.family is not None
                parts.append(sp.csr_array(inputs.phi[self.family][:, None]))
            elif block == "shared_kernel":
                assert inputs.kernel is not None
                parts.append(inputs.kernel)
        offset = np.zeros(rows.height)
        if self.spec.physics_scale == "fixed":
            assert self.family is not None
            offset = inputs.phi[self.family]
        return sp.csr_array(sp.hstack(parts, format="csr")), offset

    def penalty(self, block: str) -> Array:
        s, n = self.settings, self.turbines
        if block == "turbine_speed_cells":
            return np.eye(self.width(block))
        if block == "turbine_direction_harmonics":
            return blocks.harmonics_penalty(n, s.speed_bins, s.harmonics_order)
        if block == "turbine_direction_spline":
            return blocks.periodic_penalty(n, s.speed_bins, s.spline_knots, s.spline_ridge)
        return s.kernel.penalty(s.kernel_ridge)

    def priors(self) -> list[engine.Prior]:
        s = self.settings
        fixed = s.fixed_prior_variance
        start = 0
        out = []
        for block in self.global_blocks():
            width = self.width(block)
            columns = slice(start, start + width)
            start += width
            if block == "turbine_offset":
                out.append(
                    engine.Prior(block, columns, np.eye(width), np.zeros(width), s.offset_variance)
                )
            elif block == "physics":
                out.append(engine.Prior(block, columns, np.eye(1), np.ones(1), s.physics_variance))
            else:
                out.append(
                    engine.Prior(block, columns, self.penalty(block), np.zeros(width), fixed)
                )
        return out


def local_cells(rows: pl.DataFrame, speed_bins: int) -> Index:
    bucket = rows["direction_bucket"].to_numpy().astype(np.int64)
    return bucket * speed_bins + rows["speed_bin"].to_numpy().astype(np.int64)


class FittedEngine:
    """a fitted gaussian engine model: posterior factors, scales and the design to predict with."""

    def __init__(
        self,
        layout: Layout,
        posterior: engine.Posterior,
        stats: engine.Statistics,
        folder: Path,
        kernel_folder: Path,
        threads: int,
    ):
        self.layout = layout
        self.posterior = posterior
        self.stats = stats
        self.folder = folder
        self.kernel_folder = kernel_folder
        self.threads = threads

    @property
    def coefficients(self) -> Array:
        return self.posterior.mean

    @property
    def local_coefficients(self) -> Array:
        """local means in (cell, turbine) coordinates of the training cells."""
        s = self.stats
        return np.einsum(
            "ctk,ck->ct", s.vectors, self.posterior.local.reshape(-1, s.turbines)
        ).ravel()

    @property
    def noise_variances(self) -> Array:
        return self.posterior.noise

    @property
    def scales(self) -> dict[str, float]:
        out = dict(self.posterior.variances)
        if self.posterior.local_variance is not None:
            out["local"] = self.posterior.local_variance
        return out

    @property
    def residual_variance(self) -> Array:
        return self.posterior.residual_variance

    def facts(self) -> dict[str, Any]:
        """what the evidence chose and the posterior learned: the physics family, the scales, the
        noise of each speed bin, the log evidence, each global block's posterior mean and the
        local means of each training cell (direction bucket times speed bins plus speed bin) by
        turbine."""
        out: dict[str, Any] = {
            "family": self.layout.family,
            "scales": {name: float(value) for name, value in self.scales.items()},
            "noise": self.noise_variances.tolist(),
            "log_evidence": float(self.posterior.log_evidence),
            "blocks": {p.name: self.coefficients[p.columns].tolist() for p in self.layout.priors()},
        }
        if self.stats.local_size:
            out["local_cells"] = self.stats.local_cells.tolist()
            out["local_means"] = self.local_coefficients.reshape(-1, self.stats.turbines).tolist()
        return out

    def functionals(
        self, rows: pl.DataFrame
    ) -> tuple[sp.csr_array, sp.csr_array | None, Array, sp.csr_array]:
        """centered global and local design rows, the centered offset and the centered design of
        local cells without training rows (one column per such cell and turbine)."""
        inputs = read_inputs(
            rows, self.folder, self.kernel_folder, self.layout.spec, self.layout.settings.kernel
        )
        design, offset = self.layout.design(rows, inputs)
        group = engine.groups(rows["timestamp"].to_physical().to_numpy())
        size = np.bincount(group).astype(np.float64)
        member = sp.csr_array((np.ones(rows.height), (group, np.arange(rows.height))))
        averages = sp.diags_array(1.0 / size) @ member
        centred = sp.csr_array(design - (averages @ design)[group])
        local = None
        unseen = sp.csr_array((rows.height, 0))
        if "local" in self.layout.spec.blocks:
            n = self.layout.turbines
            known = self.stats.local_cells
            cells = local_cells(rows, self.layout.settings.speed_bins)
            position = np.clip(np.searchsorted(known, cells), 0, max(len(known) - 1, 0))
            seen = known[position] == cells if len(known) else np.zeros(rows.height, bool)
            turbine = rows["turbine_index"].to_numpy().astype(np.int64)
            raw = sp.csr_array(
                (seen.astype(np.float64), (np.arange(rows.height), position * n + turbine)),
                shape=(rows.height, max(self.stats.local_size, 1)),
            )
            local = sp.csr_array(raw - (averages @ raw)[group])
            missing = np.flatnonzero(~seen)
            _, column = np.unique(cells[missing] * n + turbine[missing], return_inverse=True)
            raw = sp.csr_array(
                (np.ones(len(missing)), (missing, column)),
                shape=(rows.height, column.max(initial=-1) + 1),
            )
            unseen = sp.csr_array(raw - (averages @ raw)[group])
        return centred, local, engine.centre(offset, group), unseen

    def variance(self, rows: pl.DataFrame, weights: sp.csr_array | None = None) -> Array:
        """predictive variance of linear functionals of the centered values at rows.

        each row of weights is one functional, with one column per row of rows (a day's sum, a
        turbine's mean); none means one functional per row. it holds the posterior uncertainty,
        the prior variance of local cells without training rows and the centered noise, with
        every cross-covariance, and never forms a rows-by-rows matrix.
        """
        with threadpool_limits(self.threads):
            return self.combine(rows, self.functionals(rows), weights)

    def combine(
        self,
        rows: pl.DataFrame,
        parts: tuple[sp.csr_array, sp.csr_array | None, Array, sp.csr_array],
        weights: sp.csr_array | None,
    ) -> Array:
        centred, local, _, unseen = parts
        group = engine.groups(rows["timestamp"].to_physical().to_numpy())
        size = np.bincount(group).astype(np.float64)
        noise = self.posterior.noise[rows["speed_bin"].to_numpy().astype(np.int64)]
        if weights is None:
            variance = engine.functional_variance(self.posterior, self.stats, centred, local)
            squares = sp.csr_array(unseen.multiply(unseen)).sum(axis=1)
            noise_part = noise * (1.0 - 1.0 / size[group])
        else:
            variance = engine.functional_variance(
                self.posterior,
                self.stats,
                sp.csr_array(weights @ centred),
                None if local is None else sp.csr_array(weights @ local),
            )
            moved = sp.csr_array(weights @ unseen)
            squares = sp.csr_array(moved.multiply(moved)).sum(axis=1)
            member = sp.csr_array((np.ones(rows.height), (np.arange(rows.height), group)))
            totals = sp.csr_array(weights @ member)
            first = np.zeros(len(size))
            first[group] = noise
            noise_part = sp.csr_array(weights.multiply(weights)) @ noise - sp.csr_array(
                totals.multiply(totals)
            ) @ (first / size)
        if self.posterior.local_variance is not None:
            variance = variance + self.posterior.local_variance * np.asarray(squares).ravel()
        return variance + noise_part

    def mean_of(self, centred: sp.csr_array, local: sp.csr_array | None, offset: Array) -> Array:
        mean = centred @ self.posterior.mean + offset
        if local is not None and self.stats.local_size:
            mean = mean + local @ self.local_coefficients
        return mean

    def predict_mean(self, rows: pl.DataFrame) -> Array:
        """the predictive mean alone, without the variance, which costs far more."""
        with threadpool_limits(self.threads):
            centred, local, offset, _ = self.functionals(rows)
            return self.mean_of(centred, local, offset)

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        with threadpool_limits(self.threads):
            parts = self.functionals(rows)
            centred, local, offset, _ = parts
            mean = self.mean_of(centred, local, offset)
            variance = self.combine(rows, parts, None)
        return pl.DataFrame({"mean": mean, "variance": variance})


class EngineModel:
    """a gaussian engine model made of named blocks."""

    def __init__(self, spec: ModelSpec, settings: EngineSettings):
        self.name = spec.name
        self.spec = spec
        self.settings = settings

    def fit(self, train: pl.DataFrame, context: FitContext) -> FittedEngine:
        check_whole_timestamps(train)
        turbines = pl.read_parquet(context.prepared_dir / "layout.parquet").height
        with threadpool_limits(context.threads):
            inputs = read_inputs(
                train, context.prepared_dir, context.kernel_design, self.spec, self.settings.kernel
            )
            family = physics.choose_family(train, inputs.phi, turbines) if inputs.phi else None
            layout = Layout(self.spec, self.settings, turbines, family)
            design, offset = layout.design(train, inputs)
            group = engine.groups(train["timestamp"].to_physical().to_numpy())
            y = engine.centre(train["y"].to_numpy() - offset, group)
            speed_bin = train["speed_bin"].to_numpy().astype(np.int64)
            has_local = "local" in self.spec.blocks
            stats = engine.statistics(
                design,
                y,
                group,
                speed_bin,
                self.settings.speed_bins,
                train["turbine_index"].to_numpy().astype(np.int64) if has_local else None,
                local_cells(train, self.settings.speed_bins) if has_local else None,
                turbines,
            )
            fixed = self.settings.fixed_prior_variance
            fitted = engine.Engine(stats, layout.priors(), has_local, self.settings.bounds, fixed)
            starts = self.settings.starts if fixed is None else self.settings.starts[:1]
            posterior = fitted.fit(starts)
        return FittedEngine(
            layout, posterior, stats, context.prepared_dir, context.kernel_design, context.threads
        )


class FittedLookup:
    def __init__(self, table: Array, fallback: Array, counts: Index, window: int):
        self.table = table
        self.fallback = fallback
        self.counts = counts
        self.window = window

    @staticmethod
    def cells(rows: pl.DataFrame) -> tuple[Index, Index, Index]:
        turbine = rows["turbine_index"].to_numpy().astype(np.int64)
        speed_bin = rows["speed_bin"].to_numpy().astype(np.int64)
        degree = np.mod(np.rint(rows["farm_direction"].to_numpy()), 360).astype(np.int64)
        return turbine, speed_bin, degree

    def support(self, rows: pl.DataFrame) -> Index:
        """training rows inside each row's window (its turbine, speed bin and ±window degrees);
        the monitoring policy abstains where this is below its threshold."""
        return self.counts[self.cells(rows)]

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        """the centered table value with its support, the training rows in its window."""
        turbine, speed_bin, degree = self.cells(rows)
        raw = self.table[turbine, speed_bin, degree]
        raw = np.where(np.isnan(raw), self.fallback[turbine, speed_bin], raw)
        raw = np.nan_to_num(raw, nan=0.0)
        group = engine.groups(rows["timestamp"].to_physical().to_numpy())
        return pl.DataFrame(
            {
                "mean": engine.centre(raw, group),
                "variance": np.full(rows.height, np.nan),
                "support": self.counts[turbine, speed_bin, degree],
            },
            schema={"mean": pl.Float64, "variance": pl.Float64, "support": pl.Int64},
        ).with_columns(pl.col("variance").fill_nan(None))


class DirectionLookup:
    """wind-up's directional method: the mean of y over 1 degree bins within the window, per
    turbine and speed bin; a cell without rows falls back to its turbine's speed-bin mean."""

    def __init__(self, spec: ModelSpec, settings: EngineSettings):
        self.name = spec.name
        self.settings = settings

    def fit(self, train: pl.DataFrame, context: FitContext) -> FittedLookup:
        check_whole_timestamps(train)
        turbines = pl.read_parquet(context.prepared_dir / "layout.parquet").height
        bins = self.settings.speed_bins
        turbine, speed_bin, degree = FittedLookup.cells(train)
        y = train["y"].to_numpy()
        cell = (turbine * bins + speed_bin) * 360 + degree
        total = np.bincount(cell, y, minlength=turbines * bins * 360).reshape(turbines, bins, 360)
        count = np.bincount(cell, minlength=turbines * bins * 360).reshape(turbines, bins, 360)
        w = self.settings.lookup_window_deg
        window_total = np.sum([np.roll(total, -o, axis=2) for o in range(-w, w + 1)], axis=0)
        window_count = np.sum([np.roll(count, -o, axis=2) for o in range(-w, w + 1)], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            table = np.where(window_count > 0, window_total / window_count, np.nan)
            fallback = np.where(count.sum(2) > 0, total.sum(2) / count.sum(2), np.nan)
        return FittedLookup(table, fallback, window_count, w)


def build_model(
    spec: ModelSpec, farm: FarmConfig, turbine: TurbineConfig, protocol: Protocol
) -> Model:
    """the blocks-based model a spec names. learner models need their tuned configuration and
    are built by their own fit jobs."""
    if spec.learner is not None:
        raise ValueError(f"{spec.name} is a {spec.learner} model; its fit job builds it")
    settings = engine_settings(protocol, spec.scales)
    return model_from_blocks(spec, settings)


def model_from_blocks(spec: ModelSpec, settings: EngineSettings) -> Model:
    if "turbine_direction_table" in spec.blocks:
        return DirectionLookup(spec, settings)
    return EngineModel(spec, settings)
