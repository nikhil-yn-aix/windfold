"""the synthetic recovery check: a known kernel on a real layout and wind record, refitted.

the truth is a member of the kernel basis (a wake-like shape projected onto it), so the
relational model is well specified. y is rebuilt from the real rows' directions, speeds, masks
and stored kernel design, plus turbine offsets and gaussian noise per speed bin; nothing else
of the real rows is used.
"""

from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from windfold.config import ModelSpec
from windfold.models import blocks, engine, registry
from windfold.models.interface import FitContext

Array = np.ndarray


def wake_shape(along: Array, across: Array, speed_bin: Array) -> Array:
    """a smooth deficit: gaussian across, decaying along, deeper at low speed."""
    depth = np.array([0.45, 0.35, 0.25])[speed_bin]
    width = 0.35 + 0.06 * along
    return (
        -depth
        * np.exp(-0.5 * (across / width) ** 2)
        * along
        / (along + 1.0)
        * np.exp(-along / 12.0)
    )


def true_coefficients(basis: blocks.KernelBasis) -> Array:
    """least-squares projection of wake_shape onto the basis on a fine relation grid."""
    along = np.linspace(0.05, basis.along_knots[-1], 120)
    across = np.linspace(basis.across_knots[0], basis.across_knots[-1], 61)
    a, c = (g.ravel() for g in np.meshgrid(along, across, indexing="ij"))
    coefficients = np.zeros(basis.columns)
    values = basis.values(a, c).toarray()
    for k in range(basis.speed_bins):
        target = wake_shape(a, c, np.full(len(a), k))
        coefficients[k * basis.per_bin : (k + 1) * basis.per_bin] = np.linalg.lstsq(
            values, target, rcond=None
        )[0]
    return coefficients


def synthetic_recovery(
    rows: pl.DataFrame,
    folder: Path,
    kernel_folder: Path,
    settings: registry.EngineSettings,
    noise_sd: tuple[float, ...],
    seed: int,
    threads: int,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """fit kernel_linear to synthetic y built on rows (rows of folder's rows.parquet).

    returns the true and recovered kernel with its posterior sd on a relation grid, and a
    summary: the coverage of the 90% kernel intervals over the whole grid and where training
    rows reached the relation, and of the 90% predictive intervals on held-out timestamps.
    """
    rng = np.random.default_rng(seed)
    basis = settings.kernel
    truth = true_coefficients(basis)
    table, key = registry.prepared_rows(
        folder / "rows.parquet", registry.modified(folder / "rows.parquet")
    )
    positions = registry.aligned(rows, table, "rows.parquet")["position"].to_numpy()
    design = sp.csr_array(blocks.read_kernel_design(kernel_folder, key, basis.key())[positions])
    turbines = pl.read_parquet(folder / "layout.parquet").height
    turbine = rows["turbine_index"].to_numpy().astype(np.int64)
    group = engine.groups(rows["timestamp"].to_physical().to_numpy())
    speed_bin = rows["speed_bin"].to_numpy().astype(np.int64)
    raw = (
        rng.normal(0, 0.03, turbines)[turbine]
        + design @ truth
        + rng.normal(0, 1, rows.height) * np.asarray(noise_sd)[speed_bin]
    )
    synthetic = rows.with_columns(y=pl.Series(engine.centre(raw, group)))
    stamps = synthetic["timestamp"].unique().sort()
    chosen = stamps.filter(pl.Series(rng.random(len(stamps)) < 0.8)).implode()
    is_train = synthetic["timestamp"].is_in(chosen).to_numpy()
    train, test = synthetic.filter(is_train), synthetic.filter(~is_train)
    spec = ModelSpec(
        name="kernel_linear",
        display_name="kernel linear",
        representation="wind_relative",
        blocks=("turbine_offset", "shared_kernel"),
    )
    context = FitContext("recovery", folder, seed, threads, None, kernel_folder)
    fitted = registry.EngineModel(spec, settings).fit(train, context)
    prediction = fitted.predict(test)
    z = (test["y"].to_numpy() - prediction["mean"].to_numpy()) / np.sqrt(
        prediction["variance"].to_numpy()
    )
    kernel = [p for p in fitted.layout.priors() if p.name == "shared_kernel"][0].columns
    support = np.asarray(abs(design[np.flatnonzero(is_train)]).sum(axis=0)).ravel()
    along = np.linspace(0.25, basis.along_knots[-1], 40)
    across = np.linspace(basis.across_knots[0], basis.across_knots[-1], 25)
    a, c = (g.ravel() for g in np.meshgrid(along, across, indexing="ij"))
    values = basis.values(a, c).toarray()
    frames = []
    for k in range(basis.speed_bins):
        columns = np.arange(k * basis.per_bin, (k + 1) * basis.per_bin)
        functional = np.zeros((len(a), fitted.posterior.mean.size))
        functional[:, kernel.start + columns] = values
        variance = engine.functional_variance(fitted.posterior, fitted.stats, functional)
        frames.append(
            pl.DataFrame(
                {
                    "speed_bin": np.full(len(a), k),
                    "along_d": a,
                    "across_d": c,
                    "true": values @ truth[columns],
                    "recovered": values @ fitted.posterior.mean[kernel.start + columns],
                    "sd": np.sqrt(variance),
                    "support_rows": values @ support[columns],
                }
            )
        )
    grid = pl.concat(frames).with_columns(
        covered=(pl.col("true") - pl.col("recovered")).abs() <= 1.6449 * pl.col("sd")
    )
    supported = grid.filter(pl.col("support_rows") >= 1.0)
    error = (supported["true"] - supported["recovered"]).to_numpy()
    summary = {
        "kernel_coverage_90": float(grid["covered"].cast(pl.Float64).mean() or 0.0),
        "supported_kernel_coverage_90": float(supported["covered"].cast(pl.Float64).mean() or 0.0),
        "supported_share": supported.height / grid.height,
        "supported_correlation": float(
            np.corrcoef(supported["true"].to_numpy(), supported["recovered"].to_numpy())[0, 1]
        ),
        "supported_rmse": float(np.sqrt(np.mean(error**2))),
        "kernel_scale": float(np.abs(grid["true"].to_numpy()).max()),
        "predictive_coverage_90": float(np.mean(np.abs(z) < 1.6449)),
        "train_rows": float(train.height),
        "test_rows": float(test.height),
    }
    return grid, summary
