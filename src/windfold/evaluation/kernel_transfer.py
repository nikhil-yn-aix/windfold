"""how windfold's shared kernel transfers to held-out sectors, read from stored fits.

windfold's prediction on a test row is the kernel's part, the row's kernel design times the
stored kernel coefficients centered within its timestamp, plus the rest. on waked rows, in points
of the no-wake model's squared error, the change from the no-kernel model to windfold is the
kernel term, what adding the kernel's part to the rest gains, plus the base shift, how much
better the rest predicts than the no-kernel model. the two cancel where the kernel takes over
wake that the physics term captures without it. reference_sse, the no-wake model's waked-row
squared error, pools the terms over arcs. the calibration slope regresses the rest's residual on
the kernel's part through the origin: 1 means the kernel's part has the right size on the
held-out rows, below 1 that it is too large. a sector's free-stream turbulence is the median
over its timestamps of the median turbulence intensity of its unwaked rows.
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from windfold.models import engine

REGIMES = ("arc_single", "arc_opposite")
TERMS = ("kernel_effect", "kernel_term", "base_shift", "slope", "correlation", "mean_abs_kernel")
KERNEL_TRANSFER = pl.Schema(
    {
        "split": pl.String,
        "regime": pl.String,
        "sensitivity": pl.String,
        "kernel_scale": pl.Float64,
        "physics_with_kernel": pl.Float64,
        "physics_without_kernel": pl.Float64,
        "test_turbulence": pl.Float64,
        "train_turbulence": pl.Float64,
        "waked_rows": pl.Int64,
        "reference_sse": pl.Float64,
    }
    | dict.fromkeys(TERMS, pl.Float64)
)


def kernel_part(
    design: sp.csr_array, coefficients: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    """each row's kernel design times the coefficients, minus its timestamp's mean."""
    values = np.asarray(design @ coefficients, dtype=np.float64)
    return engine.centre(values, engine.groups(timestamps))


def decompose(
    y: np.ndarray,
    waked: np.ndarray,
    windfold: np.ndarray,
    without_kernel: np.ndarray,
    reference: np.ndarray,
    kernel: np.ndarray,
) -> dict[str, float | None]:
    """the kernel effect, kernel term and base shift in points of the reference's waked-row
    squared error, the calibration slope and correlation and the kernel part's mean size, all
    on waked rows; none of them without waked rows."""
    if not waked.any():
        return dict.fromkeys(TERMS)
    scale = 100.0 / float(np.sum((y - reference)[waked] ** 2))
    k = kernel[waked]
    residual = (y - (windfold - kernel))[waked]
    rest = float(residual @ residual)
    joined = float(np.sum((residual - k) ** 2))
    alone = float(np.sum((y - without_kernel)[waked] ** 2))
    size = float(k @ k)
    varied = bool(k.std() > 0 and residual.std() > 0)
    return {
        "kernel_effect": scale * (alone - joined),
        "kernel_term": scale * (rest - joined),
        "base_shift": scale * (alone - rest),
        "slope": float(residual @ k) / size if size > 0 else None,
        "correlation": float(np.corrcoef(residual, k)[0, 1]) if varied else None,
        "mean_abs_kernel": float(np.abs(k).mean()),
    }


def free_stream_turbulence(rows: pl.DataFrame) -> float | None:
    """the median over timestamps of the median turbulence intensity of unwaked rows."""
    per_time = (
        rows.filter(~pl.col("waked"), pl.col("turbulence_intensity").is_not_null())
        .group_by("timestamp")
        .agg(pl.col("turbulence_intensity").median())
    )
    if per_time.is_empty():
        return None
    return float(np.median(per_time["turbulence_intensity"].to_numpy()))
