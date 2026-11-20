"""a paired calendar-block bootstrap of pooled waked-row skill, shared by the analyses.

a farm's development window is cut into blocks of BLOCK_DAYS days from its first day. a weights
matrix has one row per replicate and one column per block: row 0 weighs every block once, the
observed estimate, and row r counts how often replicate r drew each block. every model of a farm
is resampled with the same weights, so differences between models are paired, and the generator
that draws them is seeded by the caller.
"""

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from windfold.config import load_farm
from windfold.evaluation.bootstrap import block_count, multiplicity, window_days

BLOCK_DAYS = 7
REPLICATES = 2000


def window_blocks(farm: str, configs: Path | None = None) -> tuple[date, int]:
    """the first day of the farm's development window and how many blocks cover the window."""
    window = load_farm(farm, configs).windows.development
    first, days = window_days(window.start, window.end)
    return first, block_count(first, first + timedelta(days=days), BLOCK_DAYS)


def block_sse(frame: pl.DataFrame, origin: date) -> pl.DataFrame:
    """the squared errors of prediction rows summed by block from origin: block and sse."""
    block = (pl.col("timestamp").dt.date() - pl.lit(origin)).dt.total_days() // BLOCK_DAYS
    return frame.group_by(block.alias("block")).agg(sse=((pl.col("y") - pl.col("mean")) ** 2).sum())


def block_vector(parts: Iterable[pl.DataFrame], blocks: int) -> np.ndarray:
    """the sse of several block_sse frames added up by block, one entry per block."""
    frames = list(parts)
    vector = np.zeros(blocks)
    if not frames:
        return vector
    totals = pl.concat(frames).group_by("block").agg(pl.col("sse").sum())
    index = totals["block"].to_numpy()
    if index.size and (index.min() < 0 or index.max() >= blocks):
        raise ValueError(f"rows fall outside the {blocks} blocks of the window")
    vector[index] = totals["sse"].to_numpy()
    return vector


def weights(blocks: int, replicates: int, generator: np.random.Generator) -> np.ndarray:
    """replicates + 1 rows of block weights: ones, then each replicate's counts of a draw of
    blocks with replacement."""
    drawn = generator.integers(0, blocks, size=(replicates, blocks))
    return np.vstack([np.ones((1, blocks)), multiplicity(drawn, blocks)])


def replicate_sse(frames: Iterable[pl.DataFrame], origin: date, draws: np.ndarray) -> np.ndarray:
    """each replicate's pooled squared error over prediction frames, draws from weights."""
    return draws @ block_vector((block_sse(f, origin) for f in frames), draws.shape[1])


def skill(sse: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """skill in percent against a reference's squared error, replicate by replicate."""
    return 100 * (1 - sse / reference)


def interval(values: np.ndarray) -> dict[str, float]:
    """replicate 0 as the estimate and the 95% percentile interval of the other replicates."""
    low, high = np.nanquantile(values[1:], [0.025, 0.975])
    return {"estimate": float(values[0]), "low": float(low), "high": float(high)}
