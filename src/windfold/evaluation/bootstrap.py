"""synchronized calendar-block bootstrap: one shared draw per replicate for every endpoint.

each window (the study period for the cross-validation regimes, the common test window for the
histories and monitoring) is cut into non-overlapping calendar blocks from its first day. one
matrix of uniforms per bootstrap gives every replicate's blocks in every window, in draw order,
so every model, split, variant and injected copy in a window sees the same blocks. a replicate's
seed indices are drawn once and shared by every seeded model.
"""

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl


def farm_seed(study_seed: int, farm: str) -> int:
    """a farm's bootstrap seed, fixed by the study seed and the farm's name."""
    sequence = np.random.SeedSequence([study_seed, zlib.crc32(farm.encode())])
    return int(sequence.generate_state(1)[0])


def window_days(start: datetime, end: datetime) -> tuple[date, int]:
    """the first day of a window and the days it touches, the last partial day included."""
    first = start.date()
    last = (end - timedelta(microseconds=1)).date()
    return first, (last - first).days + 1


def block_count(start: date, end: date, block_days: int) -> int:
    """blocks needed to cover the days [start, end)."""
    return -(-(end - start).days // block_days)


def block_of(days: pl.Series, origin: date, block_days: int) -> np.ndarray:
    offsets = (days - origin).dt.total_days().to_numpy()
    return offsets // block_days


def generator(seed: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(stream,))))


def draw_blocks(
    block_counts: Mapping[str, int], replicates: int, seed: int
) -> dict[str, np.ndarray]:
    """each window's blocks in draw order, replicates by blocks, from one shared uniform draw.

    the k-th draw of every replicate comes from the k-th row of uniforms, so a window's draw does
    not depend on which other windows are drawn with it.
    """
    widest = max(block_counts.values())
    uniform = generator(seed, 0).random((widest, replicates)).T
    return {
        window: np.minimum((uniform[:, :blocks] * blocks).astype(np.int64), blocks - 1)
        for window, blocks in block_counts.items()
    }


def draw_seeds(replicates: int, seeds: int, seed: int) -> np.ndarray:
    """each replicate's seed indices, drawn with replacement and shared by all seeded models."""
    return generator(seed, 1).integers(0, seeds, size=(replicates, seeds))


def multiplicity(drawn: np.ndarray, blocks: int) -> np.ndarray:
    """how often each block occurs in each replicate's draw."""
    counts = np.zeros((drawn.shape[0], blocks), dtype=np.float64)
    np.add.at(counts, (np.arange(drawn.shape[0])[:, None], drawn), 1.0)
    return counts


@dataclass(frozen=True)
class Totals:
    """block sums of a reduction: values[block, column], one column per label row."""

    labels: pl.DataFrame
    values: np.ndarray

    def columns(self, quantity: str, **keys: object) -> np.ndarray:
        """column indices of a quantity whose labels match keys; a sequence matches any member."""
        chosen = self.labels.with_row_index("column").filter(pl.col("quantity") == quantity)
        for name, value in keys.items():
            if isinstance(value, Sequence) and not isinstance(value, str):
                chosen = chosen.filter(pl.col(name).is_in(pl.Series(list(value)).implode()))
            elif value is None:
                chosen = chosen.filter(pl.col(name).is_null())
            else:
                chosen = chosen.filter(pl.col(name) == value)
        return chosen["column"].to_numpy()


def block_totals(
    reduction: pl.DataFrame,
    keys: Sequence[str],
    quantities: Sequence[str],
    origin: date,
    block_days: int,
    blocks: int,
) -> Totals:
    """sum a day-keyed reduction into calendar blocks, one column per key combination and
    quantity. raises when a day falls outside the window's blocks."""
    block = block_of(reduction["day"], origin, block_days)
    if block.size and (block.min() < 0 or block.max() >= blocks):
        raise ValueError(f"days fall outside the {blocks} blocks from {origin}")
    grouped = (
        reduction.with_columns(block=pl.Series(block))
        .group_by("block", *keys)
        .agg(pl.col(q).sum() for q in quantities)
    )
    combos = grouped.select(keys).unique().sort(keys, nulls_last=True).with_row_index("combo")
    grouped = grouped.join(combos, on=list(keys), how="left", nulls_equal=True)
    values = np.zeros((blocks, combos.height * len(quantities)))
    block_index = grouped["block"].to_numpy()
    combo_index = grouped["combo"].to_numpy().astype(np.int64)
    for q, quantity in enumerate(quantities):
        values[block_index, combo_index * len(quantities) + q] = (
            grouped[quantity].fill_null(np.nan).to_numpy()
        )
    labels = (
        combos.drop("combo")
        .join(pl.DataFrame({"quantity": list(quantities)}), how="cross")
        .select(*keys, "quantity")
    )
    return Totals(labels, values)


def resample(values: np.ndarray, counts: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """replicate totals: counts (replicates by blocks) times block sums (blocks by columns)."""
    out = np.empty((counts.shape[0], values.shape[1]))
    for first in range(0, counts.shape[0], chunk):
        out[first : first + chunk] = counts[first : first + chunk] @ values
    return out


def seed_mean(per_seed: np.ndarray, seeds: np.ndarray | None) -> np.ndarray:
    """per-split metric of a seeded model: the mean over the replicate's drawn seeds.

    per_seed is replicates by seeds; seeds holds each replicate's drawn indices, or none for the
    observed estimate, the plain mean.
    """
    if seeds is None or per_seed.shape[1] == 1:
        return per_seed.mean(axis=1)
    return np.take_along_axis(per_seed, seeds, axis=1).mean(axis=1)
