"""label-free trial planning: the chance that each sector has enough support in both arms of a
toggled control trial after n days, from a block bootstrap of the multi-year direction record.

a synthetic trial strings together whole calendar blocks of the record drawn at random (the last
one cut to length). the arm of each 10-minute timestamp follows the toggle schedule on the
synthetic clock: the first toggle_minutes are one arm, the next toggle_minutes the other and
so on. a sector has enough support when each arm holds at least min_rows turbine rows on at
least min_days distinct days, the rule an arc must meet to be tested. a planning aid, not a
guaranteed duration.
"""

from collections.abc import Sequence
from datetime import date
from math import gcd

import numpy as np
import polars as pl

DAY_MINUTES = 24 * 60

TRIAL = pl.Schema(
    {
        "duration_days": pl.Int64,
        "arc": pl.Int64,
        "probability": pl.Float64,
        "median_rows": pl.Float64,
    }
)


def block_tables(
    table: pl.DataFrame,
    origin: date,
    block_days: int,
    blocks: int,
    arc_count: int,
    toggle_minutes: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """rows and distinct days per block, phase, first days kept, arc and arm.

    table is splits.timestamp_table: timestamp, direction, rows and day. a block placed at
    synthetic minute t0 has phase t0 modulo two toggles; the arrays are indexed
    [block, phase, days kept (0 to block_days), arc, arm].
    """
    cycle = 2 * toggle_minutes
    step = gcd(block_days * DAY_MINUTES, cycle)
    phases = list(range(0, cycle, step))
    day_offset = (table["day"] - origin).dt.total_days().to_numpy()
    block = day_offset // block_days
    day_in_block = day_offset % block_days
    clock = table["timestamp"].dt.hour().cast(pl.Int64) * 60 + table["timestamp"].dt.minute()
    minute = clock.to_numpy() + day_in_block * DAY_MINUTES
    arc = np.floor(table["direction"].to_numpy() % 360.0 / (360.0 / arc_count)).astype(np.int64)
    rows = table["rows"].to_numpy().astype(float)
    shape = (blocks, len(phases), block_days + 1, arc_count, 2)
    row_sums, day_counts = np.zeros(shape), np.zeros(shape)
    for p, phase in enumerate(phases):
        arm = ((minute + phase) // toggle_minutes) % 2
        for kept in range(1, block_days + 1):
            inside = day_in_block < kept
            np.add.at(row_sums[:, p, kept], (block[inside], arc[inside], arm[inside]), rows[inside])
            seen = np.zeros(shape[:1] + shape[3:] + (block_days,), dtype=bool)
            seen[block[inside], arc[inside], arm[inside], day_in_block[inside]] = True
            day_counts[:, p, kept] = seen.sum(axis=-1)
    return row_sums, day_counts, phases


def trial_support(
    table: pl.DataFrame,
    origin: date,
    blocks: int,
    *,
    block_days: int,
    arc_count: int,
    toggle_minutes: int,
    durations_days: Sequence[int],
    min_rows: int,
    min_days: int,
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    """per trial duration and arc: the share of bootstrap trials in which both arms have enough
    support, and the median over trials of the smaller arm's rows."""
    row_sums, day_counts, phases = block_tables(
        table, origin, block_days, blocks, arc_count, toggle_minutes
    )
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(4,))))
    cycle = 2 * toggle_minutes
    step = cycle // len(phases)
    records = []
    for days in durations_days:
        slots = -(-days // block_days)
        drawn = rng.integers(0, blocks, size=(replicates, slots))
        phase = np.array([(k * block_days * DAY_MINUTES) % cycle // step for k in range(slots)])
        kept = np.full(slots, block_days)
        kept[-1] = days - (slots - 1) * block_days
        rows = row_sums[drawn, phase[None, :], kept[None, :]].sum(axis=1)
        seen = day_counts[drawn, phase[None, :], kept[None, :]].sum(axis=1)
        enough = ((rows >= min_rows) & (seen >= min_days)).all(axis=-1)
        weaker = rows.min(axis=-1)
        for arc in range(arc_count):
            start = round(arc * 360 / arc_count)
            records.append(
                (days, start, float(enough[:, arc].mean()), float(np.median(weaker[:, arc])))
            )
    return pl.DataFrame(records, schema=TRIAL, orient="row")
