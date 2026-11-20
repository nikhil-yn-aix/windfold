"""split manifests: the timestamps each split trains on, tests on or leaves out.

a split lists every timestamp it uses with a role (train, test, buffer, fit, calibrate); a
timestamp it does not use is absent. every candidate split also gets a summary row with its
status and counts, so ineligible arcs, unmatched controls and short histories are declared,
never silently dropped.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import polars as pl

from windfold.config import FarmConfig, Monitoring, Splits
from windfold.schemas import (
    MANIFEST,
    MANIFEST_ROLES,
    PREPARED_ROWS,
    SPLIT_STATUSES,
    SPLIT_SUMMARY,
    conform,
)

FULL_TURN = 360.0
HALF_TURN = 180.0

INCLUDED, INELIGIBLE, UNMATCHED, INSUFFICIENT = SPLIT_STATUSES

type Parts = tuple[pl.DataFrame, pl.DataFrame]


def first_days(timestamps: pl.Series, days: int | None) -> pl.Series:
    """the timestamps on the first days calendar days that have any, or all of them for None;
    a study with a day limit fits and scores each split's train and test timestamps this way."""
    if days is None:
        return timestamps
    kept = timestamps.dt.date().unique().sort().head(days)
    return timestamps.filter(timestamps.dt.date().is_in(kept.implode()))


type Masks = dict[str, np.ndarray]
type Split = tuple[str, str, Masks, np.ndarray | None]
type Declared = tuple[str, str, str, str, np.ndarray | None, np.ndarray | None]


def timestamp_table(
    rows: pl.DataFrame, period: str, start: datetime, end: datetime
) -> pl.DataFrame:
    """one row per timestamp of a period: direction in [0, 360), speed_bin, day, rows, waked_rows.

    raises when a timestamp lies outside [start, end) or its rows disagree on farm direction
    or speed bin.
    """
    rows = conform(rows, PREPARED_ROWS, extra=True).filter(pl.col("period") == period)
    if rows.is_empty():
        raise ValueError(f"no prepared rows in the {period} period")
    bad = rows.select(
        pl.col("farm_direction").is_null().sum() + pl.col("farm_direction").is_nan().sum(),
        pl.col("speed_bin").is_null().sum(),
    ).row(0)
    if any(bad):
        raise ValueError(f"{bad[0]} rows lack farm_direction and {bad[1]} lack speed_bin")
    table = (
        rows.group_by("timestamp")
        .agg(
            direction=pl.col("farm_direction").first(),
            directions=pl.col("farm_direction").n_unique(),
            speed_bin=pl.col("speed_bin").first(),
            speed_bins=pl.col("speed_bin").n_unique(),
            rows=pl.len().cast(pl.Int64),
            waked_rows=pl.col("waked").sum().cast(pl.Int64),
        )
        .sort("timestamp")
    )
    mixed = table.filter((pl.col("directions") > 1) | (pl.col("speed_bins") > 1))
    if not mixed.is_empty():
        raise ValueError(
            f"{mixed.height} timestamps have rows that disagree on farm_direction or speed_bin, "
            f"the first at {mixed['timestamp'][0]}"
        )
    outside = table.filter((pl.col("timestamp") < start) | (pl.col("timestamp") >= end))
    if not outside.is_empty():
        raise ValueError(
            f"{outside.height} {period} timestamps lie outside the window [{start}, {end}), "
            f"the first at {outside['timestamp'][0]}"
        )
    return table.select(
        "timestamp",
        pl.col("direction") % FULL_TURN,
        "speed_bin",
        "rows",
        "waked_rows",
        day=pl.col("timestamp").dt.date(),
    )


def random_folds(table: pl.DataFrame, *, folds: int, seed: int) -> Parts:
    """random_<k>: timestamps dealt at random into folds whose sizes differ by at most one."""
    if folds < 2 or table.height < folds:
        raise ValueError(f"{folds} random folds need at least {folds} timestamps and 2 folds")
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0,)))
    fold = np.empty(table.height, dtype=np.int64)
    fold[rng.permutation(table.height)] = np.arange(table.height) % folds
    return collect(
        table,
        [
            (f"random_{k + 1}", "random", {"train": fold != k, "test": fold == k}, None)
            for k in range(folds)
        ],
    )


def time_block_edges(
    start: datetime, end: datetime, folds: int, month_aligned: bool
) -> list[datetime]:
    """fold boundaries from start to end.

    month-aligned folds are runs of whole calendar months, as equal in count as possible, the
    earlier folds taking the extra months; otherwise the folds have equal length.
    """
    if not month_aligned:
        step = (end - start) / folds
        return [start + step * k for k in range(folds)] + [end]
    months = [start]
    year, month = start.year, start.month
    while True:
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        boundary = datetime(year, month, 1, tzinfo=start.tzinfo)
        if boundary >= end:
            break
        months.append(boundary)
    months.append(end)
    if len(months) - 1 < folds:
        raise ValueError(
            f"[{start}, {end}) spans {len(months) - 1} months, fewer than {folds} folds"
        )
    sizes = [len(part) for part in np.array_split(np.arange(len(months) - 1), folds)]
    return [months[i] for i in np.cumsum([0, *sizes])]


def time_blocks(
    table: pl.DataFrame,
    *,
    start: datetime,
    end: datetime,
    folds: int,
    buffer_days: int,
    month_aligned: bool,
) -> Parts:
    """time_1 to time_<folds>: contiguous test blocks; training loses buffer_days on each side.

    a block without test or training timestamps is declared insufficient data.
    """
    instant = instants(table)
    edges = [moment(edge) for edge in time_block_edges(start, end, folds, month_aligned)]
    buffer = np.timedelta64(buffer_days, "D")
    splits: list[Split] = []
    declared: list[Declared] = []
    for k in range(folds):
        name = f"time_{k + 1}"
        test = (instant >= edges[k]) & (instant < edges[k + 1])
        near = (instant >= edges[k] - buffer) & (instant < edges[k + 1] + buffer)
        if test.any() and not near.all():
            masks = {"train": ~near, "test": test, "buffer": near & ~test}
            splits.append((name, "time_blocked", masks, None))
        else:
            reason = f"{test.sum()} test and {(~near).sum()} training timestamps"
            declared.append((name, "time_blocked", INSUFFICIENT, reason, ~near, test))
    return collect(table, splits, declared)


def stratum_codes(table: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """control strata: speed bin by month of the year, as codes and readable labels."""
    keys = table.select(
        pl.col("timestamp").dt.month().cast(pl.Int64), pl.col("speed_bin").cast(pl.Int64)
    )
    joint = keys.to_numpy() @ np.array([256, 1])
    unique, codes = np.unique(joint, return_inverse=True)
    labels = [f"speed bin {key % 256} in month {key // 256}" for key in unique]
    return codes, labels


def control_sample(
    pool: np.ndarray,
    removed: np.ndarray,
    rows: np.ndarray,
    strata: np.ndarray,
    labels: Sequence[str],
    tolerance: float,
    rng: np.random.Generator,
) -> np.ndarray | str:
    """whole timestamps from pool removing as many rows as removed, stratum by stratum.

    each stratum takes its candidates in random order while their rows stay within its target,
    then adds the subset of its remaining candidates that brings the rows closest to the target
    (closest_subset). a stratum with fewer rows than its target gives all of them. returns the
    chosen mask, or the reason when a stratum falls short by more than tolerance or the total
    misses the opposite deletion by more than tolerance.
    """
    target = np.bincount(strata[removed], weights=rows[removed], minlength=len(labels))
    available = np.bincount(strata[pool], weights=rows[pool], minlength=len(labels))
    short = np.flatnonzero(available < (1 - tolerance) * target)
    if short.size:
        worst = short[np.argmax(target[short] - available[short])]
        return (
            f"{short.size} strata lack rows; {labels[worst]} needs {target[worst]:.0f}, "
            f"has {available[worst]:.0f}"
        )
    order = rng.permutation(np.flatnonzero(pool))
    order = order[np.argsort(strata[order], kind="stable")]
    edges = np.r_[np.flatnonzero(np.r_[True, strata[order][1:] != strata[order][:-1]]), order.size]
    chosen = np.zeros(pool.size, dtype=bool)
    for start, stop in pairwise(edges if order.size else []):
        members = order[start:stop]
        goal = int(target[strata[members[0]]])
        sizes = rows[members]
        prefix = int(np.searchsorted(np.cumsum(sizes), goal, side="right"))
        chosen[members[:prefix]] = True
        deficit = goal - int(sizes[:prefix].sum())
        if deficit > 0 and prefix < members.size:
            chosen[members[prefix:][closest_subset(sizes[prefix:], deficit)]] = True
    wanted, got = int(rows[removed].sum()), int(rows[chosen].sum())
    if abs(got - wanted) > tolerance * wanted:
        return f"control removes {got} rows against {wanted}, beyond {tolerance:.0%}"
    return chosen


def closest_subset(sizes: np.ndarray, goal: int) -> list[int]:
    """positions of sizes whose sum comes closest to goal, the larger sum on a tie, built from
    the earliest positions."""
    cap = goal + int(sizes.max())
    reach: dict[int, tuple[int, ...]] = {0: ()}
    used: dict[int, int] = {}
    for position, size in enumerate(sizes.tolist()):
        if used.get(size, 0) >= cap // size:
            continue
        used[size] = used.get(size, 0) + 1
        for total, picked in list(reach.items()):
            if total + size <= cap and total + size not in reach:
                reach[total + size] = (*picked, position)
        if goal in reach:
            break
    return list(reach[min(reach, key=lambda total: (abs(total - goal), -total))])


def arcs(
    table: pl.DataFrame,
    *,
    count: int,
    width: float,
    buffer: float,
    min_rows: int,
    min_days: int,
    tolerance: float,
    seed: int,
) -> Parts:
    """arc_<start>_single, _opposite and _control for each test arc of farm direction.

    single drops the test arc and buffer degrees each side from training; opposite also drops
    the opposite arc with its buffer; control instead drops as many rows of other directions,
    matched by speed bin and month of the year (control_sample). test rows are the same in all
    three. an arc below min_rows rows or min_days days is ineligible, and a control that cannot
    be matched is declared unmatched.
    """
    if count < 1 or FULL_TURN % count or not np.isclose(count * width, FULL_TURN):
        raise ValueError(f"{count} arcs of {width} degrees do not tile the circle")
    direction = table["direction"].to_numpy()
    rows = table["rows"].to_numpy()
    day = table["day"].to_physical().to_numpy()
    strata, labels = stratum_codes(table)
    zone = width + 2 * buffer
    splits: list[Split] = []
    declared: list[Declared] = []
    for k in range(count):
        start = round(k * FULL_TURN / count)
        name = f"arc_{start:03d}"
        test = np.mod(direction - start, FULL_TURN) < width
        near = np.mod(direction - start + buffer, FULL_TURN) < zone
        opposite = np.mod(direction - start - HALF_TURN + buffer, FULL_TURN) < zone
        train = ~near
        removed = train & opposite
        test_rows, test_days = int(rows[test].sum()), np.unique(day[test]).size
        if test_rows < min_rows or test_days < min_days:
            reason = (
                f"{test_rows} rows on {test_days} days; needs {min_rows} rows and {min_days} days"
            )
            declared += [
                (f"{name}_{variant}", f"arc_{variant}", INELIGIBLE, reason, None, test)
                for variant in ("single", "opposite", "control")
            ]
            continue
        edge = near & ~test
        splits.append(
            (f"{name}_single", "arc_single", {"train": train, "test": test, "buffer": edge}, None)
        )
        masks = {"train": train & ~opposite, "test": test, "buffer": edge}
        splits.append((f"{name}_opposite", "arc_opposite", masks, removed))
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1, start)))
        chosen = control_sample(train & ~opposite, removed, rows, strata, labels, tolerance, rng)
        if isinstance(chosen, str):
            declared.append((f"{name}_control", "arc_control", UNMATCHED, chosen, None, test))
            continue
        masks = {"train": train & ~chosen, "test": test, "buffer": edge}
        splits.append((f"{name}_control", "arc_control", masks, chosen))
    return collect(table, splits, declared)


def histories(
    table: pl.DataFrame,
    *,
    starts: Sequence[datetime],
    horizons: Sequence[int],
    test_start: datetime,
    test_end: datetime,
    gap_days: int,
    min_timestamps: int,
) -> Parts:
    """history_<days>d_from_<start>: train on [start, start + days), test on the common window.

    raises when a history would end less than gap_days before the test window; a history with
    fewer than min_timestamps training timestamps is declared insufficient data.
    """
    instant = instants(table)
    test = (instant >= moment(test_start)) & (instant < moment(test_end))
    splits: list[Split] = []
    declared: list[Declared] = []
    for start in starts:
        for days in horizons:
            end = start + timedelta(days=days)
            if end + timedelta(days=gap_days) > test_start:
                raise ValueError(
                    f"a {days}-day history from {start} ends at {end}, less than {gap_days} days "
                    f"before the test window starts at {test_start}"
                )
            name = f"history_{days:03d}d_from_{start:%Y-%m-%d}"
            train = (instant >= moment(start)) & (instant < moment(end))
            reason = shortfall(int(train.sum()), int(test.sum()), min_timestamps)
            if reason:
                declared.append((name, "history", INSUFFICIENT, reason, train, test))
            else:
                splits.append((name, "history", {"train": train, "test": test}, None))
    return collect(table, splits, declared)


def thresholds(
    table: pl.DataFrame,
    *,
    starts: Sequence[datetime],
    horizons: Sequence[int],
    min_history_days: int,
    fit_fraction: float,
    calibrate_from_fraction: float,
    min_timestamps: int,
) -> Parts:
    """threshold_<days>d_from_<start> for histories of min_history_days and longer.

    fit on [0, fit_fraction H), leave [fit_fraction H, calibrate_from_fraction H) as buffer,
    calibrate on [calibrate_from_fraction H, H). declared insufficient data when the history
    has fewer than min_timestamps timestamps or the fit or calibrate window is empty.
    """
    if not 0 < fit_fraction <= calibrate_from_fraction < 1:
        raise ValueError(f"fractions {fit_fraction} and {calibrate_from_fraction} are not ordered")
    instant = instants(table)
    splits: list[Split] = []
    declared: list[Declared] = []
    for start in starts:
        for days in (h for h in horizons if h >= min_history_days):
            span = timedelta(days=days)
            bounds = [
                moment(start + offset)
                for offset in (
                    timedelta(0),
                    span * fit_fraction,
                    span * calibrate_from_fraction,
                    span,
                )
            ]
            fit, gap, calibrate = ((instant >= lo) & (instant < hi) for lo, hi in pairwise(bounds))
            name = f"threshold_{days:03d}d_from_{start:%Y-%m-%d}"
            history = fit | gap | calibrate
            if history.sum() < min_timestamps:
                reason = f"{history.sum()} history timestamps; needs {min_timestamps}"
            elif not fit.any() or not calibrate.any():
                reason = f"{fit.sum()} fit and {calibrate.sum()} calibrate timestamps"
            else:
                reason = None
            if reason:
                declared.append((name, "threshold", INSUFFICIENT, reason, fit, calibrate))
            else:
                masks = {"fit": fit, "buffer": gap, "calibrate": calibrate}
                splits.append((name, "threshold", masks, None))
    return collect(table, splits, declared)


def farm_manifest(
    rows: pl.DataFrame,
    farm: FarmConfig,
    splits: Splits,
    monitoring: Monitoring,
    period: str,
    seed: int,
) -> Parts:
    """every split of one farm and period: the manifest and the summary of every candidate split.

    the confirmatory period uses the farm's confirmatory window and histories. a farm with its
    own time_blocks gets that many blocks of equal length; otherwise folds are month-aligned.
    """
    if period not in ("development", "confirmatory"):
        raise ValueError(f"splits are built for development or confirmatory periods, not {period}")
    window = getattr(farm.windows, period)
    plan = farm.histories if period == "development" else farm.confirmatory_histories
    if window is None or plan is None:
        raise ValueError(f"{farm.name} has no {period} window and histories")
    inside = (
        window.start <= min(plan.starts)
        and max(plan.starts) < window.end
        and window.start <= plan.test.start
        and plan.test.end <= window.end
    )
    if not inside:
        raise ValueError(
            f"{farm.name} {period} histories reach outside [{window.start}, {window.end})"
        )
    table = timestamp_table(rows, period, window.start, window.end)
    horizons = plan.horizons_days or splits.history_horizons_days
    blocks = farm.time_blocks
    parts = [
        random_folds(table, folds=splits.random_folds, seed=seed),
        time_blocks(
            table,
            start=window.start,
            end=window.end,
            folds=blocks.folds if blocks else splits.time_folds,
            buffer_days=blocks.buffer_days if blocks else splits.time_buffer_days,
            month_aligned=blocks is None,
        ),
        arcs(
            table,
            count=splits.arc_count,
            width=splits.arc_width_deg,
            buffer=splits.arc_buffer_deg,
            min_rows=splits.arc_min_rows,
            min_days=splits.arc_min_days,
            tolerance=splits.control_tolerance,
            seed=seed,
        ),
        histories(
            table,
            starts=plan.starts,
            horizons=horizons,
            test_start=plan.test.start,
            test_end=plan.test.end,
            gap_days=splits.history_gap_days,
            min_timestamps=splits.history_min_timestamps,
        ),
        thresholds(
            table,
            starts=plan.starts,
            horizons=horizons,
            min_history_days=monitoring.min_history_days,
            fit_fraction=monitoring.fit_fraction,
            calibrate_from_fraction=monitoring.calibrate_from_fraction,
            min_timestamps=splits.history_min_timestamps,
        ),
    ]
    manifest = check_manifest(pl.concat([part[0] for part in parts]))
    return manifest, pl.concat([part[1] for part in parts])


def check_manifest(manifest: pl.DataFrame) -> pl.DataFrame:
    """conform to MANIFEST, with known roles and each timestamp at most once per split."""
    manifest = conform(manifest, MANIFEST)
    unknown = set(manifest["role"].unique()) - set(MANIFEST_ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}; known: {list(MANIFEST_ROLES)}")
    repeated = manifest.filter(pl.struct("split", "timestamp").is_duplicated())
    if not repeated.is_empty():
        raise ValueError(f"split {repeated['split'][0]} lists {repeated['timestamp'][0]} twice")
    return manifest


def shortfall(train: int, test: int, min_timestamps: int) -> str | None:
    if train < min_timestamps:
        return f"{train} training timestamps; needs {min_timestamps}"
    if test == 0:
        return "no test timestamps"
    return None


def moment(value: datetime) -> np.datetime64:
    if value.tzinfo is None:
        raise ValueError(f"{value} has no time zone")
    return np.datetime64(value.astimezone(UTC).replace(tzinfo=None), "us")


def instants(table: pl.DataFrame) -> np.ndarray:
    return table["timestamp"].to_numpy()


def collect(
    table: pl.DataFrame, splits: list[Split], declared: list[Declared] | None = None
) -> Parts:
    """the manifest of the included splits and a summary row for every split, declared or not."""
    timestamp = table["timestamp"]
    frames, summaries = [], []
    for name, regime, masks, removed in splits:
        frames.append(
            pl.concat(
                pl.DataFrame({"timestamp": timestamp.filter(pl.Series(mask))}).select(
                    split=pl.lit(name),
                    regime=pl.lit(regime),
                    timestamp="timestamp",
                    role=pl.lit(role),
                )
                for role, mask in masks.items()
            ).sort("timestamp")
        )
        train = masks.get("train", masks.get("fit"))
        test = masks.get("test", masks.get("calibrate"))
        summaries.append(summary(table, name, regime, INCLUDED, None, train, test, removed))
    summaries += [summary(table, *declaration) for declaration in declared or []]
    manifest = pl.concat(frames) if frames else pl.DataFrame(schema=MANIFEST)
    return manifest, pl.DataFrame(summaries, schema=SPLIT_SUMMARY, orient="row")


def summary(
    table: pl.DataFrame,
    name: str,
    regime: str,
    status: str,
    reason: str | None,
    train: np.ndarray | None,
    test: np.ndarray | None,
    removed: np.ndarray | None = None,
) -> tuple:
    rows = table["rows"].to_numpy()

    def count(mask: np.ndarray | None, values: np.ndarray | None = None) -> int | None:
        if mask is None:
            return None
        return int(mask.sum()) if values is None else int(values[mask].sum())

    days = None if test is None else np.unique(table["day"].to_physical().to_numpy()[test]).size
    return (
        name,
        regime,
        status,
        reason,
        count(train),
        count(train, rows),
        count(test),
        count(test, rows),
        count(test, table["waked_rows"].to_numpy()),
        days,
        count(removed, rows),
    )
