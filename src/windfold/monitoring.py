"""monitoring: calibration scales, daily alarms, injected drops, recall, exceedance and delay.

residuals are on the leave-one-out contrast scale, m / (m - 1) times (y - mean). a turbine's
daily mean residual over n rows is standardized by sigma * sqrt((1 + rho) / ((1 - rho) n)),
with sigma and rho (lag-1, between consecutive 10-minute rows) from its calibration residuals.
an alarm fires when the standardized mean falls below the normal quantile that gives
false_alarms_per_turbine_month alarms in a month of daily tests. alarms live on day grids
(turbines by days of the test window), which the bootstrap rebuilds from the window's drawn
calendar blocks, so every monitor and injected copy sees the same rebuilt days.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
from scipy.special import ndtri

from windfold.evaluation.schemas import EPISODES, SCALES

INTERVAL = timedelta(minutes=10)
MONTH_DAYS = 365.2425 / 12


def inject_drop(
    rows: pl.DataFrame, *, turbine: str, start: datetime, end: datetime, drop: float
) -> pl.DataFrame:
    """rows with a power drop on one turbine added to their leave-one-out contrasts in [start, end).

    rows hold timestamp, turbine, contrast and valid_turbines. masks and features stay frozen, so
    only timestamps where the turbine has a row change: its contrast moves by log(1 - drop) and
    every peer's by -log(1 - drop) / (valid_turbines - 1).
    """
    if not 0 < drop < 1:
        raise ValueError(f"a drop is a share of power in (0, 1), not {drop}")
    shift = math.log1p(-drop)
    hit = rows.filter(
        (pl.col("turbine") == turbine)
        & (pl.col("timestamp") >= start)
        & (pl.col("timestamp") < end)
    )["timestamp"]
    affected = pl.col("timestamp").is_in(hit.implode())
    peers = pl.col("valid_turbines").cast(pl.Float64) - 1
    return rows.with_columns(
        contrast=pl.when(affected & (pl.col("turbine") == turbine))
        .then(pl.col("contrast") + shift)
        .when(affected)
        .then(pl.col("contrast") - shift / peers)
        .otherwise(pl.col("contrast"))
    )


def residuals(scored: pl.DataFrame) -> pl.DataFrame:
    """timestamp, turbine, day and residual = scale * error from metrics.scored rows."""
    return scored.select(
        "timestamp", "turbine", "day", residual=pl.col("scale") * pl.col("error")
    ).sort("turbine", "timestamp")


def calibration_scales(
    rows: pl.DataFrame, min_rows_per_day: int, min_calibration_days: int
) -> pl.DataFrame:
    """per turbine: calibration rows, days with at least min_rows_per_day rows, sigma, rho and
    whether it can be monitored (enough days, sigma above zero, |rho| below one)."""
    lagged = rows.sort("turbine", "timestamp").with_columns(
        previous=pl.col("residual").shift(1).over("turbine"),
        consecutive=pl.col("timestamp").diff().over("turbine") == INTERVAL,
    )
    rho = (
        lagged.filter(pl.col("consecutive"))
        .group_by("turbine")
        .agg(rho=pl.corr("residual", "previous"))
    )
    days = (
        rows.group_by("turbine", "day")
        .len()
        .group_by("turbine")
        .agg(days=(pl.col("len") >= min_rows_per_day).sum().cast(pl.Int64))
    )
    scales = (
        rows.group_by("turbine")
        .agg(rows=pl.len().cast(pl.Int64), sigma=pl.col("residual").std())
        .join(days, on="turbine", how="left")
        .join(rho, on="turbine", how="left")
        .with_columns(pl.col("rho").fill_nan(None))
    )
    eligible = (
        (pl.col("days") >= min_calibration_days) & (pl.col("sigma") > 0) & (pl.col("rho").abs() < 1)
    ).fill_null(False)
    return (
        scales.with_columns(eligible=eligible).select(SCALES.names()).cast(SCALES).sort("turbine")
    )


def alarm_quantile(false_alarms_per_turbine_month: float) -> float:
    return float(ndtri(false_alarms_per_turbine_month / MONTH_DAYS))


def daily_means(rows: pl.DataFrame) -> pl.DataFrame:
    """per turbine and day: rows and the mean residual."""
    return (
        rows.group_by("turbine", "day")
        .agg(rows=pl.len().cast(pl.Int64), residual=pl.col("residual").mean())
        .sort("turbine", "day")
    )


def day_grids(
    daily: pl.DataFrame,
    scales: pl.DataFrame,
    turbines: Sequence[str],
    window_start: date,
    window_days: int,
    min_rows_per_day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """turbines by days of the window from daily_means: monitored, the standardized daily mean
    z, and sd, the standard deviation of the daily mean. a day is monitored when the turbine is
    eligible and has at least min_rows_per_day rows."""
    index = {name: k for k, name in enumerate(turbines)}
    usable = daily.join(scales, on="turbine", how="inner", suffix="_calibration").filter(
        pl.col("eligible") & (pl.col("rows") >= min_rows_per_day)
    )
    shape = (len(turbines), window_days)
    monitored, z, sd = np.zeros(shape, dtype=bool), np.full(shape, np.nan), np.full(shape, np.nan)
    if usable.is_empty():
        return monitored, z, sd
    turbine = np.array([index[name] for name in usable["turbine"].to_list()])
    day = (usable["day"] - window_start).dt.total_days().to_numpy()
    rho, count = usable["rho"].to_numpy(), usable["rows"].to_numpy()
    scale = usable["sigma"].to_numpy() * np.sqrt((1 + rho) / ((1 - rho) * count))
    monitored[turbine, day] = True
    sd[turbine, day] = scale
    z[turbine, day] = usable["residual"].to_numpy() / scale
    return monitored, z, sd


def alarm_grid(
    monitored: np.ndarray, z: np.ndarray, sd: np.ndarray, quantile: float, drop: float
) -> np.ndarray:
    """alarms on monitored days, with every daily mean shifted by log(1 - drop) (none for 0)."""
    with np.errstate(invalid="ignore"):
        return monitored & (z + math.log1p(-drop) / sd < quantile)


def injections(
    turbines: Sequence[str], window_days: int, copies: int, episode_days: int, seed: int
) -> pl.DataFrame:
    """matched copies: an eligible turbine and a first day (offset into the window) for an
    episode that fits in the window, shared by every monitor and drop size."""
    if not turbines or window_days < episode_days:
        raise ValueError("injections need eligible turbines and a window as long as an episode")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(3,))))
    return pl.DataFrame(
        {
            "injection": np.arange(copies),
            "turbine": np.asarray(turbines)[rng.integers(0, len(turbines), copies)],
            "offset": rng.integers(0, window_days - episode_days + 1, copies),
        },
        schema={"injection": pl.Int64, "turbine": pl.String, "offset": pl.Int64},
    )


def trajectories(drawn: np.ndarray, window_days: int, block_days: int) -> np.ndarray:
    """replicates by slots: the window days of each replicate's drawn blocks, concatenated in
    draw order and padded with -1."""
    lengths = np.minimum(block_days, window_days - drawn * block_days)
    within = np.arange(block_days)
    days = drawn[:, :, None] * block_days + within
    keep = within < lengths[:, :, None]
    slots = np.full((drawn.shape[0], int(lengths.sum(axis=1).max())), -1, dtype=np.int64)
    for r in range(drawn.shape[0]):
        rebuilt = days[r][keep[r]]
        slots[r, : rebuilt.size] = rebuilt
    return slots


def detections(
    alarms: np.ndarray,
    copies: pl.DataFrame,
    turbines: Sequence[str],
    slots: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """replicates by copies: days to the first alarm on the copy's turbine within horizon days
    of its first slot, and whether one came; an undetected copy is censored at the days it had."""
    index = {name: k for k, name in enumerate(turbines)}
    turbine = np.array([index[name] for name in copies["turbine"].to_list()])
    offset = copies["offset"].to_numpy()
    padded = np.pad(slots, ((0, 0), (0, horizon)), constant_values=-1)
    window = padded[:, offset[:, None] + np.arange(horizon)]
    extended = np.pad(alarms, ((0, 0), (0, 1)))
    fired = extended[turbine[None, :, None], np.where(window < 0, alarms.shape[1], window)]
    detected = fired.any(axis=2)
    available = (window >= 0).sum(axis=2)
    return np.where(detected, fired.argmax(axis=2) + 1, available), detected


def kaplan_meier(times: np.ndarray, events: np.ndarray, horizon: int) -> np.ndarray:
    """replicates by days 1 to horizon: the kaplan-meier share not yet detected; detections
    count before censorings on the same day."""
    survival = np.ones((times.shape[0], horizon))
    current = np.ones(times.shape[0])
    for day in range(1, horizon + 1):
        at_risk = (times >= day).sum(axis=1)
        failed = ((times == day) & events).sum(axis=1)
        current = current * (1 - failed / np.maximum(at_risk, 1))
        survival[:, day - 1] = current
    return survival


def median_delay(survival: np.ndarray) -> np.ndarray:
    """first day the survival reaches one half; nan when it does not (beyond the horizon).
    survival within sqrt(eps) of one half counts as reaching it, as in R's survival package, so
    a float product that is one half in exact arithmetic does not delay the median a day."""
    reached = survival <= 0.5 + np.sqrt(np.finfo(float).eps)
    return np.where(reached.any(axis=1), reached.argmax(axis=1) + 1.0, np.nan)


def day_counts(slots: np.ndarray, window_days: int) -> np.ndarray:
    """replicates by window days: how often each day occurs among the rebuilt slots."""
    counts = np.zeros((slots.shape[0], window_days + 1))
    np.add.at(
        counts, (np.arange(slots.shape[0])[:, None], np.where(slots < 0, window_days, slots)), 1
    )
    return counts[:, :window_days]


def exceedance(alarms: np.ndarray, monitored: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """per replicate: alarms over monitored turbine-days, with days weighted by counts."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ alarms.sum(axis=0)) / (counts @ monitored.sum(axis=0))


def coverable_share(monitored: np.ndarray, turbines: int) -> float:
    """monitored turbine-days over all turbine-days of the window."""
    return float(monitored.sum() / (turbines * monitored.shape[1]))


def episodes(
    alarms_by_drop: Mapping[float, np.ndarray],
    copies: pl.DataFrame,
    turbines: Sequence[str],
    window_start: date,
    horizon: int,
) -> pl.DataFrame:
    """the observed episodes: every copy at every drop, with its first alarm and censoring."""
    parts = []
    for drop, alarms in alarms_by_drop.items():
        slots = np.arange(alarms.shape[1])[None, :]
        times, detected = detections(alarms, copies, turbines, slots, horizon)
        starts = [window_start + timedelta(days=int(o)) for o in copies["offset"].to_list()]
        first = [
            s + timedelta(days=int(t) - 1) if d else None
            for s, t, d in zip(starts, times[0], detected[0], strict=True)
        ]
        parts.append(
            pl.DataFrame(
                {
                    "injection": copies["injection"],
                    "turbine": copies["turbine"],
                    "start": starts,
                    "drop": [drop] * copies.height,
                    "first_alarm": first,
                    "delay_days": times[0],
                    "censored": ~detected[0],
                },
                schema=EPISODES,
            )
        )
    return pl.concat(parts)


type Grid = tuple[np.ndarray, np.ndarray, np.ndarray]


def recall_and_exceedance(
    grids: Sequence[Grid],
    copies: Sequence[pl.DataFrame],
    turbines: Sequence[str],
    slots: np.ndarray,
    quantile: float,
    drop: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """per replicate, pooled over starts (one grid and one set of copies each): recall within
    horizon days, one minus the kaplan-meier share undetected, and the exceedance rate on the
    days without injections."""
    times, events = [], []
    counts = day_counts(slots, grids[0][0].shape[1])
    alarms_total, monitored_total = np.zeros(slots.shape[0]), np.zeros(slots.shape[0])
    for (monitored, z, sd), start_copies in zip(grids, copies, strict=True):
        alarms = alarm_grid(monitored, z, sd, quantile, drop)
        time, event = detections(alarms, start_copies, turbines, slots, horizon)
        times.append(time)
        events.append(event)
        alarms_total = alarms_total + counts @ alarm_grid(monitored, z, sd, quantile, 0.0).sum(
            axis=0
        )
        monitored_total = monitored_total + counts @ monitored.sum(axis=0)
    survival = kaplan_meier(np.hstack(times), np.hstack(events), horizon)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - survival[:, -1], alarms_total / monitored_total
