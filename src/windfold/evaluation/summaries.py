"""farm results built from reductions, each estimate with its bootstrap interval.

every function takes block totals (bootstrap.Totals) and a window's Draw, so all results in a
window share the same resampled blocks and seed indices. an estimate that cannot be formed
(a failed or missing fit) is left out, never filled in.
"""

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from windfold import monitoring
from windfold.config import Protocol
from windfold.evaluation import hypotheses
from windfold.evaluation.bootstrap import Totals, block_of, resample
from windfold.evaluation.hypotheses import NotEstimable, kept, one_sided_p, pooled_mse
from windfold.evaluation.metrics import direction_weighted_mse, kendall_tau_b

type Compute = Callable[[np.ndarray, np.ndarray | None], np.ndarray]

MONITOR_PARTS = ("recall_gain", "exceedance_margin")

ESTIMATE = {"estimate": pl.Float64, "lower": pl.Float64, "upper": pl.Float64}
ACCURACY = pl.Schema(
    {
        "model": pl.String,
        "regime": pl.String,
        "horizon_days": pl.Int64,
        "rows": pl.String,
        "metric": pl.String,
    }
    | ESTIMATE
)
ARC_RESULTS = pl.Schema(
    {"model": pl.String, "split": pl.String, "regime": pl.String}
    | ESTIMATE
    | {"test_rows": pl.Int64, "test_waked_rows": pl.Int64, "test_days": pl.Int64}
)
COMPARISONS = pl.Schema(
    {"model": pl.String, "against": pl.String, "regime": pl.String, "horizon_days": pl.Int64}
    | ESTIMATE
    | {"p": pl.Float64}
)
RANK_AGREEMENT = pl.Schema({"first": pl.String, "second": pl.String, "models": pl.Int64} | ESTIMATE)
MECHANISM = pl.Schema({"model": pl.String, "arcs": pl.Int64} | ESTIMATE | {"p": pl.Float64})
COVERAGE_NULL = pl.Schema({"layout": pl.String, "arcs": pl.Int64} | ESTIMATE | {"p": pl.Float64})
NONINFERIORITY = pl.Schema(
    {"model": pl.String, "horizon_days": pl.Int64, "margin": pl.Float64}
    | ESTIMATE
    | {"noninferior": pl.Boolean}
)
CALIBRATION = pl.Schema(
    {"model": pl.String, "regime": pl.String, "level": pl.Float64, "metric": pl.String} | ESTIMATE
)
SPEED_BIN_VARIANCE = pl.Schema(
    {
        "model": pl.String,
        "regime": pl.String,
        "speed_bin": pl.UInt8,
        "rows": pl.Int64,
        "residual_variance": pl.Float64,
        "predicted_variance": pl.Float64,
    }
)
ENERGY_RESULTS = pl.Schema(
    {"model": pl.String, "turbine": pl.String, "direction_bucket": pl.UInt8, "rows": pl.Int64}
    | {"observed": pl.Float64, "lower": pl.Float64, "upper": pl.Float64, "predicted": pl.Float64}
)
MONITORING = pl.Schema(
    {
        "model": pl.String,
        "horizon_days": pl.Int64,
        "drop": pl.Float64,
        "metric": pl.String,
        "starts": pl.Int64,
    }
    | ESTIMATE
)


@dataclass(frozen=True)
class Draw:
    """a window's bootstrap: block multiplicities and seed indices, one row per replicate."""

    counts: np.ndarray
    seeds: np.ndarray
    level: float
    min_finite: float

    def observed(self) -> np.ndarray:
        return np.ones((1, self.counts.shape[1]))


def summarise(
    observed: float, replicates: np.ndarray, draw: Draw
) -> tuple[float | None, float | None, float | None]:
    """the estimate and the percentile interval of its finite replicates; no interval when too
    few replicates are finite, and nothing when the estimate itself is undefined."""
    if not np.isfinite(observed):
        return None, None, None
    try:
        used = kept(replicates, draw.min_finite)
    except NotEstimable:
        return observed, None, None
    tail = (1 - draw.level) / 2
    lower, upper = np.quantile(used, [tail, 1 - tail])
    return observed, float(lower), float(upper)


def measured(
    draw: Draw, compute: Compute
) -> tuple[float | None, float | None, float | None] | None:
    """compute on every block once (seeds averaged) and on each replicate; none when a fit
    behind it is missing or the estimate is not finite."""
    try:
        observed = float(compute(draw.observed(), None)[0])
        if not np.isfinite(observed):
            return None
        return summarise(observed, compute(draw.counts, draw.seeds), draw)
    except NotEstimable:
        return None


def averaged(
    totals: Totals, model: str, splits: Sequence[str], waked: bool | None, quantity: str
) -> Compute:
    """the per-row mean of a quantity: pooled over cross-validation splits, or for histories
    (one split per start) the mean over starts."""

    def compute(counts: np.ndarray, seeds: np.ndarray | None) -> np.ndarray:
        if all(s.startswith("history_") for s in splits):
            return np.mean(
                [pooled_mse(totals, counts, seeds, model, [s], waked, quantity) for s in splits],
                axis=0,
            )
        return pooled_mse(totals, counts, seeds, model, splits, waked, quantity)

    return compute


def skill(totals: Totals, model: str, reference: str, splits: Sequence[str], waked: bool | None):
    own = averaged(totals, model, splits, waked, "sse")
    base = averaged(totals, reference, splits, waked, "sse")
    return lambda counts, seeds: 1 - own(counts, seeds) / base(counts, seeds)


def accuracy(
    totals: Totals,
    draw: Draw,
    groups: Mapping[tuple[str, int | None], Sequence[str]],
    models: Sequence[str],
    reference: str,
) -> pl.DataFrame:
    """mse, skill against the reference, mean absolute kW error and crps, on waked and all
    rows, for every model and group of splits (regime, horizon) the model has in full."""
    records = []
    for model in models:
        for (regime, horizon), splits in groups.items():
            for label, waked in (("waked", True), ("all", None)):
                metrics: dict[str, Compute] = {
                    "mse": averaged(totals, model, splits, waked, "sse"),
                    "skill": skill(totals, model, reference, splits, waked),
                    "mae_kw": averaged(totals, model, splits, waked, "abs_error_kw"),
                    "crps": averaged(totals, model, splits, waked, "crps"),
                }
                for metric, compute in metrics.items():
                    result = measured(draw, compute)
                    if result is not None:
                        records.append((model, regime, horizon, label, metric, *result))
    return pl.DataFrame(records, schema=ACCURACY, orient="row")


def arc_results(
    totals: Totals,
    draw: Draw,
    models: Sequence[str],
    summary: pl.DataFrame,
) -> pl.DataFrame:
    """per model and included arc split: waked-row mse with its interval and the arc's counts,
    then per model and variant the direction-frequency-weighted mse (split "weighted"), whose
    weights are each arc's share of test timestamps."""
    arcs = summary.filter(
        pl.col("regime").str.starts_with("arc_") & (pl.col("status") == "included")
    )
    records = []
    for model in models:
        for (regime,), part in arcs.group_by("regime", maintain_order=True):
            names, stamps = part["split"].to_list(), part["test_timestamps"].to_numpy()
            present = []
            for row in part.iter_rows(named=True):
                result = measured(draw, averaged(totals, model, [row["split"]], True, "sse"))
                if result is None:
                    continue
                present.append(row["split"])
                counts = (row["test_rows"], row["test_waked_rows"], row["test_days"])
                records.append((model, row["split"], regime, *result, *counts))
            if present and len(present) == len(names):
                result = measured(draw, direction_weighted(totals, model, names, stamps))
                if result is not None:
                    records.append((model, "weighted", regime, *result, None, None, None))
    return pl.DataFrame(records, schema=ARC_RESULTS, orient="row")


def direction_weighted(
    totals: Totals, model: str, splits: Sequence[str], timestamps: np.ndarray
) -> Compute:
    def compute(counts: np.ndarray, seeds: np.ndarray | None) -> np.ndarray:
        mse = np.column_stack(
            [averaged(totals, model, [s], True, "sse")(counts, seeds) for s in splits]
        )
        return direction_weighted_mse(mse, np.ones_like(mse), timestamps)

    return compute


def rank_agreement(
    totals: Totals, draw: Draw, groups: Mapping[str, Sequence[str]], models: Sequence[str]
) -> pl.DataFrame:
    """kendall tau-b between the waked-row mse rankings of every pair of regimes, over the
    models present in all of them."""
    usable = [
        m for m in models if all(measured_ok(totals, m, splits) for splits in groups.values())
    ]
    names = list(groups)
    records = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:

            def tau(counts: np.ndarray, seeds: np.ndarray | None, a=first, b=second):
                x = np.column_stack(
                    [pooled_mse(totals, counts, seeds, m, groups[a]) for m in usable]
                )
                y = np.column_stack(
                    [pooled_mse(totals, counts, seeds, m, groups[b]) for m in usable]
                )
                return kendall_tau_b(x, y)

            result = measured(draw, tau) if len(usable) > 1 else None
            if result is not None:
                records.append((first, second, len(usable), *result))
    return pl.DataFrame(records, schema=RANK_AGREEMENT, orient="row")


def measured_ok(totals: Totals, model: str, splits: Sequence[str]) -> bool:
    try:
        pooled_mse(totals, np.ones((1, totals.values.shape[0])), None, model, splits)
    except NotEstimable:
        return False
    return True


def paired_arcs(opposite: Sequence[str], control: Sequence[str]) -> list[tuple[str, str]]:
    """each opposite-arc split with the control split that holds out the same arc; an arc with
    only one of the two variants is left out."""
    controls = {split.rsplit("_", 1)[0]: split for split in control}
    return [
        (split, controls[arc])
        for split in sorted(opposite)
        if (arc := split.rsplit("_", 1)[0]) in controls
    ]


def deletion_contrasts(
    totals: Totals,
    draw: Draw,
    opposite: Sequence[str],
    control: Sequence[str],
    models: Sequence[str],
) -> pl.DataFrame:
    """per model, the waked-row mse on held-out arcs after deleting the opposite arc minus after
    deleting a count-matched control set, over the arcs that have both and the model fits in
    full, paired by day blocks. positive means the opposite arc carried information about the
    held-out one."""
    records = []
    for model in models:
        pairs = [
            pair
            for pair in paired_arcs(opposite, control)
            if all(measured_ok(totals, model, [split]) for split in pair)
        ]
        if not pairs:
            continue
        removed = averaged(totals, model, [a for a, _ in pairs], True, "sse")
        matched = averaged(totals, model, [b for _, b in pairs], True, "sse")
        try:
            observed = float((removed(draw.observed(), None) - matched(draw.observed(), None))[0])
            replicates = removed(draw.counts, draw.seeds) - matched(draw.counts, draw.seeds)
        except NotEstimable:
            continue
        try:
            p = one_sided_p(kept(replicates, draw.min_finite))
        except NotEstimable:
            p = None
        records.append((model, len(pairs), *summarise(observed, replicates, draw), p))
    return pl.DataFrame(records, schema=MECHANISM, orient="row")


def coverage_null(
    sector: Totals,
    time: Totals,
    draw: Draw,
    design: pl.DataFrame,
    *,
    covariates: Sequence[str],
    model: str,
    physics: str,
    time_splits: Sequence[str],
) -> pl.DataFrame:
    """the permuted-layout null of figure 6: the partial spearman of excess error with the real
    layout's coverage score, with the permuted layout's and their difference, on the arcs, excess
    errors, covariates and bootstrap of coverage predicts failure. it does not enter the decision;
    a positive difference means the real geometry predicts failure beyond what the held-out
    directions alone give."""
    judged = hypotheses.coverage_arcs(
        sector,
        draw.counts.shape[1],
        design=design,
        model=model,
        physics=physics,
    )
    if judged is None:
        return pl.DataFrame(schema=COVERAGE_NULL)
    arcs, _ = judged

    def excess(counts: np.ndarray) -> np.ndarray:
        return hypotheses.excess_errors(
            sector, time, counts, model=model, arcs=arcs, time_splits=time_splits
        )

    try:
        observed, replicates = excess(draw.observed()), excess(draw.counts)
    except NotEstimable:
        return pl.DataFrame(schema=COVERAGE_NULL)
    covariate = design.select(covariates).to_numpy()
    found: dict[str, tuple[float, np.ndarray]] = {}
    for layout, column in (("real", "score"), ("permuted", "permuted_score")):
        score = design[column].to_numpy()
        try:
            value = hypotheses.partial_spearman(score, observed, covariate)[0]
            found[layout] = (
                float(value),
                hypotheses.partial_spearman(score, replicates, covariate),
            )
        except NotEstimable:
            continue
    if len(found) == 2:
        (real, real_draws), (permuted, permuted_draws) = found["real"], found["permuted"]
        found["difference"] = (real - permuted, real_draws - permuted_draws)
    records = []
    for layout, (value, draws) in found.items():
        try:
            p = one_sided_p(kept(draws, draw.min_finite))
        except NotEstimable:
            p = None
        records.append((layout, len(arcs), *summarise(value, draws, draw), p))
    return pl.DataFrame(records, schema=COVERAGE_NULL, orient="row")


def skill_gap(
    own: Compute, theirs: Compute, base: Compute, counts: np.ndarray, seeds: np.ndarray | None
) -> np.ndarray:
    """skill of own over base minus skill of theirs over base: their mse gap over base's mse."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (theirs(counts, seeds) - own(counts, seeds)) / base(counts, seeds)


def confirmation_contrasts(
    totals: Totals,
    draw: Draw,
    splits: Sequence[str],
    model: str,
    others: Sequence[str],
    cells: tuple[str, str, str],
    reference: str,
) -> dict[str, hypotheses.Component]:
    """the model's waked-row skill over the reference minus each other model's on the splits, and
    the interaction of two changes from four cells: (model minus the first cell) minus (the second
    minus the third), every one with the same draws."""

    def gap(own: str, theirs: str) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
        base = averaged(totals, reference, splits, True, "sse")
        mine, other = (averaged(totals, name, splits, True, "sse") for name in (own, theirs))
        return lambda counts, seeds: skill_gap(mine, other, base, counts, seeds)

    found: dict[str, hypotheses.Component] = {}
    for other in others:
        contrast = gap(model, other)
        found[f"{model}_minus_{other}"] = hypotheses.attempt(
            lambda contrast=contrast: hypotheses.estimate(contrast, draw.counts, draw.seeds)
        )
    first, second, third = cells
    outer, inner = gap(model, first), gap(second, third)
    found["interaction"] = hypotheses.attempt(
        lambda: hypotheses.estimate(
            lambda counts, seeds: outer(counts, seeds) - inner(counts, seeds),
            draw.counts,
            draw.seeds,
        )
    )
    return found


def comparisons(
    totals: Totals,
    draw: Draw,
    groups: Mapping[tuple[str, int | None], Sequence[str]],
    others: Sequence[str],
    model: str,
    reference: str,
) -> pl.DataFrame:
    """the model's waked-row skill over the reference minus each other model's, on every group
    all three have in full: their mse difference over the reference's mse, per replicate, with
    its interval and one-sided p (positive favors the model)."""
    records = []
    for other in others:
        for (regime, horizon), splits in groups.items():
            own = averaged(totals, model, splits, True, "sse")
            theirs = averaged(totals, other, splits, True, "sse")
            base = averaged(totals, reference, splits, True, "sse")
            try:
                observed = float(skill_gap(own, theirs, base, draw.observed(), None)[0])
                replicates = skill_gap(own, theirs, base, draw.counts, draw.seeds)
            except NotEstimable:
                continue
            result = summarise(observed, replicates, draw)
            try:
                p = one_sided_p(kept(replicates, draw.min_finite))
            except NotEstimable:
                p = None
            records.append((other, model, regime, horizon, *result, p))
    return pl.DataFrame(records, schema=COMPARISONS, orient="row")


def noninferiority(
    totals: Totals,
    draw: Draw,
    model: str,
    baselines: Sequence[str],
    splits: Sequence[str],
    horizon: int,
    margin: float,
) -> pl.DataFrame:
    """relative waked-row mse of the model against the best baseline at a horizon (starts
    weighted equally), the best chosen afresh in each replicate; non-inferior when the upper
    bound is below the margin."""
    present = [b for b in baselines if b != model and measured_ok(totals, b, splits)]
    if not present or not measured_ok(totals, model, splits):
        return pl.DataFrame(schema=NONINFERIORITY)

    def relative(counts: np.ndarray, seeds: np.ndarray | None) -> np.ndarray:
        best = np.column_stack(
            [averaged(totals, b, splits, True, "sse")(counts, seeds) for b in present]
        ).min(axis=1)
        return averaged(totals, model, splits, True, "sse")(counts, seeds) / best - 1

    result = measured(draw, relative)
    if result is None:
        return pl.DataFrame(schema=NONINFERIORITY)
    upper = result[2]
    return pl.DataFrame(
        [(model, horizon, margin, *result, upper is not None and upper < margin)],
        schema=NONINFERIORITY,
        orient="row",
    )


def interval_calibration(
    totals: Totals,
    draw: Draw,
    groups: Mapping[str, Sequence[str]],
    models: Sequence[str],
    levels: Sequence[float],
) -> pl.DataFrame:
    """observed coverage and mean width of the central intervals at each level, waked rows."""
    records = []
    for model in models:
        for regime, splits in groups.items():
            for level in levels:
                for metric, quantity in (("coverage", "covered"), ("width", "width")):
                    compute = averaged_level(totals, model, splits, level, quantity)
                    result = measured(draw, compute)
                    if result is not None:
                        records.append((model, regime, level, metric, *result))
    return pl.DataFrame(records, schema=CALIBRATION, orient="row")


def averaged_level(
    totals: Totals, model: str, splits: Sequence[str], level: float, quantity: str
) -> Compute:
    def compute(counts: np.ndarray, seeds: np.ndarray | None) -> np.ndarray:
        return pooled_mse(totals, counts, seeds, model, splits, True, quantity, level=level)

    return compute


def speed_bin_variance(errors: pl.DataFrame, regimes: Mapping[str, str]) -> pl.DataFrame:
    """residual variance (mse) against mean predicted variance by speed bin, for every model
    with a predictive variance; regimes maps each split to its regime."""
    rows = (
        errors.filter(pl.col("variance").is_not_null())
        .with_columns(regime=pl.col("split").replace_strict(regimes, default=None))
        .group_by("model", "regime", "speed_bin")
        .agg(
            rows=pl.col("rows").sum(),
            residual_variance=pl.col("sse").sum() / pl.col("rows").sum(),
            predicted_variance=pl.col("variance").sum() / pl.col("rows").sum(),
        )
    )
    return (
        rows.select(SPEED_BIN_VARIANCE.names())
        .cast(SPEED_BIN_VARIANCE)
        .sort("model", "regime", "speed_bin")
    )


def energy_ratios(
    sums: pl.DataFrame, draw: Draw, origin: date, block_days: int, blocks: int
) -> pl.DataFrame:
    """per model, turbine and 5 degree bucket over the single-arc test rows: the observed
    energy ratio with its interval and the model's predicted ratio."""
    if sums.is_empty():
        return pl.DataFrame(schema=ENERGY_RESULTS)
    first = sums.select("model", "seed").unique().sort("model", "seed", nulls_last=True).row(0)
    observed = sums.filter((pl.col("model") == first[0]) & pl.col("seed").eq_missing(first[1]))
    block = block_of(observed["day"], origin, block_days)
    cells = (
        observed.select("turbine", "direction_bucket").unique().sort("turbine", "direction_bucket")
    )
    keyed = observed.with_columns(block=pl.Series(block)).join(
        cells.with_row_index("cell"), on=["turbine", "direction_bucket"]
    )
    values = np.zeros((blocks, 2 * cells.height))
    grouped = keyed.group_by("block", "cell").agg(
        pl.col("power_kw").sum(), pl.col("reference_kw").sum()
    )
    b, c = grouped["block"].to_numpy(), grouped["cell"].to_numpy().astype(np.int64)
    values[b, 2 * c] = grouped["power_kw"].to_numpy()
    values[b, 2 * c + 1] = grouped["reference_kw"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        replicate = resample(values, draw.counts)
        ratios = replicate[:, 0::2] / replicate[:, 1::2]
        point = values.sum(axis=0)
        observed_ratio = point[0::2] / point[1::2]
    tail = (1 - draw.level) / 2
    lower = np.nanquantile(ratios, tail, axis=0)
    upper = np.nanquantile(ratios, 1 - tail, axis=0)
    rows = observed.group_by("turbine", "direction_bucket").agg(pl.col("rows").sum())
    table = cells.with_columns(
        observed=pl.Series(observed_ratio), lower=pl.Series(lower), upper=pl.Series(upper)
    ).join(rows, on=["turbine", "direction_bucket"])
    predicted = sums.group_by("model", "turbine", "direction_bucket").agg(
        predicted=pl.col("predicted_kw").sum() / pl.col("reference_kw").sum()
    )
    joined = predicted.join(table, on=["turbine", "direction_bucket"], how="left")
    return (
        joined.select(ENERGY_RESULTS.names())
        .cast(ENERGY_RESULTS)
        .sort("model", "turbine", "direction_bucket")
    )


def seed_major(per_start: Sequence[Sequence[monitoring.Grid]]) -> list[list[monitoring.Grid]]:
    """each start's grids, one per seed, as one list of start grids per seed."""
    seeds = {len(grids) for grids in per_start}
    if len(seeds) != 1:
        raise ValueError(f"starts of one monitor have {sorted(seeds)} seeds")
    return [[grids[k] for grids in per_start] for k in range(seeds.pop())]


def seed_copies(frame: pl.DataFrame, keys: Sequence[str]) -> tuple[pl.DataFrame, int]:
    """frame with copy, the position of each row among the rows sharing keys and the number of
    copies. a seeded model's fits write their monitoring rows in seed order, as reduce_farm reads
    its sorted fits, so copy k holds the k-th seed; every key must have the same number."""
    numbered = frame.with_columns(copy=pl.int_range(pl.len()).over(keys))
    counts = sorted(numbered.group_by(keys).len()["len"].unique().to_list())
    if len(counts) != 1:
        raise ValueError(f"monitoring rows come in {counts} copies per key")
    return numbered, int(counts[0])


def monitor_grids(
    scales: pl.DataFrame,
    days: pl.DataFrame,
    turbines: Sequence[str],
    window_start: date,
    window_days: int,
    min_rows_per_day: int,
) -> dict[tuple[str, int, str], list[monitoring.Grid]]:
    """day grids per (model, horizon, start) from the calibration scales and daily residuals,
    one per seed of the model, each seed's days standardized with its own calibration."""
    grids = {}
    for (model, split), part in days.group_by("model", "split", maintain_order=True):
        horizon, start = str(split).split("_")[1], str(split).split("_from_")[1]
        threshold = f"threshold_{horizon}_from_{start}"
        own = scales.filter((pl.col("model") == model) & (pl.col("split") == threshold))
        if own.is_empty():
            continue
        own, calibrated = seed_copies(own, ["turbine"])
        part, seeds = seed_copies(part, ["turbine", "day"])
        if calibrated != seeds:
            raise ValueError(f"{model} {split}: {seeds} seeds of days, {calibrated} of calibration")
        grids[(str(model), int(horizon.removesuffix("d")), start)] = [
            monitoring.day_grids(
                part.filter(pl.col("copy") == k).select("turbine", "day", "rows", "residual"),
                own.filter(pl.col("copy") == k).drop("model", "split", "copy"),
                turbines,
                window_start,
                window_days,
                min_rows_per_day,
            )
            for k in range(seeds)
        ]
    return grids


def grids_and_copies(
    reduced: dict[str, pl.DataFrame],
    origin: tuple[date, int],
    turbines: Sequence[str],
    protocol: Protocol,
    seed: int,
    monitors: Sequence[str],
) -> tuple[dict[tuple[str, int, str], list[monitoring.Grid]], dict[tuple[int, str], pl.DataFrame]]:
    """the given monitors' day grids per (model, horizon, start), one per seed, and the matched
    copies of each usable (horizon, start): one where at least min_monitorable_share of turbines
    are eligible for every one of them."""
    settings = protocol.monitoring
    scales = reduced["monitoring_scales"].filter(pl.col("model").is_in(list(monitors)))
    days = reduced["monitoring_days"].filter(pl.col("model").is_in(list(monitors)))
    grids = monitor_grids(scales, days, turbines, origin[0], origin[1], settings.min_rows_per_day)
    copies = {}
    keyed = scales.with_columns(
        horizon=pl.col("split").str.split("_").list.get(1).str.strip_suffix("d").cast(pl.Int64),
        start=pl.col("split").str.split("_from_").list.get(1),
    )
    for (horizon, start), part in keyed.group_by("horizon", "start", maintain_order=True):
        judged = part.filter(pl.col("model") != "direction_lookup")
        eligible = sorted(
            set.intersection(
                *(
                    set(p.filter(pl.col("eligible"))["turbine"].to_list())
                    for _, p in judged.group_by("model")
                )
            )
        )
        if len(eligible) < settings.min_monitorable_share * len(turbines):
            continue
        key = (int(horizon), str(start))
        copy_seed = zlib.crc32(f"{seed}/{horizon}/{start}".encode())
        copies[key] = monitoring.injections(
            eligible, origin[1], settings.copies, settings.episode_days, copy_seed
        )
    return grids, copies


def early_history_components(
    reduced: dict[str, pl.DataFrame],
    history: Totals,
    draw: Draw,
    drawn: np.ndarray,
    horizons: dict[int, list[str]],
    origin: tuple[date, int],
    turbines: Sequence[str],
    protocol: Protocol,
    seed: int,
    *,
    accuracy_comparators: Sequence[str],
    monitor_comparators: Sequence[str],
    monitors: Sequence[str],
) -> dict[str, hypotheses.Component]:
    """early history advantage on one farm: accuracy after the endpoint horizon against each
    comparator, and the monitors' recall gain and exceedance margin on the same copies and rebuilt
    days. a comparator without results at the horizon is not estimable."""
    endpoint = protocol.endpoints.early_history_advantage
    model = endpoint.model
    splits = horizons.get(endpoint.horizon_days, [])
    components: dict[str, hypotheses.Component] = {}
    for name in accuracy_comparators:

        def compute(name: str = name) -> hypotheses.Estimate:
            if not splits:
                raise NotEstimable(f"no {endpoint.horizon_days}-day histories")
            return hypotheses.estimate(
                lambda c, s: hypotheses.early_history_accuracy(
                    history, c, s, model=model, comparator=name, splits=splits
                ),
                draw.counts,
                draw.seeds,
            )

        components[f"{name}_minus_{model}"] = hypotheses.attempt(compute)
    monitored = [f"{kind}_{name}" for name in monitor_comparators for kind in MONITOR_PARTS]
    components |= dict.fromkeys(monitored, hypotheses.NOT_ESTIMABLE)
    grids, copies = grids_and_copies(reduced, origin, turbines, protocol, seed, monitors)
    starts = [start for (days, start) in copies if days == endpoint.horizon_days]

    def usable(name: str) -> bool:
        return bool(starts) and all((name, endpoint.horizon_days, s) in grids for s in starts)

    ready = [name for name in monitor_comparators if usable(name)]
    if not usable(model) or not ready:
        return components
    slots = monitoring.trajectories(drawn, origin[1], protocol.statistics.block_days)
    settings = protocol.monitoring
    components |= hypotheses.early_history_monitoring(
        {
            m: seed_major([grids[(m, endpoint.horizon_days, s)] for s in starts])
            for m in (model, *ready)
        },
        [copies[(endpoint.horizon_days, s)] for s in starts],
        turbines,
        slots,
        draw.seeds,
        model=model,
        comparators=ready,
        quantile=monitoring.alarm_quantile(settings.false_alarms_per_turbine_month),
        drop=endpoint.drop,
        horizon=settings.censor_days,
        margin=protocol.decision.exceedance_margin,
    )
    return components


def monitoring_results(
    reduced: dict[str, pl.DataFrame],
    draw: Draw,
    drawn: np.ndarray,
    origin: tuple[date, int],
    turbines: Sequence[str],
    protocol: Protocol,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """per monitor, horizon and drop: recall, median delay, exceedance and coverable share
    pooled over usable starts, each the mean over a seeded monitor's seeds, and every observed
    episode, with its seed."""
    settings = protocol.monitoring
    grids, copies = grids_and_copies(reduced, origin, turbines, protocol, seed, settings.models)
    slots = monitoring.trajectories(drawn, origin[1], protocol.statistics.block_days)
    identity = np.arange(origin[1])[None, :]
    quantile = monitoring.alarm_quantile(settings.false_alarms_per_turbine_month)
    records, episodes = [], []
    for model in sorted({m for (m, _, _) in grids}):
        for horizon in sorted({h for (h, _) in copies}):
            starts = [s for (h, s) in copies if h == horizon and (model, h, s) in grids]
            if not starts:
                continue
            per_seed = seed_major([grids[(model, horizon, s)] for s in starts])
            sets = [copies[(horizon, s)] for s in starts]
            share = np.mean(
                [monitoring.coverable_share(g[0], len(turbines)) for own in per_seed for g in own]
            )
            records.append(
                (model, horizon, None, "coverable_share", len(starts), share, None, None)
            )
            for drop in settings.drops:
                observed, replicated = (
                    hypotheses.monitor_rates(
                        per_seed,
                        sets,
                        turbines,
                        rebuilt,
                        drawn_seeds,
                        quantile,
                        drop,
                        settings.censor_days,
                    )
                    for rebuilt, drawn_seeds in ((identity, None), (slots, draw.seeds))
                )
                recall = summarise(float(observed[0][0]), replicated[0], draw)
                records.append((model, horizon, drop, "recall", len(starts), *recall))
                if drop == settings.drops[0]:
                    exceed = summarise(float(observed[1][0]), replicated[1], draw)
                    records.append((model, horizon, None, "exceedance", len(starts), *exceed))
                medians = []
                for k, own in enumerate(per_seed):
                    label = k if len(per_seed) > 1 else None
                    for start, grid, copy in zip(starts, own, sets, strict=True):
                        alarms = monitoring.alarm_grid(*grid, quantile, drop)
                        table = monitoring.episodes(
                            {drop: alarms}, copy, turbines, origin[0], settings.censor_days
                        )
                        episodes.append(
                            table.with_columns(
                                model=pl.lit(model),
                                horizon_days=pl.lit(horizon),
                                start_of=pl.lit(start),
                                seed=pl.lit(label, dtype=pl.Int64),
                            )
                        )
                    found = [
                        monitoring.detections(
                            monitoring.alarm_grid(*g, quantile, drop),
                            c,
                            turbines,
                            identity,
                            settings.censor_days,
                        )
                        for g, c in zip(own, sets, strict=True)
                    ]
                    times = np.hstack([f[0] for f in found])
                    events = np.hstack([f[1] for f in found])
                    survival = monitoring.kaplan_meier(times, events, settings.censor_days)
                    medians.append(monitoring.median_delay(survival)[0])
                median = float(np.mean(medians))
                records.append(
                    (
                        model,
                        horizon,
                        drop,
                        "median_delay_days",
                        len(starts),
                        None if np.isnan(median) else median,
                        None,
                        None,
                    )
                )
    table = pl.DataFrame(records, schema=MONITORING, orient="row")
    episode_table = pl.concat(episodes) if episodes else pl.DataFrame()
    return {"monitoring": table, "episodes": episode_table}
