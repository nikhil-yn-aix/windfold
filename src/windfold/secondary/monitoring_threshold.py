"""thresholds for the planned monitors scaled by their own calibration days (item 58 of the
study plan), beside the standard thresholds.

at the standard threshold a test day alarms when its mean leave-one-out residual over the planned
sd of a daily mean falls below the normal quantile of one false alarm per turbine-month. k is the
mean square of that ratio on the calibration days of the eligible turbines, floored at 1, one per
farm, monitor, history, start and seed. rule c divides each test day's ratio by sqrt(k); rule ct,
the recalibrated threshold, also takes its cut from a student t with as many degrees of freedom
as the distinct calibration days behind k. recall, exceedance and windfold's gains over the other
monitors follow section 12 of the study plan under every rule, with k fixed within the bootstrap
replicates.

the table is long: one value per row, named by rule (empty for what no rule changes), metric,
monitor, history_days, start ("pooled" for a value over every usable start) and seed (empty for
a monitor without seeds or a value averaged over them).
"""

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from windfold import config, monitoring
from windfold.config import folder_of, for_farm
from windfold.evaluation import hypotheses, metrics, paths, summaries
from windfold.evaluation.bootstrap import farm_seed
from windfold.evaluation.results import bootstrap_windows, study_windows
from windfold.locations import prepared_folder
from windfold.secondary.contrasts import confirmatory_ready
from windfold.tasks.evaluation import frozen_bytes, second_freeze

MODEL = "monitoring_threshold"
RULES = ("registered", "c", "ct")
BAND = (0.5, 2.0)
POOLED = "pooled"
FIT = ["model", "split", "copy"]
TABLE = pl.Schema(
    {
        "farm": pl.String,
        "rule": pl.String,
        "metric": pl.String,
        "monitor": pl.String,
        "history_days": pl.Int64,
        "start": pl.String,
        "seed": pl.Int64,
        "drop": pl.Float64,
        "starts": pl.Int64,
        "estimate": pl.Float64,
        "lower": pl.Float64,
        "upper": pl.Float64,
        "p": pl.Float64,
        "passes": pl.Boolean,
    }
)

type Grids = Mapping[tuple[str, int, str], list[monitoring.Grid]]
type Measured = dict[tuple[str, float | None], tuple[float, np.ndarray | None]]
type Rates = dict[tuple[str, int], tuple[tuple[str, ...], tuple[str, ...], Measured]]
type Row = dict[str, Any]


def horizon_of(split: pl.Expr) -> pl.Expr:
    return split.str.split("_").list.get(1).str.strip_suffix("d").cast(pl.Int64)


def threshold_fits(fits: pl.DataFrame, models: Sequence[str], min_days: int) -> pl.DataFrame:
    """the monitors' done threshold fits of at least min_days of history, with copy, the seed's
    position among its fit's seeds, in the order the monitoring reductions hold them."""
    chosen = fits.filter(
        (pl.col("regime") == "threshold")
        & (pl.col("status") == "done")
        & pl.col("model").is_in(list(models))
        & (horizon_of(pl.col("split")) >= min_days)
    )
    return (
        chosen.sort("model", "split", "seed")
        .with_columns(copy=pl.int_range(pl.len(), dtype=pl.Int64).over("model", "split"))
        .select("model", "split", "seed", "copy")
    )


def calibration_days(
    prepared: pl.DataFrame,
    runs: Path,
    study: str,
    farm: str,
    fits: pl.DataFrame,
    protocol: config.Protocol,
    hashes: Mapping[str, str] | None,
) -> pl.DataFrame:
    """each threshold fit's mean calibration residual by turbine and day, over the days with at
    least min_rows_per_day rows, scored as the monitoring reductions score the fit."""
    settings = protocol.monitoring
    parts = []
    for model, split, seed, copy in fits.iter_rows():
        table, _ = paths.fit_inputs(study, farm, None, model, split, seed, protocol)
        predictions = pl.read_parquet(io.BytesIO(frozen_bytes(runs, table, hashes)))
        here = prepared.join(predictions.select("timestamp").unique(), on="timestamp", how="semi")
        rows = metrics.scored(predictions, here)
        if model == "direction_lookup":
            rows = rows.filter(pl.col("support") >= settings.lookup_min_rows)
        days = monitoring.daily_means(monitoring.residuals(rows))
        parts.append(
            days.filter(pl.col("rows") >= settings.min_rows_per_day).with_columns(
                model=pl.lit(model), split=pl.lit(split), copy=pl.lit(copy, dtype=pl.Int64)
            )
        )
    return pl.concat(parts)


def eligible_days(scales: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """the calibration days of eligible turbines with v, the planned variance of their mean, in
    fit, turbine and day order."""
    eligible = scales.filter(pl.col("eligible")).select(*FIT, "turbine", "sigma", "rho")
    variance = (
        pl.col("sigma") ** 2
        * (1 + pl.col("rho"))
        / ((1 - pl.col("rho")) * pl.col("rows").cast(pl.Float64))
    )
    return (
        days.join(eligible, on=[*FIT, "turbine"])
        .with_columns(v=variance)
        .sort(*FIT, "turbine", "day")
    )


def scale_factors(scales: pl.DataFrame, days: pl.DataFrame, rate: float) -> pl.DataFrame:
    """per threshold fit: k, the mean square behind it, the distinct calibration days, turbines
    and turbine-days it covers and the ratio of the student t quantile with that many degrees
    of freedom to the normal one at rate. a fit without eligible turbines keeps k at 1."""
    found = (
        eligible_days(scales, days)
        .group_by(FIT)
        .agg(
            mean_square=(pl.col("residual") ** 2 / pl.col("v")).mean(),
            calibration_days=pl.col("day").n_unique().cast(pl.Int64),
            turbines=pl.col("turbine").n_unique().cast(pl.Int64),
            turbine_days=pl.len().cast(pl.Int64),
        )
    )
    out = (
        scales.select(FIT)
        .unique()
        .join(found, on=FIT, how="left")
        .with_columns(pl.col("calibration_days", "turbines", "turbine_days").fill_null(0))
        .sort(FIT)
    )
    count = out["calibration_days"].to_numpy()
    ratio = np.ones(count.size)
    ratio[count > 0] = stats.t.ppf(rate, count[count > 0]) / stats.norm.ppf(rate)
    return out.with_columns(
        k=pl.max_horizontal(pl.col("mean_square").fill_null(1.0), pl.lit(1.0)),
        t_ratio=pl.Series(ratio),
    )


def factor(rule: str) -> pl.Expr:
    """what a rule multiplies the planned sd of a daily mean by."""
    return {
        "registered": pl.lit(1.0),
        "c": pl.col("k").sqrt(),
        "ct": pl.col("k").sqrt() * pl.col("t_ratio"),
    }[rule]


def scaled(scales: pl.DataFrame, factors: pl.DataFrame, rule: str) -> pl.DataFrame:
    """the planned scales, in their order, with sigma times the rule's factor of its fit."""
    return (
        scales.join(
            factors.select(*FIT, factor=factor(rule)), on=FIT, how="left", maintain_order="left"
        )
        .with_columns(sigma=pl.col("sigma") * pl.col("factor"))
        .drop("copy", "factor")
    )


def rates(
    grids: Grids,
    copies: Mapping[tuple[int, str], pl.DataFrame],
    turbines: Sequence[str],
    slots: np.ndarray,
    draw: summaries.Draw,
    interval_days: int | None,
    protocol: config.Protocol,
) -> Rates:
    """per monitor and history: the matched starts it watches, every matched start of the
    history and each drop's recall and the exceedance, pooled over the starts it watches and
    averaged over seeds, with their replicates at interval_days."""
    settings = protocol.monitoring
    quantile = monitoring.alarm_quantile(settings.false_alarms_per_turbine_month)
    out: Rates = {}
    for model in settings.models:
        for horizon in sorted({h for h, _ in copies}):
            matched = tuple(s for h, s in copies if h == horizon)
            starts = tuple(s for s in matched if (model, horizon, s) in grids)
            if not starts:
                continue
            per_seed = summaries.seed_major([grids[(model, horizon, s)] for s in starts])
            sets = [copies[(horizon, s)] for s in starts]
            identity = np.arange(per_seed[0][0][0].shape[1])[None, :]
            found: Measured = {}
            for drop in settings.drops:
                observed = hypotheses.monitor_rates(
                    per_seed, sets, turbines, identity, None, quantile, drop, settings.censor_days
                )
                drawn = None
                if horizon == interval_days:
                    drawn = hypotheses.monitor_rates(
                        per_seed,
                        sets,
                        turbines,
                        slots,
                        draw.seeds,
                        quantile,
                        drop,
                        settings.censor_days,
                    )
                found[("recall", drop)] = (
                    float(observed[0][0]),
                    None if drawn is None else drawn[0],
                )
                if drop == settings.drops[0]:
                    found[("exceedance", None)] = (
                        float(observed[1][0]),
                        None if drawn is None else drawn[1],
                    )
            out[(model, horizon)] = (starts, matched, found)
    return out


def summarised(value: float, replicates: np.ndarray | None, draw: summaries.Draw) -> Row:
    """the estimate, none when undefined, and its 95% interval when it has replicates."""
    if replicates is None:
        return {"estimate": value if np.isfinite(value) else None}
    estimate, lower, upper = summaries.summarise(value, replicates, draw)
    return {"estimate": estimate, "lower": lower, "upper": upper}


def one_sided(replicates: np.ndarray | None, draw: summaries.Draw) -> float | None:
    if replicates is None:
        return None
    try:
        return hypotheses.one_sided_p(hypotheses.kept(replicates, draw.min_finite))
    except hypotheses.NotEstimable:
        return None


def rate_rows(found: Rates, draw: summaries.Draw) -> list[Row]:
    return [
        {"metric": metric, "monitor": model, "history_days": horizon, "drop": drop}
        | {"start": POOLED, "starts": len(starts)}
        | summarised(value, drawn, draw)
        for (model, horizon), (starts, _, measured) in found.items()
        for (metric, drop), (value, drawn) in measured.items()
    ]


def gain_rows(found: Rates, draw: summaries.Draw, protocol: config.Protocol) -> list[Row]:
    """windfold's recall gain at the endpoint drop and exceedance margin over each other
    monitor, named by the other monitor, at every history both have. as the planned evaluation
    requires, a gain needs both monitors on every matched start of its history; without them it
    is not estimable and left empty."""
    endpoint = protocol.endpoints.early_history_advantage
    margin = protocol.decision.exceedance_margin
    out = []
    for (model, horizon), (starts, matched, own) in found.items():
        if model != endpoint.model:
            continue
        for other in protocol.monitoring.models:
            if other == model or (other, horizon) not in found:
                continue
            watched, _, theirs = found[(other, horizon)]
            paired = starts == watched == matched
            pairs = {
                ("recall_gain", endpoint.drop): (("recall", endpoint.drop), lambda a, b: a - b),
                ("exceedance_margin", None): (
                    ("exceedance", None),
                    lambda a, b: hypotheses.exceedance_benefit(a, b, margin),
                ),
            }
            for (metric, drop), (key, combine) in pairs.items():
                row = {"metric": metric, "monitor": other, "history_days": horizon, "drop": drop}
                if not paired:
                    out.append(row | {"start": POOLED})
                    continue
                (a, a_drawn), (b, b_drawn) = own[key], theirs[key]
                value = combine(a, b)
                drawn = None if a_drawn is None or b_drawn is None else combine(a_drawn, b_drawn)
                out.append(
                    row
                    | {"start": POOLED, "starts": len(starts), "p": one_sided(drawn, draw)}
                    | summarised(value, drawn, draw)
                )
    return out


def check_rows(rows: Sequence[Row], nominal: float, protocol: config.Protocol) -> list[Row]:
    """the calibration check (c1) passes when every monitor with monitored days has its
    endpoint-history exceedance within BAND times the nominal rate; the advantage check (c2) when
    windfold's recall gains and exceedance margins over the planned comparators all have
    intervals above zero and the largest p, its p, is below alpha. neither is judged without the
    endpoint history or with a component missing."""
    endpoint = protocol.endpoints.early_history_advantage
    here = [r for r in rows if r["history_days"] == endpoint.horizon_days]
    exceedance = [
        r["estimate"] for r in here if r["metric"] == "exceedance" and r["estimate"] is not None
    ]
    low, high = (share * nominal for share in BAND)
    calibrated = all(low <= e <= high for e in exceedance) if exceedance else None
    parts = [
        r
        for r in here
        if r["metric"] in ("recall_gain", "exceedance_margin")
        and r["monitor"] in endpoint.monitor_comparators
        and r.get("lower") is not None
        and r.get("p") is not None
    ]
    largest = (
        max(r["p"] for r in parts) if len(parts) == 2 * len(endpoint.monitor_comparators) else None
    )
    advantage = (
        None
        if largest is None
        else all(r["lower"] > 0 for r in parts) and largest < protocol.statistics.alpha
    )
    where = {"history_days": endpoint.horizon_days, "start": POOLED}
    return [
        {"metric": "c1", "passes": calibrated} | where,
        {"metric": "c2", "p": largest, "passes": advantage} | where,
    ]


def day_rows(
    grids: Grids, seeds: Mapping[tuple[str, str, int], int | None], quantile: float
) -> list[Row]:
    """per monitor, history, start and seed: the monitored test days of the uninjected grid and
    the mean, sd, root mean square and share below the cut of their standardized means."""
    out = []
    for (model, horizon, start), per_seed in grids.items():
        split = f"threshold_{horizon:03d}d_from_{start}"
        for copy, (monitored, z, _) in enumerate(per_seed):
            values = z[monitored]
            n = values.size
            found = {
                "test_days": float(n),
                "z_mean": float(values.mean()) if n else None,
                "z_sd": float(values.std(ddof=1)) if n > 1 else None,
                "z_rms": float(np.sqrt((values**2).mean())) if n else None,
                "below_cut": float((values < quantile).mean()) if n else None,
            }
            keys = {"monitor": model, "history_days": horizon, "start": start}
            seed = seeds[(model, split, copy)]
            out += [keys | {"seed": seed, "metric": m, "estimate": v} for m, v in found.items()]
    return out


def factor_rows(factors: pl.DataFrame, seeds: Mapping[tuple[str, str, int], int | None]):
    names = ("k", "mean_square", "calibration_days", "t_ratio", "turbines", "turbine_days")
    out = []
    for row in factors.iter_rows(named=True):
        split = row["split"]
        keys = {
            "monitor": row["model"],
            "history_days": int(split.split("_")[1].removesuffix("d")),
            "start": split.split("_from_")[1],
            "seed": seeds[(row["model"], split, row["copy"])],
        }
        out += [
            keys | {"metric": name, "estimate": None if row[name] is None else float(row[name])}
            for name in names
        ]
    return out


def variance_ratios(scales: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame:
    """per monitor and history: the variance of the calibration-day means about each turbine's
    mean at its start over their mean planned variance, both pooled over turbines and starts with
    weights of one less than their days; seeds averaged."""
    series = (
        eligible_days(scales, days)
        .group_by(*FIT, "turbine", maintain_order=True)
        .agg(
            squares=((pl.col("residual") - pl.col("residual").mean()) ** 2).sum(),
            freedom=(pl.len() - 1).cast(pl.Float64),
            v=pl.col("v").mean(),
        )
        .with_columns(horizon_days=horizon_of(pl.col("split")))
    )
    per_seed = series.group_by("model", "horizon_days", "copy", maintain_order=True).agg(
        ratio=pl.col("squares").sum() / (pl.col("freedom") * pl.col("v")).sum()
    )
    return per_seed.group_by("model", "horizon_days", maintain_order=True).agg(
        pl.col("ratio").mean()
    )


def largest_difference(
    mine: pl.DataFrame, official: pl.DataFrame, keys: Sequence[str], values: Sequence[str]
) -> float:
    """the largest absolute difference over the values both tables define; infinite when one
    defines an estimate the other leaves undefined or lacks."""
    joined = official.join(
        mine, on=list(keys), how="full", nulls_equal=True, coalesce=True, suffix="_mine"
    )
    defined = [joined[name].fill_nan(None).is_not_null() for name in ("estimate", "estimate_mine")]
    if (defined[0] != defined[1]).any():
        return float("inf")
    gaps = [
        (joined[name] - joined[f"{name}_mine"]).abs().fill_nan(None).drop_nulls().to_numpy()
        for name in values
    ]
    return max(float(gap.max(initial=0.0)) for gap in gaps)


def harness_rows(
    rows: Sequence[Row], runs: Path, study: str, farm: str, protocol: config.Protocol
) -> list[Row]:
    """the standard threshold against the study's official monitoring table and the early
    history advantage components of its hypotheses table: the largest absolute difference of any
    estimate, bound or p both give, none without those tables."""
    folder = runs / paths.results_folder(study, None) / farm
    names = ("official_monitoring_difference", "official_hypotheses_difference")
    if not all((folder / f"{name}.parquet").is_file() for name in ("monitoring", "hypotheses")):
        return [{"metric": name, "start": POOLED, "estimate": None} for name in names]
    endpoint = protocol.endpoints.early_history_advantage
    own = pl.from_dicts(list(rows), schema=TABLE)
    own_rates = own.filter(pl.col("metric").is_in(["recall", "exceedance"])).select(
        "metric", "drop", "estimate", "lower", "upper", model="monitor", horizon_days="history_days"
    )
    official_rates = pl.read_parquet(folder / "monitoring.parquet").filter(
        pl.col("metric").is_in(["recall", "exceedance"])
    )
    own_gains = own.filter(
        pl.col("metric").is_in(["recall_gain", "exceedance_margin"])
        & (pl.col("history_days") == endpoint.horizon_days)
        & pl.col("monitor").is_in(list(endpoint.monitor_comparators))
    ).select(
        "estimate",
        "lower",
        "upper",
        "p",
        component=pl.concat_str("metric", "monitor", separator="_"),
    )
    official_gains = pl.read_parquet(folder / "hypotheses.parquet").filter(
        (pl.col("hypothesis") == "early_history_advantage")
        & pl.col("component").str.contains("^(recall_gain|exceedance_margin)_")
    )
    keys = ["model", "horizon_days", "metric", "drop"]
    differences = (
        largest_difference(
            own_rates,
            official_rates.select(*keys, "estimate", "lower", "upper"),
            keys,
            ["estimate", "lower", "upper"],
        ),
        largest_difference(
            own_gains,
            official_gains.select("component", "estimate", "lower", "upper", "p"),
            ["component"],
            ["estimate", "lower", "upper", "p"],
        ),
    )
    return [
        {"metric": name, "start": POOLED, "estimate": difference}
        for name, difference in zip(names, differences, strict=True)
    ]


def score(
    data: Path, runs: Path, study: str, farm: str, configs: str | None = None
) -> pl.DataFrame:
    """every rule's recall and exceedance per monitor and history, windfold's gains over the
    other monitors, the calibration and advantage checks, k and the standardized test days per
    fit, each monitor's calibration variance ratio, matched monitors included, and the standard
    threshold's largest difference from the official tables. intervals and p come from the farm's
    planned bootstrap at the endpoint history, or its shortest history without one."""
    confirmatory_ready(runs, study, configs)
    folder = folder_of(configs)
    protocol, farm_config, _ = for_farm(farm, None, folder)
    period = config.load_study(study, folder).period
    hashes = second_freeze(runs, study)["predictions"] if period == "confirmatory" else None
    settings, statistics = protocol.monitoring, protocol.statistics
    reduced = {
        name: pl.read_parquet(runs / paths.reduction_file(study, farm, None, name))
        for name in ("monitoring_scales", "monitoring_days", "fits", "turbines")
    }
    monitors = (*settings.models, *protocol.matched_monitoring.models)
    fits = threshold_fits(reduced["fits"], monitors, settings.min_history_days)
    prepared = pl.read_parquet(
        data / "prepared" / prepared_folder(farm, None, protocol) / "rows.parquet",
        columns=["timestamp", "turbine", *metrics.PREPARED_COLUMNS],
    )
    days = calibration_days(prepared, runs, study, farm, fits, protocol, hashes)
    scales = (
        reduced["monitoring_scales"]
        .filter(pl.col("model").is_in(list(monitors)))
        .with_columns(copy=pl.int_range(pl.len(), dtype=pl.Int64).over("model", "split", "turbine"))
    )
    nominal = settings.false_alarms_per_turbine_month / monitoring.MONTH_DAYS
    factors = scale_factors(scales, days, nominal)
    seed = farm_seed(protocol.seeds.study, farm)
    windows = bootstrap_windows(
        study_windows(farm_config, period),
        statistics.block_days,
        statistics.replicates,
        len(protocol.seeds.model),
        statistics.interval,
        statistics.min_finite,
        seed,
    )
    origin, draw = windows.origins["test"], windows.draws["test"]
    slots = monitoring.trajectories(windows.drawn["test"], origin[1], statistics.block_days)
    turbines = reduced["turbines"].sort("turbine_index")["turbine"].to_list()
    quantile = monitoring.alarm_quantile(settings.false_alarms_per_turbine_month)
    seeds = {(model, split, copy): fitted for model, split, fitted, copy in fits.iter_rows()}
    endpoint = protocol.endpoints.early_history_advantage.horizon_days
    records: list[Row] = []
    for rule in RULES:
        grids, copies = summaries.grids_and_copies(
            {
                "monitoring_scales": scaled(scales, factors, rule),
                "monitoring_days": reduced["monitoring_days"],
            },
            origin,
            turbines,
            protocol,
            seed,
            settings.models,
        )
        horizons = sorted({h for h, _ in copies})
        at = endpoint if endpoint in horizons else next(iter(horizons), None)
        found = rates(grids, copies, turbines, slots, draw, at, protocol)
        rows = rate_rows(found, draw) + gain_rows(found, draw, protocol)
        rows += check_rows(rows, nominal, protocol)
        if rule == "registered":
            rows += harness_rows(rows, runs, study, farm, protocol)
        rows += day_rows(grids, seeds, quantile)
        records += [row | {"rule": rule} for row in rows]
    records += factor_rows(factors.filter(pl.col("model").is_in(list(settings.models))), seeds)
    records += [
        {"metric": "variance_ratio", "monitor": model, "history_days": horizon}
        | {"start": POOLED, "estimate": ratio}
        for model, horizon, ratio in variance_ratios(scales, days)
        .sort("model", "horizon_days")
        .iter_rows()
    ]
    return pl.from_dicts([row | {"farm": farm} for row in records], schema=TABLE)
