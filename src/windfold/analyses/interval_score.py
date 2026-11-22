"""the predictive intervals of the gaussian models on held-out arcs and in time-blocked validation.

per farm of a study, for every model whose predictions carry a gaussian variance, on the waked rows
of the single arcs and of the time-blocked folds: the share of rows inside the central interval of
N(mean, variance) at each coverage level of the protocol (90 and 99%), and the mean interval score
of gneiting and raftery (2007, eq. 43), (u - l) + (2 / alpha)(l - y)+ + (2 / alpha)(y - u)+ with
alpha one minus the level. the score is in the units of y and, beside it, on the leave-one-out
contrast scale of the official crps and interval widths, m / (m - 1) times y for a timestamp's m
valid turbines. then windfold minus every other gaussian model in coverage, interval score and the
official crps, so a negative score difference favors windfold.

each fit is scored as the reductions score it and its scores are summed by day, as crps is, then
pooled over the regime's splits (a seeded model's averaged over its seeds) with the farm's
planned bootstrap. the coverage, interval widths and crps must reproduce the farm's calibration
and accuracy tables. confirmatory predictions are read only after the second freeze, against its
hashes.
"""

import io
import math
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import ndtri

from windfold import config
from windfold.analyses.planned_tests import (
    PlannedBootstrap,
    official,
    planned_bootstrap,
    reduction,
    summary,
)
from windfold.evaluation import metrics, paths, reductions, summaries
from windfold.evaluation.results import KEYS
from windfold.evaluation.summaries import Draw
from windfold.locations import manifest_file, prepared_folder
from windfold.tasks.evaluation import frozen_bytes, second_freeze

type Compute = summaries.Compute
type Found = tuple[float | None, float | None, float | None] | None
type Value = Callable[[str, str, float | None, str], Compute]
type Measure = Callable[[str, float | None, str], Compute]

MODEL = "windfold"
REGIMES = ("arc_single", "time_blocked")
SCORES = ["rows", "covered", "width", "interval_score", "interval_score_contrast"]
REPORTED = {
    "coverage": "covered",
    "interval_score": "interval_score",
    "interval_score_contrast": "interval_score_contrast",
}
CALIBRATED = (("coverage", "covered", True), ("width", "width", False))
TOLERANCE = 1e-9


def day_scores(rows: pl.DataFrame, levels: Sequence[float]) -> pl.DataFrame:
    """a scored fit's rows, rows covered, interval widths (contrast scale) and interval scores at
    each level, summed by day and waked over the rows with a variance; covered and width as
    metrics.intervals computes them."""
    usable = rows.filter(pl.col("variance").is_not_null())
    loo_error = (usable["scale"] * usable["error"]).abs().to_numpy()
    contrast_sd = (usable["scale"] * usable["variance"].sqrt()).to_numpy()
    y, mean = usable["y"].to_numpy(), usable["mean"].to_numpy()
    sd, scale = usable["variance"].sqrt().to_numpy(), usable["scale"].to_numpy()
    parts = []
    for level in levels:
        z = float(ndtri((1 + level) / 2))
        lower, upper = mean - z * sd, mean + z * sd
        penalty = 2 / (1 - level)
        score = (
            (upper - lower)
            + penalty * np.maximum(lower - y, 0.0)
            + penalty * np.maximum(y - upper, 0.0)
        )
        parts.append(
            usable.select("day", "waked")
            .with_columns(
                level=pl.lit(level, pl.Float64),
                covered=pl.Series(loo_error <= z * contrast_sd),
                width=pl.Series(2 * z * contrast_sd),
                interval_score=pl.Series(score),
                interval_score_contrast=pl.Series(scale * score),
            )
            .group_by("day", "waked", "level")
            .agg(
                rows=pl.len().cast(pl.Int64),
                covered=pl.col("covered").sum().cast(pl.Int64),
                width=pl.col("width").sum(),
                interval_score=pl.col("interval_score").sum(),
                interval_score_contrast=pl.col("interval_score_contrast").sum(),
            )
        )
    return pl.concat(parts).sort("level", "day", "waked")


def fit_scores(
    data: Path,
    runs: Path,
    farm: str,
    chosen: PlannedBootstrap,
    fits: pl.DataFrame,
    hashes: dict[str, str] | None,
) -> pl.DataFrame:
    """day_scores of every fit, keyed by model, split and seed. each fit's predictions are
    scored against the prepared rows of its test window, as reduce_farm scores them."""
    protocol = chosen.protocol
    study = chosen.study.name
    prepared = pl.read_parquet(
        data / "prepared" / prepared_folder(farm, None, protocol) / "rows.parquet",
        columns=["timestamp", "turbine", *metrics.PREPARED_COLUMNS],
    )
    manifest = pl.read_parquet(runs / manifest_file(study, farm, None, protocol))
    days = chosen.study.limits.days if chosen.study.limits else None
    levels = protocol.statistics.coverage_levels
    parts = []
    for (split,), part in fits.sort("split").group_by("split", maintain_order=True):
        here = reductions.window_rows(prepared, manifest, str(split), days)
        for model, seed in part.sort("model", "seed").select("model", "seed").iter_rows():
            table, _ = paths.fit_inputs(study, farm, None, model, str(split), seed, protocol)
            predictions = pl.read_parquet(io.BytesIO(frozen_bytes(runs, table, hashes)))
            if predictions.height != here.height:
                raise ValueError(
                    f"{model} on {split} has {predictions.height} predictions for "
                    f"{here.height} rows"
                )
            scored = metrics.scored(predictions, here)
            parts.append(
                day_scores(scored, levels).with_columns(
                    model=pl.lit(model), split=pl.lit(split), seed=pl.lit(seed, pl.Int64)
                )
            )
    return pl.concat(parts)


def agrees(found: Found, row: dict[str, Any] | None, exact: bool) -> bool:
    """found (estimate, lower, upper) against an official row, exactly or within TOLERANCE."""
    theirs = None if row is None else (row["estimate"], row["lower"], row["upper"])
    if found is None or theirs is None:
        return found is None and theirs is None
    if exact:
        return found == theirs
    return all(
        (a is None and b is None)
        or (a is not None and b is not None and math.isclose(a, b, rel_tol=TOLERANCE))
        for a, b in zip(found, theirs, strict=True)
    )


def official_row(table: pl.DataFrame, **keys: object) -> dict[str, Any] | None:
    chosen = table
    for name, value in keys.items():
        chosen = chosen.filter(pl.col(name).is_null() if value is None else pl.col(name) == value)
    if chosen.height > 1:
        raise ValueError(f"{chosen.height} official rows match {keys}")
    return chosen.row(0, named=True) if chosen.height else None


def check_official(
    value: Value,
    draw: Draw,
    models: Sequence[str],
    levels: Sequence[float],
    calibration: pl.DataFrame,
    accuracy: pl.DataFrame,
) -> int:
    """raises unless each model's coverage (exactly), interval width and crps reproduce the
    farm's calibration and accuracy tables; the number of values checked."""
    checks = []
    for model, regime in ((m, r) for m in models for r in REGIMES):
        keys = {"model": model, "regime": regime}
        for level, (metric, quantity, exact) in ((v, c) for v in levels for c in CALIBRATED):
            row = keys | {"level": level, "metric": metric}
            checks.append((value(model, regime, level, quantity), calibration, row, exact))
        row = keys | {"horizon_days": None, "rows": "waked", "metric": "crps"}
        checks.append((value(model, regime, None, "crps"), accuracy, row, False))
    for compute, table, keys, exact in checks:
        found, row = summaries.measured(draw, compute), official_row(table, **keys)
        if not agrees(found, row, exact):
            raise ValueError(f"{keys} gives {found}; the official table has {row}")
    return len(checks)


def farm_intervals(
    data: Path, runs: Path, study: str, farm: str, configs: Path | None = None
) -> dict[str, Any]:
    chosen = planned_bootstrap(runs, study, farm, configs)
    levels = chosen.protocol.statistics.coverage_levels
    draw = chosen.draw
    errors, fits = (reduction(runs, study, farm, name) for name in ("errors", "fits"))
    gaussian = sorted(errors.filter(pl.col("crps").is_not_null())["model"].unique().to_list())
    splits = {
        regime: sorted(fits.filter(pl.col("regime") == regime)["split"].unique().to_list())
        for regime in REGIMES
    }
    every = [split for regime in REGIMES for split in splits[regime]]
    done = fits.filter(
        pl.col("status") == "done", pl.col("model").is_in(gaussian), pl.col("split").is_in(every)
    )
    hashes = None
    if chosen.study.period == "confirmatory":
        hashes = second_freeze(runs, study)["predictions"]
    scores = fit_scores(data, runs, farm, chosen, done, hashes)
    intervals = chosen.totals(scores, [*KEYS, "level"], SCORES)
    crps = chosen.totals(
        errors.filter(pl.col("model").is_in(gaussian), pl.col("split").is_in(every)),
        KEYS,
        ["crps", "rows"],
    )

    def value(model: str, regime: str, level: float | None, quantity: str) -> Compute:
        """a model's per-row mean of a quantity on the regime's waked rows; crps without level."""
        if level is None:
            return summaries.averaged(crps, model, splits[regime], True, quantity)
        return summaries.averaged_level(intervals, model, splits[regime], level, quantity)

    def difference(other: str, regime: str, level: float | None, quantity: str) -> Compute:
        own, theirs = value(MODEL, regime, level, quantity), value(other, regime, level, quantity)
        return lambda counts, seeds: own(counts, seeds) - theirs(counts, seeds)

    def described(compute: Measure) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for regime in REGIMES:
            out[regime] = {
                f"{level:g}": {
                    name: summary(summaries.measured(draw, compute(regime, level, quantity)))
                    for name, quantity in REPORTED.items()
                }
                for level in levels
            }
            out[regime]["crps"] = summary(summaries.measured(draw, compute(regime, None, "crps")))
        return out

    checked = check_official(
        value,
        draw,
        gaussian,
        levels,
        official(runs, study, farm, "calibration"),
        official(runs, study, farm, "accuracy"),
    )
    waked = scores.filter(pl.col("model") == MODEL, pl.col("waked"), pl.col("level") == levels[0])
    return {
        "waked_rows": {
            regime: int(waked.filter(pl.col("split").is_in(splits[regime]))["rows"].sum())
            for regime in REGIMES
        },
        "models": {
            model: {
                "seeds": done.filter(pl.col("model") == model)["seed"].unique().sort().to_list()
            }
            | described(partial(value, model))
            for model in gaussian
        },
        f"{MODEL}_minus": {
            other: described(partial(difference, other)) for other in gaussian if other != MODEL
        },
        "official_values_reproduced": checked,
    }


def interval_score(
    data: Path, runs: Path, study: str, farms: Sequence[str] = (), configs: Path | None = None
) -> dict[str, Any]:
    """farm_intervals over the farms given, or every farm of the study."""
    chosen = farms or config.load_study(study, configs).farms
    return {farm: farm_intervals(data, runs, study, farm, configs) for farm in chosen}
