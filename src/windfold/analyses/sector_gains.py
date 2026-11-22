"""windfold's gain over the same physics on each held-out arc, against how often the arc occurs.

per farm of a study, for each included single arc with waked test rows: its center direction, its
waked test rows and their share of the waked test rows of all single arcs and windfold minus
pywake_inflow in skill points (their mse difference in points of no_wake's mse on the arc's waked
rows) with its interval and one-sided p from the farm's planned bootstrap, as practical_scale
computes it per arc. then, over three or more such arcs, the spearman correlation between share
and gain, with an interval from the same bootstrap in which every replicate recomputes shares and
gains on the same arcs, and the arcs windfold is ahead on (interval above zero), behind on (below
zero) and not resolved on. confirmatory results are read only after the second freeze.
"""

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from windfold import config
from windfold.analyses.planned_tests import planned_bootstrap, reduction
from windfold.analyses.practical_scale import MODEL, PHYSICS, REFERENCE, endpoints
from windfold.evaluation import hypotheses, summaries
from windfold.evaluation.results import KEYS
from windfold.secondary.contrasts import single_arcs


def spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """spearman's correlation between the last axes of x and y, for any leading (replicate) axes;
    nan where either holds a nan."""
    x, y = np.atleast_2d(np.asarray(x, dtype=float)), np.atleast_2d(np.asarray(y, dtype=float))
    missing = np.isnan(x).any(axis=-1) | np.isnan(y).any(axis=-1)
    ranks = [rankdata(np.nan_to_num(a), axis=-1) for a in (x, y)]
    rx, ry = (r - r.mean(axis=-1, keepdims=True) for r in ranks)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rx * ry).sum(axis=-1) / np.sqrt((rx**2).sum(axis=-1) * (ry**2).sum(axis=-1))
    return np.where(missing, np.nan, rho)


def verdict(low: float | None, high: float | None) -> str:
    if low is not None and low > 0:
        return "ahead"
    if high is not None and high < 0:
        return "behind"
    return "not resolved"


def farm_gains(runs: Path, study: str, farm: str, configs: Path | None = None) -> dict[str, Any]:
    chosen = planned_bootstrap(runs, study, farm, configs)
    protocol, draw = chosen.protocol, chosen.draw
    statistics = protocol.statistics
    arcs = single_arcs(runs, study, farm, protocol)
    errors = reduction(runs, study, farm, "errors").filter(
        pl.col("model").is_in([MODEL, PHYSICS, REFERENCE]), pl.col("split").is_in(arcs)
    )
    totals = chosen.totals(errors, KEYS, ["sse", "rows"])
    waked = (
        errors.filter(pl.col("model") == MODEL, pl.col("waked"))
        .group_by("split")
        .agg(pl.col("rows").sum())
        .filter(pl.col("rows") > 0)
        .sort("split")
    )
    names, rows = waked["split"].to_list(), waked["rows"].to_list()
    width = protocol.splits.arc_width_deg
    found: dict[str, hypotheses.Component] = {
        split: hypotheses.attempt(
            partial(
                hypotheses.estimate,
                endpoints(totals, [split], PHYSICS)["skill_points"],
                draw.counts,
                draw.seeds,
            )
        )
        for split in names
    }
    table = hypotheses.contrast_rows(found, statistics.interval, statistics.min_finite)
    sectors = [
        {
            "split": split,
            "centre_deg": (int(split.split("_")[1]) + width / 2) % 360,
            "waked_rows": own,
            "share": own / sum(rows),
            "skill_points": {
                "estimate": row["estimate"],
                "low": row["lower"],
                "high": row["upper"],
                "p": row["p"],
            },
            "verdict": verdict(row["lower"], row["upper"]),
        }
        for split, own, row in zip(names, rows, table.iter_rows(named=True), strict=True)
    ]

    def shares(counts: np.ndarray) -> np.ndarray:
        drawn = np.column_stack(
            [
                hypotheses.total(totals, counts, "rows", model=MODEL, split=split, waked=True)
                for split in names
            ]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return drawn / drawn.sum(axis=1, keepdims=True)

    estimated = [part for part in found.values() if not isinstance(part, str)]
    correlation = None
    if len(estimated) == len(names) and len(names) > 2:
        observed = spearman(shares(draw.observed()), np.array([e[0] for e in estimated]))[0]
        replicates = spearman(shares(draw.counts), np.column_stack([e[1] for e in estimated]))
        correlation = summaries.summarise(float(observed), replicates, draw)
    estimate, low, high = correlation or (None, None, None)
    verdicts = [sector["verdict"] for sector in sectors]
    return {
        "sectors": sectors,
        "without_waked_rows": [split for split in arcs if split not in names],
        "spearman_share_gain": {"estimate": estimate, "low": low, "high": high},
        "ahead": verdicts.count("ahead"),
        "behind": verdicts.count("behind"),
        "not_resolved": verdicts.count("not resolved"),
    }


def sector_gains(
    runs: Path, study: str, farms: Sequence[str] = (), configs: Path | None = None
) -> dict[str, Any]:
    """farm_gains over the farms given, or every farm of the study."""
    chosen = farms or config.load_study(study, configs).farms
    return {farm: farm_gains(runs, study, farm, configs) for farm in chosen}
