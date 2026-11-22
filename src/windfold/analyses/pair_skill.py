"""skill differences of chosen pairs of models on held-out arcs, with the planned bootstrap.

per farm of a study, on the waked rows of the single arcs: for each pair, the first model minus the
second in skill points, 100 (mse_second - mse_first) / mse_no_wake, each mse pooled over the arcs
(a seeded model's averaged over its seeds, redrawn in every replicate) and the difference formed in
every replicate of the farm's planned bootstrap, the day blocks and seed draws of its result
tables, with the 95% percentile interval. the pairs set the physics against the peer model, the
wind-frame learners against their identity twins and kernel_linear against identity_linear, the
physics and windfold. a pair whose models lack fits on some arc is left without an estimate.
confirmatory results are read only after the second freeze.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from windfold import config
from windfold.analyses.planned_tests import planned_bootstrap, reduction
from windfold.evaluation import summaries
from windfold.evaluation.results import KEYS

REGIME, REFERENCE = "arc_single", "no_wake"
PAIRS = (
    ("pywake", "peer_automl"),
    ("geometry_gbm", "identity_gbm"),
    ("wind_frame_graph", "absolute_graph"),
    ("kernel_linear", "identity_linear"),
    ("kernel_linear", "pywake"),
    ("kernel_linear", "pywake_inflow"),
    ("kernel_linear", "windfold"),
)


def farm_pairs(
    runs: Path, study: str, farm: str, configs: Path | None = None
) -> list[dict[str, Any]]:
    """one row per pair: the farm, the pair, the estimate and its interval."""
    chosen = planned_bootstrap(runs, study, farm, configs)
    fits = reduction(runs, study, farm, "fits")
    arcs = sorted(fits.filter(pl.col("regime") == REGIME)["split"].unique().to_list())
    models = sorted({model for pair in PAIRS for model in pair} | {REFERENCE})
    errors = reduction(runs, study, farm, "errors").filter(
        pl.col("model").is_in(models), pl.col("split").is_in(arcs)
    )
    totals = chosen.totals(errors, KEYS, ["sse", "rows"])
    mse = {model: summaries.averaged(totals, model, arcs, True, "sse") for model in models}

    def gap(first: str, second: str) -> summaries.Compute:
        own, theirs, base = mse[first], mse[second], mse[REFERENCE]
        return lambda counts, seeds: (
            100 * (theirs(counts, seeds) - own(counts, seeds)) / base(counts, seeds)
        )

    rows = []
    for first, second in PAIRS:
        found = summaries.measured(chosen.draw, gap(first, second))
        estimate, lower, upper = found or (None, None, None)
        rows.append(
            {
                "farm": farm,
                "pair": f"{first}_minus_{second}",
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
            }
        )
    return rows


def pair_skill(
    runs: Path, study: str, farms: Sequence[str] = (), configs: Path | None = None
) -> dict[str, Any]:
    """the table of farm_pairs over the farms given, or every farm of the study."""
    chosen = farms or config.load_study(study, configs).farms
    return {"table": [row for farm in chosen for row in farm_pairs(runs, study, farm, configs)]}
