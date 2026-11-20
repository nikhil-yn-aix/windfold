"""what time-blocked validation chooses, and how far its ranking carries to held-out sectors.

validation misleads takes, in each replicate of the farm's planned bootstrap of the study
period, the baseline with the lowest pooled waked-row mse over the time-blocked folds and scores
it on the single arcs. per farm: each baseline's time-blocked and single-arc mse, the one chosen
on the full data and the share of replicates that choose each. then kendall's tau-b between the
time-blocked and single-arc rankings of every model the results rank, and of those models without
windfold's variants, with intervals from the same bootstrap.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.analyses.development import FARMS, STUDY
from windfold.config import for_farm, load_study
from windfold.evaluation import bootstrap, paths, summaries
from windfold.evaluation.hypotheses import pooled_mse
from windfold.evaluation.results import KEYS, RANKED_REGIMES, bootstrap_windows, study_windows

VARIANTS = "windfold_"
FIRST, SECOND = "time_blocked", "arc_single"


def baseline_mse(
    cv: bootstrap.Totals,
    counts: np.ndarray,
    seeds: np.ndarray | None,
    baselines: Sequence[str],
    splits: Sequence[str],
) -> np.ndarray:
    """replicates by baselines: each baseline's pooled waked-row mse over splits."""
    return np.column_stack([pooled_mse(cv, counts, seeds, m, splits) for m in baselines])


def farm_choice(runs: Path, farm: str, configs: Path | None) -> dict[str, Any]:
    protocol, farm_config, _ = for_farm(farm, None, configs)
    statistics = protocol.statistics
    reduced = {
        name: pl.read_parquet(runs / paths.reduction_file(STUDY, farm, None, name))
        for name in ("errors", "fits")
    }
    fits = reduced["fits"]
    splits = {
        regime: sorted(fits.filter(pl.col("regime") == regime)["split"].unique().to_list())
        for regime in RANKED_REGIMES
    }
    windows = bootstrap_windows(
        study_windows(farm_config, load_study(STUDY, configs).period),
        statistics.block_days,
        statistics.replicates,
        len(protocol.seeds.model),
        statistics.interval,
        statistics.min_finite,
        bootstrap.farm_seed(protocol.seeds.study, farm),
    )
    ranked = [s for regime in RANKED_REGIMES for s in splits[regime]]
    cv = bootstrap.block_totals(
        reduced["errors"].filter(pl.col("split").is_in(ranked)),
        KEYS,
        ["sse", "rows"],
        windows.origins["period"][0],
        statistics.block_days,
        windows.blocks["period"],
    )
    draw = windows.draws["period"]
    baselines = protocol.endpoints.validation_misleads.baselines
    observed = {
        regime: baseline_mse(cv, draw.observed(), None, baselines, splits[regime])[0]
        for regime in (FIRST, SECOND)
    }
    drawn = baseline_mse(cv, draw.counts, draw.seeds, baselines, splits[FIRST])
    chosen = np.argmin(drawn, axis=1)
    shares = np.bincount(chosen, minlength=len(baselines)) / chosen.size
    models = sorted(fits.filter(pl.col("status") == "done")["model"].unique().to_list())
    agreement = {}
    for label, kept in (
        ("all", models),
        ("without_windfold_variants", [m for m in models if not m.startswith(VARIANTS)]),
    ):
        table = summaries.rank_agreement(cv, draw, splits, kept).filter(
            (pl.col("first") == FIRST) & (pl.col("second") == SECOND)
        )
        agreement[label] = table.drop("first", "second").row(0, named=True)
    return {
        "replicates": int(chosen.size),
        "chosen": baselines[int(np.argmin(observed[FIRST]))],
        "baselines": {
            name: {
                f"{FIRST}_mse": float(observed[FIRST][k]),
                f"{SECOND}_mse": float(observed[SECOND][k]),
                "chosen_share": float(shares[k]),
            }
            for k, name in enumerate(baselines)
        },
        "rank_agreement": agreement,
    }


def validation_choice(
    runs: Path, farms: Sequence[str] = FARMS, configs: Path | None = None
) -> dict[str, Any]:
    return {farm: farm_choice(runs, farm, configs) for farm in farms}
