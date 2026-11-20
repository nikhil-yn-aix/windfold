"""how far model rankings carry between validations once windfold's variants are set aside.

the official rank agreement ranks every model a farm's results rank, windfold's variants among
them. per farm of a study: kendall's tau-b between the waked-row mse rankings of each pair of the
ranked regimes (random, time-blocked and single-arc validation), with the official estimator and
intervals from the farm's planned bootstrap, over three sets of models: the compared models
(every model but windfold's variants, windfold among them), the comparison models alone (windfold
left out too) and all models, which must reproduce the study's rank agreement table. confirmatory
results are read only after the second freeze.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from windfold import config
from windfold.analyses.planned_tests import planned_bootstrap, reduction
from windfold.evaluation import paths, summaries
from windfold.evaluation.results import KEYS, RANKED_REGIMES

MODEL, VARIANTS = "windfold", "windfold_"


def model_sets(models: Sequence[str]) -> dict[str, list[str]]:
    """the compared models, the comparison models alone and all models."""
    compared = [m for m in models if not m.startswith(VARIANTS)]
    return {
        "compared": compared,
        "comparison": [m for m in compared if m != MODEL],
        "all": list(models),
    }


def study_table(runs: Path, study: str, farm: str) -> pl.DataFrame:
    """the farm's rows of the study's rank agreement table."""
    table = pl.read_parquet(runs / paths.results_folder(study, None) / "rank_agreement.parquet")
    return table.filter(pl.col("farm") == farm).drop("farm")


def farm_ranks(runs: Path, study: str, farm: str, configs: Path | None = None) -> dict[str, Any]:
    chosen = planned_bootstrap(runs, study, farm, configs)
    errors, fits = (reduction(runs, study, farm, name) for name in ("errors", "fits"))
    splits = {
        regime: sorted(fits.filter(pl.col("regime") == regime)["split"].unique().to_list())
        for regime in RANKED_REGIMES
    }
    ranked = [split for regime in RANKED_REGIMES for split in splits[regime]]
    cv = chosen.totals(errors.filter(pl.col("split").is_in(ranked)), KEYS, ["sse", "rows"])
    models = sorted(fits.filter(pl.col("status") == "done")["model"].unique().to_list())
    found: dict[str, Any] = {}
    for name, kept in model_sets(models).items():
        table = summaries.rank_agreement(cv, chosen.draw, splits, kept)
        if name == "all" and not table.equals(study_table(runs, study, farm)):
            raise ValueError(
                f"{farm}: the rank agreement of all models does not reproduce the {study} "
                "study's rank_agreement.parquet"
            )
        usable = [m for m in kept if all(summaries.measured_ok(cv, m, s) for s in splits.values())]
        found[name] = {
            "models": usable,
            "pairs": {
                f"{row['first']}_against_{row['second']}": {
                    "models": row["models"],
                    "estimate": row["estimate"],
                    "low": row["lower"],
                    "high": row["upper"],
                }
                for row in table.iter_rows(named=True)
            },
        }
    return found


def comparison_ranks(
    runs: Path, study: str, farms: Sequence[str] = (), configs: Path | None = None
) -> dict[str, Any]:
    """farm_ranks over the farms given, or every farm of the study."""
    chosen = farms or config.load_study(study, configs).farms
    return {farm: farm_ranks(runs, study, farm, configs) for farm in chosen}
