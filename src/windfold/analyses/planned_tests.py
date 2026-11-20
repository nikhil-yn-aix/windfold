"""what the analyses of either study share: the refusal of confirmatory data before the second
freeze, a farm's reductions and official tables and the planned bootstrap of its study period,
drawn as its results draw it, so an interval here uses the same day blocks and seed redraws as the
farm's result tables."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from windfold import config
from windfold.config import Protocol, for_farm
from windfold.evaluation import bootstrap, paths
from windfold.evaluation.results import Windows, bootstrap_windows, study_windows
from windfold.evaluation.summaries import Draw
from windfold.secondary.contrasts import confirmatory_ready


@dataclass(frozen=True)
class PlannedBootstrap:
    """a farm's protocol, its study and the bootstrap of the study period."""

    protocol: Protocol
    study: config.Study
    windows: Windows

    @property
    def draw(self) -> Draw:
        return self.windows.draws["period"]

    def totals(
        self, frame: pl.DataFrame, keys: Sequence[str], quantities: Sequence[str]
    ) -> bootstrap.Totals:
        """day sums of frame in the period's calendar blocks."""
        return bootstrap.block_totals(
            frame,
            keys,
            quantities,
            self.windows.origins["period"][0],
            self.protocol.statistics.block_days,
            self.windows.blocks["period"],
        )


def planned_bootstrap(
    runs: Path, study: str, farm: str, configs: Path | None = None
) -> PlannedBootstrap:
    """refuses confirmatory data before the study's second freeze; then the farm's protocol and
    bootstrap, with the replicates and seeds its study limits, as results.farm_tables draws it."""
    confirmatory_ready(runs, study, str(configs) if configs else None)
    protocol, farm_config, _ = for_farm(farm, None, configs)
    chosen = config.load_study(study, configs)
    statistics, limits = protocol.statistics, chosen.limits
    replicates = limits.replicates if limits and limits.replicates else statistics.replicates
    seeds = len(protocol.seeds.model)
    if limits and limits.seeds:
        seeds = min(seeds, limits.seeds)
    windows = bootstrap_windows(
        study_windows(farm_config, chosen.period),
        statistics.block_days,
        replicates,
        seeds,
        statistics.interval,
        statistics.min_finite,
        bootstrap.farm_seed(protocol.seeds.study, farm),
    )
    return PlannedBootstrap(protocol, chosen, windows)


def reduction(runs: Path, study: str, farm: str, name: str) -> pl.DataFrame:
    return pl.read_parquet(runs / paths.reduction_file(study, farm, None, name))


def official(runs: Path, study: str, farm: str, name: str) -> pl.DataFrame:
    """one of the farm's result tables."""
    return pl.read_parquet(runs / paths.results_folder(study, None) / farm / f"{name}.parquet")


def summary(found: tuple[float | None, float | None, float | None] | None) -> dict[str, Any]:
    """an estimate and its interval, none where the estimate cannot be formed."""
    estimate, low, high = found or (None, None, None)
    return {"estimate": estimate, "low": low, "high": high}
