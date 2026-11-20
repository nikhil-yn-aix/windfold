"""what the report reads: configurations, prepared data, the results of its studies and the
exploratory analyses and secondary tests run beside them."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import config


@dataclass(frozen=True)
class Sources:
    """studies in reading order: the first holds the development results, the rest add to them."""

    data: Path
    runs: Path
    configs: Path | None
    studies: tuple[str, ...]

    @property
    def main(self) -> str:
        return self.studies[0]

    def results(self, study: str, name: str) -> pl.DataFrame:
        """a cross-farm result table of one study's main analysis."""
        return pl.read_parquet(self.runs / study / "results" / f"{name}.parquet")

    def protocol(self) -> config.Protocol:
        return config.load_protocol(self.configs)

    def models(self) -> dict[str, config.ModelSpec]:
        return config.load_models(self.configs)

    def farms(self, name: str) -> dict[str, list[str]]:
        """the farms each study's cross-farm results table holds."""
        return {
            study: self.results(study, name)["farm"].unique().to_list() for study in self.studies
        }

    def confirmatory(self) -> list[str]:
        """the studies of the confirmatory period, read only after its second freeze."""
        return [
            study
            for study in self.studies
            if config.load_study(study, self.configs).period == "confirmatory"
        ]

    def analysis(self, name: str, study: str | None = None) -> dict[str, Any]:
        """the result of an exploratory analysis of a study, the main one by default, as
        `windfold analyse` wrote it."""
        path = self.runs / (study or self.main) / "analyses" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run windfold analyse {name}")
        return json.loads(path.read_text(encoding="utf-8"))["result"]

    def secondary(self, study: str, test: str) -> Path:
        """the folder of a secondary test's outputs for a study."""
        return self.runs / study / "secondary" / test


def point(rows: pl.DataFrame, scale: float = 1.0) -> tuple[float, float, float] | None:
    """the one row's estimate and 95% bounds times scale, bounds nan where the bootstrap left
    them null; None when there is no single estimated row."""
    if rows.height != 1:
        return None
    found = rows.row(0, named=True)
    if found["estimate"] is None or not np.isfinite(found["estimate"]):
        return None
    low, high = (np.nan if found[k] is None else scale * found[k] for k in ("lower", "upper"))
    return scale * found["estimate"], low, high


def bounded(value: dict[str, Any]) -> tuple[float, float, float]:
    """an analysis's estimate with its low and high bounds, a missing one nan."""
    centre, low, high = (
        np.nan if value.get(k) is None else float(value[k]) for k in ("estimate", "low", "high")
    )
    return centre, low, high
