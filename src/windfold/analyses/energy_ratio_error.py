"""each model's error of predicted against observed energy ratio on held-out sectors, by farm:
the mean absolute difference over turbine and 5 degree bucket cells of at least the report's
minimum rows, weighted by their rows, with the cells and rows it covers."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from windfold.report.energy_ratio import errors
from windfold.report.style import load_report


def energy_ratio_error(
    runs: Path, study: str, farms: Sequence[str], configs: Path | None = None
) -> dict[str, Any]:
    min_rows = load_report(configs).failure.min_rows
    found = {}
    for farm in farms:
        table = pl.read_parquet(runs / study / "results" / farm / "energy_ratio.parquet")
        found[farm] = {
            row["model"]: {"error": row["error"], "cells": row["cells"], "rows": row["rows"]}
            for row in errors(table, min_rows).iter_rows(named=True)
        }
    return found
