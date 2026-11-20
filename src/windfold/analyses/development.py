"""what the exploratory analyses read of the development study: its farms, its single-arc splits
and the files of its fits. none of them reads a confirmatory result."""

import json
from pathlib import Path
from typing import Any

import polars as pl

from windfold.locations import prediction_files

STUDY = "development"
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie", "smarteole")
LARGE_FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")


def single_arcs(runs: Path, farm: str, waked: bool = True) -> list[str]:
    """the farm's included single-arc splits in name order; with waked, only those whose test
    rows include waked rows, the ones that count toward waked-row skill."""
    summary = pl.read_csv(runs / STUDY / "manifests" / f"{farm}_summary.csv")
    chosen = summary.filter(pl.col("regime") == "arc_single", pl.col("status") == "included")
    if waked:
        chosen = chosen.filter(pl.col("test_waked_rows") > 0)
    return sorted(chosen["split"].to_list())


def prediction_file(root: Path, farm: str, model: str, split: str) -> Path:
    """a fit's predictions under root, a runs folder or the output folder of a refit."""
    return root / prediction_files(STUDY, None, farm, model, split, None)[0]


def record_file(root: Path, farm: str, model: str, split: str) -> Path:
    """the record a fit wrote beside its predictions under root."""
    return root / prediction_files(STUDY, None, farm, model, split, None)[1]


def fit_record(root: Path, farm: str, model: str, split: str) -> dict[str, Any]:
    return json.loads(record_file(root, farm, model, split).read_text())


def waked_rows(root: Path, farm: str, model: str, split: str) -> pl.DataFrame:
    """a fit's waked test rows: timestamp, turbine, y and its prediction, mean."""
    frame = pl.read_parquet(
        prediction_file(root, farm, model, split),
        columns=["timestamp", "turbine", "y", "mean", "waked"],
    )
    return frame.filter(pl.col("waked")).drop("waked")


def check_paired(frames: dict[str, pl.DataFrame], where: str) -> None:
    """raises unless every frame holds the same rows, in the same order, with the same y."""
    keys = [frame.select("timestamp", "turbine", "y") for frame in frames.values()]
    if any(not key.equals(keys[0]) for key in keys[1:]):
        raise ValueError(f"{where}: {', '.join(frames)} do not predict the same rows")
