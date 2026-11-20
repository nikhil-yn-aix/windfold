"""a farm's reductions: what its results need from every fit, keyed by day.

each fit's predictions are scored once against the prepared rows of its test window. the arc
table adds each held-out arc's coverage score and covariates, read from the design alone.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold import monitoring
from windfold.config import Protocol, TurbineConfig
from windfold.evaluation import metrics
from windfold.evaluation.paths import REDUCTIONS
from windfold.evaluation.schemas import BREAKDOWN, ENERGY_RATIO, ERRORS, INTERVALS, SCALES
from windfold.schemas import SPLIT_SUMMARY
from windfold.smearing import SMEARING
from windfold.splits import first_days

TEST_ROLES = ("test", "calibrate")
BREAKDOWN_REGIMES = ("time_blocked", "arc_single")
DESIGN_REGIMES = ("arc_single", "arc_opposite")
FITS = pl.Schema(
    {
        "model": pl.String,
        "split": pl.String,
        "seed": pl.Int64,
        "regime": pl.String,
        "status": pl.String,
    }
)
ARCS = pl.Schema(
    {
        "split": pl.String,
        "regime": pl.String,
        "arc": pl.Int64,
        "score": pl.Float64,
        "permuted_score": pl.Float64,
        "wake_strength": pl.Float64,
        "angular_distance": pl.Float64,
        "training_rows": pl.Float64,
        "dissimilarity": pl.Float64,
        "relational_mse": pl.Float64,
        "physics_mse": pl.Float64,
    }
)
MONITORING_SCALES = pl.Schema({"model": pl.String, "split": pl.String} | dict(SCALES))
MONITORING_DAYS = pl.Schema(
    {
        "model": pl.String,
        "split": pl.String,
        "turbine": pl.String,
        "day": pl.Date,
        "rows": pl.Int64,
        "residual": pl.Float64,
    }
)
TURBINES = pl.Schema({"turbine": pl.String, "turbine_index": pl.UInt8})
SCHEMAS = {
    "errors": ERRORS,
    "intervals": INTERVALS,
    "breakdown": BREAKDOWN,
    "energy_ratio": ENERGY_RATIO,
    "arcs": ARCS,
    "monitoring_scales": MONITORING_SCALES,
    "monitoring_days": MONITORING_DAYS,
    "fits": FITS,
    "splits": SPLIT_SUMMARY,
    "turbines": TURBINES,
}

type Array = np.ndarray


def window_rows(
    prepared: pl.DataFrame, manifest: pl.DataFrame, split: str, days: int | None
) -> pl.DataFrame:
    """the prepared rows at a split's test (and calibration) timestamps, on the first days
    calendar days of them as its fits were scored."""
    stamps = manifest.filter((pl.col("split") == split) & pl.col("role").is_in(TEST_ROLES))
    kept = first_days(stamps["timestamp"].sort(), days)
    return prepared.join(pl.DataFrame({"timestamp": kept}), on="timestamp", how="semi")


def fit_reductions(
    predictions: pl.DataFrame,
    here: pl.DataFrame,
    record: Mapping[str, Any],
    *,
    farm: str,
    model: str,
    split: str,
    seed: int | None,
    regime: str,
    protocol: Protocol,
) -> dict[str, pl.DataFrame]:
    """one fit's errors, intervals, breakdown and energy-ratio sums, or its monitoring inputs.

    record is the fit's .json record; its smearing factors convert the energy-ratio sums.
    """
    if predictions.height != here.height:
        raise ValueError(
            f"{model} on {split} has {predictions.height} predictions for {here.height} rows"
        )
    rows = metrics.scored(predictions, here)
    out: dict[str, pl.DataFrame] = {}
    monitors = (*protocol.monitoring.models, *protocol.matched_monitoring.models)
    if regime in ("threshold", "history") and model in monitors:
        out |= monitored(rows, model, split, regime, protocol)
    if regime == "threshold":
        return out
    statistics = protocol.statistics
    keys = (farm, model, split, seed)
    out["errors"] = metrics.errors(rows, *keys)
    out["intervals"] = metrics.intervals(rows, statistics.coverage_levels, *keys)
    if regime in BREAKDOWN_REGIMES:
        out["breakdown"] = metrics.breakdown(
            rows, statistics.turbulence_edges, protocol.splits.arc_width_deg, *keys
        )
    if regime == "arc_single":
        smearing = pl.DataFrame(record["smearing"], schema=SMEARING)
        out["energy_ratio"] = metrics.energy_ratio_sums(rows, here, smearing, *keys)
    return out


def monitored(
    rows: pl.DataFrame, model: str, split: str, regime: str, protocol: Protocol
) -> dict[str, pl.DataFrame]:
    """calibration scales from a threshold fit, daily residuals from a history's test window;
    the direction lookup abstains where its support is under lookup_min_rows."""
    settings = protocol.monitoring
    horizon = int(split.split("_")[1].removesuffix("d"))
    if horizon < settings.min_history_days:
        return {}
    if model == "direction_lookup":
        rows = rows.filter(pl.col("support") >= settings.lookup_min_rows)
    residuals = monitoring.residuals(rows)
    named = {"model": pl.lit(model), "split": pl.lit(split)}
    if regime == "threshold":
        scales = monitoring.calibration_scales(
            residuals, settings.min_rows_per_day, settings.min_calibration_days
        )
        return {"monitoring_scales": scales.with_columns(**named).select(MONITORING_SCALES.names())}
    days = monitoring.daily_means(residuals)
    return {"monitoring_days": days.with_columns(**named).select(MONITORING_DAYS.names())}


def turbine_order(prepared: pl.DataFrame) -> pl.DataFrame:
    return prepared.select("turbine", "turbine_index").unique().sort("turbine_index")


def positions(turbines: pl.DataFrame, layout: pl.DataFrame) -> tuple[Array, Array]:
    """east and north by turbine_index. the coverage score reads the layout in file order, so
    the layout must list turbines in turbine_index order."""
    place = {name: k for k, name in enumerate(layout["turbine"].to_list())}
    wrong = [name for name, k in turbines.iter_rows() if place.get(name) != k]
    if wrong:
        raise ValueError(f"layout.parquet does not list {wrong} at their turbine_index")
    return layout["x_m"].to_numpy(), layout["y_m"].to_numpy()


def coverage_physics(protocol: Protocol) -> str:
    """the default physics settings the coverage score's single-wake kernel uses."""
    edges = np.asarray(protocol.processing.speed_bin_edges_ms)
    p = protocol.physics
    return json.dumps(
        {
            "kernel_family": p.kernel_family,
            "defaults": p.defaults,
            "centres": ((edges[:-1] + edges[1:]) / 2).tolist(),
            "truncate_sd": p.truncate_sd,
            "power_floor_fraction": p.power_floor_fraction,
        },
        sort_keys=True,
    )


def arc_design(
    train: pl.DataFrame,
    test: pl.DataFrame,
    prepared_dir: Path,
    protocol: Protocol,
    turbine: TurbineConfig,
    east: Array,
    north: Array,
) -> dict[str, float]:
    """one arc split's coverage scores and the covariates of coverage predicts failure, averaged
    over its waked test rows. the permuted score moves the turbines as the permuted-layout
    ablation does, the null of figure 6."""
    from windfold import coverage
    from windfold.models import blocks
    from windfold.models.registry import modified

    layout_path = prepared_dir / "layout.parquet"
    window = (turbine.rotor_diameter_m, protocol.kernel.along_max_d, protocol.kernel.across_max_d)
    rings = coverage.default_wake_rings(
        layout_path, modified(layout_path), turbine.model_dump_json(), coverage_physics(protocol)
    )
    features = [coverage.relational_features(rows, east, north, *window) for rows in (train, test)]
    order = blocks.derangement(len(east), protocol.seeds.study)
    values = {
        "score": coverage.split_scores(train, test, prepared_dir, protocol, turbine),
        "permuted_score": coverage.split_scores(
            train, test, prepared_dir, protocol, turbine, order
        ),
        "wake_strength": coverage.wake_strength(test, east, north, *window, rings),
        "angular_distance": coverage.angular_distance(train, test),
        "dissimilarity": coverage.dissimilarity(
            features[0], features[1], protocol.coverage.dissimilarity_sample, protocol.seeds.study
        ),
    }
    waked = test["waked"].to_numpy()
    design = {name: float(np.mean(value[waked])) for name, value in values.items()}
    return design | {"training_rows": float(train.height)}


def arc_table(
    splits: pl.DataFrame,
    manifest: pl.DataFrame,
    prepared: pl.DataFrame,
    prepared_dir: Path,
    errors: pl.DataFrame,
    protocol: Protocol,
    turbine: TurbineConfig,
) -> pl.DataFrame:
    """the rows of coverage predicts failure: every included single and opposite arc split with
    its coverage scores and covariates and the waked-row mse of the relational and physics
    models."""
    endpoint = protocol.endpoints.coverage_predicts_failure
    waked = errors.filter(pl.col("waked"))
    mse = waked.group_by("model", "split").agg(mse=pl.col("sse").sum() / pl.col("rows").sum())
    lookup = {(m, s): v for m, s, v in mse.iter_rows()}
    chosen = splits.filter(
        pl.col("regime").is_in(DESIGN_REGIMES) & (pl.col("status") == "included")
    )
    layout = pl.read_parquet(prepared_dir / "layout.parquet")
    east, north = positions(turbine_order(prepared), layout)
    records = []
    for split, regime in chosen.select("split", "regime").iter_rows():
        relational = lookup.get((endpoint.relational, split))
        physics = lookup.get((endpoint.physics, split))
        if relational is None or physics is None:
            continue
        roles = manifest.filter(pl.col("split") == split)
        train, test = (
            prepared.join(
                roles.filter(pl.col("role") == role).select("timestamp"),
                on="timestamp",
                how="semi",
            ).sort("timestamp", "turbine_index")
            for role in ("train", "test")
        )
        design = arc_design(train, test, prepared_dir, protocol, turbine, east, north)
        arc = round(int(split.split("_")[1]) / protocol.splits.arc_width_deg)
        records.append(
            {"split": split, "regime": regime, "arc": arc}
            | design
            | {"relational_mse": relational, "physics_mse": physics}
        )
    return pl.DataFrame(records, schema=ARCS) if records else pl.DataFrame(schema=ARCS)


def finished(parts: Mapping[str, Sequence[pl.DataFrame]]) -> dict[str, pl.DataFrame]:
    """every reduction, an empty frame of its schema where nothing contributed."""
    return {
        name: pl.concat(parts[name]) if parts.get(name) else pl.DataFrame(schema=SCHEMAS[name])
        for name in REDUCTIONS
    }
