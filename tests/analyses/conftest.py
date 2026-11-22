"""fixtures of the light analyses: a linear pair of fits on one single arc and a confirmatory
study of penmanshiel without its second freeze record. the tiny study of the heavy analyses is in
tests/conftest.py."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from windfold.evaluation.schemas import ERRORS
from windfold.evaluation.summaries import ACCURACY, ARC_RESULTS
from windfold.schemas import SPLIT_SUMMARY, conform

FARM = "kelmarsh"


@dataclass(frozen=True)
class LinearPair:
    data: Path
    runs: Path


def write_fit(runs: Path, model: str, split: str, frame: pl.DataFrame) -> None:
    target = runs / "development" / "predictions" / FARM / model / f"{split}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)


@pytest.fixture
def linear_pair(tmp_path) -> LinearPair:
    """development targets from -1 to 1 beside wild calibration ones, and one single arc with
    three waked test rows and one unwaked; a second single arc has no waked test rows and no
    fits."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    prepared = data / "prepared" / FARM
    prepared.mkdir(parents=True)
    pl.DataFrame(
        {
            "period": ["development"] * 1001 + ["calibration"] * 2,
            "y": np.r_[np.linspace(-1.0, 1.0, 1001), -50.0, 50.0],
        }
    ).write_parquet(prepared / "rows.parquet")
    manifests = runs / "development" / "manifests"
    manifests.mkdir(parents=True)
    pl.DataFrame(
        {
            "split": ["arc_000_single", "arc_030_single", "random_1"],
            "regime": ["arc_single", "arc_single", "random"],
            "status": ["included"] * 3,
            "test_waked_rows": [3, 0, 3],
        }
    ).write_csv(manifests / f"{FARM}_summary.csv")
    times = [datetime(2019, 5, 1, tzinfo=UTC) + timedelta(minutes=10 * k) for k in range(4)]
    keys = pl.DataFrame(
        {
            "timestamp": times,
            "turbine": ["T0", "T1", "T2", "T3"],
            "y": [0.5, -0.5, 0.2, 0.0],
            "waked": [True, True, True, False],
        }
    )
    for model, mean in {
        "identity_linear": [3.0, -0.4, -7.0, 40.0],
        "kernel_linear": [0.4, -0.6, 0.1, 9.0],
        "no_wake": [0.0, 0.0, 0.0, 9.0],
    }.items():
        write_fit(runs, model, "arc_000_single", keys.with_columns(mean=pl.Series(mean)))
    return LinearPair(data, runs)


@dataclass(frozen=True)
class ConfirmatoryStudy:
    data: Path
    runs: Path
    errors: pl.DataFrame
    arcs: tuple[str, ...]


CONFIRMATORY_ARCS = ("arc_000_single", "arc_030_single", "arc_060_single")
CONFIRMATORY_LEVELS = {
    "no_wake": ((0.110,), 120.0),
    "pywake_inflow": ((0.075,), 98.0),
    "windfold": ((0.070,), 95.0),
    "kernel_linear": ((0.081,), 99.0),
    "identity_gbm_inflow": ((0.078, 0.080, 0.083), 97.0),
}


def confirmatory_errors(seed: int = 6) -> pl.DataFrame:
    """penmanshiel's daily errors on every third day of the confirmation period: waked and unwaked
    rows on the first two arcs, unwaked rows only on the third."""
    rng = np.random.default_rng(seed)
    days = [datetime(2020, 1, 1).date() + timedelta(days=3 * k) for k in range(487)]
    records = []
    for model, (levels, kw) in CONFIRMATORY_LEVELS.items():
        seeds: tuple[int | None, ...] = (None,) if len(levels) == 1 else tuple(range(len(levels)))
        for split, own_seed, day in [
            (s, d, day) for s in CONFIRMATORY_ARCS for d in seeds for day in days
        ]:
            level = levels[0 if own_seed is None else own_seed]
            for waked in (True, False) if split != CONFIRMATORY_ARCS[2] else (False,):
                rows = int(rng.integers(5, 30))
                sse = rows * level * rng.uniform(0.5, 1.5)
                records.append(
                    (
                        model,
                        split,
                        own_seed,
                        day,
                        waked,
                        rows,
                        sse,
                        rows * kw * rng.uniform(0.8, 1.2),
                    )
                )
    columns = {"model": pl.String, "split": pl.String, "seed": pl.Int64, "day": pl.Date}
    columns |= {
        "waked": pl.Boolean,
        "rows": pl.Int64,
        "sse": pl.Float64,
        "abs_error_kw": pl.Float64,
    }
    frame = pl.DataFrame(records, schema=columns, orient="row")
    return frame.with_columns(
        farm=pl.lit("penmanshiel"), speed_bin=pl.lit(0, pl.UInt8), crps=pl.lit(None, pl.Float64)
    ).with_columns(variance=pl.lit(None, pl.Float64))


def job(name: str, family: str, split: str, seconds: float, **fields: object) -> dict[str, object]:
    return {
        "time": fields.pop("time", "2026-11-18T07:00:00+00:00"),
        "job": name,
        "status": fields.pop("status", "done"),
        "family": family,
        "farm": fields.pop("farm", "penmanshiel"),
        "split": split,
        "seconds": seconds,
        "peak_memory_gb": seconds / 100,
        "device": fields.pop("device", "cpu 3"),
    } | fields


@pytest.fixture
def confirmatory_study(tmp_path) -> ConfirmatoryStudy:
    """a confirmatory study of penmanshiel without its second freeze record: daily errors, the
    official accuracy and arcs tables, the split summary, fit records and the runner's job log."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    confirmatory = runs / "confirmatory"
    errors = conform(confirmatory_errors(), ERRORS)
    for folder in ("reductions", "results", "manifests"):
        (confirmatory / folder / "penmanshiel").mkdir(parents=True, exist_ok=True)
    errors.write_parquet(confirmatory / "reductions" / "penmanshiel" / "errors.parquet")
    mse = {
        "windfold_no_local": 0.060,
        "pywake": 0.065,
        "windfold": 0.070,
        "pywake_inflow": 0.075,
        "identity_gbm_inflow": 0.0803,
        "kernel_linear": 0.081,
        "no_wake": 0.110,
    }
    accuracy = [
        (model, "arc_single", None, "waked", metric, value, value * 0.9, value * 1.1)
        for model, level in mse.items()
        for metric, value in (("mse", level), ("mae_kw", level * 1000), ("skill", 1 - level))
    ]
    accuracy.append(("kernel_linear", "time_blocked", None, "waked", "mse", 0.01, 0.009, 0.011))
    pl.DataFrame(accuracy, schema=ACCURACY, orient="row").write_parquet(
        confirmatory / "results" / "penmanshiel" / "accuracy.parquet"
    )
    arcs = [
        (model, split, "arc_single", level + k / 100, level, level + 0.1, 100, 50, 10)
        for model, level in (("windfold", 0.07), ("pywake_inflow", 0.075))
        for k, split in enumerate((*CONFIRMATORY_ARCS, "weighted"))
    ]
    pl.DataFrame(arcs, schema=ARC_RESULTS, orient="row").write_parquet(
        confirmatory / "results" / "penmanshiel" / "arcs.parquet"
    )
    summary_rows = [(s, "arc_single", "included") for s in CONFIRMATORY_ARCS]
    summary_rows += [
        ("arc_090_single", "arc_single", "excluded"),
        ("time_1", "time_blocked", "included"),
    ]
    pl.DataFrame(summary_rows, schema=["split", "regime", "status"], orient="row").with_columns(
        pl.lit(None, pl.String).alias("reason"),
        *[
            pl.lit(1, pl.Int64).alias(c)
            for c in SPLIT_SUMMARY
            if c not in ("split", "regime", "status", "reason")
        ],
    ).select(list(SPLIT_SUMMARY)).write_csv(confirmatory / "manifests" / "penmanshiel_summary.csv")
    records = {"windfold": (None,), "pywake_inflow": (None,), "identity_gbm_inflow": (0, 1, 2)}
    for model, seeds in records.items():
        for k, (split, seed) in enumerate((s, d) for s in CONFIRMATORY_ARCS for d in seeds):
            name = split if seed is None else f"{split}_seed{seed}"
            path = confirmatory / "predictions" / "penmanshiel" / model / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            note = {"fit_seconds": 10.0 + k, "predict_seconds": 1.0 + k, "threads": 1, "facts": {}}
            path.write_text(json.dumps(note))
    jobs = [
        job(
            f"confirmatory/engine/penmanshiel/{s}",
            "engine",
            s,
            100.0,
            time="2026-11-18T01:00:00+00:00",
        )
        for s in CONFIRMATORY_ARCS
    ]
    jobs += [
        job(f"confirmatory/engine/penmanshiel/{s}", "engine", s, 200.0 + k)
        for k, s in enumerate(CONFIRMATORY_ARCS)
    ]
    jobs += [
        job(
            f"confirmatory/identity_gbm_inflow/penmanshiel/{s}_seed{d}",
            "gbm",
            s,
            50.0 + 3 * k + d,
            seed=d,
        )
        for k, s in enumerate(CONFIRMATORY_ARCS)
        for d in range(3)
    ]
    jobs += [
        job(
            "confirmatory/engine/penmanshiel/arc_000_single",
            "engine",
            "arc_000_single",
            900.0,
            status="failed",
            time="2026-11-18T09:00:00+00:00",
        ),
        job(
            "confirmatory/sensitivities/fixed_shrinkage/engine/penmanshiel/arc_000_single",
            "engine",
            "arc_000_single",
            900.0,
        ),
        job(
            "confirmatory/engine/kelmarsh/arc_000_single",
            "engine",
            "arc_000_single",
            900.0,
            farm="kelmarsh",
        ),
        job("confirmatory/engine/penmanshiel/time_1", "engine", "time_1", 900.0),
        job("development/engine/penmanshiel/arc_000_single", "engine", "arc_000_single", 900.0),
        job(
            "confirmatory/wind_frame_graph/penmanshiel/arc_000_single_seed0",
            "graph",
            "arc_000_single",
            80.0,
            device="gpu 0, cpu 4-7",
        ),
    ]
    (runs / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    return ConfirmatoryStudy(data, runs, errors, CONFIRMATORY_ARCS)
