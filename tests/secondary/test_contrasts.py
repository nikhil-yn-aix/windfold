import shutil
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from windfold.config import for_farm, load_protocol
from windfold.evaluation import bootstrap, metrics, paths
from windfold.evaluation.reductions import SCHEMAS
from windfold.evaluation.results import KEYS
from windfold.evaluation.summaries import Draw
from windfold.locations import manifest_file
from windfold.schemas import MANIFEST, PREDICTIONS, SPLIT_SUMMARY
from windfold.secondary.contrasts import contrast_table, contrasts, output_folder, prediction_names

ARC = "arc_000_single"
ORIGIN = date(2020, 1, 1)


def totals(only_seed: int | None = None) -> bootstrap.Totals:
    """two 7-day blocks of waked single-arc rows, ten per model, seed and block."""
    sse = {
        ("hybrid", 0): (1.0, 2.0),
        ("hybrid", 1): (2.0, 2.0),
        ("other", 0): (3.0, 2.0),
        ("other", 1): (5.0, 6.0),
        ("no_wake", None): (10.0, 4.0),
    }
    rows = [
        (model, ARC, seed, True, ORIGIN if block == 0 else date(2020, 1, 8), 10, value)
        for (model, seed), values in sse.items()
        for block, value in enumerate(values)
        if only_seed is None or seed in (None, only_seed)
    ]
    frame = pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "split": pl.String,
            "seed": pl.Int64,
            "waked": pl.Boolean,
            "day": pl.Date,
            "rows": pl.Int64,
            "sse": pl.Float64,
        },
        orient="row",
    )
    return bootstrap.block_totals(frame, KEYS, ["sse", "rows"], ORIGIN, 7, 2)


DRAW = Draw(
    counts=np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
    seeds=np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
    level=0.95,
    min_finite=0.95,
)


def test_skill_points_convert_each_replicate_over_no_wake_on_the_same_rows():
    statistics = load_protocol().statistics
    row = contrast_table(totals(), DRAW, "hybrid", ["other"], [ARC], statistics).row(0, named=True)
    assert row["component"] == "other_minus_hybrid"
    assert row["estimate"] == pytest.approx(0.4 - 0.175)
    assert row["no_wake_mse"] == pytest.approx(0.7)
    assert row["skill"] == pytest.approx(100 * 0.225 / 0.7)
    replicates = np.array([0.2 / 1.0, 0.4 / 0.4, 0.225 / 0.7, 0.25 / 1.0]) * 100
    low, high = np.quantile(replicates, [0.025, 0.975])
    assert (row["skill_lower"], row["skill_upper"]) == pytest.approx((low, high))
    assert row["skill_lower"] != pytest.approx(100 * row["lower"] / row["no_wake_mse"])


def test_a_single_seed_is_scored_without_redrawing_seeds():
    statistics = load_protocol().statistics
    seeded = contrast_table(totals(0), DRAW, "hybrid", ["other"], [ARC], statistics)
    assert seeded["status"][0] == "not estimable"
    alone = contrast_table(totals(0), DRAW, "hybrid", ["other"], [ARC], statistics, redraw=False)
    row = alone.row(0, named=True)
    assert row["estimate"] == pytest.approx(0.25 - 0.15)
    low, high = np.quantile([0.2, 0.0, 0.1, 0.2], [0.025, 0.975])
    assert (row["lower"], row["upper"]) == pytest.approx((low, high))


FARM = "kelmarsh"
ARCS = {"arc_000_single": 1, "arc_030_single": 8}
RUNNER_ERRORS = {"no_wake": 0.5, "windfold": 0.05, "pywake_inflow": 0.4, "pywake": 0.45}
SEED_ERRORS = (0.1, 0.2, 0.3)


def two_weeks(root: Path) -> tuple[Path, Path, Path]:
    """two development weeks of kelmarsh and two single arcs, one tested on the first two days
    of each week; one waked and one unwaked turbine with y of minus and plus 0.2. each runner
    model misses y by its error on every row, each hybrid seed by its own on waked rows and by
    five on unwaked ones. a third single arc is excluded and has no fits."""
    configs = shutil.copytree(Path(str(resources.files("windfold") / "configs")), root / "configs")
    farm = configs / "farms" / f"{FARM}.toml"
    window = "development = { start = 2018-01-01T00:00:00Z, end = 2025-01-01T00:00:00Z }"
    text = farm.read_text()
    assert window in text
    farm.write_text(
        text.replace(
            window, window.replace("2018-01-01", "2018-02-01").replace("2025-01-01", "2018-02-15")
        )
    )
    protocol, _, _ = for_farm(FARM, None, configs)
    data, runs = root / "data", root / "runs"
    stamps = {
        split: [datetime(2018, 2, day, tzinfo=UTC) + timedelta(hours=h) for h in range(48)]
        for split, day in ARCS.items()
    }
    prepared = pl.DataFrame(
        {
            "timestamp": [s for split in ARCS for s in stamps[split] for _ in range(2)],
            "turbine": ["T0", "T1"] * 96,
            "power_kw": [1000.0, 670.0] * 96,
            "y": [0.2, -0.2] * 96,
            "waked": [False, True] * 96,
            "valid_turbines": pl.Series([2] * 192, dtype=pl.UInt8),
            "speed_bin": pl.Series([1] * 192, dtype=pl.UInt8),
            "direction_bucket": pl.Series([54] * 192, dtype=pl.UInt8),
            "farm_direction": [270.0] * 192,
            "turbulence_intensity": [0.08] * 192,
            "steady": [True] * 192,
        }
    )
    (data / "prepared" / FARM).mkdir(parents=True)
    prepared.write_parquet(data / "prepared" / FARM / "rows.parquet")
    manifest = pl.DataFrame(
        [
            (split, "arc_single", stamp, "test" if stamp in stamps[split] else "train")
            for split in ARCS
            for stamp in prepared["timestamp"].unique().sort()
        ],
        schema=MANIFEST,
        orient="row",
    )
    path = runs / manifest_file("development", FARM, None, protocol)
    path.parent.mkdir(parents=True)
    manifest.write_parquet(path)
    names = [*ARCS, "arc_060_single"]
    pl.DataFrame(
        {"split": names, "regime": "arc_single", "status": ["included"] * 2 + ["excluded"]}
    ).select(
        pl.col(name) if name in ("split", "regime", "status") else pl.lit(None).alias(name)
        for name in SPLIT_SUMMARY
    ).write_csv(runs / paths.summary_file(manifest_file("development", FARM, None, protocol)))
    errors = []
    for split in ARCS:
        rows = prepared.filter(pl.col("timestamp").is_in(stamps[split]))
        for model, error in RUNNER_ERRORS.items():
            predicted = rows.select(
                "timestamp",
                "turbine",
                "y",
                "waked",
                mean=pl.col("y") - error,
                variance=pl.lit(None, pl.Float64),
                support=pl.lit(None, pl.Int64),
            )
            scored = metrics.scored(predicted.select(PREDICTIONS.names()), rows)
            errors.append(metrics.errors(scored, FARM, model, split, None))
        for seed, error in enumerate(SEED_ERRORS):
            table, _ = prediction_names(FARM, split, seed)
            target = output_folder(runs, "development", "hybrid_gbm") / table
            target.parent.mkdir(parents=True, exist_ok=True)
            miss = pl.when(pl.col("waked")).then(error).otherwise(5.0)
            rows.select(
                "timestamp",
                "turbine",
                "y",
                "waked",
                mean=pl.col("y") - miss,
                variance=pl.lit(None, pl.Float64),
                support=pl.lit(None, pl.Int64),
            ).select(PREDICTIONS.names()).write_parquet(target)
    reduced = runs / paths.reductions_folder("development", FARM, None)
    reduced.mkdir(parents=True)
    for name in paths.REDUCTIONS:
        frame = pl.concat(errors) if name == "errors" else pl.DataFrame(schema=SCHEMAS[name])
        frame.write_parquet(reduced / f"{name}.parquet")
    return data, runs, configs


def test_secondary_contrasts_gap_the_seed_mean_on_waked_arc_rows_and_holm_only_the_tested(
    tmp_path,
):
    data, runs, configs = two_weeks(tmp_path)
    table = contrasts(
        data,
        runs,
        "development",
        FARM,
        "hybrid_gbm",
        [0, 1, 2],
        ["pywake_inflow", "windfold"],
        ["pywake"],
        str(configs),
    )
    by = {row["component"]: row for row in table.iter_rows(named=True)}
    own = np.mean(np.square(SEED_ERRORS))
    replicates = load_protocol(configs).statistics.replicates
    expected = {
        "pywake_inflow_minus_hybrid_gbm": (0.4**2 - own, 2 / (replicates + 1), True),
        "windfold_minus_hybrid_gbm": (0.05**2 - own, 1.0, False),
        "pywake_minus_hybrid_gbm": (0.45**2 - own, None, None),
    }
    assert set(by) == set(expected)
    for component, (gap, holm_p, confirmed) in expected.items():
        row = by[component]
        assert row["estimate"] == pytest.approx(gap)
        assert row["skill"] == pytest.approx(100 * gap / 0.25)
        assert row["holm_p"] == (None if holm_p is None else pytest.approx(holm_p))
        assert row["confirmed"] is confirmed
    assert by["pywake_inflow_minus_hybrid_gbm"]["p"] == pytest.approx(1 / (replicates + 1))
    assert by["pywake_minus_hybrid_gbm"]["p"] == pytest.approx(1 / (replicates + 1))
