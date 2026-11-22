from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from windfold.analyses.interval_score import interval_score
from windfold.config import for_farm, load_study
from windfold.evaluation import bootstrap, metrics, paths, summaries
from windfold.evaluation.results import KEYS, QUANTITIES, bootstrap_windows, study_windows
from windfold.locations import prediction_files

FARM = "smarteole"
SPLITS = {"arc_000_single": "arc_single", "time_1": "time_blocked"}
TURBINES = ("T0", "T1", "T2")
DAYS = [datetime(2020, 2, 18, tzinfo=UTC) + timedelta(days=7 * k) for k in range(14)]
HOURS = {"arc_000_single": 0, "time_1": 12}
VARIANCE = {"windfold": 0.04, "pywake_inflow": 0.01, "identity_gbm": None}


def error(model: str, split: str, day: int, turbine: str) -> float:
    """y minus the mean: T2 is unwaked and missed by 5; on the arc windfold misses the waked rows
    by 0.5 every other week, pywake_inflow misses them by 0.1 throughout."""
    if turbine == "T2":
        return 5.0
    if model == "pywake_inflow":
        return 0.1
    if model == "identity_gbm":
        return 0.2
    return 0.5 if split == "arc_000_single" and day % 2 else 0.0


def write_study(root: Path) -> tuple[Path, Path]:
    """prepared rows of three valid turbines (scale 1.5), the manifest, the predictions of two
    gaussian models and a seeded one without variance and from them the reductions and official
    calibration and accuracy tables, as the evaluation computes them."""
    data, runs = root / "data", root / "runs"
    stamps = {s: [d + timedelta(hours=h) for d in DAYS] for s, h in HOURS.items()}
    every = [t for s in SPLITS for t in stamps[s]]
    prepared = pl.DataFrame(
        {
            "timestamp": [t for t in every for _ in TURBINES],
            "turbine": list(TURBINES) * len(every),
            "power_kw": 1000.0,
            "valid_turbines": 3,
            "speed_bin": 5,
            "direction_bucket": 0,
            "farm_direction": 10.0,
            "turbulence_intensity": 0.1,
            "steady": True,
        },
        schema_overrides={"valid_turbines": pl.UInt8, "speed_bin": pl.UInt8},
    ).with_columns(pl.col("direction_bucket").cast(pl.UInt8))
    (data / "prepared" / FARM).mkdir(parents=True)
    prepared.write_parquet(data / "prepared" / FARM / "rows.parquet")
    manifest = pl.DataFrame(
        [(s, "test", t) for s in SPLITS for t in stamps[s]],
        schema={"split": pl.String, "role": pl.String, "timestamp": pl.Datetime("us", "UTC")},
        orient="row",
    )
    (runs / "development" / "manifests").mkdir(parents=True)
    manifest.write_parquet(runs / "development" / "manifests" / f"{FARM}.parquet")
    protocol, farm_config, _ = for_farm(FARM, None)
    levels = protocol.statistics.coverage_levels
    errors, intervals, fits = [], [], []
    for model, variance in VARIANCE.items():
        seed = 0 if variance is None else None
        for split, regime in SPLITS.items():
            records = [
                (t, turbine, 0.1, 0.1 - error(model, split, k, turbine), variance, None)
                for k, t in enumerate(stamps[split])
                for turbine in TURBINES
            ]
            predictions = pl.DataFrame(
                records,
                schema=["timestamp", "turbine", "y", "mean", "variance", "support"],
                schema_overrides={"variance": pl.Float64, "support": pl.Int64},
                orient="row",
            ).with_columns(waked=pl.col("turbine") != "T2")
            path = runs / prediction_files("development", None, FARM, model, split, seed)[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            predictions.write_parquet(path)
            here = prepared.filter(pl.col("timestamp").is_in(stamps[split]))
            scored = metrics.scored(predictions, here)
            errors.append(metrics.errors(scored, FARM, model, split, seed))
            intervals.append(metrics.intervals(scored, levels, FARM, model, split, seed))
            fits.append((model, split, seed, regime, "done"))
    reduced = {
        "errors": pl.concat(errors),
        "intervals": pl.concat(intervals),
        "fits": pl.DataFrame(
            fits, schema=["model", "split", "seed", "regime", "status"], orient="row"
        ),
    }
    for name, frame in reduced.items():
        path = runs / paths.reduction_file("development", FARM, None, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    statistics = protocol.statistics
    windows = bootstrap_windows(
        study_windows(farm_config, load_study("development").period),
        statistics.block_days,
        statistics.replicates,
        len(protocol.seeds.model),
        statistics.interval,
        statistics.min_finite,
        bootstrap.farm_seed(protocol.seeds.study, FARM),
    )
    draw, (origin, _) = windows.draws["period"], windows.origins["period"]

    def totals(frame, extra, amounts):
        blocks = windows.blocks["period"]
        return bootstrap.block_totals(
            frame, [*KEYS, *extra], amounts, origin, statistics.block_days, blocks
        )

    gaussian = ["pywake_inflow", "windfold"]
    groups = {regime: [split] for split, regime in SPLITS.items()}
    tables = {
        "calibration": summaries.interval_calibration(
            totals(reduced["intervals"], ["level"], ["rows", "covered", "width"]),
            draw,
            groups,
            gaussian,
            levels,
        ),
        "accuracy": summaries.accuracy(
            totals(reduced["errors"], [], QUANTITIES),
            draw,
            {(regime, None): splits for regime, splits in groups.items()},
            [*gaussian, "identity_gbm"],
            "no_wake",
        ),
    }
    folder = runs / paths.results_folder("development", None) / FARM
    folder.mkdir(parents=True)
    for name, table in tables.items():
        table.write_parquet(folder / f"{name}.parquet")
    return data, runs


def test_interval_scores_and_coverage_on_waked_rows_by_hand(tmp_path):
    data, runs = write_study(tmp_path)
    found = interval_score(data, runs, "development", (FARM,))[FARM]
    assert list(found["models"]) == ["pywake_inflow", "windfold"]
    assert found["models"]["windfold"]["seeds"] == [None]
    assert found["waked_rows"] == {"arc_single": 28, "time_blocked": 28}
    assert found["official_values_reproduced"] == 2 * 2 * 5
    # z = 1.64485362695 at 90%: windfold's interval is 0.65794145078 wide and misses half its waked
    # arc rows by 0.5 - 0.32897072539, so they score 0.65794145078 + 20 * 0.17102927461
    arc = found["models"]["windfold"]["arc_single"]["0.9"]
    assert arc["coverage"]["estimate"] == 0.5
    assert arc["interval_score"]["estimate"] == pytest.approx(2.3682341969, rel=1e-9)
    assert arc["interval_score_contrast"]["estimate"] == pytest.approx(1.5 * 2.3682341969)
    time = found["models"]["windfold"]["time_blocked"]["0.9"]
    assert (time["coverage"]["estimate"], time["interval_score"]["estimate"]) == pytest.approx(
        (1.0, 0.65794145078)
    )
    wide = found["models"]["windfold"]["arc_single"]["0.99"]
    assert (wide["coverage"]["estimate"], wide["interval_score"]["estimate"]) == pytest.approx(
        (1.0, 1.03033172142)
    )
    physics = found["models"]["pywake_inflow"]["arc_single"]["0.9"]["interval_score"]["estimate"]
    assert physics == pytest.approx(0.32897072539)
    against = found["windfold_minus"]["pywake_inflow"]["arc_single"]
    assert against["0.9"]["coverage"]["estimate"] == pytest.approx(-0.5)
    assert against["0.9"]["interval_score"]["estimate"] == pytest.approx(2.3682341969 - physics)
    score = against["0.9"]["interval_score"]
    assert score["low"] < score["estimate"] < score["high"]
    assert list(found["windfold_minus"]) == ["pywake_inflow"]


@pytest.mark.parametrize(
    ("name", "metric", "change"),
    [("calibration", "coverage", 1e-12), ("accuracy", "crps", 1e-6)],
)
def test_coverage_and_crps_must_reproduce_the_official_tables(tmp_path, name, metric, change):
    data, runs = write_study(tmp_path)
    path = runs / paths.results_folder("development", None) / FARM / f"{name}.parquet"
    table = pl.read_parquet(path)
    moved = (pl.col("model") == "windfold") & (pl.col("metric") == metric)
    table.with_columns(
        estimate=pl.when(moved).then(pl.col("estimate") + change).otherwise(pl.col("estimate"))
    ).write_parquet(path)
    with pytest.raises(ValueError, match="the official table has"):
        interval_score(data, runs, "development", (FARM,))
