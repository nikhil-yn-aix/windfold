from datetime import UTC, datetime, timedelta

import polars as pl

from windfold.analyses.development import prediction_file
from windfold.analyses.sensor_count import sensor_count

SPLIT = "arc_000_single"
PARTS = [
    ("1", 1, 200, 2.0, 0),
    ("2", 2, 199, 2.0, 1),
    ("3-4", 3, 100, 3.0, 2),
    ("3-4", 4, 100, 3.0, 10),
    ("5+", 5, 100, 1.0, 6),
    ("5+", 9, 100, 1.0, 8),
]


def write_study(data, runs) -> None:
    """parts of (stratum, unwaked turbines, waked test rows, diurnal prediction, day): each
    waked single-arc test row has its own development timestamp beside that many unwaked
    turbines. y is zero, windfold predicts one and no_wake four, so every replicate of a stratum
    gives 100 (level ** 2 - 1) / 16 points."""
    start = datetime(2019, 3, 1, tzinfo=UTC)
    rows, keys = [], []
    for _, unwaked, count, level, day in PARTS:
        for t in range(count):
            stamp = start + timedelta(days=day, minutes=5 * t)
            rows += [(stamp, f"T{i}", i == 0) for i in range(unwaked + 1)]
            keys.append((stamp, level))
    prepared = data / "prepared" / "kelmarsh"
    prepared.mkdir(parents=True)
    pl.DataFrame(rows, schema=["timestamp", "turbine", "waked"], orient="row").with_columns(
        period=pl.lit("development")
    ).write_parquet(prepared / "rows.parquet")
    tests = pl.DataFrame(keys, schema=["timestamp", "level"], orient="row").with_columns(
        turbine=pl.lit("T0"), y=pl.lit(0.0), waked=pl.lit(True)
    )
    for model, mean in (("windfold", 1.0), ("windfold_diurnal_inflow", None), ("no_wake", 4.0)):
        path = prediction_file(runs, "kelmarsh", model, SPLIT)
        path.parent.mkdir(parents=True)
        predicted = pl.col("level") if mean is None else pl.lit(mean)
        tests.with_columns(mean=predicted).drop("level").write_parquet(path)
    folder = runs / "development" / "manifests"
    folder.mkdir(parents=True)
    pl.DataFrame(
        {"split": [SPLIT], "regime": ["arc_single"], "status": ["included"], "test_waked_rows": [1]}
    ).write_csv(folder / "kelmarsh_summary.csv")


def test_strata_follow_the_unwaked_sensor_count_and_need_two_hundred_rows(tmp_path):
    data, runs = tmp_path / "data", tmp_path / "runs"
    write_study(data, runs)

    def stratum(blocks: int, level: float) -> dict:
        points = 100 * (level**2 - 1) / 16
        interval = {"estimate": points, "low": points, "high": points}
        return {"rows": 200, "blocks": blocks, "measured_minus_diurnal": interval}

    assert sensor_count(data, runs, ("kelmarsh",)) == {
        "kelmarsh": {"1": stratum(1, 2.0), "3-4": stratum(2, 3.0), "5+": stratum(2, 1.0)}
    }
