import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from windfold.config import for_farm
from windfold.evaluation import paths
from windfold.evaluation.results import DECISIONS
from windfold.evaluation.schemas import HYPOTHESES
from windfold.locations import manifest_file
from windfold.schemas import MANIFEST, SPLIT_SUMMARY
from windfold.tasks.evaluation import farm_results, frozen_bytes, reduce_farm, study_decision

S, F, N = "success", "failure", "not estimable"


def test_confirmatory_files_are_read_only_as_the_second_record_hashed_them(tmp_path):
    table = tmp_path / "confirmatory" / "fit.parquet"
    table.parent.mkdir()
    table.write_bytes(b"predictions")
    name = "confirmatory/fit.parquet"
    hashes = {name: hashlib.sha256(b"predictions").hexdigest()}
    assert frozen_bytes(tmp_path, name, hashes) == b"predictions"
    assert frozen_bytes(tmp_path, name, None) == b"predictions"
    with pytest.raises(ValueError, match="not in the second freeze record"):
        frozen_bytes(tmp_path, name, {})
    table.write_bytes(b"predictions, changed")
    with pytest.raises(ValueError, match="the second freeze record has"):
        frozen_bytes(tmp_path, name, hashes)


GATHERED = paths.DECISION[2:]


def write_results(runs: Path, study: str, farm: str, outcomes: dict[str, tuple[str, int]]) -> None:
    """a farm's result tables: its outcome of each hypothesis given, repeated on as many
    component rows as given, and one row in every other table the decision gathers."""
    folder = runs / study / "results" / farm
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        {"farm": farm, "hypothesis": name, "component": f"part_{k}", "outcome": outcome}
        for name, (outcome, components) in outcomes.items()
        for k in range(components)
    ]
    pl.DataFrame(rows, schema=HYPOTHESES).write_parquet(folder / "hypotheses.parquet")
    for name in GATHERED:
        pl.DataFrame({"model": [farm], "estimate": [1.0]}).write_parquet(folder / f"{name}.parquet")


def decided(out: Path, study: str) -> pl.DataFrame:
    table = pl.read_parquet(out / study / "results" / "decision.parquet")
    assert_frame_equal(
        pl.read_csv(out / study / "results" / "decision.csv", schema=table.schema), table
    )
    return table


def test_a_study_decision_counts_each_farm_once_and_a_development_study_has_no_verdict(
    tmp_path: Path,
):
    runs, out = tmp_path / "runs", tmp_path / "out"
    farms = {
        "smarteole": {"validation_misleads": (S, 1), "wind_frame_transfers": (F, 1)},
        "kelmarsh": {
            "validation_misleads": (S, 2),
            "wind_frame_transfers": (S, 3),
            "coverage_predicts_failure": (S, 2),
            "early_history_advantage": (N, 1),
        },
        "hill_of_towie": {"validation_misleads": (F, 1), "wind_frame_transfers": (F, 1)},
        "penmanshiel": {
            "validation_misleads": (S, 1),
            "wind_frame_transfers": (F, 2),
            "coverage_predicts_failure": (S, 1),
        },
    }
    for farm, outcomes in farms.items():
        write_results(runs, "development", farm, outcomes)
    study_decision("development", None, list(farms), "", str(runs), None, out)
    development = [
        ("validation_misleads", 4, 3, S),
        ("wind_frame_transfers", 4, 1, F),
        ("coverage_predicts_failure", 2, 2, N),
        ("early_history_advantage", 0, 0, N),
    ]
    expected = pl.DataFrame(
        [(*row, None, None, None, None) for row in development], schema=DECISIONS, orient="row"
    )
    assert_frame_equal(decided(out, "development"), expected)
    folder = out / "development" / "results"
    gathered = pl.read_csv(folder / "hypotheses.csv")
    assert gathered.height == 16 and set(gathered["farm"]) == set(farms)
    assert pl.read_parquet(folder / "hypotheses.parquet").height == 16
    for name in GATHERED:
        assert sorted(pl.read_parquet(folder / f"{name}.parquet")["farm"]) == sorted(farms)

    confirmation = {
        "validation_misleads": (S, 2),
        "wind_frame_transfers": (S, 1),
        "coverage_predicts_failure": (F, 1),
    }
    write_results(runs, "confirmatory", "penmanshiel", confirmation)
    study_decision("confirmatory", None, ["penmanshiel"], "", str(runs), None, out)
    confirmatory = [
        (1, 1, S, "supported"),
        (1, 1, S, "not supported"),
        (1, 0, F, "not supported"),
        (0, 0, N, "not estimable"),
    ]
    expected = pl.DataFrame(
        [(*d, *c) for d, c in zip(development, confirmatory, strict=True)],
        schema=DECISIONS,
        orient="row",
    )
    assert_frame_equal(decided(out, "confirmatory"), expected)
    (runs / "development" / "results" / "smarteole" / "hypotheses.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="smarteole"):
        study_decision("confirmatory", None, ["penmanshiel"], "", str(runs), None, out)


FARM = "kelmarsh"
STAMPS = [datetime(2018, 2, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(96)]
SPLITS = {"random_1": ("random", 0), "arc_000_single": ("arc_single", 1)}


def write_prepared(folder: Path) -> None:
    """two turbines with y of plus and minus 0.2, hourly on four february days of 2018, the
    second one waked."""
    folder.mkdir(parents=True)
    count = 2 * len(STAMPS)
    pl.DataFrame(
        {
            "timestamp": [s for s in STAMPS for _ in range(2)],
            "turbine": ["T0", "T1"] * len(STAMPS),
            "turbine_index": pl.Series([0, 1] * len(STAMPS), dtype=pl.UInt8),
            "power_kw": [1000.0, 670.0] * len(STAMPS),
            "y": [0.2, -0.2] * len(STAMPS),
            "waked": [False, True] * len(STAMPS),
            "valid_turbines": pl.Series([2] * count, dtype=pl.UInt8),
            "speed_bin": pl.Series([1] * count, dtype=pl.UInt8),
            "direction_bucket": pl.Series([54] * count, dtype=pl.UInt8),
            "farm_direction": [270.0] * count,
            "turbulence_intensity": [0.08] * count,
            "steady": [True] * count,
        }
    ).write_parquet(folder / "rows.parquet")
    pl.DataFrame({"turbine": ["T0", "T1"], "x_m": [0.0, 500.0], "y_m": [0.0, 0.0]}).write_parquet(
        folder / "layout.parquet"
    )


def hand_made_fits(
    root: Path,
    study: str,
    errors: dict[str, float],
    splits: dict[str, tuple[str, int]] = SPLITS,
    sensitivity: str | None = None,
) -> list[list]:
    """a study's manifest and split summary of the farm, each split testing on two of the
    days, and fits whose predictions miss y by the model's error on every row; returns them."""
    protocol, _, _ = for_farm(FARM, sensitivity, None)
    folder = root / "data" / "prepared" / FARM
    if not folder.exists():
        write_prepared(folder)
    prepared = pl.read_parquet(folder / "rows.parquet")
    runs = root / "runs"
    manifest = pl.DataFrame(
        [
            (split, regime, stamp, "test" if (stamp.day - 1) // 2 == half else "train")
            for split, (regime, half) in splits.items()
            for stamp in STAMPS
        ],
        schema=MANIFEST,
        orient="row",
    )
    path = manifest_file(study, FARM, sensitivity, protocol)
    (runs / path).parent.mkdir(parents=True)
    manifest.write_parquet(runs / path)
    pl.DataFrame(
        {"split": list(splits), "regime": [regime for regime, _ in splits.values()]}
    ).with_columns(status=pl.lit("included")).select(
        pl.col(name) if name in ("split", "regime", "status") else pl.lit(None).alias(name)
        for name in SPLIT_SUMMARY
    ).write_csv(runs / paths.summary_file(path))
    smearing = [{"speed_bin": 1, "rows": 10, "factor": 1.0, "pooled": False}]
    fits = []
    for split in splits:
        tested = manifest.filter(pl.col("split") == split, pl.col("role") == "test")
        rows = prepared.join(tested, on="timestamp", how="semi")
        for model, error in errors.items():
            table, note = paths.fit_inputs(study, FARM, sensitivity, model, split, None, protocol)
            (runs / table).parent.mkdir(parents=True, exist_ok=True)
            rows.select(
                "timestamp",
                "turbine",
                "y",
                mean=pl.col("y") - error,
                variance=pl.lit(None, pl.Float64),
                support=pl.lit(None, pl.Int64),
                waked="waked",
            ).write_parquet(runs / table)
            (runs / note).write_text(json.dumps({"smearing": smearing}))
            fits.append([model, split, None])
    return fits


def waked_accuracy(runs: Path, study: str, sensitivity: str | None) -> dict[tuple, float]:
    path = runs / paths.result_outputs(study, FARM, sensitivity)[paths.RESULTS.index("accuracy")]
    table = pl.read_parquet(path).filter(
        pl.col("rows") == "waked", pl.col("metric").is_in(["mse", "skill"])
    )
    return {
        (m, r, k): v
        for m, r, k, v in table.select("model", "regime", "metric", "estimate").iter_rows()
    }


def test_a_farms_fits_reduce_into_the_result_tables_the_planner_names(tmp_path: Path):
    fits = hand_made_fits(tmp_path, "smoke", {"no_wake": 0.2, "windfold": 0.1})
    runs = tmp_path / "runs"
    folders = (str(tmp_path / "data"), str(runs), None, runs)
    lost = [["pywake", "random_1", None]]
    with pytest.raises(FileNotFoundError, match="pywake"):
        reduce_farm("smoke", FARM, None, [*fits, *lost], [], *folders)
    assert not (runs / "smoke" / "reductions").exists()
    reduce_farm("smoke", FARM, None, fits, lost, *folders)
    status = pl.read_parquet(runs / paths.reduction_file("smoke", FARM, None, "fits"))
    assert status.filter(pl.col("status") == "failed").select("model", "split").rows() == [
        ("pywake", "random_1")
    ]
    assert status.height == 5
    farm_results("smoke", FARM, None, [], *folders)
    for name, path in zip(paths.RESULTS, paths.result_outputs("smoke", FARM, None), strict=True):
        assert path.endswith(f"/{FARM}/{name}.parquet") and (runs / path).is_file()
    hypotheses = pl.read_parquet(runs / paths.result_outputs("smoke", FARM, None)[0])
    assert hypotheses.schema == HYPOTHESES and set(hypotheses["farm"]) == {FARM}
    assert waked_accuracy(runs, "smoke", None) == pytest.approx(
        {
            (model, regime, metric): value
            for regime in ("random", "arc_single")
            for model, mse in (("no_wake", 0.04), ("windfold", 0.01))
            for metric, value in (("mse", mse), ("skill", 1 - mse / 0.04))
        }
    )


def test_a_split_sensitivity_takes_the_regimes_it_does_not_refit_from_the_main_study(
    tmp_path: Path,
):
    runs = tmp_path / "runs"
    folders = (str(tmp_path / "data"), str(runs), None, runs)
    main = hand_made_fits(tmp_path, "smoke", {"no_wake": 0.2, "windfold": 0.1})
    reduce_farm("smoke", FARM, None, main, [], *folders)
    arc = {"arc_000_single": SPLITS["arc_000_single"]}
    errors = {"no_wake": 0.2, "windfold": 0.05}
    own = hand_made_fits(tmp_path, "smoke", errors, arc, "arc_buffer_0deg")
    reduce_farm("smoke", FARM, "arc_buffer_0deg", own, [], *folders)
    farm_results("smoke", FARM, "arc_buffer_0deg", [], *folders)
    found = waked_accuracy(runs, "smoke", "arc_buffer_0deg")
    assert found[("windfold", "random", "mse")] == pytest.approx(0.01)
    assert found[("windfold", "arc_single", "mse")] == pytest.approx(0.0025)


def test_confirmatory_fits_are_reduced_only_as_the_second_record_hashed_them(tmp_path: Path):
    fits = hand_made_fits(tmp_path, "confirmatory", {"no_wake": 0.2, "windfold": 0.1})
    runs = tmp_path / "runs"
    arguments = ("confirmatory", FARM, None, fits, [], str(tmp_path / "data"), str(runs), None)
    with pytest.raises(FileNotFoundError, match="second freeze record"):
        reduce_farm(*arguments, tmp_path / "out")
    hashes = {
        str(path.relative_to(runs)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (runs / "confirmatory" / "predictions").rglob("*")
        if path.is_file()
    }
    (runs / "confirmatory" / "freeze_second.json").write_text(json.dumps({"predictions": hashes}))
    reduce_farm(*arguments, tmp_path / "out")
    protocol, _, _ = for_farm(FARM, None, None)
    for written in paths.fit_inputs(
        "confirmatory", FARM, None, "windfold", "random_1", None, protocol
    ):
        original = (runs / written).read_bytes()
        (runs / written).write_bytes(original + b" ")
        with pytest.raises(ValueError, match="the second freeze record has"):
            reduce_farm(*arguments, tmp_path / "out")
        (runs / written).write_bytes(original)
