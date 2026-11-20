from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.analyses.comparison_ranks import comparison_ranks
from windfold.evaluation import paths
from windfold.evaluation.summaries import RANK_AGREEMENT

FARM = "smarteole"
SPLITS = {"random_1": "random", "time_1": "time_blocked", "arc_000_single": "arc_single"}
TIME = {
    "kernel_linear": 0.05,
    "no_wake": 0.10,
    "pywake": 0.08,
    "windfold": 0.09,
    "windfold_a": 0.01,
    "windfold_b": 0.02,
}
ARC = {
    "kernel_linear": 0.09,
    "no_wake": 0.10,
    "pywake": 0.07,
    "windfold": 0.065,
    "windfold_a": 1.0,
    "windfold_b": 1.1,
}
# by hand, time-blocked against single arcs: kernel_linear, no_wake and pywake order 2 of their 3
# pairs alike (2 - 1 of 3); windfold adds 1 pair ordered alike and 2 reversed (3 - 3 of 6); the
# variants add 1 alike and 8 reversed (4 - 11 of 15).
EXPECTED = {"compared": 0.0, "comparison": 1 / 3, "all": -7 / 15}


def write_study(runs, official_shift: float = 0.0) -> None:
    """errors and fits reductions with ten waked rows per model, split and 7-day block of the
    farm's development window at the model's mse for the regime (random as time-blocked), and the
    study's rank agreement table of all models, its time-blocked tau moved by official_shift."""
    days = [date(2020, 2, 17) + timedelta(days=7 * k) for k in range(14)]
    errors = pl.DataFrame(
        [
            (FARM, model, split, None, day, True, 10, 10 * mse[model])
            for split, regime in SPLITS.items()
            for mse in [ARC if regime == "arc_single" else TIME]
            for model in TIME
            for day in days
        ],
        schema={
            "farm": pl.String,
            "model": pl.String,
            "split": pl.String,
            "seed": pl.Int64,
            "day": pl.Date,
            "waked": pl.Boolean,
            "rows": pl.Int64,
            "sse": pl.Float64,
        },
        orient="row",
    )
    fits = pl.DataFrame(
        [(m, s, None, r, "done") for s, r in SPLITS.items() for m in TIME],
        schema={
            "model": pl.String,
            "split": pl.String,
            "seed": pl.Int64,
            "regime": pl.String,
            "status": pl.String,
        },
        orient="row",
    )
    for name, frame in (("errors", errors), ("fits", fits)):
        path = runs / paths.reduction_file("development", FARM, None, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    tau = EXPECTED["all"] + official_shift
    rows = [
        ("random", "time_blocked", 6, 1.0, 1.0, 1.0),
        ("random", "arc_single", 6, EXPECTED["all"], EXPECTED["all"], EXPECTED["all"]),
        ("time_blocked", "arc_single", 6, tau, tau, tau),
    ]
    table = pl.DataFrame(rows, schema=RANK_AGREEMENT, orient="row")
    results = runs / paths.results_folder("development", None)
    results.mkdir(parents=True)
    table.with_columns(farm=pl.lit(FARM)).write_parquet(results / "rank_agreement.parquet")


def kendall(x: list[float], y: list[float]) -> float:
    """kendall's tau for rankings without ties, from its pairs."""
    signs = [
        np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
        for i in range(len(x))
        for j in range(i + 1, len(x))
    ]
    return float(np.mean(signs))


def test_rankings_are_compared_without_windfolds_variants_and_with_them_as_the_results_are(
    tmp_path,
):
    write_study(tmp_path)
    found = comparison_ranks(tmp_path, "development", (FARM,))[FARM]
    assert found["compared"]["models"] == ["kernel_linear", "no_wake", "pywake", "windfold"]
    assert found["comparison"]["models"] == ["kernel_linear", "no_wake", "pywake"]
    assert found["all"]["models"] == list(TIME)
    for name, tau in EXPECTED.items():
        models = found[name]["models"]
        assert tau == pytest.approx(kendall([TIME[m] for m in models], [ARC[m] for m in models]))
        pairs = found[name]["pairs"]
        assert list(pairs) == [
            "random_against_time_blocked",
            "random_against_arc_single",
            "time_blocked_against_arc_single",
        ]
        assert pairs["random_against_time_blocked"]["estimate"] == pytest.approx(1.0)
        chosen = pairs["time_blocked_against_arc_single"]
        assert chosen["models"] == len(models)
        assert chosen["estimate"] == pytest.approx(tau, abs=1e-12)
        assert chosen["low"] == pytest.approx(tau, abs=1e-12)
        assert chosen["high"] == pytest.approx(tau, abs=1e-12)


def test_all_models_must_reproduce_the_studys_rank_agreement_table(tmp_path):
    write_study(tmp_path, official_shift=1 / 15)
    with pytest.raises(ValueError, match="does not reproduce"):
        comparison_ranks(tmp_path, "development", (FARM,))
