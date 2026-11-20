from datetime import date, timedelta

import polars as pl
import pytest

from windfold.analyses.pair_skill import pair_skill
from windfold.evaluation import paths

FARM = "smarteole"
ARCS = {"arc_000_single": 10, "arc_090_single": 30}
WAKED_MSE = {
    "no_wake": 0.1,
    "kernel_linear": 0.06,
    "windfold": 0.05,
    "pywake": 0.07,
    "pywake_inflow": 0.065,
}
SEEDED = {"geometry_gbm": (0.07, 0.08, 0.09), "identity_gbm": (0.05, 0.05, 0.05)}


def write_study(runs) -> None:
    """errors and fits reductions over the 14 weekly blocks of the farm's window: per block 10 and
    30 waked rows on two single arcs at each model's mse (identity_linear's 0.2 and 0.5, a seeded
    gbm's per seed), 5 unwaked rows at mse 1 and a time-blocked fold where kernel_linear is
    nearly exact. peer_automl and the graphs have no fits."""
    days = [date(2020, 2, 17) + timedelta(days=7 * k) for k in range(14)]
    levels = {(model, None): (value, value) for model, value in WAKED_MSE.items()}
    levels[("identity_linear", None)] = (0.2, 0.5)
    for model, values in SEEDED.items():
        levels |= {(model, seed): (value, value) for seed, value in enumerate(values)}
    records, fits = [], []
    for (model, seed), values in levels.items():
        for (split, rows), mse in zip(ARCS.items(), values, strict=True):
            fits.append((model, split, seed, "arc_single", "done"))
            for day in days:
                records.append((model, split, seed, day, True, rows, rows * mse))
                records.append((model, split, seed, day, False, 5, 5.0))
        fits.append((model, "time_1", seed, "time_blocked", "done"))
        tight = 0.001 if model == "kernel_linear" else 0.1
        records += [(model, "time_1", seed, day, True, 10, 10 * tight) for day in days]
    errors = pl.DataFrame(
        records,
        schema={
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
    planned = pl.DataFrame(
        fits, schema=["model", "split", "seed", "regime", "status"], orient="row"
    )
    for name, frame in (("errors", errors), ("fits", planned)):
        path = runs / paths.reduction_file("development", FARM, None, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def test_each_pair_differs_in_skill_points_pooled_over_the_single_arcs(tmp_path):
    write_study(tmp_path)
    table = pair_skill(tmp_path, "development", (FARM,))["table"]
    found = {row["pair"]: row for row in table}
    assert [row["farm"] for row in table] == [FARM] * 7
    assert list(found) == [
        "pywake_minus_peer_automl",
        "geometry_gbm_minus_identity_gbm",
        "wind_frame_graph_minus_absolute_graph",
        "kernel_linear_minus_identity_linear",
        "kernel_linear_minus_pywake",
        "kernel_linear_minus_pywake_inflow",
        "kernel_linear_minus_windfold",
    ]
    # identity_linear pools to (0.2 * 10 + 0.5 * 30) / 40 = 0.425 against no_wake's 0.1
    expected = {
        "kernel_linear_minus_identity_linear": 365.0,
        "kernel_linear_minus_pywake": 10.0,
        "kernel_linear_minus_pywake_inflow": 5.0,
        "kernel_linear_minus_windfold": -10.0,
    }
    for pair, value in expected.items():
        row = found[pair]
        assert (row["estimate"], row["lower"], row["upper"]) == pytest.approx((value,) * 3)
    seeded = found["geometry_gbm_minus_identity_gbm"]
    assert seeded["estimate"] == pytest.approx(-30.0)
    # redrawing three of geometry_gbm's seeds gives all three at 0.09 or all at 0.07 in 1 replicate
    # of 27 each, more than 2.5%, so the interval runs from -40 to -20
    assert (seeded["lower"], seeded["upper"]) == pytest.approx((-40.0, -20.0))
    for pair in ("pywake_minus_peer_automl", "wind_frame_graph_minus_absolute_graph"):
        assert (found[pair]["estimate"], found[pair]["lower"]) == (None, None)
