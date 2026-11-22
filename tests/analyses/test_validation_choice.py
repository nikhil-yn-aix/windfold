from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.analyses.validation_choice import validation_choice
from windfold.config import load_protocol
from windfold.evaluation import paths

FARM = "smarteole"
SPLITS = {"random_1": "random", "time_1": "time_blocked", "arc_000_single": "arc_single"}


def kendall(x: list[float], y: list[float]) -> float:
    """kendall's tau for rankings without ties, from its pairs."""
    signs = [
        np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
        for i in range(len(x))
        for j in range(i + 1, len(x))
    ]
    return float(np.mean(signs))


def write_study(runs, time_mse: dict[str, float], arc_mse: dict[str, float]) -> None:
    """errors and fits reductions with one row of ten waked rows per model, split and 7-day block
    of the farm's development window, each at the model's mse for the split's regime."""
    days = [date(2020, 2, 17) + timedelta(days=7 * k) for k in range(14)]
    mse = {"random": time_mse, "time_blocked": time_mse, "arc_single": arc_mse}
    errors = pl.DataFrame(
        [
            (FARM, model, split, None, day, True, 10, 10 * value)
            for split, regime in SPLITS.items()
            for model, value in mse[regime].items()
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
        [(m, s, None, r, "done") for s, r in SPLITS.items() for m in time_mse],
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


def test_time_blocked_validation_chooses_by_its_folds_and_variants_are_ranked_apart(tmp_path):
    baselines = list(load_protocol().endpoints.validation_misleads.baselines)
    order = ["peer_automl", "pywake", *[b for b in baselines if b not in ("peer_automl", "pywake")]]
    time_mse = {m: 0.1 + 0.01 * k for k, m in enumerate([*order, "windfold"])}
    arc_mse = dict(time_mse) | {
        "peer_automl": time_mse["pywake"],
        "pywake": time_mse["peer_automl"],
    }
    time_mse |= {"windfold_a": 0.01, "windfold_b": 0.02}
    arc_mse |= {"windfold_a": 1.0, "windfold_b": 1.1}
    write_study(tmp_path, time_mse, arc_mse)
    found = validation_choice(tmp_path, (FARM,))[FARM]
    assert found["chosen"] == "peer_automl"
    shares = {name: entry["chosen_share"] for name, entry in found["baselines"].items()}
    assert shares["peer_automl"] == 1.0 and sum(shares.values()) == 1.0
    assert found["baselines"]["pywake"]["arc_single_mse"] == pytest.approx(0.1)
    agreement = found["rank_agreement"]
    models = list(time_mse)
    kept = [m for m in models if not m.startswith("windfold_")]
    assert (agreement["all"]["models"], agreement["without_windfold_variants"]["models"]) == (
        len(models),
        len(kept),
    )
    for label, chosen in (("all", models), ("without_windfold_variants", kept)):
        expected = kendall([time_mse[m] for m in chosen], [arc_mse[m] for m in chosen])
        assert agreement[label]["estimate"] == pytest.approx(expected)
