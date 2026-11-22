from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.analyses.practical_scale import practical_scale
from windfold.analyses.sector_gains import sector_gains, spearman
from windfold.evaluation import paths
from windfold.schemas import SPLIT_SUMMARY

FARM = "smarteole"
WAKED = {"arc_000_single": 10, "arc_090_single": 20, "arc_180_single": 30, "arc_270_single": 40}
WINDFOLD = {"arc_000_single": 0.045, "arc_090_single": 0.04, "arc_270_single": 0.06}


def write_study(runs) -> None:
    """a development study of single arcs over the 14 weekly blocks of the farm's window: per
    block 10, 20, 30 and 40 waked rows on four arcs, at no_wake's mse 0.1 and pywake_inflow's
    0.05; windfold's 0.045, 0.04, then 0.049 and 0.051 in alternate blocks, then 0.06. every arc
    has 5 unwaked rows per block at mse 1, and a fifth arc has only those."""
    days = [date(2020, 2, 17) + timedelta(days=7 * k) for k in range(14)]
    records = []
    for k, day in enumerate(days):
        for split, rows in WAKED.items():
            mse = {"no_wake": 0.1, "pywake_inflow": 0.05}
            mse["windfold"] = WINDFOLD.get(split, 0.049 if k % 2 == 0 else 0.051)
            for model, value in mse.items():
                records.append((model, split, day, True, rows, rows * value))
        for split in (*WAKED, "arc_030_single"):
            for model in ("no_wake", "pywake_inflow", "windfold"):
                records.append((model, split, day, False, 5, 5.0))
    errors = pl.DataFrame(
        records,
        schema={
            "model": pl.String,
            "split": pl.String,
            "day": pl.Date,
            "waked": pl.Boolean,
            "rows": pl.Int64,
            "sse": pl.Float64,
        },
        orient="row",
    ).with_columns(seed=pl.lit(None, pl.Int64))
    path = runs / paths.reduction_file("development", FARM, None, "errors")
    path.parent.mkdir(parents=True)
    errors.write_parquet(path)
    summary = [(split, "arc_single", "included") for split in (*WAKED, "arc_030_single")]
    summary.append(("arc_120_single", "arc_single", "excluded"))
    manifests = runs / "development" / "manifests"
    manifests.mkdir(parents=True)
    pl.DataFrame(summary, schema=["split", "regime", "status"], orient="row").with_columns(
        pl.lit(None, pl.String).alias("reason"),
        *[pl.lit(1, pl.Int64).alias(c) for c in list(SPLIT_SUMMARY)[4:]],
    ).select(list(SPLIT_SUMMARY)).write_csv(manifests / f"{FARM}_summary.csv")


def test_spearman_ranks_each_replicate_and_leaves_a_replicate_with_a_gap_undefined():
    shares = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
    found = spearman(shares, np.array([[5.0, 10.0, 0.0, -10.0], [1.0, 2.0, 3.0, 4.0]]))
    assert found.tolist() == pytest.approx([-0.8, 1.0])
    assert np.isnan(spearman(np.array([0.1, 0.2, 0.3]), np.array([1.0, np.nan, 2.0]))).all()


def test_each_arcs_gain_follows_its_share_of_waked_rows(tmp_path):
    write_study(tmp_path)
    found = sector_gains(tmp_path, "development", (FARM,))[FARM]
    assert [s["split"] for s in found["sectors"]] == list(WAKED)
    assert found["without_waked_rows"] == ["arc_030_single"]
    assert [s["centre_deg"] for s in found["sectors"]] == [15.0, 105.0, 195.0, 285.0]
    assert [s["waked_rows"] for s in found["sectors"]] == [140, 280, 420, 560]
    assert [s["share"] for s in found["sectors"]] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    gains = [s["skill_points"]["estimate"] for s in found["sectors"]]
    assert gains == pytest.approx([5.0, 10.0, 0.0, -10.0], abs=1e-9)
    unsure = found["sectors"][2]["skill_points"]
    assert unsure["low"] < 0 < unsure["high"]
    verdicts = [s["verdict"] for s in found["sectors"]]
    assert verdicts == ["ahead", "ahead", "not resolved", "behind"]
    assert (found["ahead"], found["behind"], found["not_resolved"]) == (2, 1, 1)
    # gains rank 3, 4, 2, 1 against shares 1 to 4: 1 - 6 * 18 / (4 * 15)
    assert found["spearman_share_gain"] == pytest.approx(
        {"estimate": -0.8, "low": -0.8, "high": -0.8}
    )


def test_a_confirmatory_farms_gains_are_practical_scales_per_arc(confirmatory_study):
    (confirmatory_study.runs / "confirmatory" / "freeze_second.json").write_text("{}")
    found = sector_gains(confirmatory_study.runs, "confirmatory")["penmanshiel"]
    practical = practical_scale(confirmatory_study.runs, "confirmatory")["penmanshiel"]["per_arc"]
    assert [(s["split"], s["waked_rows"], s["skill_points"]) for s in found["sectors"]] == [
        (a["split"], a["waked_rows"], a["against_pywake_inflow"]["skill_points"]) for a in practical
    ]
    assert found["spearman_share_gain"]["estimate"] is None
