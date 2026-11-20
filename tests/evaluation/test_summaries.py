from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.config import load_protocol
from windfold.evaluation.bootstrap import block_totals, draw_blocks, multiplicity
from windfold.evaluation.hypotheses import (
    NOT_ESTIMABLE,
    CoverageConfirmation,
    coverage_predicts_failure,
)
from windfold.evaluation.reductions import MONITORING_DAYS, MONITORING_SCALES
from windfold.evaluation.summaries import (
    Draw,
    accuracy,
    comparisons,
    confirmation_contrasts,
    coverage_null,
    grids_and_copies,
    monitor_grids,
    paired_arcs,
)

ORIGIN = date(2020, 1, 1)
BLOCKS = 6
COVARIATES = ["wake_strength", "angular_distance"]


def test_deletion_variants_pair_on_the_arc_they_hold_out():
    opposite = ["arc_030_opposite", "arc_000_opposite", "arc_090_opposite"]
    control = ["arc_000_control", "arc_060_control", "arc_030_control"]
    assert paired_arcs(opposite, control) == [
        ("arc_000_opposite", "arc_000_control"),
        ("arc_030_opposite", "arc_030_control"),
    ]


def arcs_and_totals(seed: int = 4):
    """eight held-out arcs, a relational model losing to physics on four, block totals of both
    on the arc splits and of the relational model's time-blocked fit by arc."""
    rng = np.random.default_rng(seed)
    splits = [f"arc_{30 * arc:03d}_single" for arc in range(8)]
    sector, time = [], []
    for block in range(BLOCKS):
        day = ORIGIN + timedelta(days=7 * block)
        for arc, split in enumerate(splits):
            rows = float(rng.integers(20, 40))
            physics = rng.uniform(1.0, 2.0) * rows
            relational = physics * (1.3 if arc % 2 else 0.7) + rng.uniform(0, 0.2) * rows
            sector += [("rel", split, True, day, relational, rows)]
            sector += [("phys", split, True, day, physics, rows)]
            time += [("rel", "time_1", True, "arc", arc, day, rng.uniform(0.5, 1.5) * rows, rows)]
    keys = {"model": pl.String, "split": pl.String, "seed": pl.Int64, "waked": pl.Boolean}
    amounts = {"day": pl.Date, "sse": pl.Float64, "rows": pl.Float64}
    sector_frame = pl.DataFrame(
        [(m, s, None, w, d, e, r) for m, s, w, d, e, r in sector],
        schema=keys | amounts,
        orient="row",
    )
    time_frame = pl.DataFrame(
        [(m, s, None, w, f, lv, d, e, r) for m, s, w, f, lv, d, e, r in time],
        schema=keys | {"factor": pl.String, "level": pl.Int64} | amounts,
        orient="row",
    )
    sector_totals = block_totals(sector_frame, list(keys), ["sse", "rows"], ORIGIN, 7, BLOCKS)
    time_totals = block_totals(
        time_frame, [*keys, "factor", "level"], ["sse", "rows"], ORIGIN, 7, BLOCKS
    )
    design = pl.DataFrame(
        {
            "split": splits,
            "arc": list(range(8)),
            "score": rng.uniform(0, 1, 8),
            "wake_strength": rng.uniform(0, 1, 8),
            "angular_distance": rng.uniform(0, 30, 8),
        }
    )
    counts = multiplicity(draw_blocks({"w": BLOCKS}, 400, seed=seed)["w"], BLOCKS)
    draw = Draw(counts, np.zeros((400, 1), int), 0.95, 0.95)
    return sector_totals, time_totals, design, draw


def nulls(design, sector, time, draw):
    return coverage_null(
        sector,
        time,
        draw,
        design,
        covariates=COVARIATES,
        model="rel",
        physics="phys",
        time_splits=["time_1"],
    )


def test_the_permuted_layout_null_shares_the_arcs_and_bootstrap_of_coverage_predicts_failure():
    sector, time, design, draw = arcs_and_totals()
    same = design.with_columns(permuted_score=pl.col("score"))
    table = nulls(same, sector, time, draw)
    by = {row["layout"]: row for row in table.iter_rows(named=True)}
    assert set(by) == {"real", "permuted", "difference"}
    assert by["difference"]["estimate"] == 0.0
    assert by["difference"]["lower"] == by["difference"]["upper"] == 0.0
    h3 = coverage_predicts_failure(
        sector,
        time,
        draw.counts,
        design=same,
        covariates=COVARIATES,
        confirmation=None,
        model="rel",
        physics="phys",
        time_splits=["time_1"],
        min_class=3,
    )["partial_spearman"]
    assert not isinstance(h3, str)
    observed, replicates = h3
    assert by["real"]["estimate"] == pytest.approx(observed)
    assert (by["real"]["lower"], by["real"]["upper"]) == pytest.approx(
        tuple(np.quantile(replicates, [0.025, 0.975]))
    )
    reversed_layout = design.with_columns(permuted_score=-pl.col("score"))
    by = {r["layout"]: r for r in nulls(reversed_layout, sector, time, draw).iter_rows(named=True)}
    assert by["difference"]["estimate"] == pytest.approx(
        by["real"]["estimate"] - by["permuted"]["estimate"]
    )
    assert by["permuted"]["estimate"] == pytest.approx(-by["real"]["estimate"])


def test_too_few_arcs_in_a_class_leave_the_auc_gain_not_estimable_and_keep_the_correlation():
    sector, time, design, draw = arcs_and_totals()
    same = design.with_columns(permuted_score=pl.col("score"))
    confirmation = CoverageConfirmation(0.5, 0.3, 0.0, 1.0, "wake_strength", 1.0)

    def h3(min_class):
        return coverage_predicts_failure(
            sector,
            time,
            draw.counts,
            design=same,
            covariates=COVARIATES,
            confirmation=confirmation,
            model="rel",
            physics="phys",
            time_splits=["time_1"],
            min_class=min_class,
        )

    judged, gated = h3(3), h3(5)
    assert not isinstance(judged["auc_gain"], str)
    assert gated["auc_gain"] == NOT_ESTIMABLE
    gated_rho, judged_rho = gated["partial_spearman"], judged["partial_spearman"]
    assert not isinstance(gated_rho, str) and not isinstance(judged_rho, str)
    assert gated_rho[0] == judged_rho[0]
    assert np.array_equal(gated_rho[1], judged_rho[1], equal_nan=True)
    assert not nulls(same, sector, time, draw).is_empty()


def test_comparisons_give_the_skill_a_model_keeps_over_another_against_the_reference():
    rng = np.random.default_rng(7)
    records = []
    for block in range(BLOCKS):
        day = ORIGIN + timedelta(days=7 * block)
        rows = float(rng.integers(20, 40))
        for model, level in (("windfold", 1.0), ("other", 1.5), ("no_wake", 4.0)):
            records.append(
                (model, "random_1", None, True, day, level * rng.uniform(0.8, 1.2) * rows, rows)
            )
    frame = pl.DataFrame(
        records,
        schema={
            "model": pl.String,
            "split": pl.String,
            "seed": pl.Int64,
            "waked": pl.Boolean,
            "day": pl.Date,
            "sse": pl.Float64,
            "rows": pl.Float64,
        },
        orient="row",
    )
    totals = block_totals(
        frame, ["model", "split", "seed", "waked"], ["sse", "rows"], ORIGIN, 7, BLOCKS
    )
    counts = multiplicity(draw_blocks({"w": BLOCKS}, 300, seed=7)["w"], BLOCKS)
    draw = Draw(counts, np.zeros((300, 1), int), 0.95, 0.95)
    table = comparisons(
        totals, draw, {("random", None): ["random_1"]}, ["other"], "windfold", "no_wake"
    )
    mse = {m: float(g["sse"].sum()) / float(g["rows"].sum()) for (m,), g in frame.group_by("model")}
    row = table.row(0, named=True)
    assert row["estimate"] == pytest.approx((mse["other"] - mse["windfold"]) / mse["no_wake"])
    assert row["lower"] < row["estimate"] < row["upper"]
    assert row["p"] < 0.05


def test_skill_is_one_minus_the_mse_ratio_and_history_starts_weigh_equally():
    starts = ["history_030d_from_2018-01-01", "history_030d_from_2018-02-01"]
    sums = {
        ("reference", "random_1"): (20.0, 10.0),
        ("half", "random_1"): (10.0, 10.0),
        ("same", "random_1"): (20.0, 10.0),
        ("double", "random_1"): (40.0, 10.0),
        ("perfect", "random_1"): (0.0, 10.0),
        ("half", starts[0]): (20.0, 10.0),
        ("half", starts[1]): (60.0, 30.0),
        ("reference", starts[0]): (10.0, 10.0),
        ("reference", starts[1]): (270.0, 30.0),
    }
    frame = pl.DataFrame(
        [
            (model, split, None, True, ORIGIN, sse, rows)
            for (model, split), (sse, rows) in sums.items()
        ],
        schema={
            "model": pl.String,
            "split": pl.String,
            "seed": pl.Int64,
            "waked": pl.Boolean,
            "day": pl.Date,
            "sse": pl.Float64,
            "rows": pl.Float64,
        },
        orient="row",
    )
    totals = block_totals(frame, ["model", "split", "seed", "waked"], ["sse", "rows"], ORIGIN, 7, 1)
    draw = Draw(np.ones((3, 1)), np.zeros((3, 1), int), 0.95, 0.95)
    groups = {("random", None): ["random_1"], ("history", 30): starts}

    def skills(table: pl.DataFrame) -> dict[tuple[str, str], float]:
        chosen = table.filter(pl.col("metric") == "skill", pl.col("rows") == "waked")
        return {(m, r): e for m, r, e in chosen.select("model", "regime", "estimate").iter_rows()}

    found = skills(accuracy(totals, draw, groups, ["half", "same", "double"], "reference"))
    assert found == pytest.approx(
        {
            ("half", "random"): 0.5,
            ("same", "random"): 0.0,
            ("double", "random"): -1.0,
            ("half", "history"): 1 - 2 / 5,
        }
    )
    with np.errstate(divide="ignore"):
        against_zero = accuracy(totals, draw, {("random", None): ["random_1"]}, ["half"], "perfect")
    assert skills(against_zero) == {}
    assert against_zero.filter(pl.col("metric") == "mse")["estimate"].to_list() == [1.0, 1.0]


def monitoring_inputs(models: dict[str, set[str]]) -> dict[str, pl.DataFrame]:
    split = "threshold_030d_from_2018-01-01"
    rows = [
        (model, split, turbine, 100, 20, 0.05, 0.3, turbine not in ineligible)
        for model, ineligible in models.items()
        for turbine in ("T1", "T2", "T3", "T4")
    ]
    return {
        "monitoring_scales": pl.DataFrame(rows, schema=MONITORING_SCALES, orient="row"),
        "monitoring_days": pl.DataFrame(schema=MONITORING_DAYS),
    }


def test_planned_copies_ignore_matched_monitors_and_matched_copies_need_every_monitor():
    protocol = load_protocol()
    turbines = ["T1", "T2", "T3", "T4"]
    planned = ("windfold", "cyclic_gam", "direction_lookup")
    alone = monitoring_inputs({"windfold": set(), "cyclic_gam": set(), "direction_lookup": {"T1"}})
    beside = monitoring_inputs(
        {"windfold": set(), "cyclic_gam": set(), "direction_lookup": {"T1"}, "matched": {"T4"}}
    )
    origin = (date(2018, 1, 1), 120)
    key = (30, "2018-01-01")
    _, before = grids_and_copies(alone, origin, turbines, protocol, 7, planned)
    _, after = grids_and_copies(beside, origin, turbines, protocol, 7, planned)
    _, matched = grids_and_copies(beside, origin, turbines, protocol, 7, (*planned, "matched"))
    assert before[key].equals(after[key])
    assert "T1" in set(before[key]["turbine"]) and "T4" in set(before[key]["turbine"])
    assert "T4" not in set(matched[key]["turbine"])


def test_each_seed_of_a_monitor_is_standardised_with_its_own_calibration():
    turbines = ["T1", "T2"]
    start = date(2018, 1, 31)
    threshold, history = "threshold_030d_from_2018-01-01", "history_030d_from_2018-01-01"
    sigmas, residuals = (0.1, 0.2, 0.4), (0.01, 0.02, 0.03)
    scales = pl.DataFrame(
        [
            ("identity_gbm", threshold, turbine, 100, 20, sigma, 0.0, True)
            for sigma in sigmas
            for turbine in turbines
        ],
        schema=MONITORING_SCALES,
        orient="row",
    )
    days = pl.DataFrame(
        [
            ("identity_gbm", history, turbine, start + timedelta(days=d), 100, residual)
            for residual in residuals
            for turbine in turbines
            for d in range(3)
        ],
        schema=MONITORING_DAYS,
        orient="row",
    )
    grids = monitor_grids(scales, days, turbines, start, 3, 36)[("identity_gbm", 30, "2018-01-01")]
    assert isinstance(grids, list) and len(grids) == 3
    for (monitored, z, _), sigma, residual in zip(grids, sigmas, residuals, strict=True):
        assert monitored.all()
        assert np.allclose(z, residual / (sigma / np.sqrt(100)))
    with pytest.raises(ValueError, match="seeds"):
        monitor_grids(scales.head(4), days, turbines, start, 3, 36)


def test_confirmation_contrasts_give_each_gap_and_the_interaction_of_the_four_cells():
    rng = np.random.default_rng(11)
    levels = {"w": 1.0, "c": 1.4, "e": 1.2, "ce": 1.9, "other": 1.6, "no_wake": 4.0}
    records = []
    for block in range(BLOCKS):
        day = ORIGIN + timedelta(days=7 * block)
        rows = float(rng.integers(20, 40))
        for model, level in levels.items():
            sse = level * rng.uniform(0.8, 1.2) * rows
            records.append((model, "arc_000_single", None, True, day, sse, rows))
    schema = {
        "model": pl.String,
        "split": pl.String,
        "seed": pl.Int64,
        "waked": pl.Boolean,
        "day": pl.Date,
        "sse": pl.Float64,
        "rows": pl.Float64,
    }
    frame = pl.DataFrame(records, schema=schema, orient="row")
    keys = ["model", "split", "seed", "waked"]
    totals = block_totals(frame, keys, ["sse", "rows"], ORIGIN, 7, BLOCKS)
    counts = multiplicity(draw_blocks({"w": BLOCKS}, 300, seed=7)["w"], BLOCKS)
    draw = Draw(counts, np.zeros((300, 1), int), 0.95, 0.95)
    found = confirmation_contrasts(
        totals, draw, ["arc_000_single"], "w", ["other"], ("c", "e", "ce"), "no_wake"
    )
    mse = {m: float(g["sse"].sum()) / float(g["rows"].sum()) for (m,), g in frame.group_by("model")}
    skill = {m: 1 - mse[m] / mse["no_wake"] for m in mse}
    gap = found["w_minus_other"]
    interaction = found["interaction"]
    assert not isinstance(gap, str) and not isinstance(interaction, str)
    assert gap[0] == pytest.approx(skill["w"] - skill["other"])
    expected = (skill["w"] - skill["c"]) - (skill["e"] - skill["ce"])
    assert interaction[0] == pytest.approx(expected)
    assert interaction[1].shape == (300,)
