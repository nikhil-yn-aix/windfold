from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from windfold.evaluation.bootstrap import (
    Totals,
    block_totals,
    draw_blocks,
    draw_seeds,
    multiplicity,
)
from windfold.evaluation.hypotheses import (
    NotEstimable,
    auc,
    auc_gain,
    confirmatory_part,
    contrast_rows,
    decision,
    development_part,
    early_history_accuracy,
    early_history_monitoring,
    exceedance_benefit,
    farm_outcome,
    fit_confirmation,
    holm,
    hypothesis_rows,
    intersection_union,
    losses,
    one_sided_p,
    partial_spearman,
    percentile_interval,
    pooled_mse,
    validation_misleads,
)
from windfold.monitoring import trajectories

NAMES = (
    "validation_misleads",
    "wind_frame_transfers",
    "coverage_predicts_failure",
    "early_history_advantage",
)
S, F, N, X = "success", "failure", "not estimable", "not in this sensitivity"


def family(*values: float | None) -> dict[str, float | None]:
    return dict(zip(NAMES, values, strict=True))


@pytest.mark.parametrize(
    ("raw", "adjusted"),
    [
        (family(0.01, 0.04, 0.03, 0.005), family(0.03, 0.06, 0.06, 0.02)),
        (family(0.0001, 0.02, None, 0.004), family(0.0004, 0.04, None, 0.012)),
        (family(0.5, 0.6, 0.001, 0.002), family(1.0, 1.0, 0.004, 0.006)),
        (family(None, None, None, 0.01), family(None, None, None, 0.04)),
        ({"a": 0.01, "b": 0.04}, {"a": 0.02, "b": 0.04}),
        ({"a": 0.02, "b": 0.5, "c": 0.02}, {"a": 0.06, "b": 0.5, "c": 0.06}),
        ({"a": 0.01, "b": None, "c": 0.03}, {"a": 0.03, "b": None, "c": 0.06}),
    ],
)
def test_holm_matches_hand_values(raw, adjusted):
    for names in (list(raw), list(reversed(raw))):
        result = holm({name: raw[name] for name in names})
        assert set(result) == set(adjusted)
        for name, expected in adjusted.items():
            assert result[name] == (None if expected is None else pytest.approx(expected))


def test_intersection_union_takes_the_largest_component_p():
    assert intersection_union([0.01, 0.03, 0.002]) == 0.03
    assert intersection_union([0.01, None]) is None


def test_one_sided_p_counts_replicates_at_or_below_zero():
    assert one_sided_p(np.array([-1.0, 0.0, 1.0, 2.0])) == pytest.approx(3 / 5)
    assert one_sided_p(np.array([1.0, 2.0, 3.0])) == pytest.approx(1 / 4)
    with pytest.raises(ValueError, match="finite"):
        one_sided_p(np.array([1.0, np.nan]))


def test_exceedance_enters_as_margin_minus_the_rate_difference():
    assert exceedance_benefit(0.012, 0.010, 0.005) == pytest.approx(0.003)
    assert exceedance_benefit(0.017, 0.010, 0.005) == pytest.approx(-0.002)
    difference = np.linspace(-0.0036, 0.0044, 4001)
    benefit = exceedance_benefit(difference, np.zeros_like(difference), 0.005)
    lower = percentile_interval(benefit, 0.95)[0]
    assert percentile_interval(difference, 0.95)[1] == pytest.approx(0.0042)
    assert lower == pytest.approx(0.0008)
    assert one_sided_p(benefit) == pytest.approx(1 / 4002)
    assert farm_outcome([lower], one_sided_p(benefit), 0.05) == S
    worse = exceedance_benefit(difference + 0.001, np.zeros_like(difference), 0.005)
    assert percentile_interval(worse, 0.95)[0] == pytest.approx(-0.0002)
    assert farm_outcome([percentile_interval(worse, 0.95)[0]], 0.001, 0.05) == F


@pytest.mark.parametrize(
    ("lower_bounds", "adjusted_p", "outcome"),
    [
        ([0.1, 0.2], 0.01, S),
        ([0.1, -0.01], 0.01, F),
        ([0.1, 0.0], 0.01, F),
        ([0.1], 0.05, F),
        ([0.1, None], None, N),
        ([0.1], None, N),
    ],
)
def test_farm_outcome_truth_table(lower_bounds, adjusted_p, outcome):
    assert farm_outcome(lower_bounds, adjusted_p, alpha=0.05) == outcome


@pytest.mark.parametrize(
    ("development", "confirmatory", "verdict", "minima"),
    [
        ([S, S, S, S], [S], "supported", (3, 1)),
        ([S, S, S, F], [S], "supported", (3, 1)),
        ([S, S, S, N], [S], "supported", (3, 1)),
        ([S, S, F, F], [S], "not supported", (3, 1)),
        ([S, S, F, N], [S], "not supported", (3, 1)),
        ([S, S, N, N], [S], "not estimable", (3, 1)),
        ([S, S, S, S], [N], "not estimable", (3, 1)),
        ([S, S, S, S], [F], "not supported", (3, 1)),
        ([S, S, S, S], [S, F], "not supported", (3, 1)),
        ([S, S, S, S], [S, N], "supported", (3, 1)),
        ([S, S, N, N], [F], "not supported", (3, 1)),
        ([N, N, N, N], [N], "not estimable", (3, 1)),
        ([S, S, F, F], [N], "not supported", (3, 1)),
        ([S, S, N, N], [S, S], "supported", (2, 2)),
        ([S, S, F, F], [S, S], "supported", (2, 2)),
        ([S, S, F, F], [S, N], "not estimable", (2, 2)),
    ],
)
def test_decision_rule_truth_table(development, confirmatory, verdict, minima):
    least = dict(zip(("min_development", "min_confirmatory"), minima, strict=True))
    assert decision(development, confirmatory, **least) == verdict


def totals_of(table: dict) -> Totals:
    """block totals with sse and rows per (model, split, seed) and block, all on waked rows."""
    records = [
        (model, split, seed, True, date(2020, 1, 1) + timedelta(days=7 * block), sse, rows)
        for (model, split, seed), blocks in table.items()
        for block, (sse, rows) in enumerate(blocks)
    ]
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
    blocks = max(len(b) for b in table.values())
    return block_totals(
        frame, ["model", "split", "seed", "waked"], ["sse", "rows"], date(2020, 1, 1), 7, blocks
    )


def test_pooled_mse_averages_redrawn_seeds_like_a_loop():
    rng = np.random.default_rng(3)
    sse = rng.uniform(1, 2, size=(3, 4))
    rows = np.array([10.0, 20.0, 30.0, 40.0])
    table = {("g", "s", k): [(sse[k, b], rows[b]) for b in range(4)] for k in range(3)}
    totals = totals_of(table)
    counts = multiplicity(draw_blocks({"w": 4}, 30, seed=5)["w"], 4)
    seeds = draw_seeds(30, 3, seed=5)
    got = pooled_mse(totals, counts, seeds, "g", ["s"])
    for r in range(30):
        per_seed = [(counts[r] * sse[k]).sum() / (counts[r] * rows).sum() for k in range(3)]
        assert got[r] == pytest.approx(np.mean([per_seed[k] for k in seeds[r]]))
    observed = pooled_mse(totals, np.ones((1, 4)), None, "g", ["s"])[0]
    assert observed == pytest.approx(np.mean(sse.sum(axis=1)) / rows.sum())
    with pytest.raises(NotEstimable):
        pooled_mse(totals, counts, seeds, "g", ["s", "missing"])


def test_validation_misleads_picks_the_time_blocked_winner_in_each_replicate():
    table = {
        ("a", "time_1", None): [(1.0, 10.0), (1.0, 10.0)],
        ("b", "time_1", None): [(0.5, 10.0), (3.0, 10.0)],
        ("a", "arc_000_single", None): [(5.0, 10.0), (5.0, 10.0)],
        ("b", "arc_000_single", None): [(2.0, 10.0), (2.0, 10.0)],
        ("p", "arc_000_single", None): [(3.0, 10.0), (3.0, 10.0)],
    }
    totals = totals_of(table)
    counts = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    got = validation_misleads(
        totals,
        counts,
        None,
        baselines=["a", "b"],
        physics="p",
        time_splits=["time_1"],
        arc_splits=["arc_000_single"],
    )
    assert got.tolist() == pytest.approx([0.5 - 0.3, 0.2 - 0.3, 0.5 - 0.3])


def test_early_history_accuracy_weights_each_start_equally():
    table = {
        ("w", "h1", None): [(1.0, 10.0)],
        ("w", "h2", None): [(4.0, 100.0)],
        ("c", "h1", None): [(2.0, 10.0)],
        ("c", "h2", None): [(9.0, 100.0)],
    }
    got = early_history_accuracy(
        totals_of(table), np.ones((1, 1)), None, model="w", comparator="c", splits=["h1", "h2"]
    )
    assert got[0] == pytest.approx((0.2 + 0.09) / 2 - (0.1 + 0.04) / 2)


def test_partial_spearman_matches_an_independent_residualisation():
    rng = np.random.default_rng(8)
    score = rng.normal(size=14)
    covariates = rng.normal(size=(14, 2))
    outcome = score[None, :] + rng.normal(size=(5, 14))
    got = partial_spearman(score, outcome, covariates)
    ranked = np.column_stack([np.ones(14), stats.rankdata(covariates, axis=0)])
    for r in range(5):
        residual = []
        for values in (stats.rankdata(score), stats.rankdata(outcome[r])):
            coef = np.linalg.lstsq(ranked, values, rcond=None)[0]
            residual.append(values - ranked @ coef)
        assert got[r] == pytest.approx(stats.pearsonr(*residual)[0])
    plain = partial_spearman(score, outcome, np.empty((14, 0)))
    assert plain[0] == pytest.approx(stats.spearmanr(score, outcome[0])[0])


def test_auc_matches_scikit_learn_with_ties():
    rng = np.random.default_rng(2)
    scores = rng.integers(0, 4, size=20).astype(float)
    labels = rng.random((6, 20)) < 0.5
    got = auc(scores, labels)
    for r in range(6):
        assert got[r] == pytest.approx(roc_auc_score(labels[r], scores))
    assert np.isnan(auc(scores, np.zeros((1, 20), dtype=bool))[0])


def test_confirmation_orients_the_map_and_the_simple_predictor_on_the_fitting_farms():
    labels = np.array([True] * 6 + [False] * 6)
    scores = np.r_[np.linspace(2, 3, 6), np.linspace(0, 1, 6)]
    predictors = {"rows": np.r_[np.linspace(0, 1, 6), np.linspace(2, 3, 6)], "noise": np.zeros(12)}
    confirmation = fit_confirmation(scores, predictors, labels)
    assert confirmation.slope > 0
    assert (confirmation.predictor, confirmation.direction) == ("rows", -1.0)
    assert auc_gain(confirmation, scores, predictors, labels[None, :])[0] == pytest.approx(0.0)
    assert auc_gain(confirmation, scores, predictors, np.ones((1, 12), dtype=bool))[0] == 0.0
    with pytest.raises(NotEstimable):
        fit_confirmation(scores, predictors, np.ones(12, dtype=bool))


def test_a_replicate_that_leaves_an_arc_unlabelled_has_no_auc_gain():
    labels = np.array([1.0] * 6 + [0.0] * 6)
    scores = np.r_[np.linspace(2, 3, 6), np.linspace(0, 1, 6)]
    predictors = {"rows": np.r_[np.linspace(0, 1, 6), np.linspace(2, 3, 6)]}
    confirmation = fit_confirmation(scores, predictors, labels.astype(bool))
    unlabelled = labels.copy()
    unlabelled[0] = np.nan
    one_class = np.where(np.arange(12) == 0, np.nan, 1.0)
    gains = auc_gain(confirmation, scores, predictors, np.vstack([labels, unlabelled, one_class]))
    assert np.isfinite(gains[0])
    assert np.isnan(gains[1]) and np.isnan(gains[2])


def test_losses_are_undefined_where_a_replicate_draws_no_row_of_the_split():
    rows = [
        ("rel", split, None, True, day, sse, 10.0)
        for split, day, sse in [
            ("arc_000_single", date(2020, 1, 1), 30.0),
            ("arc_030_single", date(2020, 1, 8), 5.0),
        ]
    ]
    rows += [("phys", s, None, w, d, 10.0, n) for _, s, _, w, d, _, n in rows]
    schema = {
        "model": pl.String,
        "split": pl.String,
        "seed": pl.Int64,
        "waked": pl.Boolean,
        "day": pl.Date,
        "sse": pl.Float64,
        "rows": pl.Float64,
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row")
    keys = ["model", "split", "seed", "waked"]
    sector = block_totals(frame, keys, ["sse", "rows"], date(2020, 1, 1), 7, 2)
    counts = np.array([[1, 1], [2, 0]])
    got = losses(
        sector, counts, model="rel", physics="phys", splits=["arc_000_single", "arc_030_single"]
    )
    assert got[0].tolist() == [1.0, 0.0]
    assert got[1, 0] == 1.0 and np.isnan(got[1, 1])


def test_hypothesis_rows_combine_components_holm_and_outcomes():
    rising = np.linspace(0.001, 1.0, 1000)
    mixed = np.linspace(-0.5, 1.0, 1000)
    rows = hypothesis_rows(
        "kelmarsh",
        {
            "validation_misleads": {"d": (0.4, rising)},
            "wind_frame_transfers": {"linear": (0.4, rising), "gbm": (0.1, mixed), "graph": X},
            "coverage_predicts_failure": {"partial_spearman": N, "auc_gain": (0.4, rising)},
            "early_history_advantage": {"power_curve": X, "cyclic_gam": X},
        },
        0.95,
        0.05,
        0.95,
    )
    by = {(r["hypothesis"], r["component"]): r for r in rows.iter_rows(named=True)}
    assert rows.height == 8
    assert by[("validation_misleads", "d")]["replicates"] == 1000
    assert by[("validation_misleads", "d")]["p"] == pytest.approx(1 / 1001)
    assert by[("wind_frame_transfers", "gbm")]["p"] == pytest.approx((1 + 334) / 1001)
    assert by[("wind_frame_transfers", "linear")]["hypothesis_p"] == pytest.approx(335 / 1001)
    assert by[("validation_misleads", "d")]["holm_p"] == pytest.approx(4 / 1001)
    assert by[("wind_frame_transfers", "gbm")]["holm_p"] == 1.0
    assert by[("validation_misleads", "d")]["outcome"] == S
    assert by[("wind_frame_transfers", "graph")]["outcome"] == F
    assert by[("wind_frame_transfers", "graph")]["status"] == X
    assert by[("wind_frame_transfers", "graph")]["estimate"] is None
    assert by[("coverage_predicts_failure", "auc_gain")]["status"] == "estimated"
    assert by[("coverage_predicts_failure", "auc_gain")]["outcome"] == N
    assert by[("coverage_predicts_failure", "partial_spearman")]["hypothesis_p"] is None
    assert by[("early_history_advantage", "cyclic_gam")]["outcome"] == X
    assert by[("early_history_advantage", "cyclic_gam")]["holm_p"] is None


def test_contrast_rows_match_the_hypothesis_statistics_outside_the_decision_rule():
    mixed = np.linspace(-0.5, 1.0, 1000)
    sparse = np.where(np.arange(1000) % 3 == 0, 0.2, np.nan)
    rows = contrast_rows({"a": (0.1, mixed), "b": X, "c": (0.2, sparse)}, 0.95, 0.95)
    by = {r["component"]: r for r in rows.iter_rows(named=True)}
    same = hypothesis_rows(
        "kelmarsh", {"wind_frame_transfers": {"a": (0.1, mixed)}}, 0.95, 0.05, 0.95
    )
    reference = same.row(0, named=True)
    for key in ("estimate", "lower", "upper", "p", "replicates"):
        assert by["a"][key] == reference[key]
    assert by["b"]["status"] == X and by["b"]["estimate"] is None
    assert by["c"]["status"] == N and by["c"]["p"] is None
    assert "holm_p" not in rows.columns


@pytest.mark.parametrize(
    ("outcomes", "development", "confirmatory"),
    [
        ([X, X, X, X], X, X),
        ([S, S, S, X], S, S),
        ([S, S, X, X], N, S),
        ([S, F, X, X], N, F),
        ([N, X, X, X], N, N),
        ([], N, N),
    ],
)
def test_parts_leave_out_farms_outside_a_sensitivity(outcomes, development, confirmatory):
    assert development_part(outcomes, 3) == development
    assert confirmatory_part(outcomes, 1) == confirmatory


def test_a_verdict_refuses_a_part_outside_scope():
    with pytest.raises(ValueError, match="scope"):
        decision([X, X, X], [S], min_development=3, min_confirmatory=1)


def test_monitoring_components_compare_monitors_on_the_same_copies():
    days = 28
    monitored = np.ones((1, days), dtype=bool)
    sd = np.full((1, days), 0.01)
    sharp = (monitored, np.zeros((1, days)), sd)
    blind = (monitored, np.full((1, days), 50.0), sd)
    copies = [pl.DataFrame({"injection": [0, 1], "turbine": ["A", "A"], "offset": [0, 10]})]
    drawn = draw_blocks({"w": 4}, 30, seed=1)["w"]
    result = early_history_monitoring(
        {"windfold": [[sharp]], "cyclic_gam": [[blind]]},
        copies,
        ["A"],
        trajectories(drawn, days, 7),
        draw_seeds(30, 3, seed=2),
        model="windfold",
        comparators=["cyclic_gam"],
        quantile=-1.84,
        drop=0.05,
        horizon=14,
        margin=0.005,
    )
    assert result["recall_gain_cyclic_gam"][0] == pytest.approx(1.0)
    assert result["exceedance_margin_cyclic_gam"][0] == pytest.approx(0.005)
    assert np.allclose(result["exceedance_margin_cyclic_gam"][1], 0.005)


def test_a_seeded_monitor_is_averaged_over_its_seeds_and_each_replicates_drawn_seeds():
    days = 28
    monitored = np.ones((1, days), dtype=bool)
    sd = np.full((1, days), 0.01)
    sharp = (monitored, np.zeros((1, days)), sd)
    blind = (monitored, np.full((1, days), 50.0), sd)
    copies = [pl.DataFrame({"injection": [0, 1], "turbine": ["A", "A"], "offset": [0, 10]})]
    slots = trajectories(draw_blocks({"w": 4}, 30, seed=1)["w"], days, 7)
    seeds = draw_seeds(30, 3, seed=2)
    result = early_history_monitoring(
        {"windfold": [[sharp]], "identity_gbm": [[sharp], [blind], [blind]]},
        copies,
        ["A"],
        slots,
        seeds,
        model="windfold",
        comparators=["identity_gbm"],
        quantile=-1.84,
        drop=0.05,
        horizon=14,
        margin=0.005,
    )
    observed, replicates = result["recall_gain_identity_gbm"]
    assert observed == pytest.approx(2 / 3)
    assert np.allclose(replicates, 1 - (seeds == 0).mean(axis=1))
    with pytest.raises(NotEstimable, match="seeds"):
        early_history_monitoring(
            {"windfold": [[sharp]], "identity_gbm": [[sharp], [blind]]},
            copies,
            ["A"],
            slots,
            seeds,
            model="windfold",
            comparators=["identity_gbm"],
            quantile=-1.84,
            drop=0.05,
            horizon=14,
            margin=0.005,
        )


def test_a_seeded_model_missing_a_seed_is_not_estimable():
    table = {("g", "s", k): [(1.0, 10.0)] for k in (0, 2)}
    with pytest.raises(NotEstimable, match="seeds"):
        pooled_mse(totals_of(table), np.ones((4, 1)), np.zeros((4, 3), int), "g", ["s"])


def test_partial_spearman_is_not_estimable_when_a_covariate_orders_the_score():
    score = np.linspace(0, 1, 10)
    with pytest.raises(NotEstimable, match="explained"):
        partial_spearman(score, score[None, :], np.column_stack([2 * score + 1]))


def test_replicates_that_drew_no_rows_are_dropped_up_to_a_share():
    rising = np.linspace(0.001, 1.0, 1000)
    few = rising.copy()
    few[:30] = np.nan
    many = rising.copy()
    many[:60] = np.nan
    rows = hypothesis_rows(
        "kelmarsh",
        {"validation_misleads": {"d": (0.4, few)}, "wind_frame_transfers": {"linear": (0.4, many)}},
        0.95,
        0.05,
        0.95,
    )
    by = {r["hypothesis"]: r for r in rows.iter_rows(named=True)}
    assert by["validation_misleads"]["replicates"] == 970
    assert by["validation_misleads"]["p"] == pytest.approx(1 / 971)
    assert by["wind_frame_transfers"]["status"] == N
    assert by["wind_frame_transfers"]["outcome"] == N
