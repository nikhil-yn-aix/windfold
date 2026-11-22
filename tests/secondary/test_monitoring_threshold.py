from datetime import date

import numpy as np
import polars as pl
import pytest
from scipy import stats

from windfold import monitoring
from windfold.config import load_protocol
from windfold.evaluation import summaries
from windfold.secondary import monitoring_threshold as mt

SPLIT = "threshold_030d_from_2020-01-01"
RATE = 1 / monitoring.MONTH_DAYS


def scales_of(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "split": pl.String,
            "copy": pl.Int64,
            "turbine": pl.String,
            "rows": pl.Int64,
            "days": pl.Int64,
            "sigma": pl.Float64,
            "rho": pl.Float64,
            "eligible": pl.Boolean,
        },
        orient="row",
    )


def days_of(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "split": pl.String,
            "copy": pl.Int64,
            "turbine": pl.String,
            "day": pl.Date,
            "rows": pl.Int64,
            "residual": pl.Float64,
        },
        orient="row",
    )


def test_k_is_the_floored_mean_square_of_the_eligible_standardised_calibration_days():
    scales = scales_of(
        [
            ("windfold", SPLIT, 0, "T0", 150, 3, 0.2, 0.5, True),
            ("windfold", SPLIT, 0, "T1", 76, 3, 0.1, 0.0, True),
            ("windfold", SPLIT, 0, "T2", 100, 3, 0.1, 0.0, False),
            ("pywake", SPLIT, 0, "T0", 100, 3, 1.0, 0.0, True),
            ("cyclic_gam", SPLIT, 0, "T0", 100, 1, 0.1, 0.0, False),
        ]
    )
    first, second, third = date(2020, 1, 25), date(2020, 1, 26), date(2020, 1, 27)
    days = days_of(
        [
            ("windfold", SPLIT, 0, "T0", first, 100, 0.1),
            ("windfold", SPLIT, 0, "T0", second, 50, -0.2),
            ("windfold", SPLIT, 0, "T1", first, 40, 0.05),
            ("windfold", SPLIT, 0, "T1", third, 36, 0.0),
            ("windfold", SPLIT, 0, "T2", first, 100, 5.0),
            ("pywake", SPLIT, 0, "T0", first, 100, 0.01),
        ]
    )
    found = {row["model"]: row for row in mt.scale_factors(scales, days, RATE).to_dicts()}
    squares = [0.1**2 / 0.0012, 0.2**2 / 0.0024, 0.05**2 / 0.00025, 0.0]
    assert found["windfold"]["mean_square"] == pytest.approx(np.mean(squares))
    assert found["windfold"]["k"] == pytest.approx(8.75)
    assert (found["windfold"]["calibration_days"], found["windfold"]["turbines"]) == (3, 2)
    assert found["windfold"]["turbine_days"] == 4
    assert found["windfold"]["t_ratio"] == pytest.approx(
        stats.t.ppf(RATE, 3) / stats.norm.ppf(RATE)
    )
    assert found["pywake"]["mean_square"] == pytest.approx(0.01**2 / 0.01)
    assert found["pywake"]["k"] == 1.0
    assert found["cyclic_gam"]["mean_square"] is None
    assert (found["cyclic_gam"]["k"], found["cyclic_gam"]["t_ratio"]) == (1.0, 1.0)


def test_rules_c_and_ct_divide_the_standardised_day_by_sqrt_k_and_the_student_t_ratio():
    scales = scales_of([("windfold", SPLIT, 0, "T0", 150, 3, 0.2, 0.0, True)])
    factors = pl.DataFrame(
        {"model": ["windfold"], "split": [SPLIT], "copy": [0], "k": [4.0], "t_ratio": [1.25]}
    )
    test_day = pl.DataFrame(
        {"turbine": ["T0"], "day": [date(2022, 1, 1)], "rows": [100], "residual": [-0.06]}
    )
    grids = {
        rule: monitoring.day_grids(
            test_day,
            mt.scaled(scales, factors, rule).drop("model", "split"),
            ["T0"],
            date(2022, 1, 1),
            1,
            36,
        )
        for rule in mt.RULES
    }
    assert {rule: grid[1][0, 0] for rule, grid in grids.items()} == pytest.approx(
        {"registered": -3.0, "c": -1.5, "ct": -1.2}
    )
    alarms = {
        rule: bool(monitoring.alarm_grid(*grid, -1.3, 0.0)[0, 0]) for rule, grid in grids.items()
    }
    assert alarms == {"registered": True, "c": True, "ct": False}


def test_the_variance_ratio_pools_turbines_and_starts_with_one_less_than_their_days():
    other = "threshold_030d_from_2020-04-01"
    scales = scales_of(
        [
            ("windfold", SPLIT, 0, "T0", 150, 3, 0.1, 0.0, True),
            ("windfold", other, 0, "T0", 150, 3, 0.2, 0.0, True),
        ]
    )
    stamps = [date(2020, 1, d) for d in (25, 26, 27)]
    days = days_of(
        [
            ("windfold", SPLIT, 0, "T0", d, 100, r)
            for d, r in zip(stamps, [0.1, 0.2, 0.6], strict=True)
        ]
        + [
            ("windfold", other, 0, "T0", d, 50, r)
            for d, r in zip(stamps[:2], [0.0, 0.4], strict=True)
        ]
    )
    squares = np.var([0.1, 0.2, 0.6]) * 3 + np.var([0.0, 0.4]) * 2
    expected = squares / (2 * 0.1**2 / 100 + 1 * 0.2**2 / 50)
    ratio = mt.variance_ratios(scales, days).row(0, named=True)
    assert (ratio["model"], ratio["horizon_days"]) == ("windfold", 30)
    assert ratio["ratio"] == pytest.approx(expected)


def test_k_and_the_variance_ratio_ignore_the_order_the_calibration_days_arrive_in():
    starts = (SPLIT, "threshold_030d_from_2020-04-01")
    turbines = ("T0", "T1", "T2")
    scales = scales_of(
        [("windfold", s, 0, t, 100, 9, 0.1, 0.2, True) for s in starts for t in turbines]
    )
    residuals = iter(np.random.default_rng(1).normal(0.0, 0.05, 54))
    days = days_of(
        [
            ("windfold", s, 0, t, date(2020, 1, 1 + d), 40 + d, float(next(residuals)))
            for s in starts
            for t in turbines
            for d in range(9)
        ]
    )
    found = [
        (
            mt.scale_factors(scales, d, RATE)["k"].to_list(),
            mt.variance_ratios(scales, d)["ratio"][0],
        )
        for d in (days, days.reverse())
    ]
    assert found[0] == found[1]


def rows_at(
    exceedance: dict[str, float | None], parts: dict[tuple[str, str], tuple[float, float]]
) -> list[dict]:
    rows = [
        {"metric": "exceedance", "monitor": m, "history_days": 30, "estimate": e}
        for m, e in exceedance.items()
    ]
    rows += [
        {"metric": metric, "monitor": monitor, "history_days": 30}
        | dict(zip(("lower", "p"), v, strict=True))
        for (metric, monitor), v in parts.items()
    ]
    return rows


def test_calibration_needs_every_exceedance_in_the_band_and_advantage_every_gain_resolved():
    protocol = load_protocol()
    nominal = 1 / monitoring.MONTH_DAYS
    resolved = {
        (metric, other): (0.01, 0.002)
        for metric in ("recall_gain", "exceedance_margin")
        for other in ("cyclic_gam", "identity_gbm")
    }
    inside = {"windfold": 0.6 * nominal, "pywake": 1.9 * nominal, "direction_lookup": None}
    calibration, advantage = mt.check_rows(rows_at(inside, resolved), nominal, protocol)
    assert (calibration["passes"], advantage["passes"], advantage["p"]) == (True, True, 0.002)
    outside = inside | {"cyclic_gam": 2.1 * nominal}
    unresolved = resolved | {("recall_gain", "identity_gbm"): (-0.001, 0.04)}
    calibration, advantage = mt.check_rows(rows_at(outside, unresolved), nominal, protocol)
    assert (calibration["passes"], advantage["passes"], advantage["p"]) == (False, False, 0.04)
    missing = {key: value for key, value in resolved.items() if key[1] == "cyclic_gam"}
    calibration, advantage = mt.check_rows(rows_at({}, missing), nominal, protocol)
    assert (calibration["passes"], advantage["passes"], advantage["p"]) == (None, None, None)


def test_a_gain_needs_both_monitors_on_every_matched_start():
    protocol = load_protocol()
    first, second = "2020-01-01", "2020-04-01"
    quiet = (np.ones((1, 28), dtype=bool), np.zeros((1, 28)), np.ones((1, 28)))
    alarming = (quiet[0], quiet[1], np.full((1, 28), 0.01))
    episodes = pl.DataFrame({"injection": [0, 1], "turbine": ["T0", "T0"], "offset": [0, 7]})
    copies = {(30, start): episodes for start in (first, second)}
    draw = summaries.Draw(
        counts=np.ones((100, 4)),
        seeds=np.zeros((100, 1), dtype=np.int64),
        level=0.95,
        min_finite=0.95,
    )
    slots = np.tile(np.arange(28), (100, 1))

    def scored(windfold: dict, comparators: dict) -> tuple[dict, bool | None]:
        grids = {("windfold", 30, start): [grid] for start, grid in windfold.items()}
        grids |= {
            (model, 30, start): [grid]
            for model in ("cyclic_gam", "identity_gbm")
            for start, grid in comparators.items()
        }
        found = mt.rates(grids, copies, ["T0"], slots, draw, 30, protocol)
        rows = mt.rate_rows(found, draw) + mt.gain_rows(found, draw, protocol)
        gains = {
            (r["monitor"], r.get("starts")): (r.get("estimate"), r.get("lower"), r.get("p"))
            for r in rows
            if r["metric"] == "recall_gain"
        }
        return gains, mt.check_rows(rows, RATE, protocol)[1]["passes"]

    unpaired = {(m, None): (None, None, None) for m in ("cyclic_gam", "identity_gbm")}
    assert scored({first: quiet, second: alarming}, {first: quiet}) == (unpaired, None)
    assert scored({first: quiet}, {first: quiet}) == (unpaired, None)
    both = {first: quiet, second: alarming}
    paired = {(m, 2): (0.0, 0.0, 1.0) for m in ("cyclic_gam", "identity_gbm")}
    assert scored(both, both) == (paired, False)


def test_the_harness_difference_is_infinite_when_one_table_leaves_a_value_undefined():
    official = pl.DataFrame({"key": [1, 2], "estimate": [0.5, None], "lower": [0.5, None]})
    same = pl.DataFrame({"key": [1, 2], "estimate": [0.5, float("nan")], "lower": [0.375, None]})
    assert mt.largest_difference(same, official, ["key"], ["estimate", "lower"]) == 0.125
    defined = same.with_columns(estimate=pl.Series([0.5, 0.1]))
    assert mt.largest_difference(defined, official, ["key"], ["estimate"]) == float("inf")
    assert mt.largest_difference(same[:1], official[:1], ["key"], ["estimate"]) == 0.0


def test_confirmatory_thresholds_wait_for_the_second_freeze_record(tmp_path):
    with pytest.raises(FileNotFoundError, match="second freeze"):
        mt.score(tmp_path / "data", tmp_path, "confirmatory", "penmanshiel")
