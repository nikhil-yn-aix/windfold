import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats

from windfold.evaluation.bootstrap import draw_blocks
from windfold.evaluation.schemas import EPISODES, SCALES
from windfold.monitoring import (
    alarm_grid,
    alarm_quantile,
    calibration_scales,
    daily_means,
    day_counts,
    day_grids,
    detections,
    episodes,
    exceedance,
    inject_drop,
    injections,
    kaplan_meier,
    median_delay,
    recall_and_exceedance,
    trajectories,
)


def at(day: int, hour: int = 0) -> datetime:
    return datetime(2020, 1, day, hour, tzinfo=UTC)


def contrasts(powers: dict[str, float]) -> dict[str, float]:
    """leave-one-out contrasts straight from powers: own log power minus the peers' mean."""
    logs = {name: math.log(power) for name, power in powers.items()}
    return {
        name: logs[name] - sum(logs[peer] for peer in logs if peer != name) / (len(logs) - 1)
        for name in logs
    }


def frame(powers_at: dict[datetime, dict[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            (timestamp, name, value, len(powers))
            for timestamp, powers in powers_at.items()
            for name, value in contrasts(powers).items()
        ],
        schema=["timestamp", "turbine", "contrast", "valid_turbines"],
        orient="row",
    ).with_columns(pl.col("valid_turbines").cast(pl.UInt8))


def test_a_five_percent_drop_on_three_turbines_matches_hand_values():
    rows = frame({at(1): {"A": 1000.0, "B": 800.0, "C": 600.0}})
    assert rows["contrast"].to_list() == pytest.approx([0.366984588, 0.032269261, -0.399253848])
    injected = inject_drop(rows, turbine="A", start=at(1), end=at(2), drop=0.05)
    assert injected["contrast"].to_list() == pytest.approx(
        [0.315691293, 0.057915908, -0.373607201], abs=1e-9
    )
    shifts = (injected["contrast"] - rows["contrast"]).to_list()
    assert shifts == pytest.approx([math.log(0.95), -math.log(0.95) / 2, -math.log(0.95) / 2])


def test_injection_equals_recomputing_contrasts_from_dropped_power():
    powers_at = {
        at(1): {"A": 1000.0, "B": 800.0, "C": 600.0},
        at(2): {"B": 700.0, "C": 650.0},
        at(3): {"A": 900.0, "B": 950.0, "C": 400.0, "D": 1200.0},
        at(8): {"A": 610.0, "B": 640.0},
        at(9): {"A": 500.0, "B": 450.0, "C": 520.0},
    }
    injected = inject_drop(frame(powers_at), turbine="A", start=at(1), end=at(8), drop=0.02)
    dropped = {
        t: {name: p * 0.98 if name == "A" and t < at(8) else p for name, p in powers.items()}
        for t, powers in powers_at.items()
    }
    assert injected["contrast"].to_list() == pytest.approx(frame(dropped)["contrast"].to_list())
    assert injected.filter(pl.col("timestamp") == at(2)).equals(
        frame(powers_at).filter(pl.col("timestamp") == at(2))
    )


@pytest.mark.parametrize("drop", [0.0, 1.0, -0.05])
def test_a_drop_outside_zero_to_one_fails(drop):
    with pytest.raises(ValueError, match="share of power"):
        inject_drop(
            frame({at(1): {"A": 1.0, "B": 2.0}}), turbine="A", start=at(1), end=at(2), drop=drop
        )


def stamp(day: int, minute: int) -> datetime:
    return datetime(2020, 1, day, tzinfo=UTC) + timedelta(minutes=minute)


def residual_rows(values_by_turbine: dict[str, list[tuple[datetime, float]]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(t, name, value) for name, rows in values_by_turbine.items() for t, value in rows],
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "turbine": pl.String,
            "residual": pl.Float64,
        },
        orient="row",
    ).with_columns(day=pl.col("timestamp").dt.date())


def test_calibration_scales_use_consecutive_ten_minute_pairs():
    rng = np.random.default_rng(6)
    minutes = [*range(0, 400, 10), *range(1440, 1440 + 400, 10), *range(2880 + 30, 2880 + 300, 10)]
    values = rng.normal(0, 0.1, len(minutes))
    rows = residual_rows({"A": list(zip([stamp(1, m) for m in minutes], values, strict=True))})
    scales = calibration_scales(rows, min_rows_per_day=36, min_calibration_days=3)
    row = scales.row(0, named=True)
    pairs = [
        (values[k], values[k - 1])
        for k in range(1, len(minutes))
        if minutes[k] - minutes[k - 1] == 10
    ]
    expected_rho = np.corrcoef(np.array(pairs).T)[0, 1]
    assert row["rho"] == pytest.approx(expected_rho)
    assert row["sigma"] == pytest.approx(np.std(values, ddof=1))
    assert (row["rows"], row["days"], row["eligible"]) == (len(minutes), 2, False)
    assert scales.schema == SCALES
    loose = calibration_scales(rows, min_rows_per_day=27, min_calibration_days=3)
    assert loose["eligible"].to_list() == [True]


def test_day_grid_standardises_daily_means_and_alarms_at_the_monthly_quantile():
    rows = residual_rows({"A": [(stamp(2, 10 * k), -0.05) for k in range(40)]})
    scales = pl.DataFrame(
        {
            "turbine": ["A"],
            "rows": [500],
            "days": [5],
            "sigma": [0.2],
            "rho": [0.5],
            "eligible": [True],
        },
        schema=SCALES,
    )
    monitored, z, sd = day_grids(daily_means(rows), scales, ["A", "B"], date(2020, 1, 1), 3, 36)
    expected_sd = 0.2 * math.sqrt(1.5 / (0.5 * 40))
    assert monitored.tolist() == [[False, True, False], [False, False, False]]
    assert sd[0, 1] == pytest.approx(expected_sd) and z[0, 1] == pytest.approx(-0.05 / expected_sd)
    quantile = alarm_quantile(1.0)
    assert quantile == pytest.approx(stats.norm.ppf(1 / (365.2425 / 12)))
    assert not alarm_grid(monitored, z, sd, quantile, 0.0).any()
    shifted = -0.05 / expected_sd + math.log(0.95) / expected_sd
    assert alarm_grid(monitored, z, sd, quantile, 0.05)[0, 1] == (shifted < quantile)


def test_trajectories_concatenate_drawn_blocks_in_draw_order():
    slots = trajectories(np.array([[1, 0], [1, 1]]), window_days=10, block_days=7)
    assert slots[0].tolist() == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
    assert slots[1].tolist() == [7, 8, 9, 7, 8, 9, -1, -1, -1, -1]


def first_alarm_by_loop(alarms, turbine, offset, slots_row, horizon):
    days = 0
    for k in range(horizon):
        position = offset + k
        if position >= slots_row.size or slots_row[position] < 0:
            break
        days += 1
        if alarms[turbine, slots_row[position]]:
            return k + 1, True
    return days, False


def test_detections_match_a_loop_over_rebuilt_days():
    rng = np.random.default_rng(12)
    alarms = rng.random((3, 30)) < 0.08
    copies = pl.DataFrame(
        {"injection": [0, 1, 2, 3], "turbine": ["A", "B", "C", "A"], "offset": [0, 5, 16, 10]}
    )
    drawn = draw_blocks({"w": 5}, 60, seed=4)["w"]
    slots = trajectories(drawn, window_days=30, block_days=7)
    times, detected = detections(alarms, copies, ["A", "B", "C"], slots, 14)
    for r in range(60):
        for j, (turbine, offset) in enumerate([(0, 0), (1, 5), (2, 16), (0, 10)]):
            expected = first_alarm_by_loop(alarms, turbine, offset, slots[r], 14)
            assert (times[r, j], detected[r, j]) == expected


def test_overlapping_episodes_share_their_rebuilt_weather():
    rng = np.random.default_rng(21)
    windfold = rng.random((2, 30)) < 0.15
    gam = windfold.copy()
    copies = pl.DataFrame({"injection": [0, 1], "turbine": ["A", "A"], "offset": [0, 3]})
    slots = trajectories(draw_blocks({"w": 5}, 300, seed=8)["w"], 30, 7)
    times, detected = detections(windfold, copies, ["A", "B"], slots, 14)
    gam_times, gam_detected = detections(gam, copies, ["A", "B"], slots, 14)
    assert np.array_equal(times, gam_times) and np.array_equal(detected, gam_detected)
    late = detected[:, 0] & (times[:, 0] > 3)
    assert late.any()
    assert np.array_equal(times[late, 1] + 3, times[late, 0])
    assert detected[late, 1].all()


def test_kaplan_meier_matches_scipy_and_reports_an_unreached_median():
    times = np.array([[1, 3, 3, 5, 14, 14, 2, 7]])
    events = np.array([[True, True, False, True, False, False, True, False]])
    survival = kaplan_meier(times, events, 14)
    reference = stats.ecdf(
        stats.CensoredData(uncensored=times[0][events[0]], right=times[0][~events[0]])
    )
    for day in range(1, 15):
        assert survival[0, day - 1] == pytest.approx(reference.sf.evaluate(day))
    assert median_delay(survival)[0] == 5
    few = kaplan_meier(np.array([[1, 14, 14, 14]]), np.array([[True, False, False, False]]), 14)
    assert few[0, -1] == pytest.approx(0.75) and np.isnan(median_delay(few))[0]


def test_a_survival_of_one_half_in_exact_arithmetic_sets_the_median_on_that_day():
    daily = [10, 7, 5, 4, 13, 14, 10, 11, 9, 5, 12]
    times = np.concatenate([np.repeat(np.arange(1, 12), daily), np.full(100, 14)])[None, :]
    events = np.concatenate([np.ones(100, dtype=bool), np.zeros(100, dtype=bool)])[None, :]
    survival = kaplan_meier(times, events, 14)
    assert survival[0, 10] > 0.5
    assert median_delay(survival)[0] == 11


def test_exceedance_weights_days_by_their_rebuilt_counts():
    alarms = np.array([[True, False, False], [False, False, True]])
    monitored = np.array([[True, True, False], [True, True, True]])
    slots = np.array([[0, 0, 2, -1], [1, 2, 2, 2]])
    counts = day_counts(slots, 3)
    assert counts.tolist() == [[2, 0, 1], [0, 1, 3]]
    assert exceedance(alarms, monitored, counts).tolist() == pytest.approx([3 / 5, 3 / 5])


def test_injected_copies_and_observed_episodes():
    copies = injections(["A", "B"], window_days=30, copies=50, episode_days=14, seed=1)
    assert copies.height == 50 and set(copies["turbine"]) <= {"A", "B"}
    assert copies["offset"].is_between(0, 16).all()
    assert copies.equals(injections(["A", "B"], 30, 50, 14, seed=1))
    alarms = np.zeros((2, 30), dtype=bool)
    alarms[0, 6] = True
    one = pl.DataFrame({"injection": [0, 1], "turbine": ["A", "B"], "offset": [2, 2]})
    table = episodes({0.05: alarms}, one, ["A", "B"], date(2020, 1, 1), 14)
    assert table.schema == EPISODES
    assert table["delay_days"].to_list() == [5, 14]
    assert table["censored"].to_list() == [False, True]
    assert table["first_alarm"].to_list() == [date(2020, 1, 7), None]


def test_recall_and_exceedance_pool_starts_like_a_loop():
    rng = np.random.default_rng(30)
    days, turbines = 28, ["A", "B"]
    grids, copies = [], []
    for start in range(2):
        monitored = rng.random((2, days)) < 0.9
        z = rng.normal(0, 1, (2, days))
        sd = np.full((2, days), 0.05)
        grids.append((monitored, z, sd))
        copies.append(injections(turbines, days, 6, 14, seed=start))
    quantile, drop = alarm_quantile(1.0), 0.05
    slots = trajectories(draw_blocks({"w": 4}, 40, seed=2)["w"], days, 7)
    recall, exceed = recall_and_exceedance(grids, copies, turbines, slots, quantile, drop, 14)
    for r in range(40):
        times, events, alarms, monitored_days = [], [], 0, 0
        for (monitored, z, sd), start_copies in zip(grids, copies, strict=True):
            shifted = monitored & (z + math.log(1 - drop) / sd < quantile)
            plain = monitored & (z < quantile)
            for turbine, offset in zip(
                start_copies["turbine"], start_copies["offset"], strict=True
            ):
                time, event = first_alarm_by_loop(
                    shifted, turbines.index(turbine), offset, slots[r], 14
                )
                times.append(time)
                events.append(event)
            for day in slots[r][slots[r] >= 0]:
                alarms += plain[:, day].sum()
                monitored_days += monitored[:, day].sum()
        times, events = np.array(times), np.array(events)
        reference = stats.ecdf(stats.CensoredData(uncensored=times[events], right=times[~events]))
        assert recall[r] == pytest.approx(1 - reference.sf.evaluate(14))
        assert exceed[r] == pytest.approx(alarms / monitored_days)
