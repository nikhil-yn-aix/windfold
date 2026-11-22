import os
import subprocess
import sys
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from windfold.evaluation import bootstrap
from windfold.evaluation.results import bootstrap_windows


def test_blocks_count_whole_steps_from_the_window_start():
    assert bootstrap.block_count(date(2020, 1, 1), date(2021, 1, 1), 7) == 53
    assert bootstrap.block_count(date(2020, 4, 25), date(2020, 5, 25), 7) == 5
    days = pl.Series([date(2020, 1, 1), date(2020, 1, 7), date(2020, 1, 8), date(2020, 12, 31)])
    assert bootstrap.block_of(days, date(2020, 1, 1), 7).tolist() == [0, 0, 1, 52]
    assert bootstrap.block_of(days, date(2020, 1, 1), 14).tolist() == [0, 0, 0, 26]


def test_a_window_counts_its_last_partial_day():
    start = datetime(2023, 9, 1, tzinfo=UTC)
    assert bootstrap.window_days(start, datetime(2025, 10, 28, 13, tzinfo=UTC)) == (
        date(2023, 9, 1),
        789,
    )
    assert bootstrap.window_days(start, datetime(2023, 9, 3, tzinfo=UTC)) == (date(2023, 9, 1), 2)


def test_one_uniform_draw_serves_every_window():
    drawn = bootstrap.draw_blocks({"period": 365, "test": 53, "same": 365}, 400, seed=9)
    assert drawn["period"].shape == (400, 365) and drawn["test"].shape == (400, 53)
    assert drawn["period"].min() == 0 and drawn["period"].max() == 364
    assert drawn["test"].min() == 0 and drawn["test"].max() == 52
    assert np.array_equal(drawn["period"], drawn["same"])
    alone = bootstrap.draw_blocks({"test": 53}, 400, seed=9)
    assert np.array_equal(alone["test"], drawn["test"])
    assert not np.array_equal(
        bootstrap.draw_blocks({"test": 53}, 400, seed=10)["test"], drawn["test"]
    )
    seeds = bootstrap.draw_seeds(400, 3, seed=9)
    assert seeds.shape == (400, 3) and set(np.unique(seeds)) == {0, 1, 2}


def test_a_farm_seed_follows_the_study_seed_and_farm_name_in_any_process_and_order():
    farms = ["kelmarsh", "penmanshiel", "smarteole"]
    seeds = [bootstrap.farm_seed(20261116, farm) for farm in farms]
    assert len(set(seeds)) == 3 and bootstrap.farm_seed(20261117, farms[0]) != seeds[0]
    assert [bootstrap.farm_seed(20261116, farm) for farm in reversed(farms)] == seeds[::-1]
    script = (
        "from windfold.evaluation.bootstrap import farm_seed\n"
        "print(farm_seed(20261116, 'kelmarsh'))"
    )
    for hash_seed in ("1", "2"):
        run = subprocess.run(
            [sys.executable, "-c", script],
            env=os.environ | {"PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(run.stdout) == seeds[0]


def test_every_window_of_a_farm_shares_one_seed_draw_that_its_seed_alone_fixes():
    spans = {
        "period": (datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 3, 1, tzinfo=UTC)),
        "test": (datetime(2020, 2, 1, tzinfo=UTC), datetime(2020, 3, 1, tzinfo=UTC)),
    }
    windows = bootstrap_windows(spans, 7, 50, 3, 0.95, 0.95, seed=9)
    assert np.array_equal(windows.draws["period"].seeds, windows.draws["test"].seeds)
    again = bootstrap_windows(spans, 7, 50, 3, 0.95, 0.95, seed=9)
    assert np.array_equal(again.seeds, windows.seeds)
    other = bootstrap_windows(spans, 7, 50, 3, 0.95, 0.95, seed=10)
    assert not np.array_equal(other.seeds, windows.seeds)
    assert not np.array_equal(other.drawn["period"], windows.drawn["period"])


def test_multiplicity_counts_each_drawn_block():
    drawn = np.array([[0, 0, 2], [1, 2, 2]])
    assert bootstrap.multiplicity(drawn, 3).tolist() == [[2, 0, 1], [0, 1, 2]]


def reduction() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": ["a", "a", "a", "b", "b"],
            "split": ["s", "s", "t", "s", "s"],
            "seed": [None, None, None, 0, 1],
            "day": [
                date(2020, 1, 1),
                date(2020, 1, 9),
                date(2020, 1, 2),
                date(2020, 1, 15),
                date(2020, 1, 3),
            ],
            "sse": [1.0, 2.0, 4.0, 8.0, 16.0],
            "rows": [10, 20, 40, 80, 160],
        },
        schema_overrides={"seed": pl.Int64},
    )


def test_block_totals_sum_days_into_blocks_per_key():
    totals = bootstrap.block_totals(
        reduction(), ["model", "split", "seed"], ["sse", "rows"], date(2020, 1, 1), 7, 3
    )
    assert totals.values.shape == (3, 8)
    a_s = totals.columns("sse", model="a", split="s", seed=None)
    assert totals.values[:, a_s].ravel().tolist() == [1.0, 2.0, 0.0]
    b_rows = totals.columns("rows", model="b", split=["s"])
    assert totals.values[:, b_rows].sum(axis=1).tolist() == [160.0, 0.0, 80.0]
    with pytest.raises(ValueError, match="outside"):
        bootstrap.block_totals(reduction(), ["model"], ["sse"], date(2020, 1, 1), 7, 2)


def test_resampled_totals_equal_a_loop_over_drawn_blocks():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(6, 4))
    drawn = bootstrap.draw_blocks({"w": 6}, 50, seed=2)["w"]
    fast = bootstrap.resample(values, bootstrap.multiplicity(drawn, 6), chunk=7)
    slow = np.array([[values[drawn[r], k].sum() for k in range(4)] for r in range(50)])
    assert fast == pytest.approx(slow)


def test_percentile_intervals_cover_a_known_effect_near_the_nominal_rate():
    rng = np.random.default_rng(11)
    blocks, days, effect, simulations, replicates = 40, 7, 0.3, 400, 1000
    counts = bootstrap.multiplicity(
        bootstrap.draw_blocks({"w": blocks}, replicates, seed=3)["w"], blocks
    )
    covered = 0
    for _ in range(simulations):
        shared = rng.normal(0, 1.0, size=blocks)
        daily = effect + shared[:, None] + rng.normal(0, 1.0, size=(blocks, days))
        sums = np.column_stack([daily.sum(axis=1), np.full(blocks, days)])
        totals = bootstrap.resample(sums, counts)
        means = totals[:, 0] / totals[:, 1]
        lower, upper = np.quantile(means, [0.025, 0.975])
        covered += lower <= effect <= upper
    assert 0.9 <= covered / simulations <= 0.98
