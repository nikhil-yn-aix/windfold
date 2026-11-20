from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from windfold import splits
from windfold.config import load_farm, load_protocol
from windfold.schemas import MANIFEST, PREPARED_ROWS, conform

STEP = timedelta(minutes=10)
START = datetime(2020, 1, 1, tzinfo=UTC)


def at(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def prepared(
    timestamps: list[datetime],
    directions: list[float],
    speed_bins: list[int],
    counts: list[int],
    period: str = "development",
) -> pl.DataFrame:
    """prepared rows with counts[i] turbines at timestamps[i]; turbine 0 is the waked one."""
    records = [
        (t, f"T{i}", i, period, d, b, i == 0, m)
        for t, d, b, m in zip(timestamps, directions, speed_bins, counts, strict=True)
        for i in range(m)
    ]
    frame = pl.DataFrame(
        records,
        schema=[
            "timestamp",
            "turbine",
            "turbine_index",
            "period",
            "farm_direction",
            "speed_bin",
            "waked",
            "valid_turbines",
        ],
        orient="row",
    )
    filled = frame.with_columns(
        power_kw=pl.lit(1000.0),
        speed=pl.lit(7.0),
        speed_std=pl.lit(0.5),
        direction_raw=pl.col("farm_direction"),
        direction=pl.col("farm_direction"),
        nacelle_position=pl.col("farm_direction"),
        reference_speed=pl.lit(7.0),
        direction_bucket=(pl.col("farm_direction") // 5),
        steady=pl.lit(True),
        operating_mask=pl.lit(0),
        y=pl.lit(0.0),
        temperature=pl.lit(10.0),
        turbulence_intensity=pl.lit(None),
    )
    return conform(filled.cast(PREPARED_ROWS), PREPARED_ROWS)


def grid(start: datetime, end: datetime, every: timedelta = STEP) -> list[datetime]:
    count = (end - start) // every
    return [start + every * k for k in range(count)]


def uniform(timestamps: list[datetime], direction: float = 10.0, rows: int = 2) -> pl.DataFrame:
    n = len(timestamps)
    return prepared(timestamps, [direction] * n, [0] * n, [rows] * n)


def table_of(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return splits.timestamp_table(frame, "development", start, end)


def roles(manifest: pl.DataFrame, split: str) -> dict[str, list[datetime]]:
    chosen = manifest.filter(pl.col("split") == split)
    return {
        role: chosen.filter(pl.col("role") == role)["timestamp"].to_list()
        for role in chosen["role"].unique()
    }


def times(manifest: pl.DataFrame, split: str, role: str) -> np.ndarray:
    chosen = manifest.filter((pl.col("split") == split) & (pl.col("role") == role))
    return chosen["timestamp"].to_numpy()


def angular_gap(direction: np.ndarray, start: float, width: float) -> np.ndarray:
    """degrees from each direction to the arc [start, start + width), zero inside."""
    offset = np.mod(direction - start, 360.0)
    outside = np.minimum(offset - width, 360.0 - offset)
    return np.where(offset < width, 0.0, outside)


@pytest.mark.parametrize(
    ("start", "end", "edges"),
    [
        (
            at(2017, 3, 1),
            at(2020, 1, 1),
            [at(2017, 3, 1), at(2017, 10, 1), at(2018, 5, 1), at(2018, 12, 1), at(2019, 7, 1)],
        ),
        (
            at(2018, 1, 1),
            at(2025, 1, 1),
            [at(2018, 1, 1), at(2019, 6, 1), at(2020, 11, 1), at(2022, 4, 1), at(2023, 9, 1)],
        ),
        (
            at(2023, 9, 1),
            at(2025, 10, 28, 13),
            [at(2023, 9, 1), at(2024, 3, 1), at(2024, 8, 1), at(2025, 1, 1), at(2025, 6, 1)],
        ),
        (
            at(2020, 1, 1),
            at(2024, 1, 1),
            [at(2020, 1, 1), at(2020, 11, 1), at(2021, 9, 1), at(2022, 7, 1), at(2023, 4, 1)],
        ),
        (
            at(2024, 3, 14, 17),
            at(2025, 10, 28, 13),
            [at(2024, 3, 14, 17), at(2024, 7, 1), at(2024, 11, 1), at(2025, 3, 1), at(2025, 7, 1)],
        ),
    ],
)
def test_month_aligned_edges_split_whole_months_evenly(start, end, edges):
    assert splits.time_block_edges(start, end, 5, month_aligned=True) == [*edges, end]


def test_equal_blocks_split_the_window_into_equal_lengths():
    edges = splits.time_block_edges(
        at(2020, 2, 17, 16, 30), at(2020, 5, 25), 4, month_aligned=False
    )
    assert edges == [
        at(2020, 2, 17, 16, 30),
        datetime(2020, 3, 13, 0, 22, 30, tzinfo=UTC),
        at(2020, 4, 6, 8, 15),
        datetime(2020, 4, 30, 16, 7, 30, tzinfo=UTC),
        at(2020, 5, 25),
    ]


def test_too_few_months_for_the_folds_fails():
    with pytest.raises(ValueError, match="spans 4 months, fewer than 5 folds"):
        splits.time_block_edges(at(2020, 2, 17), at(2020, 5, 25), 5, month_aligned=True)


def test_time_block_buffers_on_a_full_grid():
    stamps = grid(at(2020, 1, 1), at(2020, 4, 1))
    table = table_of(uniform(stamps), at(2020, 1, 1), at(2020, 4, 1))
    manifest, summary = splits.time_blocks(
        table, start=at(2020, 1, 1), end=at(2020, 4, 1), folds=3, buffer_days=7, month_aligned=True
    )
    february = roles(manifest, "time_2")
    assert (min(february["test"]), max(february["test"])) == (
        at(2020, 2, 1),
        at(2020, 2, 29, 23, 50),
    )
    assert max(t for t in february["train"] if t < at(2020, 2, 1)) == at(2020, 1, 24, 23, 50)
    assert min(t for t in february["train"] if t > at(2020, 2, 1)) == at(2020, 3, 8)
    assert len(february["buffer"]) == 2 * 7 * 144
    assert summary["test_timestamps"].to_list() == [31 * 144, 29 * 144, 31 * 144]
    assert summary["test_rows"].to_list() == [2 * 31 * 144, 2 * 29 * 144, 2 * 31 * 144]
    assert summary["test_days"].to_list() == [31, 29, 31]


@st.composite
def time_cases(draw):
    first = START + STEP * draw(st.integers(0, 144 * 40))
    span_days = draw(st.integers(60, 400))
    end = first + timedelta(days=span_days, minutes=10 * draw(st.integers(0, 143)))
    offsets = draw(st.sets(st.integers(0, (end - first) // STEP - 1), min_size=10, max_size=400))
    month_aligned = draw(st.booleans())
    folds = draw(
        st.integers(2, max(2, min(6, span_days // 31))) if month_aligned else st.integers(2, 6)
    )
    buffer_days = draw(st.integers(0, 10))
    return first, end, sorted(first + STEP * o for o in offsets), folds, buffer_days, month_aligned


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(time_cases())
def test_no_test_timestamp_inside_a_training_buffer(case):
    first, end, stamps, folds, buffer_days, month_aligned = case
    table = table_of(uniform(stamps), first, end)
    manifest, summary = splits.time_blocks(
        table,
        start=first,
        end=end,
        folds=folds,
        buffer_days=buffer_days,
        month_aligned=month_aligned,
    )
    tested = []
    for k in range(folds):
        test = times(manifest, f"time_{k + 1}", "test")
        train = times(manifest, f"time_{k + 1}", "train")
        tested += test.tolist()
        if test.size:
            gap = np.abs(train[:, None] - test[None, :]).min()
            assert gap >= np.timedelta64(timedelta(days=buffer_days) + STEP)
    assert len(tested) == len(set(tested))
    assert summary["test_timestamps"].sum() == len(stamps)


def test_random_folds_test_every_timestamp_once():
    stamps = grid(at(2020, 1, 1), at(2020, 1, 8))
    table = table_of(uniform(stamps), at(2020, 1, 1), at(2020, 2, 1))
    manifest, summary = splits.random_folds(table, folds=5, seed=7)
    tested = manifest.filter(pl.col("role") == "test")
    assert sorted(tested["timestamp"].to_list()) == stamps
    sizes = summary["test_timestamps"].to_list()
    assert max(sizes) - min(sizes) <= 1
    assert (summary["train_timestamps"] + summary["test_timestamps"]).to_list() == [len(stamps)] * 5
    again, _ = splits.random_folds(table, folds=5, seed=7)
    other, _ = splits.random_folds(table, folds=5, seed=8)
    assert again.equals(manifest)
    assert not other.equals(manifest)


def arc_table(directions, speed_bins=None, counts=None, days=None) -> pl.DataFrame:
    """one timestamp per direction, on the given day of 2020 (default: spread over 28 days)."""
    n = len(directions)
    days = days or [k % 28 for k in range(n)]
    stamps = [at(2020, 1, 1) + timedelta(days=d, minutes=k) for k, d in enumerate(days)]
    frame = prepared(stamps, directions, speed_bins or [0] * n, counts or [3] * n)
    return table_of(frame, at(2020, 1, 1), at(2023, 1, 1))


def stamp_of(table: pl.DataFrame) -> dict[float, datetime]:
    return dict(zip(table["direction"].to_list(), table["timestamp"].to_list(), strict=True))


def run_arcs(
    table: pl.DataFrame,
    count: int = 12,
    width: float = 30.0,
    buffer: float = 5.0,
    min_rows: int = 1,
    seed: int = 3,
):
    return splits.arcs(
        table,
        count=count,
        width=width,
        buffer=buffer,
        min_rows=min_rows,
        min_days=1,
        tolerance=0.05,
        seed=seed,
    )


def test_arc_roles_at_the_edges():
    directions = [120.0, 149.9, 150.0, 117.0, 115.0, 114.9, 155.0, 300.0, 295.0, 334.9, 335.0]
    table = arc_table(directions)
    manifest, summary = run_arcs(table)
    time_of = stamp_of(table)
    single = roles(manifest, "arc_120_single")
    assert sorted(single["test"]) == sorted(time_of[d] for d in (120.0, 149.9))
    assert sorted(single["buffer"]) == sorted(time_of[d] for d in (150.0, 117.0, 115.0))
    assert sorted(single["train"]) == sorted(
        time_of[d] for d in (114.9, 155.0, 300.0, 295.0, 334.9, 335.0)
    )
    opposite = roles(manifest, "arc_120_opposite")
    assert opposite["test"] == single["test"] and opposite["buffer"] == single["buffer"]
    assert sorted(opposite["train"]) == sorted(time_of[d] for d in (114.9, 155.0, 335.0))
    row = summary.filter(pl.col("split") == "arc_120_opposite").row(0, named=True)
    assert (row["removed_rows"], row["test_rows"], row["test_timestamps"]) == (9, 6, 2)


def test_control_removes_the_matched_stratum_rows():
    directions = [130.0, 131.0, 310.0, 311.0, 320.0, 60.0, 61.0, 200.0, 201.0, 202.0, 30.0]
    speed_bins = [0, 0, 1, 1, 2, 1, 1, 2, 2, 2, 0]
    counts = [3, 3, 3, 4, 6, 4, 3, 6, 6, 6, 5]
    table = arc_table(directions, speed_bins, counts, days=[0] * 11)
    manifest, summary = run_arcs(table)
    time_of = stamp_of(table)
    control = roles(manifest, "arc_120_control")
    removed = set(roles(manifest, "arc_120_single")["train"]) - set(control["train"])
    assert {time_of[60.0], time_of[61.0]} <= removed
    assert len(removed & {time_of[d] for d in (200.0, 201.0, 202.0)}) == 1
    assert len(removed) == 3
    assert control["test"] == roles(manifest, "arc_120_single")["test"]
    rows = summary.filter(pl.col("split").is_in(["arc_120_opposite", "arc_120_control"]))
    assert rows["removed_rows"].to_list() == [13, 13]


def test_control_stops_at_the_whole_timestamp_closest_to_each_target():
    opposite = [310.0, 311.0] + [312.0 + k for k in range(20)]
    pool = [60.0, 61.0] + [200.0 + k for k in range(20)]
    directions = [130.0, *opposite, *pool]
    speed_bins = [0] + [1, 1] + [2] * 20 + [1, 1] + [2] * 20
    counts = [3] + [3, 4] + [5] * 20 + [6, 6] + [5] * 20
    table = arc_table(directions, speed_bins, counts, days=[0] * len(directions))
    manifest, summary = run_arcs(table)
    time_of = stamp_of(table)
    single = set(roles(manifest, "arc_120_single")["train"])
    removed = single - set(roles(manifest, "arc_120_control")["train"])
    assert len(removed & {time_of[60.0], time_of[61.0]}) == 1
    assert {time_of[d] for d in pool[2:]} <= removed
    rows = summary.filter(pl.col("split").is_in(["arc_120_opposite", "arc_120_control"]))
    assert rows["removed_rows"].to_list() == [107, 106]


def test_control_without_enough_rows_in_a_stratum_is_declared_unmatched():
    directions = [130.0, 310.0, 311.0, 312.0, 60.0, 61.0, 200.0]
    speed_bins = [0, 1, 1, 1, 1, 1, 0]
    counts = [3, 4, 4, 4, 4, 4, 6]
    table = arc_table(directions, speed_bins, counts, days=[40, 40, 41, 42, 43, 44, 45])
    manifest, summary = run_arcs(table)
    assert "arc_120_control" not in manifest["split"].to_list()
    assert {"arc_120_single", "arc_120_opposite"} <= set(manifest["split"].to_list())
    row = summary.filter(pl.col("split") == "arc_120_control").row(0, named=True)
    assert row["status"] == "unmatched"
    assert "speed bin 1 in month 2 needs 12, has 8" in row["reason"]


def single_stratum(removed_sizes: list[int], pool_sizes: list[int]):
    rows = np.array(removed_sizes + pool_sizes)
    removed = np.arange(rows.size) < len(removed_sizes)
    return ~removed, removed, rows, np.zeros(rows.size, dtype=np.int64), ["speed bin 0 in month 1"]


def test_a_stratum_short_by_less_than_the_tolerance_is_still_matched():
    pool, removed, rows, strata, labels = single_stratum([6] * 50, [6] * 49)
    chosen = splits.control_sample(
        pool, removed, rows, strata, labels, 0.05, np.random.default_rng(20261116)
    )
    assert not isinstance(chosen, str)
    assert rows[chosen].sum() == 294 and (chosen == pool).all()
    pool, removed, rows, strata, labels = single_stratum([6] * 50, [6] * 47)
    reason = splits.control_sample(
        pool, removed, rows, strata, labels, 0.05, np.random.default_rng(20261116)
    )
    assert isinstance(reason, str) and "needs 300, has 282" in reason


@pytest.mark.parametrize("seed", [20261116, 0])
def test_a_control_refills_from_the_remaining_timestamps_before_giving_up(seed):
    pool, removed, rows, strata, labels = single_stratum([4], [6, 4])
    chosen = splits.control_sample(
        pool, removed, rows, strata, labels, 0.05, np.random.default_rng(seed)
    )
    assert not isinstance(chosen, str)
    assert chosen.tolist() == [False, False, True]


def test_small_arcs_are_declared_ineligible_with_their_counts():
    table = arc_table([125.0] * 5 + [10.0] * 20, counts=[3] * 25)
    manifest, summary = run_arcs(table, min_rows=16)
    assert not manifest["split"].str.starts_with("arc_120").any()
    declared = summary.filter(pl.col("split").str.starts_with("arc_120"))
    assert declared["status"].to_list() == ["ineligible"] * 3
    assert declared["test_rows"].to_list() == [15] * 3
    assert declared["test_days"].to_list() == [5] * 3
    assert "15 rows on 5 days" in declared["reason"][0]


@st.composite
def arc_cases(draw):
    n = draw(st.integers(20, 80))
    prevailing = draw(st.floats(0, 359))
    spread = draw(st.floats(5, 120))
    share = draw(st.floats(0, 1))
    directions = [
        (prevailing + draw(st.floats(-spread, spread))) % 360
        if draw(st.floats(0, 1)) < share
        else draw(st.floats(0, 359.999))
        for _ in range(n)
    ]
    speed_bins = draw(st.lists(st.integers(0, 2), min_size=n, max_size=n))
    counts = draw(st.lists(st.integers(1, 6), min_size=n, max_size=n))
    days = draw(st.lists(st.integers(0, 800), min_size=n, max_size=n))
    buffer = draw(st.sampled_from([0.0, 5.0, 10.0]))
    count = draw(st.sampled_from([12, 6]))
    return directions, speed_bins, counts, days, buffer, count


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(arc_cases())
def test_arc_variants_keep_their_contract(case):
    directions, speed_bins, counts, days, buffer, count = case
    width, tolerance = 360.0 / count, 0.05
    table = arc_table(directions, speed_bins, counts, days)
    manifest, summary = run_arcs(table, count=count, width=width, buffer=buffer, seed=11)
    direction = table["direction"].to_numpy()
    rows = table["rows"].to_numpy()
    stamps = table["timestamp"].to_numpy()
    stratum = (
        table.select(pl.col("timestamp").dt.month() * 3 + pl.col("speed_bin"))
        .to_series()
        .to_numpy()
    )
    index = {t: k for k, t in enumerate(stamps)}

    def mask(split: str, role: str) -> np.ndarray:
        chosen = manifest.filter((pl.col("split") == split) & (pl.col("role") == role))
        out = np.zeros(len(stamps), dtype=bool)
        out[[index[t] for t in chosen["timestamp"].to_numpy()]] = True
        return out

    tested = np.zeros(len(stamps), dtype=int)
    for k in range(count):
        start = k * width
        name = f"arc_{round(start):03d}"
        if f"{name}_single" not in manifest["split"].to_list():
            continue
        test, train = mask(f"{name}_single", "test"), mask(f"{name}_single", "train")
        edge = mask(f"{name}_single", "buffer")
        tested += test
        gap = angular_gap(direction, start, width)
        assert (gap[test] == 0).all()
        assert (gap[train] >= buffer - 1e-9).all() and (gap[edge] <= buffer + 1e-9).all()
        assert (test | train | edge).all()
        opposite_train = mask(f"{name}_opposite", "train")
        opposite_gap = angular_gap(direction, start + 180.0, width)
        assert not (opposite_train & ~train).any()
        assert not (opposite_train & (opposite_gap < buffer - 1e-9)).any()
        assert (opposite_train | ~train | (opposite_gap <= buffer + 1e-9)).all()
        assert np.array_equal(mask(f"{name}_opposite", "test"), test)
        removed = train & ~opposite_train
        status = summary.filter(pl.col("split") == f"{name}_control")["status"][0]
        target = np.bincount(stratum[removed], weights=rows[removed], minlength=40)
        pool = train & ~removed
        available = np.bincount(stratum[pool], weights=rows[pool], minlength=40)
        wanted = rows[removed].sum()
        if status == "included":
            control_train = mask(f"{name}_control", "train")
            assert np.array_equal(mask(f"{name}_control", "test"), test)
            assert not (control_train & ~train).any()
            taken = train & ~control_train
            assert not (taken & removed).any()
            got = np.bincount(stratum[taken], weights=rows[taken], minlength=40)
            assert abs(rows[taken].sum() - wanted) <= tolerance * wanted
            short = available < target
            assert (np.abs(got - target)[~short] <= rows.max() / 2).all()
            assert (got[short] == available[short]).all()
            assert (available[short] >= (1 - tolerance) * target[short]).all()
        else:
            assert status == "unmatched"
            feasible = (available >= (1 - tolerance) * target).all()
            shortfall = np.maximum(target - available, 0).sum()
            rounding = np.count_nonzero((target > 0) & (available >= target)) * rows.max() / 2
            assert not (feasible and shortfall + rounding <= tolerance * wanted)
    assert (tested == 1).all()


def test_histories_train_on_their_window_and_test_on_the_common_one():
    stamps = grid(at(2020, 1, 1), at(2020, 4, 1), timedelta(hours=1))
    table = table_of(uniform(stamps), at(2020, 1, 1), at(2020, 4, 1))
    manifest, summary = splits.histories(
        table,
        starts=[at(2020, 1, 1), at(2020, 1, 15)],
        horizons=[7, 14, 30],
        test_start=at(2020, 3, 1),
        test_end=at(2020, 4, 1),
        gap_days=7,
        min_timestamps=300,
    )
    split = roles(manifest, "history_014d_from_2020-01-15")
    assert (min(split["train"]), max(split["train"])) == (at(2020, 1, 15), at(2020, 1, 28, 23))
    assert (min(split["test"]), max(split["test"])) == (at(2020, 3, 1), at(2020, 3, 31, 23))
    short = summary.filter(pl.col("split").str.starts_with("history_007d"))
    assert short["status"].to_list() == ["insufficient data"] * 2
    assert short["train_timestamps"].to_list() == [168, 168]
    assert "history_007d_from_2020-01-01" not in manifest["split"].to_list()
    assert summary.filter(pl.col("status") == "included").height == 4


def test_a_history_reaching_into_the_test_gap_fails():
    table = table_of(uniform(grid(at(2020, 1, 1), at(2020, 1, 2))), at(2020, 1, 1), at(2021, 1, 1))
    with pytest.raises(ValueError, match="less than 7 days before the test window"):
        splits.histories(
            table,
            starts=[at(2020, 1, 1)],
            horizons=[30],
            test_start=at(2020, 2, 6),
            test_end=at(2020, 3, 1),
            gap_days=7,
            min_timestamps=1,
        )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.sets(st.integers(0, 144 * 120 - 1), min_size=5, max_size=300),
    st.lists(st.integers(1, 60), min_size=1, max_size=4, unique=True),
    st.integers(0, 10),
)
def test_histories_never_touch_the_test_window(offsets, horizons, gap_days):
    stamps = sorted(START + STEP * o for o in offsets)
    table = table_of(uniform(stamps), START, START + timedelta(days=120))
    test_start = START + timedelta(days=max(horizons) + gap_days + 1)
    manifest, _ = splits.histories(
        table,
        starts=[START],
        horizons=horizons,
        test_start=test_start,
        test_end=START + timedelta(days=120),
        gap_days=gap_days,
        min_timestamps=1,
    )
    train = manifest.filter(pl.col("role") == "train")["timestamp"].to_list()
    test = manifest.filter(pl.col("role") == "test")["timestamp"].to_list()
    assert all(t + STEP <= test_start - timedelta(days=gap_days) for t in train)
    assert all(t >= test_start for t in test)


def test_threshold_windows_of_a_thirty_day_history():
    stamps = grid(at(2020, 1, 1), at(2020, 2, 1))
    table = table_of(uniform(stamps), at(2020, 1, 1), at(2020, 3, 1))
    manifest, summary = splits.thresholds(
        table,
        starts=[at(2020, 1, 1)],
        horizons=[14, 30],
        min_history_days=30,
        fit_fraction=0.75,
        calibrate_from_fraction=0.8,
        min_timestamps=300,
    )
    assert summary["split"].to_list() == ["threshold_030d_from_2020-01-01"]
    split = roles(manifest, "threshold_030d_from_2020-01-01")
    assert (min(split["fit"]), max(split["fit"])) == (at(2020, 1, 1), at(2020, 1, 23, 11, 50))
    assert (min(split["buffer"]), max(split["buffer"])) == (
        at(2020, 1, 23, 12),
        at(2020, 1, 24, 23, 50),
    )
    assert (min(split["calibrate"]), max(split["calibrate"])) == (
        at(2020, 1, 25),
        at(2020, 1, 30, 23, 50),
    )


def test_prepared_rows_that_disagree_within_a_timestamp_fail():
    frame = prepared([at(2020, 1, 1)] * 2, [10.0, 12.0], [0, 0], [1, 1])
    with pytest.raises(ValueError, match="disagree on farm_direction or speed_bin"):
        table_of(frame, at(2020, 1, 1), at(2020, 2, 1))


def test_rows_of_other_periods_stay_out_and_outside_rows_fail():
    calibration = prepared([at(2019, 12, 31)], [10.0], [0], [2], period="calibration")
    development = prepared([at(2020, 1, 1)], [10.0], [0], [3])
    table = table_of(pl.concat([calibration, development]), at(2020, 1, 1), at(2020, 2, 1))
    assert table["timestamp"].to_list() == [at(2020, 1, 1)]
    assert table["rows"].to_list() == [3] and table["waked_rows"].to_list() == [1]
    with pytest.raises(ValueError, match="outside the window"):
        table_of(development, at(2020, 1, 2), at(2020, 2, 1))


def synthetic_farm(start: datetime, end: datetime, every: timedelta, period: str) -> pl.DataFrame:
    stamps = grid(start, end, every)
    rng = np.random.default_rng(5)
    directions = (rng.normal(230.0, 70.0, len(stamps)) % 360).tolist()
    speed_bins = rng.integers(0, 3, len(stamps)).tolist()
    counts = rng.integers(2, 5, len(stamps)).tolist()
    return prepared(stamps, directions, speed_bins, counts, period=period)


def test_smarteole_manifest_follows_its_farm_config():
    farm, protocol = load_farm("smarteole"), load_protocol()
    window = farm.windows.development
    rows = synthetic_farm(window.start, window.end, STEP, "development")
    manifest, summary = splits.farm_manifest(
        rows, farm, protocol.splits, protocol.monitoring, "development", seed=1
    )
    assert manifest.schema == MANIFEST
    names = summary["split"].to_list()
    assert sum(n.startswith("time_") for n in names) == 4
    assert sum(n.startswith("arc_") for n in names) == 36
    assert sum(n.startswith("history_") for n in names) == 4
    assert [n for n in names if n.startswith("threshold_")] == [
        "threshold_030d_from_2020-02-17",
        "threshold_060d_from_2020-02-17",
    ]
    included = summary.filter(pl.col("status") == "included")["split"].to_list()
    assert sorted(set(manifest["split"].to_list())) == sorted(included)
    time_2 = roles(manifest, "time_2")
    assert min(time_2["test"]) == at(2020, 3, 13, 0, 30)
    assert max(t for t in time_2["train"] if t < at(2020, 3, 13)) == at(2020, 3, 10, 0, 20)


def test_confirmatory_manifest_keeps_2023_out_of_every_history():
    farm, protocol = load_farm("penmanshiel"), load_protocol()
    window = farm.windows.confirmatory
    assert window is not None
    rows = synthetic_farm(window.start, window.end, timedelta(hours=1), "confirmatory")
    manifest, summary = splits.farm_manifest(
        rows, farm, protocol.splits, protocol.monitoring, "confirmatory", seed=1
    )
    learning = manifest.filter(pl.col("regime").is_in(["history", "threshold"]))
    fitted = learning.filter(pl.col("role").is_in(["train", "fit", "calibrate", "buffer"]))
    assert (fitted["timestamp"] < at(2022, 1, 1)).all()
    tested = learning.filter(pl.col("role") == "test")["timestamp"]
    assert (tested.min(), tested.max()) == (at(2022, 1, 1), at(2022, 12, 31, 23))
    assert manifest["timestamp"].is_between(window.start, window.end, closed="left").all()
    short = summary.filter(pl.col("split").str.starts_with("history_007d"))
    assert short["status"].to_list() == ["insufficient data"] * 4
    blocks = summary.filter(pl.col("regime") == "time_blocked")["test_days"].to_list()
    assert blocks == [305, 304, 303, 274, 275]


@pytest.mark.parametrize(
    ("sizes", "goal", "positions"),
    [
        ([6, 4], 4, [1]),
        ([3, 5], 4, [1]),
        ([2, 2, 2, 6], 6, [0, 1, 2]),
        ([6, 6], 4, [0]),
        ([5, 5], 2, []),
    ],
)
def test_closest_subset_prefers_exact_then_earliest_then_larger(sizes, goal, positions):
    assert splits.closest_subset(np.array(sizes), goal) == positions
