from datetime import UTC, datetime, timedelta

import polars as pl

from windfold.analyses.arc_variants import shifted_arcs, time_buffered

START = datetime(2020, 1, 1, 12, tzinfo=UTC)


def manifest(entries: list[tuple[int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "split": ["arc_000_single"] * len(entries),
            "regime": ["arc_single"] * len(entries),
            "timestamp": [START + timedelta(minutes=m) for m, _ in entries],
            "role": [role for _, role in entries],
        },
        schema={
            "split": pl.String,
            "regime": pl.String,
            "timestamp": pl.Datetime("us", "UTC"),
            "role": pl.String,
        },
    )


def test_time_buffer_moves_only_training_rows_near_a_test_row():
    entries = [(-70, "train"), (-60, "train"), (-30, "buffer"), (0, "test"), (10, "test")]
    entries += [(70, "train"), (80, "train")]
    roles = time_buffered(manifest(entries), timedelta(hours=1))["role"].to_list()
    assert roles == ["train", "buffer", "buffer", "test", "test", "buffer", "train"]


def table(directions: list[float]) -> pl.DataFrame:
    n = len(directions)
    stamps = [START + timedelta(days=k % 20, minutes=k) for k in range(n)]
    return pl.DataFrame(
        {
            "timestamp": stamps,
            "direction": directions,
            "speed_bin": [0] * n,
            "rows": [3] * n,
            "waked_rows": [1] * n,
            "day": [s.date() for s in stamps],
        },
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "direction": pl.Float64,
            "speed_bin": pl.UInt8,
            "rows": pl.Int64,
            "waked_rows": pl.Int64,
            "day": pl.Date,
        },
    )


def test_shifted_grid_starts_arcs_half_an_arc_later():
    directions = [float(d) for d in range(0, 360, 2)]
    shifted = shifted_arcs(
        table(directions),
        15.0,
        count=12,
        width=30.0,
        buffer=5.0,
        min_rows=1,
        min_days=1,
        tolerance=0.05,
        seed=3,
    )
    assert sorted(shifted["split"].unique()) == [f"arc_{s:03d}_single" for s in range(15, 360, 30)]
    tested = shifted.filter(pl.col("role") == "test")
    at = dict(zip(table(directions)["timestamp"], directions, strict=True))
    for split, group in tested.group_by("split"):
        start = float(split[0].split("_")[1])
        assert all((at[t] - start) % 360.0 < 30.0 for t in group["timestamp"])
