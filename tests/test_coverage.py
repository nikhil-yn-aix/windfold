import numpy as np
import polars as pl
import pytest

from windfold import coverage, geometry
from windfold.models import blocks

DIAMETER = 82.0
EAST = np.array([0.0, 410.0, 820.0, 200.0, 610.0])
NORTH = np.array([0.0, 20.0, -30.0, 400.0, 420.0])
BASIS = blocks.kernel_basis(30.0, 3.0, 1.5, 9, 0.45, 4, 3, 3)


def rows(
    timestamps: int, seed: int, directions: tuple[float, float] = (0.0, 360.0)
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(EAST)
    full = (1 << n) - 1
    masks = np.where(
        rng.random(timestamps) < 0.2, full & ~(1 << rng.integers(0, n, timestamps)), full
    )
    direction = rng.uniform(*directions, timestamps)
    speed_bin = rng.integers(0, 3, timestamps)
    t, i = np.nonzero((masks[:, None] >> np.arange(n)) & 1)
    stamps = np.datetime64("2018-01-01T00:00", "us") + (t + seed * 100_000) * np.timedelta64(
        10, "m"
    )
    return pl.DataFrame(
        {
            "timestamp": pl.Series(stamps).dt.replace_time_zone("UTC"),
            "turbine_index": pl.Series(i, dtype=pl.UInt8),
            "operating_mask": pl.Series(masks[t], dtype=pl.UInt32),
            "farm_direction": direction[t],
            "speed_bin": pl.Series(speed_bin[t], dtype=pl.UInt8),
            "reference_speed": 6.0 + speed_bin[t] * 1.5,
            "power_kw": rng.uniform(100, 1500, len(t)),
            "y": rng.normal(0, 0.1, len(t)),
        }
    )


def kernel_rings() -> np.ndarray:
    ring = np.zeros((360, 3, len(EAST), len(EAST)))
    for degree in range(360):
        along, across = geometry.relations(EAST, NORTH, float(degree), DIAMETER)
        ring[degree] = -0.5 * np.exp(-0.5 * (across / 0.5) ** 2) * np.exp(-along / 10) * (along > 0)
    return ring


KERNEL = kernel_rings()


def score(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    return coverage.design_only_score(
        train, test, EAST, NORTH, DIAMETER, BASIS, 30.0, 3.0, KERNEL, 0.01, 0.01, 0.1
    )


def test_design_only_score_is_unchanged_when_power_values_are_permuted():
    train, test = rows(300, 1), rows(60, 2)
    rng = np.random.default_rng(3)
    shuffled = train.with_columns(
        power_kw=pl.Series(rng.permutation(train["power_kw"].to_numpy())),
        y=pl.Series(rng.permutation(train["y"].to_numpy())),
    )
    assert np.array_equal(score(train, test), score(shuffled, test))


def test_more_training_data_never_raises_a_rows_score():
    held_out = rows(300, 4, directions=(60.0, 330.0))
    everything = pl.concat([held_out, rows(200, 5, directions=(0.0, 60.0))])
    test = rows(80, 6, directions=(10.0, 50.0))
    less, more = score(held_out, test), score(everything, test)
    assert np.all(more <= less + 1e-12)
    assert more.sum() < less.sum()


def test_sparse_quadratic_forms_equal_dense_ones():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(BASIS.columns, BASIS.columns))
    matrix = a @ a.T
    along, across = rng.uniform(0.1, 30, 50), rng.uniform(-3, 3, 50)
    speed_bin = rng.integers(0, 3, 50)
    columns, values = coverage.pair_basis(BASIS, along, across, speed_bin)
    dense = np.zeros((50, BASIS.columns))
    np.add.at(dense, (np.arange(50)[:, None], columns), values)
    expected = np.einsum("pi,ij,pj->p", dense, matrix, dense)
    np.testing.assert_allclose(
        coverage.quadratic(matrix, columns, values, chunk=7), expected, rtol=1e-12
    )


def test_dissimilarity_index_is_the_nearest_distance_over_the_mean_training_distance():
    rng = np.random.default_rng(8)
    train, test = rng.normal(size=(40, 3)), rng.normal(size=(5, 3))
    got = coverage.dissimilarity(train, test, sample=40, seed=0)
    centre, spread = train.mean(0), train.std(0)
    a, b = (train - centre) / spread, (test - centre) / spread
    pair = [np.linalg.norm(a[i] - a[j]) for i in range(40) for j in range(i + 1, 40)]
    nearest = np.array([min(np.linalg.norm(t - a, axis=1)) for t in b])
    np.testing.assert_allclose(got, nearest / np.mean(pair), rtol=1e-12)


def test_dissimilarity_is_undefined_when_training_rows_are_identical():
    train = np.ones((3, 6))
    got = coverage.dissimilarity(train, np.array([np.ones(6), np.zeros(6)]), sample=3, seed=0)
    assert np.isnan(got).all()


def test_angular_distance_wraps_around_north():
    train = pl.DataFrame({"farm_direction": [10.0, 20.0, 355.4]})
    test = pl.DataFrame({"farm_direction": [15.0, 200.0, 2.0]})
    assert coverage.angular_distance(train, test).tolist() == pytest.approx([5.0, 155.0, 7.0])
