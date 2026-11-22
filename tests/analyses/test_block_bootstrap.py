from datetime import UTC, date, datetime, timedelta
from importlib import resources

import numpy as np
import polars as pl
import pytest

from windfold.analyses import block_bootstrap as bb

ORIGIN = date(2020, 1, 1)


def errors_on(days: list[int], errors: list[float]) -> pl.DataFrame:
    """prediction rows at noon utc of the given days from ORIGIN, y minus mean as given."""
    noon = datetime(2020, 1, 1, 12, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [noon + timedelta(days=d) for d in days],
            "y": errors,
            "mean": np.zeros(len(days)),
        }
    )


def scratch_weights(blocks: int, replicates: int, seed: int) -> np.ndarray:
    """the scratch scripts' replicate weights, one replicate at a time."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates + 1, blocks))
    counts[0] = 1
    draws = rng.integers(0, blocks, size=(replicates, blocks))
    for r in range(replicates):
        np.add.at(counts[r + 1], draws[r], 1)
    return counts


def test_weights_redraw_the_scratch_bootstrap_from_the_same_seed():
    drawn = bb.weights(14, 300, np.random.default_rng(20261121))
    assert np.array_equal(drawn, scratch_weights(14, 300, 20261121))
    assert (drawn[0] == 1).all() and (drawn.sum(axis=1) == 14).all()


def test_rows_add_up_by_block_from_the_window_start_across_splits():
    first = errors_on([0, 7], [1.0, 2.0])
    second = errors_on([6, 13, 14], [3.0, 4.0, 5.0])
    vector = bb.block_vector([bb.block_sse(f, ORIGIN) for f in (first, second)], 3)
    assert vector.tolist() == [1.0 + 9.0, 4.0 + 16.0, 25.0]
    with pytest.raises(ValueError, match="outside"):
        bb.block_vector([bb.block_sse(second, ORIGIN)], 2)


def test_models_of_a_farm_are_resampled_with_the_same_blocks():
    reference = errors_on([0, 7, 14], [np.sqrt(10.0)] * 3)
    same = errors_on([0, 7, 14], [np.sqrt(5.0)] * 3)
    better_late = errors_on([0, 7, 14], [np.sqrt(5.0), np.sqrt(5.0), np.sqrt(2.0)])
    draws = bb.weights(3, 400, np.random.default_rng(1))
    total = bb.replicate_sse([reference], ORIGIN, draws)
    gain = bb.skill(bb.replicate_sse([better_late], ORIGIN, draws), total) - bb.skill(
        bb.replicate_sse([same], ORIGIN, draws), total
    )
    np.testing.assert_allclose(gain, 30.0 * draws[:, 2] / draws.sum(axis=1))
    assert (gain[draws[:, 2] == 0] == 0).all() and gain[0] == pytest.approx(10.0)


def test_the_interval_takes_replicate_zero_as_its_estimate_and_the_rest_for_its_bounds():
    got = bb.interval(np.r_[1000.0, np.arange(100.0)])
    assert got == {"estimate": 1000.0, "low": pytest.approx(2.475), "high": pytest.approx(96.525)}


def test_seven_whole_days_make_one_block_and_touching_an_eighth_makes_two(tmp_path):
    packaged = resources.files("windfold") / "configs" / "farms" / "smarteole.toml"
    text = packaged.read_text()
    window = "development = { start = 2020-02-17T16:30:00Z, end = 2020-05-25T00:00:00Z }"
    assert window in text
    (tmp_path / "farms").mkdir()
    for name, end in (("seven", "2020-02-24T00:00:00Z"), ("eight", "2020-02-24T00:10:00Z")):
        farm = text.replace('name = "smarteole"', f'name = "{name}"').replace(
            window, f"development = {{ start = 2020-02-17T00:00:00Z, end = {end} }}"
        )
        (tmp_path / "farms" / f"{name}.toml").write_text(farm)
    assert bb.window_blocks("seven", tmp_path) == (date(2020, 2, 17), 1)
    assert bb.window_blocks("eight", tmp_path) == (date(2020, 2, 17), 2)
