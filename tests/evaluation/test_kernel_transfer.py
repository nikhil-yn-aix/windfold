from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from windfold.evaluation.kernel_transfer import decompose, free_stream_turbulence, kernel_part

RNG = np.random.default_rng(3)
Y = RNG.normal(size=200)
WAKED = RNG.random(200) < 0.6


@pytest.mark.parametrize(("size", "slope", "term"), [(0.5, 1.0, 25.0), (1.0, 0.5, 0.0)])
def test_a_kernel_part_over_the_physics_base_is_scored_by_its_size(size, slope, term):
    without_kernel = 0.5 * Y
    kernel = size * Y
    out = decompose(Y, WAKED, without_kernel + kernel, without_kernel, np.zeros(200), kernel)
    assert out["slope"] == pytest.approx(slope)
    assert out["kernel_term"] == pytest.approx(term)
    assert out["base_shift"] == pytest.approx(0.0, abs=1e-12)
    assert out["correlation"] == pytest.approx(1.0)


def test_the_kernel_effect_is_its_term_plus_the_base_shift_and_equals_the_skill_change():
    reference = RNG.normal(size=200)
    windfold, without_kernel, kernel = RNG.normal(size=(3, 200))
    out = decompose(Y, WAKED, windfold, without_kernel, reference, kernel)
    sse = {
        name: np.sum((Y - f)[WAKED] ** 2)
        for name, f in (
            ("windfold", windfold),
            ("without", without_kernel),
            ("reference", reference),
        )
    }
    skill_change = 100 * (sse["without"] - sse["windfold"]) / sse["reference"]
    term, shift = out["kernel_term"], out["base_shift"]
    assert term is not None and shift is not None
    assert out["kernel_effect"] == pytest.approx(skill_change)
    assert term + shift == pytest.approx(skill_change)


def test_a_split_without_waked_rows_has_no_terms():
    out = decompose(Y, np.zeros(200, bool), Y, Y, Y, Y)
    assert all(value is None for value in out.values())


def test_the_kernel_part_is_centred_within_each_timestamp():
    timestamps = np.array([5, 5, 5, 9, 9])
    design = sp.csr_array(np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [3.0, 1.0], [0.0, 0.0]]))
    part = kernel_part(design, np.array([1.0, 1.0]), timestamps)
    np.testing.assert_allclose(part, [0.0, 1.0, -1.0, 2.0, -2.0])


def test_free_stream_turbulence_is_the_median_of_unwaked_timestamp_medians():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    rows = pl.DataFrame(
        {
            "timestamp": [start] * 3 + [start + timedelta(minutes=10)] * 3,
            "waked": [False, False, True, False, True, True],
            "turbulence_intensity": [0.10, 0.20, 0.90, 0.08, 0.70, None],
        }
    )
    assert free_stream_turbulence(rows) == pytest.approx(np.median([0.15, 0.08]))
    assert free_stream_turbulence(rows.filter(pl.col("waked"))) is None
