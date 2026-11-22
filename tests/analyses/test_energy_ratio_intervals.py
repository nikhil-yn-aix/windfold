import numpy as np
import pytest

from windfold.analyses.energy_ratio_intervals import replicate_errors

OBSERVED = np.array(
    [
        [[8.0, 3.0], [2.0, 1.0]],
        [[10.0, 4.0], [2.0, 2.0]],
        [[20.0, 30.0], [10.0, 0.0]],
    ]
)
SEEDS = np.array(
    [
        [[[6.0, 3.0], [2.0, 1.0]], [[10.0, 4.0], [2.0, 2.0]]],
        [[[10.0, 2.0], [2.0, 1.0]], [[10.0, 4.0], [2.0, 2.0]]],
    ]
)


def test_the_point_estimate_pools_blocks_and_seeds_and_weights_cells_by_rows():
    found = replicate_errors(OBSERVED, SEEDS, np.ones((1, 2)), None)
    observed = np.array([10 / 12, 4 / 6])
    predicted = np.array([20 / 24, 9 / 12])
    expected = (np.abs(predicted - observed) * [30, 30]).sum() / 60
    assert found == pytest.approx([expected])


def test_a_replicate_uses_its_drawn_blocks_and_seeds():
    counts = np.array([[2.0, 0.0]])
    found = replicate_errors(OBSERVED, SEEDS, counts, np.array([[1, 1]]))
    gap = np.abs(np.array([10 / 10, 2 / 4]) - np.array([8 / 10, 3 / 4]))
    assert found == pytest.approx([(gap * [20, 30]).sum() / 50])
