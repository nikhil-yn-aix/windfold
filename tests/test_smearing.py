import numpy as np
import pytest

from windfold.smearing import smearing_factors


def test_smearing_factors_fall_back_to_pooled_in_small_bins():
    residuals = np.log(np.array([1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 6.0]))
    bins = np.array([0, 0, 1, 0, 0, 3, 3])
    table = smearing_factors(residuals, bins, min_rows=2, bin_count=4)
    pooled = (1 + 2 + 3 + 1 + 1 + 4 + 6) / 7
    assert table["factor"].to_list() == pytest.approx([5 / 4, pooled, pooled, 5.0])
    assert table["pooled"].to_list() == [False, True, True, False]
    assert table["rows"].to_list() == [4, 1, 0, 2]
    with pytest.raises(ValueError, match="speed bins"):
        smearing_factors(residuals, bins + 2, min_rows=2, bin_count=4)
