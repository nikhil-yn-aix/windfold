import math

import polars as pl
import pytest

from windfold.prepare.target import centred_log_power


def test_target_is_log_power_centred_per_timestamp():
    rows = pl.DataFrame(
        {"timestamp": [1, 1, 1, 2, 2], "power_kw": [1000.0, 500.0, 250.0, 100.0, 400.0]}
    )
    out = centred_log_power(rows)
    ln2 = math.log(2.0)
    assert out.get_column("y").to_list() == pytest.approx([ln2, 0.0, -ln2, -ln2, ln2])
    assert out.get_column("valid_turbines").to_list() == [3, 3, 3, 2, 2]
    sums = out.group_by("timestamp").agg(pl.col("y").sum()).get_column("y")
    assert max(abs(v) for v in sums.to_list()) < 1e-12


def test_non_positive_power_is_refused():
    with pytest.raises(ValueError, match="positive power"):
        centred_log_power(pl.DataFrame({"timestamp": [1, 1], "power_kw": [10.0, 0.0]}))
