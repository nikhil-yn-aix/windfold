import polars as pl
import pytest

from windfold.analyses.development import prediction_file
from windfold.analyses.linear_clipping import linear_clipping


def test_identity_predictions_are_clipped_to_the_development_target_range(linear_pair):
    got = linear_clipping(linear_pair.data, linear_pair.runs, ("kelmarsh",))["kelmarsh"]
    assert got["clip"] == [pytest.approx(-0.998), pytest.approx(0.998)]
    assert got["waked_rows"] == 3
    reference = 0.25 + 0.25 + 0.04
    sse = {
        "identity_linear": 2.5**2 + 0.1**2 + 7.2**2,
        "identity_linear_clipped": 0.498**2 + 0.1**2 + 1.198**2,
        "kernel_linear": 3 * 0.1**2,
        "no_wake": reference,
    }
    assert got["skill"] == pytest.approx({m: 100 * (1 - v / reference) for m, v in sse.items()})
    assert got["clipped_identity_minus_kernel_mse"] == pytest.approx(
        (sse["identity_linear_clipped"] - sse["kernel_linear"]) / 3
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda frame: frame.with_columns(pl.col("turbine").replace({"T0": "T9"})),
        lambda frame: frame.reverse(),
        lambda frame: frame.with_columns(pl.col("y") + 0.1),
    ],
    ids=["key", "order", "target"],
)
def test_predictions_of_other_rows_are_refused(linear_pair, change):
    path = prediction_file(linear_pair.runs, "kelmarsh", "kernel_linear", "arc_000_single")
    change(pl.read_parquet(path)).write_parquet(path)
    with pytest.raises(ValueError, match="do not predict the same rows"):
        linear_clipping(linear_pair.data, linear_pair.runs, ("kelmarsh",))
