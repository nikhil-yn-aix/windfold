import numpy as np
import polars as pl
import pytest

from windfold.prepare.validity import disturbed_half_width, disturbing_sources, pair_sectors

LAYOUT = pl.DataFrame(
    {
        "turbine": ["A", "B", "C"],
        "x_m": [0.0, 500.0, 0.0],
        "y_m": [0.0, 0.0, 4000.0],
        "rotor_diameter_m": [100.0, 100.0, 100.0],
    }
)


@pytest.mark.parametrize(("distance", "half"), [(500.0, 26.46551), (3000.0, 13.53712)])
def test_disturbed_sector_follows_the_iec_formula(distance, half):
    assert disturbed_half_width(np.array([distance]), np.array([100.0]))[0] == pytest.approx(
        half, abs=1e-4
    )


def test_pairs_beyond_thirty_diameters_are_left_out_and_bearings_point_upwind():
    pairs = pair_sectors(LAYOUT, 30.0).sort("source")
    assert pairs.select("source", "target").rows() == [("A", "B"), ("B", "A")]
    assert pairs.get_column("bearing_deg").to_list() == pytest.approx([270.0, 90.0])
    assert pairs.get_column("distance_d").to_list() == pytest.approx([5.0, 5.0])


def test_a_source_disturbs_its_target_only_inside_the_sector():
    rows = pl.DataFrame(
        {
            "turbine_index": [1, 1, 1, 1, 1, 0, 0],
            "farm_direction": [270.0, 296.0, 297.0, 90.0, None, 90.0, 270.0],
        },
        schema={"turbine_index": pl.UInt8, "farm_direction": pl.Float64},
    )
    bits = disturbing_sources(rows, pair_sectors(LAYOUT, 30.0))
    assert bits.to_list() == [1, 1, 0, 0, 0, 2, 0]
    assert bits.dtype == pl.UInt32
