from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from windfold.analyses import turbine_levels as tl
from windfold.config import load_protocol
from windfold.locations import search_file
from windfold.schemas import LAYOUT, conform

FARM = "kelmarsh"


def test_removing_turbine_levels_leaves_only_the_difference_within_turbines():
    turbine = np.array(["A", "A", "A", "B"])
    identity = np.array([[0.1, 0.2], [-0.1, -0.2], [0.0, 0.0], [0.0, 0.0]])
    levels = np.array([[0.3, 0.1], [0.3, 0.1], [0.3, 0.1], [-0.5, -0.3]])
    geometry = identity + levels
    total, part = tl.mse_parts(turbine, geometry)
    assert part == pytest.approx(np.mean([(3 * 0.3**2 + 0.5**2) / 4, (3 * 0.1**2 + 0.3**2) / 4]))
    assert total == pytest.approx((geometry**2).mean())
    own_total, own_part = tl.mse_parts(turbine, identity)
    assert own_part == pytest.approx(0.0)
    assert (own_total - own_part) - (total - part) == pytest.approx(0.0)
    assert tl.difference(identity, geometry) == pytest.approx(own_total - total)


def test_the_level_correlation_pools_full_arcs_against_each_turbines_whole_offset():
    cells = [
        ("A", "first", 150, 0.2, 0.3),
        ("A", "second", 120, -0.1, 0.3),
        ("B", "first", 150, 0.05, -0.2),
        ("B", "second", 50, 0.9, 0.4),
        ("C", "first", 130, -0.3, 0.1),
        ("C", "second", 20, 0.7, 0.1),
        ("D", "first", 30, 2.0, 5.0),
        ("D", "second", 30, -2.0, 5.0),
    ]
    frame = pl.DataFrame(
        {
            "turbine": [t for t, _, n, _, _ in cells for _ in range(n)],
            "split": [s for _, s, n, _, _ in cells for _ in range(n)],
            tl.REFERENCE: [o for _, _, n, _, o in cells for _ in range(n)],
        }
    )
    mean = np.array([m for _, _, n, m, _ in cells for _ in range(n)])
    error = np.column_stack([mean + 0.1, mean - 0.1])
    levels = [(150 * 0.2 - 120 * 0.1) / 270, 0.05, -0.3]
    offsets = [0.3, (150 * -0.2 + 50 * 0.4) / 200, 0.1]
    assert tl.level_correlation(frame, error) == pytest.approx(np.corrcoef(levels, offsets)[0, 1])


def test_edge_rows_have_an_operating_upstream_source_near_the_windows_across_edge(tmp_path):
    folder = tmp_path / "prepared" / FARM
    folder.mkdir(parents=True)
    east, north = [0.0, -280.0, 100.0, -350.0], [0.0, -1000.0, -1000.0, -500.0]
    layout = pl.DataFrame(
        {
            "turbine": [f"T{k}" for k in range(4)],
            "latitude": np.zeros(4),
            "longitude": np.zeros(4),
            "x_m": east,
            "y_m": north,
            "hub_height_m": np.full(4, 80.0),
        }
    )
    conform(layout, LAYOUT).write_parquet(folder / "layout.parquet")
    stamps = [datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, 0, 10, tzinfo=UTC)]
    rows = pl.DataFrame(
        {
            "timestamp": [stamps[0]] * 4 + [stamps[1]] * 4,
            "turbine": [f"T{k}" for k in range(4)] * 2,
            "turbine_index": pl.Series(list(range(4)) * 2, dtype=pl.UInt8),
            "farm_direction": [0.0] * 8,
            "operating_mask": pl.Series([15] * 4 + [14] * 4, dtype=pl.UInt32),
        }
    )
    rows.write_parquet(folder / "rows.parquet")
    kernel = load_protocol().kernel
    frame = rows.select("timestamp", "turbine").reverse()
    edge = tl.edge_rows(tmp_path, FARM, frame, kernel, 100.0)
    assert edge.tolist() == [False] * 4 + [True, False, True, False]


def test_speed_levels_come_from_development_rows_centred_on_each_timestamp(tmp_path):
    folder = tmp_path / "prepared" / FARM
    folder.mkdir(parents=True)
    level = np.array([0.1, 0.0, -0.1])
    parts = []
    for k, (period, slope) in enumerate(
        [("development", 2.5), ("development", 2.5), ("calibration", -9.0)]
    ):
        common = 0.2 * k
        parts.append(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2020, 1, 1 + k, tzinfo=UTC)] * 3,
                    "turbine": ["A", "B", "C"],
                    "period": [period] * 3,
                    "speed": np.exp(1.5 + common + level),
                    "y": slope * level - slope * level.mean(),
                }
            )
        )
    pl.concat(parts).write_parquet(folder / "rows.parquet")
    found = tl.speed_levels(tmp_path, FARM)
    assert found["speed_level_sd"] == pytest.approx(np.std(level))
    assert found["power_on_speed_slope"] == pytest.approx(2.5)
    assert found["power_on_speed_correlation"] == pytest.approx(1.0)


def test_the_tuning_menu_counts_configurations_where_geometry_validates_worse(tmp_path):
    for model, mse in {"identity_gbm": [0.1, 0.2, 0.3], "geometry_gbm": [0.15, 0.18, 0.35]}.items():
        path = tmp_path / search_file(FARM, model)
        path.parent.mkdir(parents=True)
        pl.DataFrame({"index": [0, 1, 2], "mse": mse}).write_parquet(path)
    found = tl.tuning_menu(tmp_path, FARM)
    assert found == {
        "configurations": 3,
        "geometry_worse": 2,
        "median_identity_minus_geometry": pytest.approx(-0.05),
    }
