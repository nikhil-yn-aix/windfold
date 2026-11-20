from pathlib import Path

import numpy as np
import polars as pl
import pytest

from windfold.models import blocks, engine, recovery, registry

EAST = np.array([0.0, 400.0, 800.0, 0.0, 400.0, 800.0])
NORTH = np.array([0.0, 0.0, 0.0, 450.0, 450.0, 450.0])
BASIS = blocks.kernel_basis(30.0, 3.0, 1.5, 9, 0.45, 4, 3, 3)
SETTINGS = registry.EngineSettings(
    speed_bins=3,
    buckets=72,
    harmonics_order=4,
    spline_knots=24,
    spline_ridge=0.1,
    offset_variance=1.0,
    physics_variance=0.25,
    starts=(1e-4, 1e-2, 1.0),
    bounds=engine.Bounds(1e-6, 1e-8, 100.0),
    fixed_prior_variance=None,
    lookup_window_deg=5,
    kernel=BASIS,
    kernel_ridge=0.1,
)


def prepared_folder(folder: Path, timestamps: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = len(EAST)
    full = (1 << n) - 1
    masks = np.where(
        rng.random(timestamps) < 0.1, full & ~(1 << rng.integers(0, n, timestamps)), full
    )
    direction = rng.uniform(0, 360, timestamps)
    t, i = np.nonzero((masks[:, None] >> np.arange(n)) & 1)
    rows = pl.DataFrame(
        {
            "timestamp": pl.Series(
                np.datetime64("2018-01-01T00:00", "us") + t * np.timedelta64(10, "m")
            ).dt.replace_time_zone("UTC"),
            "turbine": [f"T{k}" for k in i],
            "turbine_index": pl.Series(i, dtype=pl.UInt8),
            "operating_mask": pl.Series(masks[t], dtype=pl.UInt32),
            "farm_direction": direction[t],
            "speed_bin": pl.Series(rng.integers(0, 3, timestamps)[t], dtype=pl.UInt8),
            "direction_bucket": pl.Series((direction[t] // 5).astype(int), dtype=pl.UInt8),
            "valid_turbines": pl.Series(np.bincount(t)[t], dtype=pl.UInt8),
            "y": np.zeros(len(t)),
        }
    )
    rows.write_parquet(folder / "rows.parquet")
    pl.DataFrame(
        {
            "turbine": [f"T{k}" for k in range(n)],
            "latitude": np.zeros(n),
            "longitude": np.zeros(n),
            "x_m": EAST,
            "y_m": NORTH,
            "hub_height_m": np.full(n, 59.0),
        }
    ).write_parquet(folder / "layout.parquet")
    blocks.build_kernel_designs(folder, folder / "kernel_design", 82.0, BASIS, 30.0, 3.0, 5)
    return rows


def test_a_known_kernel_is_recovered_with_near_nominal_intervals(tmp_path):
    rows = prepared_folder(tmp_path, 4000)
    grid, summary = recovery.synthetic_recovery(
        rows, tmp_path, tmp_path / "kernel_design", SETTINGS, (0.06, 0.05, 0.04), 1, 1
    )
    assert grid.height == 3 * 40 * 25
    assert summary["supported_correlation"] > 0.95
    assert summary["supported_rmse"] < 0.1 * summary["kernel_scale"]
    assert summary["predictive_coverage_90"] == pytest.approx(0.9, abs=0.03)
    assert 0.8 < summary["supported_kernel_coverage_90"] <= 1.0


def test_a_design_built_for_other_rows_is_refused(tmp_path):
    rows = prepared_folder(tmp_path, 300)
    changed = rows.with_columns(farm_direction=pl.col("farm_direction") + 1.0)
    changed.write_parquet(tmp_path / "rows.parquet")
    table, key = registry.prepared_rows(
        tmp_path / "rows.parquet", registry.modified(tmp_path / "rows.parquet")
    )
    with pytest.raises(ValueError, match="built for other rows"):
        blocks.read_kernel_design(tmp_path / "kernel_design", key)
