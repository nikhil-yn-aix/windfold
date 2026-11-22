import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from windfold.analyses.development import record_file
from windfold.analyses.kernel_geometry import (
    kernel_geometry,
    kernel_summary,
    neighbour_pairs,
    protocol_basis,
)
from windfold.config import load_farm, load_protocol, load_turbine

DIAMETER = 100.0
TIMES = [
    (270.0, 0b111, 1, "development"),
    (270.0, 0b101, 1, "development"),
    (0.0, 0b111, 1, "development"),
    (90.0, 0b111, 1, "development"),
    (270.0, 0b111, 0, "development"),
    (270.0, 0b111, 1, "calibration"),
]


def rows_of(times: list[tuple[float, int, int, str]]) -> pl.DataFrame:
    """three turbine rows per timestamp: (direction, operating mask, speed bin, period)."""
    stamps = [datetime(2019, 1, 1, tzinfo=UTC) + timedelta(hours=k) for k in range(len(times))]
    return pl.DataFrame(
        {
            "timestamp": [s for s in stamps for _ in range(3)],
            "period": [t[3] for t in times for _ in range(3)],
            "farm_direction": [t[0] for t in times for _ in range(3)],
            "speed_bin": pl.Series([t[2] for t in times for _ in range(3)], dtype=pl.UInt8),
            "operating_mask": pl.Series([t[1] for t in times for _ in range(3)], dtype=pl.UInt32),
        }
    )


LAYOUT = pl.DataFrame(
    {
        "turbine": ["T0", "T1", "T2"],
        "latitude": [0.0] * 3,
        "longitude": [0.0] * 3,
        "x_m": [0.0, 500.0, 1000.0],
        "y_m": [0.0, 0.0, 0.0],
        "hub_height_m": [80.0] * 3,
    }
)


def by_rows(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values, 12)
    return rounded[np.lexsort(rounded.T[::-1])]


def test_pairs_are_operating_upstream_neighbours_at_development_middle_bin_timestamps():
    protocol = load_protocol()
    design = neighbour_pairs(rows_of(TIMES), LAYOUT, DIAMETER, protocol)
    westerly, stopped, easterly = [5.0, 10.0, 5.0], [10.0], [5.0, 10.0, 5.0]
    along = np.array(westerly + stopped + easterly)
    expected = protocol_basis(protocol).values(along, np.zeros(along.size)).toarray()
    np.testing.assert_allclose(by_rows(design), by_rows(expected))


def test_the_sample_does_not_depend_on_the_order_of_the_rows():
    protocol = load_protocol()
    rows = rows_of(TIMES)
    shuffled = rows.sample(fraction=1.0, shuffle=True, seed=3)
    first = neighbour_pairs(rows, LAYOUT, DIAMETER, protocol, sample=2)
    assert np.array_equal(first, neighbour_pairs(shuffled, LAYOUT, DIAMETER, protocol, sample=2))


def test_the_kernel_is_read_at_the_middle_speed_bin_and_averaged_over_arcs():
    design = np.array([[1.0, 0.0], [0.0, 1.0]])
    records = [
        {
            "facts": {
                "blocks": {"shared_kernel": [9.0, 9.0, 0.01, 0.03, 9.0, 9.0], "physics": [0.8]}
            }
        },
        {
            "facts": {
                "blocks": {"shared_kernel": [5.0, 5.0, 0.03, 0.01, 5.0, 5.0], "physics": [1.2]}
            }
        },
    ]
    summary, mean = kernel_summary(design, records, speed_bin=1, per_bin=2)
    np.testing.assert_allclose(mean, [0.02, 0.02])
    assert summary["arcs"] == 2 and summary["beta_median"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        [summary["kernel_mean_percent"], summary["kernel_rms_percent"]], [2.0, 2.0]
    )
    np.testing.assert_allclose(summary["arc_to_arc_spread_percent"], 1.0)
    np.testing.assert_allclose(summary["arc_to_arc_variability"], 0.5)


def test_each_variant_is_summarised_over_every_included_arc_of_the_farm_asked_for(tmp_path):
    """a row of three turbines five diameters apart under westerlies gives three pairs per
    timestamp, along five, ten and five; each arc's middle-bin coefficients put the kernel at
    chosen values there, and every other bin's are wild."""
    protocol = load_protocol()
    diameter = load_turbine(load_farm("kelmarsh").turbine_type).rotor_diameter_m
    middle = (len(protocol.processing.speed_bin_edges_ms) - 1) // 2
    times = [(270.0, 0b111, middle, "development")] * 2
    times += [(270.0, 0b111, middle + 1, "development"), (270.0, 0b111, middle, "calibration")]
    folder = tmp_path / "data" / "prepared" / "kelmarsh"
    folder.mkdir(parents=True)
    rows_of(times).write_parquet(folder / "rows.parquet")
    LAYOUT.with_columns(x_m=pl.Series([0.0, 5 * diameter, 10 * diameter])).write_parquet(
        folder / "layout.parquet"
    )
    runs = tmp_path / "runs"
    (runs / "development" / "manifests").mkdir(parents=True)
    pl.DataFrame(
        {
            "split": ["arc_000_single", "arc_030_single", "arc_060_single"],
            "regime": ["arc_single"] * 3,
            "status": ["included", "included", "excluded"],
            "test_waked_rows": [5, 0, 5],
        }
    ).write_csv(runs / "development" / "manifests" / "kelmarsh_summary.csv")
    basis = protocol_basis(protocol)
    at = basis.values(np.array([5.0, 10.0]), np.zeros(2)).toarray()
    kernels = {
        "windfold": ((0.02, 0.01), (0.04, 0.01)),
        "windfold_constant_inflow": ((0.01, 0.03), (0.01, 0.03)),
        "windfold_evidence_scales": ((0.06, 0.02), (0.06, 0.02)),
        "windfold_constant_inflow_evidence_scales": ((0.0, 0.03), (0.0, 0.03)),
    }
    arcs = (("arc_000_single", 0.8), ("arc_030_single", 1.0))
    for model, values in kernels.items():
        for (split, beta), value in zip(arcs, values, strict=True):
            coefficients = np.full(basis.columns, 9.0)
            chosen = slice(middle * basis.per_bin, (middle + 1) * basis.per_bin)
            coefficients[chosen] = np.linalg.lstsq(at, np.array(value), rcond=None)[0]
            facts = {"blocks": {"shared_kernel": coefficients.tolist(), "physics": [beta]}}
            path = record_file(runs, "kelmarsh", model, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"facts": facts}))
    got = kernel_geometry(tmp_path / "data", runs, ("kelmarsh",))
    assert list(got) == ["kelmarsh"] and got["kelmarsh"]["pairs"] == 6
    first = got["kelmarsh"]["windfold"]
    mean = np.array([0.03, 0.01, 0.03] * 2)
    rms = np.sqrt(np.mean(mean**2))
    spread = np.sqrt(np.mean(np.array([0.01, 0.0, 0.01] * 2) ** 2))
    assert first == pytest.approx(
        {
            "arcs": 2,
            "beta_median": 0.9,
            "kernel_mean_percent": 100 * mean.mean(),
            "kernel_rms_percent": 100 * rms,
            "arc_to_arc_spread_percent": 100 * spread,
            "arc_to_arc_variability": spread / rms,
        }
    )
    correlations = {
        model: summary["correlation_with_windfold"]
        for model, summary in got["kelmarsh"].items()
        if model not in ("pairs", "windfold")
    }
    assert correlations == pytest.approx(
        {
            "windfold_constant_inflow": -1.0,
            "windfold_evidence_scales": 1.0,
            "windfold_constant_inflow_evidence_scales": -1.0,
        }
    )
