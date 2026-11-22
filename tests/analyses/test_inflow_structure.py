from datetime import UTC, datetime, timedelta
from importlib import resources

import numpy as np
import polars as pl
import pytest

from windfold.analyses.inflow_structure import (
    donors,
    explained,
    inflow_structure,
    measured_times,
)
from windfold.config import load_protocol

DAY = datetime(2021, 6, 1, 12, tzinfo=UTC)


def rows_of(entries: list[tuple[float, float, float | None, str]]) -> pl.DataFrame:
    """two turbine rows per timestamp: (day offset, direction, turbulence, period)."""
    times = [DAY + timedelta(days=day) for day, *_ in entries]
    return pl.DataFrame(
        {
            "timestamp": [t for t in times for _ in range(2)],
            "period": [e[3] for e in entries for _ in range(2)],
            "farm_direction": [e[1] for e in entries for _ in range(2)],
            "speed_bin": pl.Series([1] * 2 * len(entries), dtype=pl.UInt8),
            "turbulence_intensity": pl.Series([e[2] for e in entries for _ in range(2)]),
        }
    )


def test_only_measured_development_timestamps_count_once_each():
    rows = rows_of(
        [
            (0, 10.0, 0.1, "development"),
            (1, 10.0, None, "development"),
            (2, 10.0, 0.0, "development"),
            (3, 10.0, float("nan"), "development"),
            (4, 10.0, 0.2, "calibration"),
            (5, 40.0, 0.3, "development"),
        ]
    )
    times = measured_times(rows, load_protocol().inflow)
    assert times["timestamp"].to_list() == [DAY, DAY + timedelta(days=5)]
    assert times["ratio"].to_list() == [0.1, 0.3] and times["sector"].to_list() == [0, 1]


def test_the_variance_share_is_what_lies_between_cells():
    by_sector = rows_of([(d, 30.0 * (d % 2), 0.1 * (1 + d % 2), "development") for d in range(6)])
    times = measured_times(by_sector, load_protocol().inflow)
    assert explained(times, ["sector"]) == pytest.approx(1.0)
    assert explained(times, ["speed_bin"]) == pytest.approx(0.0)
    within = rows_of([(d, 30.0 * (d % 2), 0.1 * (1 + d // 2), "development") for d in range(4)])
    assert explained(measured_times(within, load_protocol().inflow), ["sector"]) == pytest.approx(
        0.0
    )


def test_every_other_timestamp_of_a_cell_is_a_donor_and_none_is_its_own():
    rows = rows_of(
        [
            (2, 70.0, 0.1, "development"),
            (4, 10.0, 0.1, "development"),
            (5, 70.0, 0.1, "development"),
            (0, 40.0, 0.1, "development"),
            (1, 70.0, 0.1, "development"),
            (3, 40.0, 0.1, "development"),
        ]
    )
    got = donors(measured_times(rows, load_protocol().inflow))
    distances = [3.0, 3.0, np.median([1.0, 3.0]), np.median([3.0, 4.0]), np.median([1.0, 4.0])]
    assert got == {
        "development_days": 5.0,
        "cells": 3,
        "median_donors": 1.5,
        "share_with_few_donors": 1.0,
        "median_days_to_donors": float(np.median(distances)),
    }


def test_a_farm_asked_for_gets_its_shares_sector_medians_and_donors_under_the_given_configs(
    tmp_path,
):
    packaged = resources.files("windfold") / "configs" / "protocol.toml"
    text = packaged.read_text()
    assert text.count("climatology_sector_deg = 30.0") == 1
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "protocol.toml").write_text(
        text.replace("climatology_sector_deg = 30.0", "climatology_sector_deg = 60.0")
    )
    farms = {
        "kelmarsh": [
            (0, 10.0, 0.2, "development"),
            (1, 50.0, 0.2, "development"),
            (2, 70.0, 0.1, "development"),
            (3, 130.0, 0.4, "development"),
            (4, 20.0, None, "development"),
            (5, 20.0, 0.9, "calibration"),
        ],
        "penmanshiel": [(0, 10.0, 0.5, "development"), (1, 10.0, 0.6, "development")],
    }
    for farm, entries in farms.items():
        folder = tmp_path / "prepared" / farm
        folder.mkdir(parents=True)
        rows_of(entries).write_parquet(folder / "rows.parquet")
    assert inflow_structure(tmp_path, ("kelmarsh",), configs) == {
        "kelmarsh": {
            "measured_timestamps": 4,
            "variance_share": {
                "sector": pytest.approx(1.0),
                "speed_bin": 0.0,
                "sector_and_speed_bin": pytest.approx(1.0),
                "diurnal_cell": pytest.approx(1.0),
            },
            "median_turbulence_by_sector": {0: 0.2, 1: 0.1, 2: 0.4},
            "donors": {
                "development_days": 3.0,
                "cells": 3,
                "median_donors": 0.5,
                "share_with_few_donors": 1.0,
                "median_days_to_donors": 1.0,
            },
        }
    }
