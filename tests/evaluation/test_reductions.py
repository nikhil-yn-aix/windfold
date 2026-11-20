from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.config import load_protocol
from windfold.evaluation import reductions

START = datetime(2020, 3, 1, tzinfo=UTC)


def window() -> tuple[pl.DataFrame, pl.DataFrame]:
    stamps = [START + timedelta(hours=12 * k) for k in range(6)]
    manifest = pl.DataFrame(
        {
            "split": "time_1",
            "regime": "time_blocked",
            "timestamp": stamps,
            "role": ["train", "train", "test", "test", "calibrate", "test"],
        }
    )
    prepared = pl.DataFrame(
        {
            "timestamp": [s for s in stamps for _ in range(2)],
            "turbine": ["T1", "T2"] * len(stamps),
            "y": [0.1, -0.1] * len(stamps),
        }
    )
    return prepared, manifest


def test_a_day_limit_scores_the_same_first_days_its_fits_tested_on():
    prepared, manifest = window()
    every = reductions.window_rows(prepared, manifest, "time_1", None)
    assert every["timestamp"].unique().to_list() == [
        START + timedelta(hours=h) for h in (24, 36, 48, 60)
    ]
    first = reductions.window_rows(prepared, manifest, "time_1", 1)
    assert first["timestamp"].unique().sort().to_list() == [
        START + timedelta(hours=24),
        START + timedelta(hours=36),
    ]


def test_a_fit_that_skips_rows_of_its_window_is_refused():
    prepared, manifest = window()
    here = reductions.window_rows(prepared, manifest, "time_1", None)
    predictions = here.select("timestamp", "turbine", mean=pl.lit(0.0)).head(here.height - 1)
    with pytest.raises(ValueError, match="7 predictions for 8 rows"):
        reductions.fit_reductions(
            predictions,
            here,
            {},
            farm="f",
            model="m",
            split="time_1",
            seed=None,
            regime="time_blocked",
            protocol=load_protocol(),
        )
