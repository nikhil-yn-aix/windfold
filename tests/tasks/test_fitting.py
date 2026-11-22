from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from windfold import config
from windfold.models.interface import FitContext
from windfold.schemas import PREDICTIONS
from windfold.tasks.fitting import fit_one, predicted, split_rows

START = datetime(2020, 1, 1, tzinfo=UTC)


def rows(days: int, turbines: int = 2) -> pl.DataFrame:
    stamps = [START + timedelta(days=d, minutes=10 * k) for d in range(days) for k in range(2)]
    frame = pl.DataFrame(
        {
            "timestamp": [t for t in stamps for _ in range(turbines)],
            "turbine": [f"T{i}" for _ in stamps for i in range(turbines)],
            "turbine_index": [i for _ in stamps for i in range(turbines)],
        }
    )
    sign = np.where(frame["turbine_index"].to_numpy() == 0, 1.0, -1.0)
    return frame.with_columns(
        y=pl.Series(0.1 * sign),
        waked=pl.col("turbine_index") == 1,
        valid_turbines=pl.lit(turbines),
        speed_bin=pl.lit(0, pl.UInt8),
    )


def manifest(roles: dict[str, list[int]], split: str) -> pl.DataFrame:
    stamps = sorted(rows(4)["timestamp"].unique())
    entries = [(split, "random", stamps[i], role) for role, idx in roles.items() for i in idx]
    return pl.DataFrame(
        entries, schema=["split", "regime", "timestamp", "role"], orient="row"
    ).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))


def test_a_split_trains_on_train_or_fit_and_tests_on_test_or_calibrate():
    prepared = rows(4)
    plain = manifest({"train": [0, 1, 2], "buffer": [3], "test": [4, 5, 6, 7]}, "random_1")
    train, test = split_rows(prepared, plain, "random_1", None)
    assert train["timestamp"].n_unique() == 3 and test["timestamp"].n_unique() == 4
    assert train.height == 6 and test["turbine_index"].to_list()[:2] == [0, 1]
    threshold = manifest({"fit": [0, 1], "buffer": [2, 3], "calibrate": [6, 7]}, "threshold_1")
    train, test = split_rows(prepared, threshold, "threshold_1", None)
    stamps = sorted(prepared["timestamp"].unique())
    assert train["timestamp"].unique().sort().to_list() == stamps[:2]
    assert test["timestamp"].unique().sort().to_list() == stamps[6:]


def test_days_keeps_the_first_calendar_days_of_each_part():
    prepared = rows(4)
    plain = manifest({"train": [0, 1, 2, 3, 4, 5], "test": [6, 7]}, "random_1")
    train, _ = split_rows(prepared, plain, "random_1", 2)
    assert train["timestamp"].dt.date().n_unique() == 2 and train.height == 8


class Zero:
    name: str = "zero"

    def fit(self, train: pl.DataFrame, context: FitContext) -> "Zero":
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame({"mean": np.zeros(rows.height)}).with_columns(
            variance=pl.lit(None, pl.Float64)
        )


class Short(Zero):
    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        return super().predict(rows.head(1))


def test_a_fit_gives_conforming_predictions_and_a_record(tmp_path):
    protocol = config.load_protocol()
    train, test = rows(3), rows(2)
    context = FitContext("farm", tmp_path, 0, 1, None)
    predictions, record, _ = fit_one(Zero(), train, test, context, protocol)
    assert predictions.schema == PREDICTIONS
    assert predictions["support"].null_count() == test.height
    assert predictions["y"].to_list() == test["y"].to_list()
    assert record["train_rows"] == train.height and record["test_timestamps"] == 4
    factor = next(s for s in record["smearing"] if s["speed_bin"] == 0)["factor"]
    assert factor == pytest.approx(np.mean(np.exp(2 * 0.1 * np.array([1.0, -1.0]))))


def test_a_prediction_of_the_wrong_length_is_refused(tmp_path):
    with pytest.raises(ValueError, match="predict returned"):
        predicted(Short(), rows(1))
