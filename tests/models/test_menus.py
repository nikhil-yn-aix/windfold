import math
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from windfold.config import load_menu
from windfold.models.interface import FitContext
from windfold.models.menus import (
    best_mean_rank,
    calibration_split,
    choose,
    draw,
    score,
    write_menu,
    write_menus,
)

SPACE = {
    "num_leaves": [15, 31, 63],
    "learning_rate": {"log": [0.01, 0.2]},
    "feature_fraction": {"uniform": [0.5, 1.0]},
    "rounds": {"integer": [20, 22]},
    "edges": ["wind_frame"],
}


def test_menus_are_seeded_and_stay_in_their_ranges():
    menu = draw(SPACE, size=200, seed=20261116)
    assert menu == draw(SPACE, size=200, seed=20261116)
    assert menu != draw(SPACE, size=200, seed=1)
    assert {c["num_leaves"] for c in menu} == {15, 31, 63}
    assert {c["rounds"] for c in menu} == {20, 21, 22}
    assert all(0.01 <= c["learning_rate"] <= 0.2 for c in menu)
    assert all(0.5 <= c["feature_fraction"] <= 1.0 for c in menu)
    assert all(isinstance(c["rounds"], int) and c["edges"] == "wind_frame" for c in menu)
    logs = np.log([c["learning_rate"] for c in menu])
    assert abs(np.median(logs) - 0.5 * (math.log(0.01) + math.log(0.2))) < 0.3
    with pytest.raises(ValueError, match="search range"):
        draw({"x": {"normal": [0, 1]}}, size=1, seed=0)


def test_a_menu_is_written_once_and_reads_back(tmp_path):
    menu = draw(SPACE, size=30, seed=7)
    path = tmp_path / "menus" / "gbm.toml"
    write_menu(path, menu, seed=7)
    text = path.read_text()
    assert tomllib.loads(text)["seed"] == 7
    assert list(load_menu("gbm", tmp_path)) == menu
    with pytest.raises(FileExistsError):
        write_menu(path, draw(SPACE, size=30, seed=8), seed=8)
    assert path.read_text() == text
    assert not list(tmp_path.glob("menus/*.part"))


def test_write_menus_writes_each_learner_once(tmp_path):
    spaces = {"gbm": {"num_leaves": [15, 31]}, "graph": {"layers": [2, 3, 4, 8]}}
    paths = write_menus(tmp_path / "menus", spaces, size=30, seed=20261116)
    assert [p.name for p in paths] == ["gbm.toml", "graph.toml"]
    graph = load_menu("graph", tmp_path)
    assert len(graph) == 30 and {c["layers"] for c in graph} <= {2, 3, 4, 8}
    with pytest.raises(FileExistsError):
        write_menus(tmp_path / "menus", {"graph": spaces["graph"], "other": spaces["gbm"]}, 30, 1)
    assert not (tmp_path / "menus" / "other.toml").exists()


def rows_at(stamps: list[datetime], waked: list[bool], y: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"timestamp": stamps, "waked": waked, "y": y},
        schema_overrides={"timestamp": pl.Datetime("us", "UTC")},
    )


def test_calibration_split_by_hand():
    start = datetime(2017, 6, 12, tzinfo=UTC)
    end = datetime(2018, 1, 1, tzinfo=UTC)
    cut = datetime(2017, 11, 21, 9, 36, tzinfo=UTC)
    embargo = datetime(2017, 11, 14, 9, 36, tzinfo=UTC)
    minute = timedelta(minutes=10)
    stamps = [
        start,
        embargo - minute,
        embargo,
        cut - minute,
        cut,
        end - minute,
    ]
    split = calibration_split(rows_at(stamps, [False] * 6, [0.0] * 6), start, end, 0.2, 7)
    assert split.record["validation_start"] == cut.isoformat()
    assert split.record["embargo_start"] == embargo.isoformat()
    assert split.fit["timestamp"].to_list() == [start, embargo - minute]
    assert split.validation["timestamp"].to_list() == [cut, end - minute]
    assert (split.record["fit_rows"], split.record["embargo_rows"]) == (2, 2)
    assert split.record["validation_timestamps"] == 2
    with pytest.raises(ValueError, match="outside the calibration window"):
        calibration_split(rows_at([end], [False], [0.0]), start, end, 0.2, 7)


class Constant:
    name = "constant"

    def __init__(self, value: float):
        self.value = value

    def fit(self, train: pl.DataFrame, context: FitContext) -> "Constant":
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame({"mean": [self.value] * rows.height, "variance": [None] * rows.height})


def test_score_choose_and_mean_rank():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    stamps = [start + timedelta(days=d) for d in range(10)]
    rows = rows_at(
        stamps, [True, False] * 5, [0.1, -0.1, 0.3, -0.3, 0.0, 0.2, -0.2, 0.4, -0.4, 0.1]
    )
    split = calibration_split(rows, start, start + timedelta(days=10), 0.3, 1)
    assert split.validation["y"].to_list() == [0.4, -0.4, 0.1]
    assert split.fit.height == 6
    context = FitContext("synthetic", Path("prepared"), 0, 1, None)
    row, _ = score(Constant(0.1), split, context)
    assert row["mse"] == pytest.approx((0.3**2 + 0.5**2 + 0.0**2) / 3)
    assert row["waked_mse"] == pytest.approx(0.5**2)
    assert choose([0.3, 0.1, 0.1, 0.2]) == 1
    assert best_mean_rank([[0.3, 0.1, 0.2], [0.1, 0.3, 0.2]]) == 0
    assert best_mean_rank([[0.3, 0.1, 0.2], [0.25, 0.2, 0.1]]) == 1
    with pytest.raises(ValueError, match="same menu"):
        best_mean_rank([[0.1, 0.2], [0.1]])
