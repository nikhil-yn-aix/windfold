import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from windfold.config import AutoML, load_farm, load_models, load_protocol, load_turbine
from windfold.models import automl
from windfold.models.automl import PeerAutoml, centred, day_folds, peer_contrast, peer_features
from windfold.models.interface import FitContext, Model
from windfold.schemas import PREPARED_ROWS, conform

TURBINES = ("T1", "T2", "T3", "T4")
FEATURES = ("peer_power", "direction", "reference_speed", "turbine")
START = datetime(2024, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=10)


def prepared(
    powers: list[dict[str, float]],
    directions: list[float],
    speeds: list[float],
    start: datetime = START,
) -> pl.DataFrame:
    """prepared rows with y centered over each timestamp's valid turbines.

    each turbine's own anemometer reads 1.5 m/s plus its index above the reference speed, so a
    leak of it into a feature would show.
    """
    records = []
    for k, (power, direction, speed) in enumerate(zip(powers, directions, speeds, strict=True)):
        mean_log = sum(math.log(p) for p in power.values()) / len(power)
        mask = sum(1 << TURBINES.index(t) for t in power)
        for turbine, p in power.items():
            index = TURBINES.index(turbine)
            records.append(
                {
                    "timestamp": start + k * STEP,
                    "turbine": turbine,
                    "turbine_index": index,
                    "period": "development",
                    "power_kw": p,
                    "speed": speed + 1.5 + index,
                    "speed_std": 0.5,
                    "direction_raw": direction,
                    "direction": direction,
                    "nacelle_position": direction,
                    "farm_direction": direction,
                    "reference_speed": speed,
                    "speed_bin": 1,
                    "direction_bucket": int(direction // 5),
                    "steady": True,
                    "operating_mask": mask,
                    "waked": False,
                    "valid_turbines": len(power),
                    "y": math.log(p) - mean_log,
                    "temperature": 10.0,
                    "turbulence_intensity": 0.1,
                }
            )
    return conform(pl.DataFrame(records, schema=PREPARED_ROWS), PREPARED_ROWS)


def kw(rows: pl.DataFrame, mean: np.ndarray) -> np.ndarray:
    """the plan's median kW: the peers' geometric mean times exp of the leave-one-out contrast."""
    m = rows["valid_turbines"].cast(pl.Float64).to_numpy()
    log_power = pl.col("power_kw").log()
    peers = (log_power.sum().over("timestamp") - log_power) / (pl.col("valid_turbines") - 1)
    return np.exp(rows.select(peers).to_series().to_numpy() + m / (m - 1) * mean)


def test_peer_features_hold_only_the_other_valid_turbines_log_power():
    rows = prepared(
        [
            {"T1": 1000.0, "T2": 800.0, "T3": 500.0, "T4": 900.0},
            {"T1": 1200.0, "T3": 700.0, "T4": 650.0},
        ],
        directions=[90.0, 180.0],
        speeds=[7.0, 8.0],
    )
    features = peer_features(rows, TURBINES, FEATURES)
    assert features.columns == [
        "peer_power_T1",
        "peer_power_T2",
        "peer_power_T3",
        "peer_power_T4",
        "direction_sin",
        "direction_cos",
        "reference_speed",
        "turbine",
    ]
    ln = math.log
    expected = [
        [None, ln(800), ln(500), ln(900)],
        [ln(1000), None, ln(500), ln(900)],
        [ln(1000), ln(800), None, ln(900)],
        [ln(1000), ln(800), ln(500), None],
        [None, None, ln(700), ln(650)],
        [ln(1200), None, None, ln(650)],
        [ln(1200), None, ln(700), None],
    ]
    peers = features.select(pl.selectors.starts_with("peer_power_")).rows()
    for got_row, want_row in zip(peers, expected, strict=True):
        for got, want in zip(got_row, want_row, strict=True):
            assert got is None if want is None else got == pytest.approx(want, abs=1e-12)
    own = rows.select(pl.col("power_kw").log(), "speed")
    for column in features.drop("turbine").columns:
        for own_column in own.columns:
            same = np.isclose(features[column].to_numpy(), own[own_column].to_numpy(), atol=1e-12)
            assert not same.any(), (column, own_column)
    assert features["direction_sin"].to_list() == pytest.approx([1.0] * 4 + [0.0] * 3, abs=1e-12)
    assert features["direction_cos"].to_list() == pytest.approx([0.0] * 4 + [-1.0] * 3, abs=1e-12)
    assert features["reference_speed"].to_list() == [7.0] * 4 + [8.0] * 3
    assert features["turbine"].cast(pl.String).to_list() == rows["turbine"].to_list()


def test_peer_features_refuse_a_timestamp_missing_some_of_its_valid_rows():
    rows = prepared([{"T1": 1000.0, "T2": 800.0, "T3": 500.0}], directions=[90.0], speeds=[7.0])
    with pytest.raises(ValueError, match="valid_turbines"):
        peer_features(rows.filter(pl.col("turbine") != "T2"), TURBINES, ("peer_power",))


def test_contrast_is_leave_one_out_and_the_conversion_never_centres():
    rows = prepared([{"T1": 1000.0, "T2": 800.0, "T3": 500.0}], directions=[90.0], speeds=[7.0])
    contrast = peer_contrast(rows).to_numpy()
    by_hand = [
        math.log(1000 / math.sqrt(800 * 500)),
        math.log(800 / math.sqrt(1000 * 500)),
        math.log(500 / math.sqrt(1000 * 800)),
    ]
    assert contrast == pytest.approx(by_hand, abs=1e-12)
    m = rows["valid_turbines"].to_numpy()
    assert centred(contrast, m) == pytest.approx(rows["y"].to_list(), abs=1e-12)
    assert kw(rows, centred(contrast, m)) == pytest.approx([1000.0, 800.0, 500.0], rel=1e-12)
    guess = centred(np.array([0.3, 0.1, -0.5]), m)
    assert guess == pytest.approx([0.2, 0.2 / 3, -1 / 3], abs=1e-12)
    assert guess.sum() == pytest.approx(-0.2 / 3, abs=1e-12)


def test_day_folds_keep_each_calendar_day_in_one_fold():
    stamps = [
        datetime(2024, 3, day, hour, 50, tzinfo=UTC)
        for day in range(1, 11)
        for hour in range(23 - day % 4, 24)
    ]
    timestamps = pl.Series("timestamp", stamps, dtype=pl.Datetime("us", "UTC"))
    folds = day_folds(timestamps, folds=4, seed=7)
    frame = pl.DataFrame({"day": timestamps.dt.date(), "fold": folds})
    per_day = frame.group_by("day").agg(
        pl.col("fold").n_unique().alias("folds"), pl.col("fold").first()
    )
    assert per_day["folds"].to_list() == [1] * 10
    assert sorted(per_day["fold"].value_counts()["count"].to_list()) == [2, 2, 3, 3]
    assert day_folds(timestamps, folds=4, seed=7).to_list() == folds.to_list()
    with pytest.raises(ValueError, match="calendar days"):
        day_folds(timestamps, folds=11, seed=7)


def synthetic_farm(days: int, start: datetime, seed: int) -> pl.DataFrame:
    """four turbines; each loses power when the wind blows from within 30 degrees of its bearing."""
    rng = np.random.default_rng(seed)
    bearings = {"T1": 0.0, "T2": 90.0, "T3": 180.0, "T4": 270.0}
    powers, directions, speeds = [], [], []
    for _ in range(days * 144):
        direction = float(rng.uniform(0, 360))
        speed = float(rng.uniform(5, 10))
        power = {}
        for turbine, bearing in bearings.items():
            if rng.random() < 0.1:
                continue
            gap = abs((direction - bearing + 180) % 360 - 180)
            deficit = 0.3 * max(0.0, 1 - gap / 30)
            noise = math.exp(rng.normal(0, 0.02))
            power[turbine] = 2000 * (speed / 12) ** 3 * (1 - deficit) * noise
        if len(power) >= 2:
            powers.append(power)
            directions.append(direction)
            speeds.append(speed)
    return prepared(powers, directions, speeds, start=start)


def fit_small(folder: Path, every: int = 1):
    """a small peer automl fit on four days, of which it keeps every given whole timestamp."""
    model: Model = PeerAutoml(
        "peer_automl",
        features=FEATURES,
        settings=AutoML(
            preset="medium_quality",
            hyperparameters="toy",
            models=("GBM",),
            gpu_models=(),
            bag_folds=2,
            stack_levels=1,
        ),
        minutes=2.0,
        workdir=folder / "work",
    )
    train = synthetic_farm(days=4, start=START, seed=1)
    kept = train["timestamp"].unique().sort()[::every]
    train = train.filter(pl.col("timestamp").is_in(kept.implode()))
    context = FitContext(farm="synthetic", prepared_dir=folder, seed=0, threads=1, device=None)
    return model.fit(train, context), train


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("autogluon.tabular")
    return fit_small(tmp_path_factory.mktemp("automl"))


def test_peer_automl_never_sees_the_target_power(fitted):
    model, _ = fitted
    test = synthetic_farm(days=1, start=START + timedelta(days=10), seed=2).with_row_index()
    target = test.filter(pl.col("valid_turbines") == 4).row(5, named=True)
    same_time = pl.col("timestamp") == target["timestamp"]
    is_target = same_time & (pl.col("turbine") == target["turbine"])
    power = pl.when(is_target).then(pl.col("power_kw") * 0.7).otherwise(pl.col("power_kw"))
    changed = test.with_columns(power_kw=power).with_columns(
        y=pl.col("power_kw").log() - pl.col("power_kw").log().mean().over("timestamp")
    )
    assert changed.filter(same_time)["y"].to_list() != test.filter(same_time)["y"].to_list()
    assert changed["valid_turbines"].to_list() == test["valid_turbines"].to_list()

    before = model.predict(test.drop("index"))["mean"].to_numpy()
    after = model.predict(changed.drop("index"))["mean"].to_numpy()
    index = target["index"]
    others = test.filter(same_time & ~is_target)["index"].to_numpy()
    assert after[index] == pytest.approx(before[index], abs=1e-12)
    assert kw(changed, after)[index] == pytest.approx(kw(test, before)[index], rel=1e-12)
    assert not np.allclose(after[others], before[others])


def test_peer_automl_predicts_in_input_order_without_variance(fitted):
    model, train = fitted
    forward = model.predict(train)
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=3)
    backward = model.predict(shuffled)
    assert forward.schema == pl.Schema({"mean": pl.Float64, "variance": pl.Float64})
    assert forward.height == train.height
    assert forward["variance"].null_count() == train.height
    keys = list(zip(train["timestamp"], train["turbine"], strict=True))
    lookup = dict(zip(keys, forward["mean"], strict=True))
    order = zip(shuffled["timestamp"], shuffled["turbine"], strict=True)
    assert backward["mean"].to_list() == pytest.approx([lookup[key] for key in order], abs=1e-12)
    y = train["y"].to_numpy()
    assert np.mean((forward["mean"].to_numpy() - y) ** 2) < 0.8 * np.mean(y**2)


def test_peer_automl_records_its_fit(fitted):
    model, train = fitted
    record = model.record
    assert record["features"][:4] == [f"peer_power_{t}" for t in TURBINES]
    assert set(record["features"]) == set(model.predictor.features())
    assert record["train_rows"] == train.height
    assert record["bag_folds"] == 2 and record["stack_levels"] == 1
    assert record["fit_seconds"] > 0 and record["completed_models"]
    assert {row["model"] for row in record["leaderboard"]} == set(record["completed_models"])


def test_finish_writes_the_record_and_removes_the_work_folder(tmp_path):
    pytest.importorskip("autogluon.tabular")
    model, _ = fit_small(tmp_path, every=4)
    assert (tmp_path / "work").is_dir()
    model.finish(tmp_path / "split.json")
    record = json.loads((tmp_path / "split.json").read_text())
    assert record["completed_models"] == model.record["completed_models"]
    assert not (tmp_path / "work").exists()


class Scored:
    def __init__(self, minutes: float):
        self.record = {"minutes": minutes}
        self.removed = False

    def remove(self) -> None:
        self.removed = True


@pytest.mark.parametrize(("long_mse", "chosen"), [(49.0, 6.0), (48.95, 24.0), (60.0, 6.0)])
def test_the_longer_budget_needs_more_than_two_percent(monkeypatch, tmp_path, long_mse, chosen):
    fits = []

    def fake_score(model, split, context):
        fits.append(Scored(model.minutes))
        return {"mse": {6.0: 50.0, 24.0: long_mse}[model.minutes]}, fits[-1]

    monkeypatch.setattr(automl, "score", fake_score)
    calibration = prepared([{"T1": 900.0, "T2": 1000.0}] * 2, [10.0, 10.0], [8.0, 8.0])
    result = automl.choose_budget(
        calibration,
        window=(START, START + timedelta(days=10)),
        validation_fraction=0.2,
        embargo_days=1,
        candidate_minutes=(24.0, 6.0),
        min_relative_gain=0.02,
        features=FEATURES,
        settings=AutoML(
            preset="best_quality",
            hyperparameters="zeroshot",
            models=("GBM",),
            gpu_models=(),
            bag_folds=8,
            stack_levels=1,
        ),
        workdir=tmp_path,
        context=FitContext("synthetic", tmp_path, 0, 1, None),
    )
    assert result["minutes"] == chosen
    assert [c["minutes"] for c in result["candidates"]] == [6.0, 24.0]
    assert all(fit.removed for fit in fits)
    assert result["split"]["validation_start"] == (START + timedelta(days=8)).isoformat()


def test_build_model_reads_the_packaged_spec(tmp_path):
    protocol = load_protocol()
    farm = load_farm("penmanshiel")
    spec = load_models()["peer_automl"]
    model = automl.build_model(
        spec, farm, load_turbine(farm.turbine_type), protocol, minutes=6.0, workdir=tmp_path
    )
    assert model.settings == protocol.automl and model.features == FEATURES
    assert (model.minutes, model.workdir) == (6.0, tmp_path)
