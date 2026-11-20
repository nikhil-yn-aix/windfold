import json
import math
import shutil
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from windfold import fingerprints
from windfold.config import for_farm, load_menu, load_models
from windfold.geometry import relations
from windfold.locations import (
    budget_file,
    chosen_file,
    pretraining_files,
    search_file,
    tuning_file,
)
from windfold.models import graph
from windfold.models.menus import write_menu
from windfold.schemas import PREPARED_ROWS, conform
from windfold.tasks import calibration

KELMARSH = [f"KWF{i}" for i in range(1, 7)]
SMARTEOLE = [f"SMV{i}" for i in range(1, 8)]
GBM_MENU = [
    {
        "num_boost_round": 20,
        "learning_rate": rate,
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l2": 0.0,
        "min_data_per_group": 5,
        "cat_smooth": 1.0,
    }
    for rate in (0.1, 0.05)
]
GRAPH_MENU = [
    {
        "width": 8,
        "layers": layers,
        "gnn_dropout": 0.0,
        "encoder_dropout": 0.0,
        "rounds": 2,
        "batch_size": 64,
        "learning_rate": 0.003,
        "learning_rate_decay": 0.99,
        "weight_decay": 0.01,
    }
    for layers in (2, 3)
]
TOY_AUTOML = """preset = "medium_quality"
hyperparameters = "toy"
models = ["GBM"]
gpu_models = []
bag_folds = 2
stack_levels = 0"""
PACKAGED_AUTOML = """preset = "best_quality"
hyperparameters = "zeroshot"
models = ["GBM", "CAT", "XGB", "NN_TORCH", "FASTAI"]
gpu_models = ["NN_TORCH", "FASTAI"]
bag_folds = 8
stack_levels = 1"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


def layout(names: list[str]) -> pl.DataFrame:
    angle = np.linspace(0, 2 * math.pi, len(names), endpoint=False)
    return pl.DataFrame(
        {
            "turbine": names,
            "latitude": 0.0,
            "longitude": 0.0,
            "x_m": 400.0 * np.cos(angle),
            "y_m": 400.0 * np.sin(angle),
            "hub_height_m": 80.0,
        }
    )


def prepared_rows(
    frame: pl.DataFrame, start: datetime, end: datetime, period: str, seed: int
) -> pl.DataFrame:
    """rows every 4 hours over [start, end) with wake deficits, some turbines stopped."""
    rng = np.random.default_rng(seed)
    names = frame["turbine"].to_list()
    east, north = frame["x_m"].to_numpy(), frame["y_m"].to_numpy()
    count = len(names)
    records = []
    stamp = start
    while stamp < end:
        direction = float(rng.uniform(0, 360))
        speed = float(rng.uniform(5, 10))
        on = np.ones(count, dtype=bool)
        if rng.random() < 0.3:
            on[rng.integers(count)] = False
        mask = int((on * (1 << np.arange(count))).sum())
        along, across = relations(east, north, direction, 92.0)
        source = on[None, :] & (along > 0) & (np.abs(across) < 1.5)
        deficit = np.minimum(np.where(source, 0.3 * np.exp(-along / 6), 0).sum(axis=1), 0.7)
        log_power = np.log(1500 * (1 - deficit)) + rng.normal(0, 0.01, count)
        level = log_power[on].mean()
        for i in np.flatnonzero(on):
            records.append(
                {
                    "timestamp": stamp,
                    "turbine": names[i],
                    "turbine_index": int(i),
                    "period": period,
                    "power_kw": float(np.exp(log_power[i])),
                    "speed": speed,
                    "speed_std": 0.5,
                    "direction_raw": direction,
                    "direction": direction,
                    "nacelle_position": direction,
                    "farm_direction": direction,
                    "reference_speed": speed,
                    "speed_bin": int(np.digitize(speed, [6.5, 8.0])),
                    "direction_bucket": int(direction // 5),
                    "steady": True,
                    "operating_mask": mask,
                    "waked": bool(deficit[i] > 0),
                    "valid_turbines": int(on.sum()),
                    "y": float(log_power[i] - level),
                    "temperature": 10.0,
                    "turbulence_intensity": 0.1,
                }
            )
        stamp += timedelta(hours=4)
    return conform(pl.DataFrame(records).cast(dict(PREPARED_ROWS.items())), PREPARED_ROWS)


@pytest.fixture
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """the packaged configs with small menus, training and budgets, and the data, runs and
    output folders, all still empty."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    configs = tmp_path / "configs"
    shutil.copytree(Path(str(resources.files("windfold") / "configs")), configs)
    for learner, menu in (("gbm", GBM_MENU), ("graph", GRAPH_MENU)):
        (configs / "menus" / f"{learner}.toml").unlink()
        write_menu(configs / "menus" / f"{learner}.toml", menu, seed=1)
    protocol = configs / "protocol.toml"
    replace_once(protocol, "steps_per_round = 100", "steps_per_round = 3")
    replace_once(protocol, "pretrain_rounds = 100", "pretrain_rounds = 2")
    replace_once(protocol, "candidate_minutes = [6.0, 24.0]", "candidate_minutes = [0.5, 1.0]")
    replace_once(protocol, PACKAGED_AUTOML, TOY_AUTOML)
    return {
        "configs": configs,
        "data": tmp_path / "data",
        "runs": tmp_path / "runs",
        "out": tmp_path / "out",
    }


@pytest.fixture
def world(configured: dict[str, Path]) -> dict[str, Path]:
    """configured, with prepared calibration rows of kelmarsh and development rows of smarteole."""
    data = configured["data"]
    farms = {
        "kelmarsh": (KELMARSH, datetime(2017, 6, 12, tzinfo=UTC), datetime(2018, 1, 1, tzinfo=UTC)),
        "smarteole": (
            SMARTEOLE,
            datetime(2020, 3, 1, tzinfo=UTC),
            datetime(2020, 3, 15, tzinfo=UTC),
        ),
    }
    for farm, (names, start, end) in farms.items():
        folder = data / "prepared" / farm
        folder.mkdir(parents=True)
        frame = layout(names)
        frame.write_parquet(folder / "layout.parquet")
        period = "calibration" if farm == "kelmarsh" else "development"
        prepared_rows(frame, start, end, period, seed=len(farm)).write_parquet(
            folder / "rows.parquet"
        )
        fit = {"teacher": "jensen", "teacher_parameters": {"k": 0.05, "spread_deg": 2.0}}
        (folder / "physics_fit.json").write_text(json.dumps(fit))
    return configured


def written(folder: Path) -> set[str]:
    return {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}


def args(world: dict[str, Path]) -> dict[str, str]:
    return {"runs_dir": str(world["runs"]), "configs": str(world["configs"])}


def test_a_change_to_one_learner_reruns_only_its_own_calibration_jobs(tmp_path):
    source = tmp_path / "windfold"
    shutil.copytree(fingerprints.SOURCE, source, ignore=shutil.ignore_patterns("__pycache__"))
    jobs = ("tune_gbm", "tune_graph", "choose", "pretrain", "budget")

    def hashes() -> dict[str, str]:
        fingerprints.forget()
        return {
            job: fingerprints.function_hash(f"windfold.tasks.calibration:{job}", source)
            for job in jobs
        }

    before = hashes()
    for module, rerun in (
        ("gbm", {"tune_gbm"}),
        ("graph", {"tune_graph", "pretrain"}),
        ("automl", {"budget"}),
        ("menus", {"tune_gbm", "tune_graph", "choose", "budget"}),
    ):
        path = source / "models" / f"{module}.py"
        path.write_text(path.read_text() + "\n\nCHANGED = 1\n")
        after = hashes()
        assert {job for job in jobs if after[job] != before[job]} == rerun, module
        before = after
    fingerprints.forget()


@pytest.mark.parametrize(
    ("job", "model", "index"),
    [
        ("tune_gbm", "identity_gbm", 1),
        ("tune_graph", "wind_frame_graph", 0),
    ],
)
def test_tune_scores_one_configuration_on_the_calibration_split(world, job, model, index):
    getattr(calibration, job)(
        "kelmarsh", model, index, str(world["data"]), **args(world), out_dir=world["out"]
    )
    assert written(world["out"]) == {tuning_file("kelmarsh", model, index)}
    record = json.loads((world["out"] / tuning_file("kelmarsh", model, index)).read_text())
    learner = "gbm" if model.endswith("gbm") else "graph"
    assert record["configuration"] == load_menu(learner, world["configs"])[index]
    assert record["split"]["validation_start"] == "2017-11-21T09:36:00+00:00"
    assert record["split"]["embargo_start"] == "2017-11-14T09:36:00+00:00"
    assert 0 < record["mse"] and record["validation_rows"] == record["split"]["validation_rows"]
    if learner == "graph":
        assert len(record["training_loss"]) == GRAPH_MENU[index]["rounds"]


@pytest.mark.parametrize(
    ("job", "model"),
    [
        ("tune_gbm", "wind_frame_graph"),
        ("tune_graph", "identity_gbm"),
        ("tune_graph", "pretrained_graph"),
    ],
)
def test_tune_refuses_a_model_without_a_search_of_its_own(configured, job, model):
    with pytest.raises(ValueError, match=model):
        getattr(calibration, job)(
            "kelmarsh",
            model,
            0,
            str(configured["data"]),
            **args(configured),
            out_dir=configured["out"],
        )


def test_choose_takes_the_lowest_mse_or_the_best_mean_rank(configured):
    runs = configured["runs"]
    for i, mse in enumerate([0.2, 0.1]):
        row = {"index": i, "mse": mse, "waked_mse": mse, "fit_seconds": 1.0, "fit_rows": 10}
        target = runs / tuning_file("kelmarsh", "identity_gbm", i)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(row | {"validation_rows": 5, "split": {"cut": i}}))
    calibration.choose(
        "kelmarsh", "identity_gbm", [], **args(configured), out_dir=configured["out"]
    )
    chosen = json.loads((configured["out"] / chosen_file("kelmarsh", "identity_gbm")).read_text())
    assert chosen["index"] == 1 and chosen["configuration"] == GBM_MENU[1]
    assert chosen["evidence"]["split"] == {"cut": 1}
    assert written(configured["out"]) == {
        search_file("kelmarsh", "identity_gbm"),
        chosen_file("kelmarsh", "identity_gbm"),
    }
    for farm, mse in (("kelmarsh", [0.2, 0.1]), ("penmanshiel", [0.3, 0.1])):
        target = runs / search_file(farm, "identity_gbm")
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"index": [1, 0], "mse": mse[::-1]}).write_parquet(target)
    out = configured["out"] / "smarteole"
    calibration.choose(
        "smarteole", "identity_gbm", ["kelmarsh", "penmanshiel"], **args(configured), out_dir=out
    )
    chosen = json.loads((out / chosen_file("smarteole", "identity_gbm")).read_text())
    assert chosen["index"] == 1 and chosen["evidence"]["mean_rank"] == 1.0
    search = pl.read_parquet(out / search_file("smarteole", "identity_gbm"))
    assert search["mean_rank"].to_list() == [2.0, 1.0]
    with pytest.raises(ValueError, match="wind_frame_graph"):
        calibration.choose("kelmarsh", "pretrained_graph", [], **args(configured), out_dir=out)


@pytest.mark.parametrize("farm", ["kelmarsh", "smarteole"])
def test_pretrain_learns_from_the_teacher_on_calibration_operating_sets(world, farm):
    source = world["runs"] / chosen_file(farm, "wind_frame_graph")
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"index": 1, "configuration": GRAPH_MENU[1]}))
    folder = world["data"] / "prepared" / farm
    rows = pl.read_parquet(folder / "rows.parquet")
    masks = set(rows["operating_mask"].unique().to_list())
    if farm == "kelmarsh":
        all_on, two_stopped = 63, 63 & ~3
        later = rows.filter(pl.col("operating_mask") == all_on, pl.col("turbine_index") >= 2)
        later = later.with_columns(
            pl.col("timestamp") + timedelta(days=400),
            period=pl.lit("development"),
            operating_mask=pl.lit(two_stopped, rows.schema["operating_mask"]),
            valid_turbines=pl.lit(4, rows.schema["valid_turbines"]),
        )
        assert two_stopped not in masks and later.height > 0
        pl.concat([rows, later]).write_parquet(folder / "rows.parquet")
    calibration.pretrain(farm, str(world["data"]), **args(world), out_dir=world["out"])
    assert written(world["out"]) == set(pretraining_files(farm))
    weights, curve, note = (world["out"] / p for p in pretraining_files(farm))
    record = json.loads(note.read_text())
    if farm == "kelmarsh":
        assert record["calibration_operating_sets"] == len(masks)
        assert record["operating_sets"] == len(masks | {63})
    else:
        assert (record["calibration_operating_sets"], record["operating_sets"]) == (0, 1)
    assert record["cases"] == record["operating_sets"] * 360 * 3
    assert record["teacher"] == "jensen" and record["seed"] == 0
    assert (record["rounds"], record["steps_per_round"]) == (2, 3)
    assert pl.read_parquet(curve)["round"].to_list() == [1, 2]
    protocol, farm_config, turbine = for_farm(farm, None, world["configs"])
    network = graph.build_model(
        load_models()["wind_frame_graph"],
        farm_config,
        turbine,
        protocol,
        configuration=GRAPH_MENU[1],
    ).network()
    network.load_state_dict(torch.load(weights))


def test_budget_from_the_calibration_window_or_by_majority(world):
    calibration.budget("kelmarsh", [], str(world["data"]), **args(world), out_dir=world["out"])
    assert written(world["out"]) == {budget_file("kelmarsh")}
    choice = json.loads((world["out"] / budget_file("kelmarsh")).read_text())
    assert choice["minutes"] in (0.5, 1.0) and len(choice["candidates"]) == 2
    assert all(c["mse"] > 0 and c["record"]["completed_models"] for c in choice["candidates"])
    assert not (world["out"] / "work").exists()
    for chosen, expected in (
        ({"a": 6.0, "b": 24.0, "c": 24.0}, 24.0),
        ({"a": 24.0, "b": 6.0}, 6.0),
    ):
        runs = world["runs"] / str(expected) / str(len(chosen))
        for farm, minutes in chosen.items():
            target = runs / budget_file(farm)
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"minutes": minutes}))
        out = runs / "out"
        calibration.budget(
            "smarteole", list(chosen), "", str(runs), str(world["configs"]), out_dir=out
        )
        assert json.loads((out / budget_file("smarteole")).read_text())["minutes"] == expected
