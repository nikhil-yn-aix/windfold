import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from windfold import inflow
from windfold.config import (
    Curve,
    TurbineConfig,
    load_farm,
    load_models,
    load_protocol,
    load_turbine,
)
from windfold.geometry import relations
from windfold.models.farm import farm_from_layout
from windfold.models.graph import (
    Edges,
    FittedGraph,
    Graph,
    GraphSettings,
    batches,
    build_model,
    centred_loss,
    contrasts,
    operating_sets,
    pretrain,
    pretraining_cases,
)
from windfold.models.interface import FitContext, Model

START = datetime(2024, 1, 1, tzinfo=UTC)
TURBINE = TurbineConfig(
    name="synthetic",
    rated_power_kw=2000.0,
    rotor_diameter_m=100.0,
    source="synthetic",
    power_curve=Curve(speed_ms=(3.0, 12.0), value=(0.0, 2000.0), source="synthetic"),
    thrust_curve=Curve(speed_ms=(3.0, 12.0), value=(0.8, 0.4), source="synthetic"),
)
EAST = [0.0, 500.0, 1000.0, 1500.0, 400.0]
NORTH = [0.0, 0.0, 0.0, 0.0, 600.0]
SETTINGS = GraphSettings(
    width=16,
    layers=2,
    gnn_dropout=0.0,
    encoder_dropout=0.0,
    rounds=40,
    batch_size=32,
    learning_rate=3e-3,
    learning_rate_decay=0.99,
    weight_decay=0.01,
)
STEPS = 12


def layout(east: list[float], north: list[float]) -> pl.DataFrame:
    count = len(east)
    return pl.DataFrame(
        {
            "turbine": [f"T{i + 1}" for i in range(count)],
            "latitude": [0.0] * count,
            "longitude": [0.0] * count,
            "x_m": east,
            "y_m": north,
            "hub_height_m": [80.0] * count,
        }
    )


def graph(edges: Edges, steps: int = STEPS, measured_inflow: bool = False, **changes) -> Graph:
    settings = GraphSettings(**{**SETTINGS.__dict__, **changes})
    return Graph(
        f"{edges}_graph",
        edges=edges,
        settings=settings,
        steps_per_round=steps,
        turbine=TURBINE,
        edge_distance_d=30.0,
        speed_scale_ms=10.0,
        measured_inflow=measured_inflow,
    )


@pytest.fixture(scope="module")
def cpu(tmp_path_factory: pytest.TempPathFactory) -> FitContext:
    prepared = tmp_path_factory.mktemp("prepared")
    layout(EAST, NORTH).write_parquet(prepared / "layout.parquet")
    return FitContext(farm="synthetic", prepared_dir=prepared, seed=0, threads=1, device=None)


def synthetic_rows(
    timestamps: int, seed: int, east=EAST, north=NORTH, turn=0.0, turbulent=False
) -> pl.DataFrame:
    """five turbines; y from wake deficits of upstream turbines, plus noise. turn rotates the
    farm direction, so with a layout turned the same way the wake pattern is unchanged. when
    turbulent, winds blow along the row and each timestamp's turbulence_ratio q scales the
    deficit by q ** -0.5."""
    rng = np.random.default_rng(seed)
    records = []
    for k in range(timestamps):
        direction = float(rng.uniform(255, 285) if turbulent else rng.uniform(0, 360))
        speed = float(rng.uniform(5, 10))
        ratio = float(np.exp2(rng.uniform(-1, 1))) if turbulent else 1.0
        valid = rng.random(len(east)) > 0.15
        if valid.sum() < 2:
            valid[:2] = True
        along, across = relations(np.array(east), np.array(north), direction, 100.0)
        source = valid[None, :] & (along > 0) & (np.abs(across) < 1.5)
        deficit = np.where(source, 0.3 * np.exp(-along / 6), 0.0).sum(axis=1) * ratio**-0.5
        log_power = np.log1p(-np.minimum(deficit, 0.7)) + rng.normal(0, 0.01, len(east))
        level = log_power[valid].mean()
        for i in np.flatnonzero(valid):
            records.append(
                {
                    "timestamp": START + timedelta(minutes=10 * k),
                    "turbine": f"T{i + 1}",
                    "turbine_index": int(i),
                    "farm_direction": (direction + turn) % 360,
                    "reference_speed": speed,
                    "speed": speed * (1 - deficit[i]),
                    "power_kw": 1000 * math.exp(log_power[i]),
                    "valid_turbines": int(valid.sum()),
                    "y": float(log_power[i] - level),
                    "turbulence_ratio": ratio,
                }
            )
    return pl.DataFrame(records).with_columns(
        pl.col("turbine_index").cast(pl.UInt8), pl.col("valid_turbines").cast(pl.UInt8)
    )


def test_loss_is_the_projected_loss_and_ignores_a_shift_per_timestamp():
    rng = np.random.default_rng(0)
    valid = torch.tensor([[True, True, True, False], [True, False, True, True]])
    y = torch.zeros(2, 4)
    for s in range(2):
        raw = torch.tensor(rng.normal(size=4), dtype=torch.float32) * valid[s]
        y[s] = (raw - raw[valid[s]].mean()) * valid[s]
    f = torch.tensor(rng.normal(size=(2, 4)), dtype=torch.float32)
    total = 0.0
    for s in range(2):
        m = int(valid[s].sum())
        h = np.eye(m) - np.ones((m, m)) / m
        total += float(np.sum((h @ f[s][valid[s]].numpy() - y[s][valid[s]].numpy()) ** 2))
    assert float(centred_loss(f, y, valid)) == pytest.approx(total / 6, rel=1e-6)
    shifted = f + torch.tensor([[2.5], [-7.0]])
    assert float(centred_loss(shifted, y, valid)) == pytest.approx(total / 6, rel=1e-5)


def rotated(turn_deg: float) -> tuple[list[float], list[float]]:
    turn = math.radians(turn_deg)
    east = [e * math.cos(turn) + n * math.sin(turn) for e, n in zip(EAST, NORTH, strict=True)]
    north = [-e * math.sin(turn) + n * math.cos(turn) for e, n in zip(EAST, NORTH, strict=True)]
    return east, north


@pytest.mark.parametrize("edges", ["wind_frame", "absolute"])
def test_only_the_wind_frame_graph_ignores_turning_layout_and_wind_together(edges, cpu):
    rows = synthetic_rows(8, seed=3)
    fitted = graph(edges, rounds=0).fit(synthetic_rows(4, seed=4), cpu)
    east, north = rotated(73.0)
    turned_farm = farm_from_layout(layout(east, north), 100.0)
    turned = FittedGraph(fitted.network, turned_farm, fitted.inputs, fitted.device, [], 0)
    turned_rows = rows.with_columns((pl.col("farm_direction") + 73.0) % 360)
    before = fitted.predict(rows)["mean"].to_numpy()
    after = turned.predict(turned_rows)["mean"].to_numpy()
    if edges == "wind_frame":
        assert np.abs(after - before).max() < 1e-5
    else:
        assert np.abs(after - before).max() > 1e-3


def test_graph_learns_predicts_in_order_and_repeats_exactly(cpu):
    train = synthetic_rows(400, seed=1)
    test = synthetic_rows(100, seed=2)
    model = graph("wind_frame")
    conforming: Model = model
    fitted = conforming.fit(train, cpu)
    assert isinstance(fitted, FittedGraph)
    curve = fitted.curve
    assert curve.height == SETTINGS.rounds
    assert curve["loss"][-1] < 0.5 * np.mean(train["y"].to_numpy() ** 2)
    prediction = fitted.predict(test)
    assert prediction.schema == pl.Schema({"mean": pl.Float64, "variance": pl.Float64})
    assert prediction["variance"].null_count() == test.height
    sums = test.with_columns(prediction["mean"]).group_by("timestamp").agg(pl.col("mean").sum())
    assert np.abs(sums["mean"].to_numpy()).max() < 1e-5
    y = test["y"].to_numpy()
    assert np.mean((prediction["mean"].to_numpy() - y) ** 2) < 0.6 * np.mean(y**2)
    shuffled = test.sample(fraction=1.0, shuffle=True, seed=5)
    lookup = dict(
        zip(zip(test["timestamp"], test["turbine"], strict=True), prediction["mean"], strict=True)
    )
    keys = zip(shuffled["timestamp"], shuffled["turbine"], strict=True)
    assert fitted.predict(shuffled)["mean"].to_list() == pytest.approx(
        [lookup[key] for key in keys], abs=1e-6
    )
    again = model.fit(train, cpu).predict(test)
    assert again["mean"].to_list() == prediction["mean"].to_list()
    target = test.filter(pl.col("speed") < pl.col("reference_speed")).row(0, named=True)
    same_time = pl.col("timestamp") == target["timestamp"]
    is_target = same_time & (pl.col("turbine") == target["turbine"])
    log_power = pl.col("power_kw").log()
    changed = test.with_columns(
        speed=pl.when(is_target).then(pl.col("speed") * 1.5).otherwise(pl.col("speed")),
        power_kw=pl.when(is_target).then(pl.col("power_kw") * 0.5).otherwise(pl.col("power_kw")),
    ).with_columns(
        y=pl.when(same_time)
        .then(log_power - log_power.mean().over("timestamp"))
        .otherwise(pl.col("y"))
    )
    assert changed.filter(same_time)["y"].to_list() != test.filter(same_time)["y"].to_list()
    assert fitted.predict(changed)["mean"].to_list() == prediction["mean"].to_list()


def test_measured_inflow_lets_a_graph_follow_wakes_that_change_with_turbulence(tmp_path):
    layout(EAST, NORTH).write_parquet(tmp_path / "layout.parquet")
    train = synthetic_rows(400, seed=1, turbulent=True)
    test = synthetic_rows(100, seed=2, turbulent=True).with_columns(
        pl.col("timestamp") + timedelta(days=30)
    )
    table = pl.concat([train, test]).select("timestamp", "turbine", ratio="turbulence_ratio")
    table.write_parquet(tmp_path / inflow.TABLE)
    train, test = train.drop("turbulence_ratio"), test.drop("turbulence_ratio")
    context = FitContext(farm="synthetic", prepared_dir=tmp_path, seed=0, threads=1, device=None)
    errors = {}
    for measured_inflow in (False, True):
        model = graph("wind_frame", measured_inflow=measured_inflow, rounds=200)
        fitted = model.fit(train, context)
        assert fitted.network.nodes[0].in_features == 1 + int(measured_inflow)
        prediction = fitted.predict(test)["mean"]
        shuffled = test.sample(fraction=1.0, shuffle=True, seed=5)
        pairs = zip(test["timestamp"], test["turbine"], strict=True)
        lookup = dict(zip(pairs, prediction, strict=True))
        keys = zip(shuffled["timestamp"], shuffled["turbine"], strict=True)
        assert fitted.predict(shuffled)["mean"].to_list() == pytest.approx(
            [lookup[key] for key in keys], abs=1e-6
        )
        errors[measured_inflow] = float(
            np.mean((prediction.to_numpy() - test["y"].to_numpy()) ** 2)
        )
    assert errors[True] < 0.6 * errors[False]


def test_batches_are_slices_of_a_permutation_reshuffled_whenever_a_pass_ends():
    stream = batches(10, 4, torch.Generator().manual_seed(0))
    taken = [next(stream).tolist() for _ in range(6)]
    assert all(len(batch) == 4 for batch in taken)
    passes = [taken[0] + taken[1], taken[2] + taken[3], taken[4] + taken[5]]
    assert all(len(set(one)) == 8 and set(one) <= set(range(10)) for one in passes)
    assert passes[0] != passes[1] != passes[2]
    again = batches(10, 4, torch.Generator().manual_seed(0))
    assert [next(again).tolist() for _ in range(6)] == taken
    few = batches(3, 4, torch.Generator().manual_seed(0))
    assert sorted(next(few).tolist()) == sorted(next(few).tolist()) == [0, 1, 2]


def test_a_round_is_a_fixed_number_of_steps_whatever_the_data_size(cpu, monkeypatch):
    rates: list[float] = []
    step = torch.optim.AdamW.step

    def counted(self, *args, **kwargs):
        rates.append(self.param_groups[0]["lr"])
        return step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted)
    model = graph("wind_frame", rounds=3, learning_rate=1e-3, learning_rate_decay=0.5)
    for timestamps in (20, 300):
        rates.clear()
        fitted = model.fit(synthetic_rows(timestamps, seed=9), cpu)
        assert fitted.curve["round"].to_list() == [1, 2, 3]
        assert rates == pytest.approx([1e-3 * 0.5**r for r in range(3) for _ in range(STEPS)])


def masks_of(rows: pl.DataFrame) -> np.ndarray:
    group = np.unique(rows["timestamp"].to_physical().to_numpy(), return_inverse=True)[1]
    masks = np.zeros(group.max() + 1, dtype=np.int64)
    np.add.at(masks, group, 1 << rows["turbine_index"].to_numpy().astype(np.int64))
    return masks


def test_pretraining_cases_come_from_conditions_and_phi_only():
    calibration = synthetic_rows(30, seed=6)
    sets = operating_sets(masks_of(calibration), 5)
    assert sets.all(axis=1).any()
    assert len(np.unique(sets, axis=0)) == len(sets)
    masks, degrees, bins, valid = pretraining_cases(sets, np.arange(0, 360, 90), 2)
    assert len(masks) == len(sets) * 4 * 2
    assert set(masks.tolist()) == set(masks_of(calibration).tolist()) | {31}
    assert ((masks[:, None] >> np.arange(5)) & 1 == valid).all()
    assert set(degrees.tolist()) == {0, 90, 180, 270} and set(bins.tolist()) == {0, 1}
    phi = np.where(valid, np.log(0.5 + valid.cumsum(axis=1)), np.nan)
    target = contrasts(phi, valid)
    assert np.allclose(np.where(valid, target, 0).sum(axis=1), 0.0)
    assert (target[~valid] == 0).all()
    louder = calibration.with_columns(pl.col("power_kw") * 3, pl.col("y") * 0)
    assert (operating_sets(masks_of(louder), 5) == sets).all()


def synthetic_phi(masks: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    on = ((masks[:, None] >> np.arange(5)) & 1).astype(bool)
    along, across = relations(np.array(EAST), np.array(NORTH), degrees.astype(float), 100.0)
    source = on[:, None, :] & (along > 0) & (np.abs(across) < 1.5)
    deficit = np.where(source, 0.3 * np.exp(-along / 6), 0.0).sum(axis=2)
    return np.where(on, np.log1p(-np.minimum(deficit, 0.7)), np.nan)


def test_pretrained_weights_seed_the_fine_tuning(cpu):
    rng = np.random.default_rng(7)
    sets = operating_sets(rng.integers(3, 32, size=6), 5)
    masks, degrees, bins, valid = pretraining_cases(sets, np.arange(360), 1)
    target = contrasts(synthetic_phi(masks, degrees), valid)
    cases = (degrees.astype(float), np.full(len(degrees), 7.25), valid, target)
    teacher = graph("wind_frame", steps=100)
    pretrained = pretrain(teacher, cases, rounds=3, context=cpu)
    assert pretrained.curve.height == 3
    assert pretrained.curve["loss"][-1] < pretrained.curve["loss"][0]
    rows = synthetic_rows(60, seed=8)
    student = Graph(
        "pretrained_graph",
        edges="wind_frame",
        settings=GraphSettings(**{**SETTINGS.__dict__, "rounds": 0}),
        steps_per_round=STEPS,
        turbine=TURBINE,
        edge_distance_d=30.0,
        speed_scale_ms=10.0,
        initial=pretrained.state(),
    )
    untrained = student.fit(rows, cpu).predict(rows)["mean"].to_numpy()
    assert np.allclose(untrained, pretrained.predict(rows)["mean"].to_numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="wind-frame"):
        pretrain(graph("absolute"), cases, rounds=1, context=cpu)


def test_build_model_reads_the_packaged_specs():
    protocol = load_protocol()
    farm = load_farm("kelmarsh")
    turbine = load_turbine(farm.turbine_type)
    specs = load_models()
    configuration = {**SETTINGS.__dict__, "batch_size": 128}
    absolute = build_model(
        specs["absolute_graph"], farm, turbine, protocol, configuration=configuration
    )
    assert absolute.inputs.edges == "absolute" and absolute.settings.batch_size == 128
    assert absolute.steps_per_round == protocol.graph.steps_per_round
    assert absolute.inputs.edge_distance_d == protocol.processing.validity_distance_d
    wind_frame = build_model(
        specs["wind_frame_graph"], farm, turbine, protocol, configuration=configuration
    )
    state = wind_frame.network().state_dict()
    pretrained = build_model(
        specs["pretrained_graph"],
        farm,
        turbine,
        protocol,
        configuration=configuration,
        initial=state,
    )
    assert pretrained.inputs.edges == "wind_frame" and pretrained.initial is state
    with pytest.raises(ValueError, match="pretrained"):
        build_model(specs["pretrained_graph"], farm, turbine, protocol, configuration=configuration)
