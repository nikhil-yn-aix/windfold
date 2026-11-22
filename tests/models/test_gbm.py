import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

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
from windfold.models.gbm import (
    FittedGradientBoosting,
    GeometrySettings,
    GradientBoosting,
    blockage,
    build_model,
    projected_objective,
    wind_frame_geometry,
)
from windfold.models.interface import FitContext, Model
from windfold.schemas import PHYSICS, PHYSICS_FAMILIES, PREPARED_ROWS, conform

GROUP = np.array([0, 0, 0, 1, 1])
Y = np.array([0.05, -0.15, 0.10, 0.12, -0.12])


def projected_loss(f: np.ndarray, group: np.ndarray, y: np.ndarray) -> float:
    total = 0.0
    for t in np.unique(group):
        rows = group == t
        m = int(rows.sum())
        h = np.eye(m) - np.ones((m, m)) / m
        total += float(np.sum((h @ f[rows] - y[rows]) ** 2))
    return total


def test_projected_objective_matches_a_hand_calculation_on_two_timestamps():
    f = np.array([0.10, -0.20, 0.40, 0.30, 0.10])
    gradient, hessian = projected_objective(GROUP, Y)(f, lgb.Dataset(np.zeros((5, 1))))
    assert gradient == pytest.approx([-0.10, -0.30, 0.40, -0.04, 0.04], abs=1e-12)
    assert hessian == pytest.approx([4 / 3, 4 / 3, 4 / 3, 1.0, 1.0], abs=1e-12)


def test_projected_objective_is_the_derivative_of_the_dense_projected_loss():
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(4), [3, 5, 2, 4])
    y = rng.normal(size=group.size)
    f = rng.normal(size=group.size)
    gradient, hessian = projected_objective(group, y)(f, lgb.Dataset(np.zeros((group.size, 1))))
    step = 1e-5
    for i in range(group.size):
        bump = np.zeros(group.size)
        bump[i] = step
        up, mid, down = (projected_loss(f + s * bump, group, y) for s in (1, 0, -1))
        assert gradient[i] == pytest.approx((up - down) / (2 * step), abs=1e-6)
        assert hessian[i] == pytest.approx((up - 2 * mid + down) / step**2, rel=1e-4)
    shift = np.repeat(rng.normal(size=4), [3, 5, 2, 4])
    shifted, _ = projected_objective(group, y)(f + shift, lgb.Dataset(np.zeros((group.size, 1))))
    assert shifted == pytest.approx(gradient, abs=1e-12)


def test_lightgbm_descends_the_projected_loss():
    rng = np.random.default_rng(1)
    group = np.repeat(np.arange(200), 3)
    x = rng.normal(size=(group.size, 2))
    raw = np.sin(x[:, 0]) + 0.5 * x[:, 1] + np.repeat(rng.normal(size=200), 3)
    y = raw - (np.bincount(group, weights=raw) / 3)[group]
    params = {
        "objective": projected_objective(group, y),
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": 0,
        "verbosity": -1,
    }
    booster = lgb.train(params, lgb.Dataset(x), num_boost_round=30)
    fitted = np.asarray(booster.predict(x), dtype=np.float64)
    assert np.isfinite(fitted).all()
    assert projected_loss(fitted, group, y) < 0.5 * projected_loss(np.zeros(group.size), group, y)


TURBINES = ("T1", "T2", "T3", "T4")
START = datetime(2024, 1, 1, tzinfo=UTC)
GEOMETRY = GeometrySettings(
    along_max_d=30.0, across_max_d=3.0, upstream_neighbours=2, rotor_points=4000
)


def layout(east: list[float], north: list[float], hub: list[float]) -> pl.DataFrame:
    count = len(east)
    return pl.DataFrame(
        {
            "turbine": list(TURBINES[:count]),
            "latitude": [0.0] * count,
            "longitude": [0.0] * count,
            "x_m": [float(e) for e in east],
            "y_m": [float(n) for n in north],
            "hub_height_m": [float(h) for h in hub],
        }
    )


TURBINE = TurbineConfig(
    name="synthetic",
    rated_power_kw=2000.0,
    rotor_diameter_m=100.0,
    source="synthetic",
    power_curve=Curve(speed_ms=(3.0, 12.0), value=(0.0, 2000.0), source="synthetic"),
    thrust_curve=Curve(speed_ms=(3.0, 12.0), value=(0.8, 0.4), source="synthetic"),
)


def rows_at(direction: float, receivers: list[int], mask: int, turbines: int) -> pl.DataFrame:
    """minimal prepared rows of one timestamp for the geometry features."""
    return pl.DataFrame(
        {
            "timestamp": [START] * len(receivers),
            "turbine": [TURBINES[i] for i in receivers],
            "turbine_index": receivers,
            "farm_direction": [direction] * len(receivers),
            "operating_mask": [mask] * len(receivers),
            "valid_turbines": [len(receivers)] * len(receivers),
        },
        schema_overrides={"turbine_index": pl.UInt8, "operating_mask": pl.UInt32},
    ).with_columns(pl.col("valid_turbines").cast(pl.UInt8))


def close(got: float | None, want: float | None) -> bool:
    if got is None or want is None:
        return got is None and want is None
    return math.isclose(got, want, abs_tol=1e-9)


def same(got: list[tuple], want: list[tuple]) -> bool:
    return len(got) == len(want) and all(
        close(g, w)
        for got_row, want_row in zip(got, want, strict=True)
        for g, w in zip(got_row, want_row, strict=True)
    )


def test_wind_frame_geometry_by_hand():
    farm = farm_from_layout(layout([0, 500, 1000, 1100], [0, 0, 0, 100], [80.0] * 4), 100.0)
    rows = rows_at(270.0, [0, 1, 2, 3], 0b1111, 4)
    got = wind_frame_geometry(rows, farm, 30.0, 3.0, 2).rows()
    assert same(
        got,
        [
            (None, None, None, None, 0.0),
            (5.0, 0.0, None, None, 1.0),
            (5.0, 0.0, 10.0, 0.0, 2.0),
            (1.0, 1.0, 6.0, 1.0, 3.0),
        ],
    )
    without_t2 = wind_frame_geometry(rows_at(270.0, [2], 0b1101, 4), farm, 30.0, 3.0, 2).rows()
    assert same(without_t2, [(10.0, 0.0, None, None, 1.0)])
    narrow = wind_frame_geometry(rows_at(270.0, [3], 0b1111, 4), farm, 30.0, 0.5, 2).rows()
    assert same(narrow, [(None, None, None, None, 0.0)])
    from_north = wind_frame_geometry(rows_at(0.0, [2], 0b1111, 4), farm, 30.0, 3.0, 2).rows()
    assert same(from_north, [(1.0, -1.0, None, None, 1.0)])


def test_blockage_matches_closed_forms():
    def mean_inverse_distance(along: float) -> float:
        return 8.0 * (math.sqrt(along**2 + 0.25) - along)

    lens = 2 * 0.25 * math.acos(0.5) - 0.25 * math.sqrt(0.75)
    lens_share = lens / (math.pi * 0.25)
    cases = [
        (layout([0, 500], [0, 0], [80, 80]), 0b11, 1.0, mean_inverse_distance(5.0)),
        (layout([0, 500], [0, 50], [80, 80]), 0b11, lens_share, None),
        (layout([0, 500], [0, 0], [130, 80]), 0b11, lens_share, None),
        (layout([0, 500], [0, 0], [80, 80]), 0b10, 0.0, 0.0),
        (layout([0, 500], [0, 150], [80, 80]), 0b11, 0.0, 0.0),
    ]
    for frame, mask, ratio, inverse in cases:
        farm = farm_from_layout(frame, 100.0)
        got_ratio, got_inverse = blockage(rows_at(270.0, [1], mask, 2), farm, 4000)
        assert got_ratio[0] == pytest.approx(ratio, abs=2e-3)
        if inverse is not None:
            assert got_inverse[0] == pytest.approx(inverse, abs=1e-4)
    farm = farm_from_layout(layout([0, 500, 1500], [0, 0, 0], [80, 80, 80]), 100.0)
    ratio, inverse = blockage(rows_at(270.0, [2], 0b111, 3), farm, 4000)
    assert ratio[0] == pytest.approx(1.0)
    assert inverse[0] == pytest.approx(
        mean_inverse_distance(10.0) + mean_inverse_distance(15.0), abs=1e-4
    )


def synthetic_rows(
    timestamps: int, seed: int, farm_layout: pl.DataFrame, turbulent: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """four turbines in an east-west row 5 D apart; y from a wake deficit, turbine offsets and
    noise. each turbine's own anemometer and power are columns the models must ignore. when
    turbulent, winds blow along the row and each timestamp's turbulence ratio q scales the deficit
    by q ** -0.5."""
    rng = np.random.default_rng(seed)
    farm = farm_from_layout(farm_layout, 100.0)
    offsets = np.array([0.03, -0.01, 0.02, -0.04])
    records = []
    for k in range(timestamps):
        direction = float(rng.uniform(255, 285) if turbulent else rng.uniform(0, 360))
        speed = float(rng.uniform(5, 10))
        ratio = float(np.exp2(rng.uniform(-1, 1))) if turbulent else 1.0
        along, across = relations(farm.east, farm.north, direction, farm.diameter)
        deficit = (
            np.where((along > 0) & (np.abs(across) < 1), 0.35 * np.exp(-along / 8), 0.0).sum(axis=1)
            * ratio**-0.5
        )
        log_power = (
            np.log(2000 * (speed / 12) ** 3)
            + np.log1p(-np.minimum(deficit, 0.7))
            + offsets
            + rng.normal(0, 0.01, 4)
        )
        y = log_power - log_power.mean()
        for i in range(4):
            records.append(
                {
                    "timestamp": START + timedelta(minutes=10 * k),
                    "turbine": TURBINES[i],
                    "turbine_index": i,
                    "period": "development",
                    "power_kw": float(np.exp(log_power[i])),
                    "speed": speed * (1 - deficit[i]) + 0.3,
                    "speed_std": 0.5,
                    "direction_raw": direction,
                    "direction": direction,
                    "nacelle_position": direction,
                    "farm_direction": direction,
                    "reference_speed": speed,
                    "speed_bin": 1,
                    "direction_bucket": int(direction // 5),
                    "steady": True,
                    "operating_mask": 0b1111,
                    "waked": bool(deficit[i] > 0),
                    "valid_turbines": 4,
                    "y": float(y[i]),
                    "temperature": 10.0,
                    "turbulence_intensity": 0.1 * ratio,
                    "true_phi": float(np.log1p(-min(deficit[i], 0.7))),
                    "ratio": ratio,
                }
            )
    frame = pl.DataFrame(records)
    phi = frame.select("timestamp", "turbine", "true_phi", "ratio")
    rows = frame.drop("true_phi", "ratio").cast(dict(PREPARED_ROWS.items()))
    return conform(rows, PREPARED_ROWS), phi


ROW_LAYOUT = layout([0, 500, 1000, 1500], [0, 0, 0, 0], [80.0] * 4)
MENU = {
    "num_boost_round": 60,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l2": 0.0,
    "min_data_per_group": 10,
    "cat_smooth": 10.0,
}
FEATURES = {
    "identity_gbm": ("turbine", "direction", "reference_speed"),
    "geometry_gbm": ("wind_frame_geometry", "reference_speed"),
    "physics_guided_gbm": (
        "blockage_ratio",
        "inverse_blocking_distance",
        "physics_efficiency",
        "reference_speed",
    ),
}


def write_physics(folder: Path, phi: pl.DataFrame) -> None:
    """the constant-inflow table holds the true phi as niayifar's and the measured-inflow table
    as turbopark's, so the family a fit chooses shows which table it read."""
    rng = np.random.default_rng(5)
    true_phi = pl.col("true_phi")
    shared = {"physics_jensen": true_phi * 1.8 + pl.Series(rng.normal(0, 0.05, phi.height))}
    shared |= {f"physics_{family}_default": true_phi for family in PHYSICS_FAMILIES}
    constant = phi.select(
        "timestamp",
        "turbine",
        physics_niayifar=true_phi,
        physics_turbopark=true_phi * 0.3,
        **shared,
    )
    conform(constant, PHYSICS).write_parquet(folder / "physics.parquet")
    measured = phi.select(
        "timestamp",
        "turbine",
        "ratio",
        physics_niayifar=true_phi * 0.3,
        physics_turbopark=true_phi,
        **shared,
    )
    conform(measured, inflow.INFLOW_PHYSICS).write_parquet(folder / inflow.TABLE)


def later(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.with_columns(pl.col("timestamp") + timedelta(days=30))


@pytest.mark.parametrize("measured_inflow", [False, True])
@pytest.mark.parametrize("name", list(FEATURES))
def test_gbm_fits_predicts_centred_and_never_reads_the_target_sensors(
    name, measured_inflow, tmp_path
):
    train, phi_train = synthetic_rows(1500, seed=1, farm_layout=ROW_LAYOUT)
    test, phi_test = synthetic_rows(300, seed=2, farm_layout=ROW_LAYOUT)
    test, phi_test = later(test), later(phi_test)
    write_physics(tmp_path, pl.concat([phi_train, phi_test]))
    ROW_LAYOUT.write_parquet(tmp_path / "layout.parquet")
    spec = load_models()[f"{name}_inflow" if measured_inflow else name]
    model = build_model(
        spec, load_farm("hill_of_towie"), TURBINE, load_protocol(), configuration=MENU
    )
    conforming: Model = model
    context = FitContext(farm="synthetic", prepared_dir=tmp_path, seed=0, threads=1, device=None)
    fitted = conforming.fit(train, context)
    assert isinstance(fitted, FittedGradientBoosting)
    prediction = fitted.predict(test)
    assert prediction.schema == pl.Schema({"mean": pl.Float64, "variance": pl.Float64})
    assert prediction.height == test.height and prediction["variance"].null_count() == test.height
    sums = test.with_columns(prediction["mean"]).group_by("timestamp").agg(pl.col("mean").sum())
    assert np.abs(sums["mean"].to_numpy()).max() < 1e-12
    y = test["y"].to_numpy()
    assert np.mean((prediction["mean"].to_numpy() - y) ** 2) < 0.5 * np.mean(y**2)
    again = model.fit(train, context).predict(test)
    assert again["mean"].to_list() == prediction["mean"].to_list()
    target = test.filter(pl.col("waked")).row(0, named=True)
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
    reversed_rows = fitted.predict(test.reverse())["mean"].to_list()
    assert reversed_rows == pytest.approx(prediction["mean"].to_list()[::-1], abs=1e-12)
    if name == "physics_guided_gbm":
        assert fitted.family == ("turbopark" if measured_inflow else "niayifar")
    assert ("inflow" in fitted.booster.feature_name()) == measured_inflow


def test_measured_inflow_lets_a_gbm_follow_wakes_that_change_with_turbulence(tmp_path):
    train, phi_train = synthetic_rows(1500, seed=1, farm_layout=ROW_LAYOUT, turbulent=True)
    test, phi_test = synthetic_rows(300, seed=2, farm_layout=ROW_LAYOUT, turbulent=True)
    test, phi_test = later(test), later(phi_test)
    write_physics(tmp_path, pl.concat([phi_train, phi_test]))
    ROW_LAYOUT.write_parquet(tmp_path / "layout.parquet")
    context = FitContext(farm="synthetic", prepared_dir=tmp_path, seed=0, threads=1, device=None)
    errors = {}
    for measured_inflow in (False, True):
        model = GradientBoosting(
            "geometry_gbm",
            features=FEATURES["geometry_gbm"],
            parameters=MENU,
            turbine=TURBINE,
            geometry=GEOMETRY,
            measured_inflow=measured_inflow,
        )
        fitted = model.fit(train, context)
        table = model.table(test.reverse(), fitted.farm, None, tmp_path)
        expected = np.log(phi_test.reverse()["ratio"].to_numpy())
        assert ("inflow" in table.columns) == measured_inflow
        if measured_inflow:
            np.testing.assert_allclose(table["inflow"].to_numpy(), expected, rtol=0, atol=1e-15)
        residual = fitted.predict(test)["mean"].to_numpy() - test["y"].to_numpy()
        errors[measured_inflow] = float(np.mean(residual**2))
    assert errors[True] < 0.5 * errors[False]


def test_build_model_reads_the_packaged_specs():
    protocol = load_protocol()
    farm = load_farm("hill_of_towie")
    turbine = load_turbine(farm.turbine_type)
    for name, spec in load_models().items():
        if spec.learner != "gbm":
            continue
        model = build_model(spec, farm, turbine, protocol, configuration=MENU)
        assert model.name == name and model.features == spec.features
        assert model.geometry.along_max_d == protocol.kernel.along_max_d
        assert model.geometry.upstream_neighbours == protocol.gbm.upstream_neighbours
        assert model.diameter_m == turbine.rotor_diameter_m
