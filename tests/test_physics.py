import json
from typing import Any

import numpy as np
import polars as pl
import pytest
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.ground_models.ground_models import Mirror
from py_wake.literature.gaussian_models import Niayifar_PorteAgel_2016
from py_wake.literature.noj import Jensen_1983
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.site import UniformSite
from py_wake.superposition_models import SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

from windfold import physics
from windfold.config import Curve, TurbineConfig
from windfold.schemas import PHYSICS

SPEED = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 25.0)
POWER = (0.0, 64.0, 159.0, 314.0, 511.0, 767.0, 1096.0, 1439.0, 1700.0, 1912.0, 2000.0, 2050.0)
THRUST = (0.0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.05)
TURBINE = TurbineConfig(
    name="test",
    rated_power_kw=2050.0,
    rotor_diameter_m=82.0,
    source="test",
    power_curve=Curve(speed_ms=SPEED, value=POWER, source="test"),
    thrust_curve=Curve(speed_ms=SPEED, value=THRUST, source="test"),
)
SPEEDS = np.array([5.75, 7.25, 9.0])
FLOOR = 0.01 * 2050.0 * 1000.0
EAST = np.array([0.0, 410.0, 780.0, 120.0, 560.0, -300.0])
NORTH = np.array([0.0, 30.0, -60.0, 420.0, 470.0, 260.0])
HUB = np.array([59.0, 59.0, 69.0, 59.0, 69.0, 59.0])
PARAMETERS = {
    "jensen": {"k": 0.06, "spread_deg": 0.0},
    "niayifar": {"ti": 0.09, "spread_deg": 0.0},
    "turbopark": {"a": 0.04, "ti": 0.09, "spread_deg": 0.0},
}


def plain_turbine() -> WindTurbines:
    return WindTurbine(
        "plain", 82.0, 59.0, powerCtFunction=PowerCtTabular(SPEED, POWER, "kW", THRUST)
    )


def literature_model(family: str, parameters: dict):
    if family == "jensen":
        return Jensen_1983(UniformSite(), plain_turbine(), k=parameters["k"])
    if family == "niayifar":
        return Niayifar_PorteAgel_2016(UniformSite(ti=parameters["ti"]), plain_turbine())
    assert parameters["a"] == 0.04
    return Nygaard_2022(UniformSite(ti=parameters["ti"]), plain_turbine())


def direct_power(family, parameters, members, direction, speed):
    model = literature_model(family, parameters)
    result: Any = model(EAST[members], NORTH[members], h=HUB[members], wd=[direction], ws=[speed])
    out = np.full(len(EAST), np.nan)
    out[members] = result.Power.values[:, 0, 0]
    return out


def direct_phi(power_w, speed):
    free = plain_turbine().power(speed)
    return np.log(np.maximum(power_w, FLOOR)) - np.log(max(free, FLOOR))


def mask_of(members) -> int:
    return int(sum(1 << int(i) for i in members))


def prepared_rows(seed: int, timestamps: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    full = (1 << len(EAST)) - 1
    masks = np.where(
        rng.random(timestamps) < 0.3, full & ~(1 << rng.integers(0, 6, timestamps)), full
    )
    direction = rng.uniform(0.0, 360.0, timestamps)
    speed_bin = rng.integers(0, 3, timestamps)
    t, i = np.nonzero((masks[:, None] >> np.arange(len(EAST))) & 1)
    stamps = np.datetime64("2023-06-01T00:00", "us") + t * np.timedelta64(10, "m")
    return pl.DataFrame(
        {
            "timestamp": pl.Series(stamps).dt.replace_time_zone("UTC"),
            "turbine": [f"T{k}" for k in i],
            "turbine_index": pl.Series(i, dtype=pl.UInt8),
            "operating_mask": pl.Series(masks[t], dtype=pl.UInt32),
            "farm_direction": direction[t],
            "speed_bin": pl.Series(speed_bin[t], dtype=pl.UInt8),
        }
    )


def rows_phi(rows, family, parameters, truncate_sd=4.0):
    masks, degrees, bins, inverse = physics.row_keys(rows)
    model = physics.wake_model(family, parameters, physics.pywake_turbine(TURBINE))
    phi = physics.phi_keys(
        model,
        EAST,
        NORTH,
        HUB,
        masks,
        degrees,
        bins,
        SPEEDS,
        parameters["spread_deg"],
        truncate_sd,
        FLOOR,
    )
    return phi[inverse, rows["turbine_index"].to_numpy().astype(int)]


@pytest.mark.parametrize("family", physics.WAKE_PARAMETERS)
def test_cached_phi_equals_a_direct_pywake_call(family):
    rows = prepared_rows(seed=1, timestamps=60)
    phi = rows_phi(rows, family, PARAMETERS[family])
    rng = np.random.default_rng(2)
    for r in rng.choice(rows.height, 25, replace=False):
        row = rows.row(int(r), named=True)
        members = np.flatnonzero((row["operating_mask"] >> np.arange(len(EAST))) & 1)
        speed = SPEEDS[row["speed_bin"]]
        degree = float(np.mod(np.rint(row["farm_direction"]), 360))
        expected = direct_phi(
            direct_power(family, PARAMETERS[family], members, degree, speed), speed
        )
        assert phi[r] == pytest.approx(expected[row["turbine_index"]], rel=1e-9, abs=1e-12)


def test_turbopark_is_nygaard_with_the_expansion_set():
    thrust = Curve(speed_ms=SPEED, value=(0.0, 0.99, 0.98, *THRUST[3:]), source="test")
    wt = physics.pywake_turbine(TURBINE.model_copy(update={"thrust_curve": thrust}))
    deficit = TurboGaussianDeficit(
        A=0.06,
        ct2a=ct2a_mom1d,
        groundModel=Mirror(superpositionModel=SquaredSum()),
        rotorAvgModel=GaussianOverlapAvgModel(),
        ctlim=0.96,
    )
    deficit.WS_key = "WS_jlk"
    superposition: Any = SquaredSum()
    oracle = PropagateDownwind(
        UniformSite(ti=0.07), wt, wake_deficitModel=deficit, superpositionModel=superposition
    )
    ours = physics.wake_model("turbopark", {"a": 0.06, "ti": 0.07}, wt)
    published = physics.wake_model("turbopark", {"a": 0.04, "ti": 0.07}, wt)
    wd, ws = np.arange(0.0, 360.0, 7.0), np.array([4.5, *SPEEDS])
    a: Any = ours(EAST, NORTH, h=HUB, wd=wd, ws=ws)
    b: Any = oracle(EAST, NORTH, h=HUB, wd=wd, ws=ws)
    c: Any = published(EAST, NORTH, h=HUB, wd=wd, ws=ws)
    d: Any = Nygaard_2022(UniformSite(ti=0.07), wt)(EAST, NORTH, h=HUB, wd=wd, ws=ws)
    assert np.array_equal(a.Power.values, b.Power.values)
    assert np.array_equal(c.Power.values, d.Power.values)
    assert not np.array_equal(a.Power.values, c.Power.values)


@pytest.mark.parametrize("family", physics.WAKE_PARAMETERS)
def test_phi_is_zero_for_a_free_stream_turbine(family):
    east, north, hub = np.array([0.0, 400.0, 800.0]), np.zeros(3), np.full(3, 59.0)
    parameters = {**PARAMETERS[family], "spread_deg": 3.0}
    model = physics.wake_model(family, parameters, physics.pywake_turbine(TURBINE))
    still = physics.phi_keys(model, east, north, hub, [7], [270], [1], SPEEDS, 0.0, 4.0, FLOOR)
    assert still[0, 0] == 0.0 and np.all(still[0, 1:] < -0.01)
    spread = physics.phi_keys(
        model, east, north, hub, [7, 7], [270, 0], [1, 1], SPEEDS, 3.0, 4.0, FLOOR
    )
    assert spread[0, 0] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(spread[1], 0.0, atol=1e-14)
    assert np.all(spread[0, 1:] < -0.01)


@pytest.mark.parametrize("family", physics.WAKE_PARAMETERS)
def test_direction_spread_equals_a_brute_force_average_over_directions(family):
    spread, truncate = 2.6, 4.0
    parameters = {**PARAMETERS[family], "spread_deg": spread}
    members = np.array([0, 1, 2, 4, 5])
    model = physics.wake_model(family, parameters, physics.pywake_turbine(TURBINE))
    degree, speed_bin = 263, 1
    phi = physics.phi_keys(
        model,
        EAST,
        NORTH,
        HUB,
        [mask_of(members)],
        [degree],
        [speed_bin],
        SPEEDS,
        spread,
        truncate,
        FLOOR,
    )
    offsets = np.arange(-11, 12)
    assert int(truncate * spread + 0.5) == 10
    weights = np.where(np.abs(offsets) <= 10, norm.pdf(offsets, scale=spread), 0.0)
    weights /= weights.sum()
    average = sum(
        w * direct_power(family, PARAMETERS[family], members, float(degree + o), SPEEDS[speed_bin])
        for o, w in zip(offsets, weights, strict=True)
    )
    np.testing.assert_allclose(
        phi[0], direct_phi(average, SPEEDS[speed_bin]), rtol=1e-9, atol=1e-12
    )


def test_partial_rings_equal_full_rings_and_scipy_smoothing():
    parameters = {"k": 0.05, "spread_deg": 4.3}
    model = physics.wake_model("jensen", parameters, physics.pywake_turbine(TURBINE))
    full = (1 << len(EAST)) - 1
    ring = np.arange(360)
    whole = physics.phi_keys(
        model,
        EAST,
        NORTH,
        HUB,
        np.full(360, full),
        ring,
        np.zeros(360, int),
        SPEEDS,
        4.3,
        4.0,
        FLOOR,
    )
    some = np.array([3, 97, 98, 355])
    sparse = physics.phi_keys(
        model, EAST, NORTH, HUB, np.full(4, full), some, np.zeros(4, int), SPEEDS, 4.3, 4.0, FLOOR
    )
    np.testing.assert_allclose(sparse, whole[some], rtol=1e-12, atol=0)
    raw = physics.power(
        model,
        EAST,
        NORTH,
        HUB,
        np.ones((360, len(EAST)), int),
        ring.astype(float),
        np.full(360, SPEEDS[0]),
    )
    smoothed = gaussian_filter1d(raw, 4.3, axis=0, mode="wrap", truncate=4.0)
    free = physics.free_power(model, SPEEDS[:1])[0]
    np.testing.assert_allclose(
        whole, np.log(np.maximum(smoothed, FLOOR)) - np.log(free), rtol=0, atol=1e-12
    )


def test_physics_mse_equals_a_dense_least_squares_fit():
    rows = prepared_rows(seed=3, timestamps=40)
    rng = np.random.default_rng(4)
    phi = rng.normal(0, 0.2, rows.height)
    group = physics.timestamp_groups(rows)
    index = rows["turbine_index"].to_numpy().astype(int)
    y = rng.normal(0, 0.1, rows.height)
    y = y - (np.bincount(group, y) / np.bincount(group))[group]
    gram = physics.offset_gram(group, index, len(EAST))
    centring = np.zeros((rows.height, rows.height))
    for t in np.unique(group):
        members = np.flatnonzero(group == t)
        centring[np.ix_(members, members)] = np.eye(len(members)) - 1.0 / len(members)
    design = centring @ np.eye(len(EAST))[index]
    target = y - centring @ phi
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    expected = np.mean((target - design @ coefficients) ** 2)
    assert physics.physics_mse(phi, y, group, index, gram) == pytest.approx(expected, rel=1e-10)


def test_rows_the_tables_cannot_describe_are_refused():
    rows = prepared_rows(seed=5, timestamps=5)
    physics.check_rows(rows, len(EAST), 3)
    with pytest.raises(ValueError, match="outside their operating mask"):
        physics.check_rows(
            rows.with_columns(operating_mask=pl.lit(1, dtype=pl.UInt32)), len(EAST), 3
        )
    with pytest.raises(ValueError, match="finite farm_direction"):
        physics.check_rows(rows.with_columns(farm_direction=pl.lit(float("nan"))), len(EAST), 3)
    with pytest.raises(ValueError, match="speed_bin 2"):
        physics.check_rows(rows.with_columns(speed_bin=pl.lit(2, dtype=pl.UInt8)), len(EAST), 2)
    with pytest.raises(ValueError, match="turbine_index 5"):
        physics.check_rows(rows, 5, 3)


def test_teacher_and_smarteole_consensus():
    fits = [
        {
            "teacher": "niayifar",
            "mse": {"jensen": 3.8, "niayifar": 2.0, "turbopark": 2.5},
            "no_wake_mse": 4.0,
        },
        {
            "teacher": "turbopark",
            "mse": {"jensen": 3.8, "niayifar": 2.9, "turbopark": 2.0},
            "no_wake_mse": 4.0,
        },
        {
            "teacher": "niayifar",
            "mse": {"jensen": 1.0, "niayifar": 0.9, "turbopark": 1.2},
            "no_wake_mse": 2.0,
        },
    ]
    assert [physics.teacher(fit["mse"]) for fit in fits] == ["niayifar", "turbopark", "niayifar"]
    assert physics.teacher({"jensen": 1.0, "niayifar": 1.0, "turbopark": 2.0}) == "jensen"
    assert physics.consensus_teacher(fits) == "niayifar"
    fits[2] = {
        "teacher": "jensen",
        "mse": {"jensen": 1.0, "niayifar": 1.9, "turbopark": 1.2},
        "no_wake_mse": 2.0,
    }
    relative = {
        "jensen": (0.95 + 0.95 + 0.5) / 3,
        "niayifar": (0.5 + 0.725 + 0.95) / 3,
        "turbopark": (0.625 + 0.5 + 0.6) / 3,
    }
    assert min(relative, key=lambda family: relative[family]) == "turbopark"
    assert physics.consensus_teacher(fits) == "turbopark"
    fits = [
        {
            "parameters": {
                "jensen": {"k": k, "spread_deg": s},
                "niayifar": {"ti": t, "spread_deg": s},
                "turbopark": {"a": a, "ti": t, "spread_deg": s},
            }
        }
        for k, t, a, s in [
            (0.05, 0.08, 0.03, 1.0),
            (0.09, 0.06, 0.05, 4.0),
            (0.07, 0.12, 0.02, 2.0),
        ]
    ]
    medians = physics.median_parameters(fits)
    assert medians["jensen"] == {"k": 0.07, "spread_deg": 2.0}
    assert medians["turbopark"] == {"a": 0.03, "ti": 0.08, "spread_deg": 2.0}


@pytest.mark.parametrize("family", physics.WAKE_PARAMETERS)
def test_single_wake_kernel_equals_two_turbine_simulations(family):
    kernel = physics.single_wake_kernel(
        family, PARAMETERS[family], TURBINE, EAST, NORTH, HUB, SPEEDS, 4.0, 0.01
    )
    assert kernel.shape == (360, 3, len(EAST), len(EAST))
    assert np.all(kernel[:, :, np.arange(6), np.arange(6)] == 0.0)
    for receiver, source, degree, speed_bin in [
        (1, 0, 265, 0),
        (2, 1, 270, 2),
        (4, 3, 262, 1),
        (0, 1, 265, 1),
    ]:
        pair = np.array([source, receiver])
        expected = direct_phi(
            direct_power(family, PARAMETERS[family], pair, float(degree), SPEEDS[speed_bin]),
            SPEEDS[speed_bin],
        )
        assert kernel[degree, speed_bin, receiver, source] == pytest.approx(
            expected[receiver], rel=1e-9, abs=1e-12
        )
    assert kernel[265, 0, 1, 0] < -0.05
    assert kernel[265, 0, 0, 1] == 0.0


def test_physics_table_rows_follow_the_prepared_rows():
    rows = prepared_rows(seed=6, timestamps=30)
    defaults = {f: {**p, "spread_deg": 0.0} for f, p in PARAMETERS.items()}
    calibrated = {f: {**p, "spread_deg": 2.0} for f, p in PARAMETERS.items()}
    table = physics.physics_table(
        rows, TURBINE, EAST, NORTH, HUB, SPEEDS, calibrated, defaults, 4.0, 0.01
    )
    assert table.schema == PHYSICS
    assert table["timestamp"].equals(rows["timestamp"]) and table["turbine"].equals(rows["turbine"])
    np.testing.assert_array_equal(
        table["physics_jensen_default"].to_numpy(), rows_phi(rows, "jensen", defaults["jensen"])
    )
    np.testing.assert_array_equal(
        table["physics_turbopark"].to_numpy(), rows_phi(rows, "turbopark", calibrated["turbopark"])
    )
    assert (
        table.select(pl.all().exclude("timestamp", "turbine").is_nan().any()).row(0) == (False,) * 6
    )


SEARCH = {
    "jensen": {"k": (0.01, 0.15), "spread_deg": (0.0, 10.0)},
    "niayifar": {"ti": (0.02, 0.3), "spread_deg": (0.0, 10.0)},
    "turbopark": {"a": (0.01, 0.15), "ti": (0.02, 0.3), "spread_deg": (0.0, 10.0)},
}
DEFAULTS = {
    "jensen": {"k": 0.1, "spread_deg": 0.0},
    "niayifar": {"ti": 0.1, "spread_deg": 0.0},
    "turbopark": {"a": 0.04, "ti": 0.1, "spread_deg": 0.0},
}


def generated_rows(family, truth, timestamps, seed):
    rows = prepared_rows(seed=seed, timestamps=timestamps)
    group = physics.timestamp_groups(rows)
    index = rows["turbine_index"].to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    raw = rows_phi(rows, family, truth) + rng.normal(0, 0.03, len(EAST))[index]
    raw = raw + rng.normal(0, 0.02, rows.height)
    y = raw - (np.bincount(group, raw) / np.bincount(group))[group]
    return rows.with_columns(y=pl.Series(y))


@pytest.mark.parametrize(
    ("family", "truth", "tolerance", "size"),
    [
        ("jensen", {"k": 0.055, "spread_deg": 3.0}, {"k": 0.005, "spread_deg": 1.0}, (600, 100)),
        ("niayifar", {"ti": 0.11, "spread_deg": 3.0}, {"ti": 0.015, "spread_deg": 1.0}, (200, 60)),
        (
            "turbopark",
            {"a": 0.06, "ti": 0.10, "spread_deg": 3.0},
            {"a": 0.01, "spread_deg": 1.0},
            (200, 60),
        ),
    ],
)
def test_calibration_recovers_parameters_of_pywake_generated_data(family, truth, tolerance, size):
    timestamps, trials = size
    rows = generated_rows(family, truth, timestamps=timestamps, seed=11)
    best, mse, record = physics.calibrate(
        family,
        rows,
        TURBINE,
        EAST,
        NORTH,
        HUB,
        SPEEDS,
        SEARCH[family],
        DEFAULTS[family],
        trials,
        0,
        4.0,
        0.01,
    )
    for name, allowed in tolerance.items():
        assert abs(best[name] - truth[name]) < allowed, (name, best)
    group = physics.timestamp_groups(rows)
    index = rows["turbine_index"].to_numpy().astype(int)
    gram = physics.offset_gram(group, index, len(EAST))
    truth_mse = physics.physics_mse(
        rows_phi(rows, family, truth), rows["y"].to_numpy(), group, index, gram
    )
    assert truth_mse * 0.98 < mse < truth_mse * 1.1
    assert record.height == trials and record["mse"].min() == mse
    assert record.row(0, named=True) | DEFAULTS[family] == record.row(0, named=True)


def test_farm_fit_starts_from_defaults_and_names_the_best_family_teacher():
    rows = generated_rows("niayifar", {"ti": 0.11, "spread_deg": 2.0}, timestamps=120, seed=13)
    fit = physics.fit_farm(
        rows, TURBINE, EAST, NORTH, HUB, SPEEDS, SEARCH, DEFAULTS, 6, 0, 4.0, 0.01
    )
    for family in physics.WAKE_PARAMETERS:
        assert fit["mse"][family] <= fit["default_mse"][family] < fit["no_wake_mse"]
        assert len(fit["trials"][family]) == 6
        assert set(fit["parameters"][family]) == set(physics.parameter_names(family))
    assert fit["teacher"] == min(fit["mse"], key=lambda family: fit["mse"][family])
    assert fit["teacher_parameters"] == fit["parameters"][fit["teacher"]]
    assert fit["timestamps"] == 120 and fit["rows"] == rows.height
    with pytest.raises(ValueError, match="outside its search range"):
        physics.fit_farm(
            rows,
            TURBINE,
            EAST,
            NORTH,
            HUB,
            SPEEDS,
            SEARCH,
            DEFAULTS | {"jensen": {"k": 0.3, "spread_deg": 0.0}},
            1,
            0,
            4.0,
            0.01,
        )


def test_power_does_not_depend_on_workers(monkeypatch):
    monkeypatch.setattr(physics, "CASES_PER_CALL", 8)
    model = physics.wake_model("niayifar", {"ti": 0.08}, physics.pywake_turbine(TURBINE))
    rng = np.random.default_rng(7)
    codes = (rng.random((17, len(EAST))) < 0.8).astype(int)
    direction, speed = rng.uniform(0, 360, 17), rng.uniform(4, 11, 17)
    alone = physics.power(model, EAST, NORTH, HUB, codes, direction, speed)
    with physics.worker_pool(2) as pool:
        shared = physics.power(model, EAST, NORTH, HUB, codes, direction, speed, pool)
    assert np.array_equal(alone, shared)
    assert np.all(alone[codes == 0] == 0.0) and np.all(alone[codes == 1] > 0.0)


def test_physics_files_round_trip_and_smarteole_takes_the_median_fit(tmp_path):
    rows = prepared_rows(seed=8, timestamps=10)
    fits = {
        farm: {
            "parameters": {f: {**p, "spread_deg": float(s)} for f, p in PARAMETERS.items()},
            "teacher": "niayifar",
            "mse": {"jensen": 2.0, "niayifar": 1.0, "turbopark": 1.5},
            "no_wake_mse": 3.0,
        }
        for farm, s in (("kelmarsh", 1), ("penmanshiel", 5), ("hill_of_towie", 2))
    }
    fit = physics.fit_from_farms(fits, DEFAULTS)
    assert fit["source"] == "median of kelmarsh, penmanshiel, hill_of_towie"
    assert fit["teacher"] == "niayifar" and fit["teacher_parameters"]["spread_deg"] == 2.0
    table = physics.physics_table(
        rows, TURBINE, EAST, NORTH, HUB, SPEEDS, fit["parameters"], fit["defaults"], 4.0, 0.01
    )
    physics.write_physics(tmp_path, table, fit)
    assert pl.read_parquet(tmp_path / "physics.parquet").equals(table)
    assert (
        json.loads((tmp_path / "physics_fit.json").read_text())["parameters"] == fit["parameters"]
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["physics.parquet", "physics_fit.json"]


def test_farm_fit_does_not_depend_on_workers():
    rows = generated_rows("jensen", {"k": 0.05, "spread_deg": 2.0}, timestamps=12, seed=17)
    alone = physics.fit_farm(
        rows, TURBINE, EAST, NORTH, HUB, SPEEDS, SEARCH, DEFAULTS, 2, 3, 4.0, 0.01
    )
    shared = physics.fit_farm(
        rows, TURBINE, EAST, NORTH, HUB, SPEEDS, SEARCH, DEFAULTS, 2, 3, 4.0, 0.01, workers=2
    )
    for family in physics.WAKE_PARAMETERS:
        assert alone["parameters"][family] == shared["parameters"][family]
        assert [t["mse"] for t in alone["trials"][family]] == [
            t["mse"] for t in shared["trials"][family]
        ]
