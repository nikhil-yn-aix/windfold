from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from windfold import inflow, physics
from windfold.config import Curve, Inflow, TurbineConfig
from windfold.schemas import PHYSICS_FAMILIES

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
EAST = np.array([0.0, 410.0, 780.0, 120.0])
NORTH = np.array([0.0, 30.0, -60.0, 420.0])
HUB = np.array([59.0, 59.0, 69.0, 59.0])
CALIBRATED = {
    "jensen": {"k": 0.06, "spread_deg": 1.0},
    "niayifar": {"ti": 0.09, "spread_deg": 1.0},
    "turbopark": {"a": 0.04, "ti": 0.09, "spread_deg": 1.0},
}
DEFAULTS = {
    "jensen": {"k": 0.1, "spread_deg": 0.0},
    "niayifar": {"ti": 0.1, "spread_deg": 0.0},
    "turbopark": {"a": 0.04, "ti": 0.1, "spread_deg": 0.0},
}
SETTINGS = Inflow(
    log2_ratio_low=-1.0,
    log2_ratio_high=1.0,
    log2_ratio_step=0.5,
    climatology_sector_deg=30.0,
    diurnal_hours=6,
)
GRID = inflow.ratio_grid(SETTINGS)


def prepared_rows(turbulence: list[float | None], periods: list[str] | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    timestamps = len(turbulence)
    n = len(EAST)
    full = (1 << n) - 1
    masks = np.where(
        rng.random(timestamps) < 0.3, full & ~(1 << rng.integers(0, n, timestamps)), full
    )
    direction = rng.uniform(0.0, 360.0, timestamps)
    speed_bin = rng.integers(0, 3, timestamps)
    t, i = np.nonzero((masks[:, None] >> np.arange(n)) & 1)
    stamps = np.datetime64("2023-06-01T00:00", "us") + t * np.timedelta64(10, "m")
    return pl.DataFrame(
        {
            "timestamp": pl.Series(stamps).dt.replace_time_zone("UTC"),
            "turbine": [f"T{k}" for k in i],
            "turbine_index": pl.Series(i, dtype=pl.UInt8),
            "period": [(periods or ["development"] * timestamps)[k] for k in t],
            "operating_mask": pl.Series(masks[t], dtype=pl.UInt32),
            "farm_direction": direction[t],
            "speed_bin": pl.Series(speed_bin[t], dtype=pl.UInt8),
            "turbulence_intensity": pl.Series([turbulence[k] for k in t], dtype=pl.Float64),
        }
    )


def tables(rows: pl.DataFrame, reference: float) -> dict[str, pl.DataFrame]:
    layout = (TURBINE, EAST, NORTH, HUB, SPEEDS)
    return inflow.tables(rows, *layout, CALIBRATED, DEFAULTS, 4.0, 0.01, reference, SETTINGS)


def table(rows: pl.DataFrame, reference: float) -> pl.DataFrame:
    return tables(rows, reference)["measured"]


def direct_phi(rows: pl.DataFrame, family: str, parameters: dict) -> np.ndarray:
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
        4.0,
        0.01 * TURBINE.rated_power_kw * 1000.0,
    )
    return phi[inverse, rows["turbine_index"].to_numpy().astype(int)]


def test_the_ratio_grid_holds_one_exactly_and_ends_on_its_bounds():
    settings = SETTINGS.model_copy(
        update={"log2_ratio_low": -1.5, "log2_ratio_high": 1.5, "log2_ratio_step": 0.25}
    )
    grid = inflow.ratio_grid(settings)
    assert grid.size == 13 and grid[6] == 1.0
    assert grid[0] == pytest.approx(2**-1.5) and grid[-1] == pytest.approx(2**1.5)
    fields = SETTINGS.model_dump()
    for change, message in (
        ({"log2_ratio_low": -1.4, "log2_ratio_step": 0.25}, "ratio one"),
        ({"climatology_sector_deg": 25.0}, "circle"),
        ({"diurnal_hours": 5}, "day"),
    ):
        with pytest.raises(ValueError, match=message):
            Inflow.model_validate(fields | change)


def test_ratios_divide_by_the_reference_take_one_when_unmeasured_and_clip_to_the_grid():
    rows = pl.DataFrame({"turbulence_intensity": [0.05, 0.1, None, 1.0, 0.01, 0.0, float("nan")]})
    np.testing.assert_allclose(inflow.ratios(rows, 0.1, GRID), [0.5, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError, match="not positive"):
        inflow.ratios(rows, float("nan"), GRID)


def test_the_reference_is_the_median_over_calibration_timestamps_not_rows():
    times = [
        (0.10, 4, "calibration"),
        (0.20, 2, "calibration"),
        (0.30, 2, "calibration"),
        (None, 1, "calibration"),
        (0.0, 1, "calibration"),
        (float("nan"), 1, "calibration"),
        (0.90, 1, "development"),
    ]
    start = datetime(2023, 6, 1, tzinfo=UTC)
    rows = pl.DataFrame(
        [
            (start + timedelta(minutes=10 * k), period, turbulence)
            for k, (turbulence, turbines, period) in enumerate(times)
            for _ in range(turbines)
        ],
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "period": pl.String,
            "turbulence_intensity": pl.Float64,
        },
        orient="row",
    )
    assert inflow.reference(rows) == pytest.approx(0.20)
    with pytest.raises(ValueError, match="calibration-window"):
        inflow.reference(rows.filter(pl.col("period") == "development"))


def test_at_the_reference_inflow_the_table_is_the_constant_inflow_physics():
    rows = prepared_rows([0.12] * 40)
    measured = table(rows, 0.12)
    constant = physics.physics_table(
        rows, TURBINE, EAST, NORTH, HUB, SPEEDS, CALIBRATED, DEFAULTS, 4.0, 0.01
    )
    assert measured.select("timestamp", "turbine").equals(constant.select("timestamp", "turbine"))
    for column in constant.columns[2:]:
        np.testing.assert_array_equal(measured[column].to_numpy(), constant[column].to_numpy())
    assert (measured["ratio"] == 1.0).all()


@pytest.fixture(scope="module")
def between_grid_points() -> tuple[pl.DataFrame, pl.DataFrame]:
    """rows measured at 2 ** 0.2 times the reference, two fifths of the way from ratio one to
    the next grid ratio, and their measured table."""
    rows = prepared_rows([0.12 * 2**0.2] * 25)
    return rows, table(rows, 0.12)


@pytest.mark.parametrize("family", PHYSICS_FAMILIES)
def test_between_grid_points_phi_is_linear_in_the_log_of_the_ratio(family, between_grid_points):
    rows, measured = between_grid_points
    name = inflow.TURBULENCE_PARAMETER[family]
    parameters = CALIBRATED[family]
    below = direct_phi(rows, family, {**parameters, name: parameters[name]})
    above = direct_phi(rows, family, {**parameters, name: parameters[name] * 2**0.5})
    expected = (1 - 0.4) * below + 0.4 * above
    np.testing.assert_allclose(measured[f"physics_{family}"].to_numpy(), expected, rtol=1e-12)


def test_wakes_get_shallower_on_average_as_measured_turbulence_rises():
    low, high = table(prepared_rows([0.06] * 40), 0.12), table(prepared_rows([0.24] * 40), 0.12)
    for family in PHYSICS_FAMILIES:
        before, after = low[f"physics_{family}"].to_numpy(), high[f"physics_{family}"].to_numpy()
        waked = before < -1e-6
        assert waked.any() and after[waked].mean() > before[waked].mean()


def test_the_record_counts_missing_and_clipped_timestamps_by_period():
    rows = prepared_rows(
        [0.1, None, 0.5, 0.01, 0.1, 0.1],
        ["calibration", "calibration", "development", "development", "development", "development"],
    )
    note = inflow.record(rows, 0.1, "calibration window", GRID)
    assert note["reference_ti"] == 0.1 and note["grid"] == GRID.tolist()
    assert note["shares"]["calibration"] == {
        "timestamps": 2,
        "missing": 0.5,
        "clipped_low": 0.0,
        "clipped_high": 0.0,
    }
    assert note["shares"]["development"] == {
        "timestamps": 4,
        "missing": 0.0,
        "clipped_low": 0.25,
        "clipped_high": 0.25,
    }


def test_a_control_averages_its_group_without_the_timestamp_itself():
    times = pl.DataFrame(
        {
            "period": ["development"] * 4 + ["calibration"],
            "ratio": [0.5, 1.0, 2.0, 1.0, 2.0],
            "sector": [0, 0, 0, 1, 1],
        }
    )
    own = inflow.grid_weights(times["ratio"].to_numpy(), GRID)
    grouped = inflow.expected_weights(times, ["sector"], GRID)
    np.testing.assert_allclose(grouped[0], (own[1] + own[2]) / 2)
    np.testing.assert_allclose(grouped[3], (own[0] + own[1] + own[2]) / 3)
    np.testing.assert_allclose(grouped[4], GRID == 1.0)
    everything = inflow.expected_weights(times, [], GRID)
    np.testing.assert_allclose(everything[1], (own[0] + own[2] + own[3]) / 3)
    np.testing.assert_allclose(grouped.sum(axis=1), 1.0)


def test_the_average_control_takes_phi_over_the_other_timestamps_measured_ratios():
    turbulence = [0.06, 0.24, 0.24, None, 0.06, 0.24, 0.06, 0.24]
    rows = prepared_rows(turbulence)
    average = tables(rows, 0.12)["average"]
    low, high = 0.5, 2.0
    for family in PHYSICS_FAMILIES:
        name = inflow.TURBULENCE_PARAMETER[family]
        parameters = CALIBRATED[family]
        at_low = direct_phi(rows, family, {**parameters, name: parameters[name] * low})
        at_high = direct_phi(rows, family, {**parameters, name: parameters[name] * high})
        at_one = direct_phi(rows, family, parameters)
        stamp = rows["timestamp"].rank("dense").cast(pl.Int64).to_numpy() - 1
        measured = np.array([turbulence[k] is not None for k in stamp])
        share_high = np.array([3 / 6 if turbulence[k] == 0.24 else 4 / 6 for k in stamp])
        averaged = share_high * at_high + (1 - share_high) * at_low
        expected = np.where(measured, averaged, at_one)
        np.testing.assert_allclose(average[f"physics_{family}"].to_numpy(), expected, rtol=1e-12)


def test_with_one_shared_ratio_every_control_is_the_measured_physics():
    built = tables(prepared_rows([0.24] * 30), 0.12)
    for name in ("average", "climatology", "diurnal"):
        for column in built["measured"].columns[2:]:
            np.testing.assert_allclose(
                built[name][column].to_numpy(), built["measured"][column].to_numpy(), rtol=1e-12
            )


def test_without_any_measurement_every_control_keeps_ratio_one():
    rows = prepared_rows([None, 0.0, None])
    ratio = inflow.ratios(rows, 0.1, GRID)
    for weights in inflow.control_weights(rows, ratio, SETTINGS).values():
        np.testing.assert_array_equal(weights, np.tile(GRID == 1.0, (rows.height, 1)))
