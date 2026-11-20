import numpy as np
import pytest

from windfold.config import Curve
from windfold.prepare.power_curves import curve_power, fit_power_curve, outside_curve_bounds

CURVE = Curve(speed_ms=(3, 5, 7, 9, 11), value=(0, 200, 600, 1200, 2000), source="hand-made")


def test_curve_is_linear_between_points_zero_below_and_flat_above():
    power = curve_power(np.array([2.0, 4.0, 6.0, 12.0]), CURVE)
    assert power.tolist() == [0.0, 100.0, 400.0, 2000.0]
    published = Curve(speed_ms=(4, 5), value=(64, 159), source="starts above cut-in")
    assert curve_power(np.array([3.9, 4.0]), published).tolist() == [0.0, 64.0]


@pytest.mark.parametrize(
    ("power", "expected"),
    [(619.0, False), (621.0, True), (181.0, False), (179.0, True)],
)
def test_bounds_shift_the_curve_in_speed_and_power(power, expected):
    flags = outside_curve_bounds(np.array([6.0]), np.array([power]), CURVE, 2000.0, 0.1, 0.05)
    assert flags.tolist() == [expected]


def test_rows_without_speed_are_not_power_outliers():
    flags = outside_curve_bounds(np.array([np.nan]), np.array([5000.0]), CURVE, 2000.0, 0.1, 0.05)
    assert flags.tolist() == [False]


def test_fitted_curve_takes_bin_medians_at_bin_centres():
    speed = np.array([5.1, 5.2, 5.3, 6.1, 6.2, 9.9, np.nan])
    power = np.array([100.0, 300.0, 200.0, 500.0, 700.0, 1500.0, 50.0])
    curve = fit_power_curve(speed, power, 1.0, 2, "calibration window")
    assert curve.speed_ms == (5.5, 6.5)
    assert curve.value == (200.0, 600.0)
