import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from windfold.geometry import bearings, distances, relations, upstream

DIAMETER = 82.0


def offset_layout(bearing_deg: float, distance_m: float) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(bearing_deg)
    east = np.array([0.0, distance_m * np.sin(radians)])
    north = np.array([0.0, distance_m * np.cos(radians)])
    return east, north


@pytest.mark.parametrize(
    ("bearing", "direction", "along", "across"),
    [
        (0.0, 0.0, 5.0, 0.0),
        (0.0, 180.0, -5.0, 0.0),
        (0.0, 90.0, 0.0, 5.0),
        (0.0, 270.0, 0.0, -5.0),
        (270.0, 270.0, 5.0, 0.0),
        (225.0, 255.0, 5.0 * np.cos(np.deg2rad(30.0)), 5.0 * np.sin(np.deg2rad(30.0))),
    ],
)
def test_source_at_a_known_bearing(bearing, direction, along, across):
    east, north = offset_layout(bearing, 5.0 * DIAMETER)
    a, c = relations(east, north, direction, DIAMETER)
    assert a[0, 1] == pytest.approx(along, abs=1e-12)
    assert c[0, 1] == pytest.approx(across, abs=1e-12)
    assert a[0, 0] == 0.0 and c[0, 0] == 0.0


@settings(max_examples=200, deadline=None)
@given(
    bearing=st.floats(0.0, 360.0, exclude_max=True),
    direction=st.floats(0.0, 360.0, exclude_max=True),
    distance=st.floats(0.5, 40.0),
)
def test_relations_follow_the_angle_between_wind_and_bearing(bearing, direction, distance):
    east, north = offset_layout(bearing, distance * DIAMETER)
    a, c = relations(east, north, direction, DIAMETER)
    delta = np.deg2rad(direction - bearing)
    assert a[0, 1] == pytest.approx(distance * np.cos(delta), abs=1e-9)
    assert c[0, 1] == pytest.approx(distance * np.sin(delta), abs=1e-9)


layouts = st.integers(2, 9).flatmap(
    lambda n: st.tuples(
        arrays(np.float64, n, elements=st.floats(-4000.0, 4000.0)),
        arrays(np.float64, n, elements=st.floats(-4000.0, 4000.0)),
    )
)


@settings(max_examples=200, deadline=None)
@given(layout=layouts, steps=st.integers(0, 180 * 64 - 1))
def test_half_turn_swaps_receiver_and_source_exactly(layout, steps):
    east, north = layout
    direction = steps / 64.0
    along, across = relations(east, north, direction, DIAMETER)
    along_turned, across_turned = relations(east, north, direction + 180.0, DIAMETER)
    assert np.array_equal(along_turned, along.T)
    assert np.array_equal(across_turned, across.T)
    turned = upstream(along_turned, across_turned, 30.0, 3.0)
    assert np.array_equal(turned, upstream(along, across, 30.0, 3.0).T)


@settings(max_examples=100, deadline=None)
@given(layout=layouts, direction=st.floats(0.0, 180.0, exclude_max=True))
def test_half_turn_holds_for_any_direction(layout, direction):
    east, north = layout
    along, across = relations(east, north, direction, DIAMETER)
    along_turned, across_turned = relations(east, north, direction + 180.0, DIAMETER)
    np.testing.assert_allclose(along_turned, along.T, rtol=0, atol=1e-9)
    np.testing.assert_allclose(across_turned, across.T, rtol=0, atol=1e-9)


def test_relations_broadcast_over_directions():
    east, north = np.array([0.0, 300.0, -150.0]), np.array([0.0, 120.0, 400.0])
    directions = np.array([[10.0, 200.0], [359.5, 45.25]])
    along, across = relations(east, north, directions, DIAMETER)
    assert along.shape == (2, 2, 3, 3)
    single_along, single_across = relations(east, north, 359.5, DIAMETER)
    assert np.array_equal(along[1, 0], single_along)
    assert np.array_equal(across[1, 0], single_across)


def test_bearing_puts_the_source_straight_upstream():
    rng = np.random.default_rng(3)
    east, north = rng.uniform(-2000, 2000, 6), rng.uniform(-2000, 2000, 6)
    toward = bearings(east, north)
    separation = distances(east, north, DIAMETER)
    for i in range(6):
        for j in range(6):
            if i == j:
                assert np.isnan(toward[i, j])
                continue
            along, across = relations(east, north, toward[i, j], DIAMETER)
            assert along[i, j] == pytest.approx(separation[i, j], rel=1e-12)
            assert across[i, j] == pytest.approx(0.0, abs=1e-9)
            assert np.mod(toward[j, i] - toward[i, j], 360.0) == pytest.approx(180.0)


def test_upstream_window_edges():
    along = np.array([[0.0, 30.0, 30.0 + 1e-9, 12.0, 12.0, -4.0]])
    across = np.array([[0.0, 0.0, 0.0, 3.0, -3.0 - 1e-9, 0.0]])
    assert upstream(along, across, 30.0, 3.0).tolist() == [[False, True, False, True, False, False]]


def test_a_turbine_exactly_side_by_side_is_not_upstream():
    east, north = np.array([0.0, 0.0]), np.array([0.0, 200.0])
    along, across = relations(east, north, 90.0, DIAMETER)
    assert 0.0 < abs(along[0, 1]) < 1e-12 and abs(across[0, 1]) < 3.0
    assert not upstream(along, across, 30.0, 3.0).any()
