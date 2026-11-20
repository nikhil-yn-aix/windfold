"""where turbines sit relative to each other in the wind frame.

directions are meteorological: degrees clockwise from north, the way the wind comes from.
positions are east and north in meters. pair arrays are indexed [..., receiver, source].
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def downwind(direction_deg: ArrayLike) -> tuple[Array, Array]:
    """east and north parts of the unit vector the wind blows toward.

    the direction is folded onto [0, 180) before the trigonometry, so a half turn negates the
    vector exactly whenever the turned direction is itself exact, as whole degrees are.
    """
    direction = np.mod(np.asarray(direction_deg, dtype=np.float64), 360.0)
    turned = direction >= 180.0
    radians = np.deg2rad(np.where(turned, direction - 180.0, direction))
    sign = np.where(turned, 1.0, -1.0)
    return sign * np.sin(radians), sign * np.cos(radians)


def relations(
    east: ArrayLike, north: ArrayLike, direction_deg: ArrayLike, diameter_m: float
) -> tuple[Array, Array]:
    """along and across of every source as seen from every receiver, in rotor diameters.

    along is positive when the source is upstream of the receiver. across is positive when the
    receiver sits left of the source's wake axis, facing downwind. the shape is the direction's
    shape followed by (receiver, source); a half turn of the wind swaps receiver and source.
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    to_east, to_north = downwind(direction_deg)
    to_east, to_north = to_east[..., None, None], to_north[..., None, None]
    d_east = east[:, None] - east[None, :]
    d_north = north[:, None] - north[None, :]
    along = (d_east * to_east + d_north * to_north) / diameter_m
    across = (d_north * to_east - d_east * to_north) / diameter_m
    return along, across


SIDE_BY_SIDE_D = 1e-9


def upstream(along: Array, across: Array, along_max_d: float, across_max_d: float) -> NDArray:
    """true where the source lies upstream of the receiver inside the kernel window.

    along must exceed SIDE_BY_SIDE_D, so rounding cannot put a side-by-side turbine upstream.
    """
    return (along > SIDE_BY_SIDE_D) & (along <= along_max_d) & (np.abs(across) <= across_max_d)


def bearings(east: ArrayLike, north: ArrayLike) -> Array:
    """wind direction, in degrees, that puts each source straight upstream of each receiver.

    this is the compass bearing from receiver to source; the diagonal is nan.
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    out = np.mod(
        np.rad2deg(np.arctan2(east[None, :] - east[:, None], north[None, :] - north[:, None])),
        360.0,
    )
    np.fill_diagonal(out, np.nan)
    return out


def distances(east: ArrayLike, north: ArrayLike, diameter_m: float) -> Array:
    """distance between every receiver and source, in rotor diameters."""
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    return np.hypot(east[:, None] - east[None, :], north[:, None] - north[None, :]) / diameter_m
