"""turbine layouts: positions and hub heights, with east and north meters from the farm centroid."""

import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from windfold.config import FarmConfig
from windfold.schemas import LAYOUT, conform

WGS84_SEMI_MAJOR_M = 6_378_137.0
WGS84_FLATTENING = 1 / 298.257223563
SMARTEOLE_FOLDER = "SMARTEOLE-WFC-open-dataset/"
SMARTEOLE_COORDINATES = SMARTEOLE_FOLDER + "SMARTEOLE_WakeSteering_Coordinates_staticData.csv"
SMARTEOLE_TURBINE = re.compile(r"SMV\d+")


def east_north(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """east and north meters of wgs84 points from their centroid, on the tangent plane at their
    mean latitude and longitude."""
    e2 = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
    lat, lon = np.radians(latitude), np.radians(longitude)
    radius = WGS84_SEMI_MAJOR_M / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * (1 - e2) * np.sin(lat)
    x, y, z = x - x.mean(), y - y.mean(), z - z.mean()
    lat0, lon0 = lat.mean(), lon.mean()
    east = -np.sin(lon0) * x + np.cos(lon0) * y
    north = -np.sin(lat0) * np.cos(lon0) * x - np.sin(lat0) * np.sin(lon0) * y + np.cos(lat0) * z
    return east, north


def layout_frame(
    turbines: list[str], latitude: list[float], longitude: list[float], hub_height: list[float]
) -> pl.DataFrame:
    east, north = east_north(np.array(latitude), np.array(longitude))
    frame = pl.DataFrame(
        {
            "turbine": turbines,
            "latitude": latitude,
            "longitude": longitude,
            "x_m": east,
            "y_m": north,
            "hub_height_m": hub_height,
        },
        schema=LAYOUT,
    )
    return conform(frame.sort("turbine"), LAYOUT)


def read_table(path: Path, name: str) -> pl.DataFrame:
    """a turbine table with a name column, Latitude, Longitude and Hub Height (m)."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        listed = [row for row in csv.DictReader(stream) if row[name]]
    return layout_frame(
        [row[name] for row in listed],
        [float(row["Latitude"]) for row in listed],
        [float(row["Longitude"]) for row in listed],
        [float(row["Hub Height (m)"]) for row in listed],
    )


def read_smarteole(path: Path, hub_height: float) -> pl.DataFrame:
    with zipfile.ZipFile(path) as package:
        text = package.read(SMARTEOLE_COORDINATES).decode("utf-8-sig")
    listed = [
        r for r in csv.DictReader(io.StringIO(text)) if SMARTEOLE_TURBINE.fullmatch(r["Turbine"])
    ]
    return layout_frame(
        [row["Turbine"] for row in listed],
        [float(row["Latitude"]) for row in listed],
        [float(row["Longitude"]) for row in listed],
        [hub_height] * len(listed),
    )


def read_layout(farm: FarmConfig, raw_dir: Path) -> pl.DataFrame:
    """the layout of farm from its turbine file: one row per turbine, in turbine order."""
    path = raw_dir / farm.source.turbines
    match farm.reader:
        case "greenbyte":
            return read_table(path, "Alternative Title")
        case "hill_of_towie":
            return read_table(path, "Turbine Name")
        case "smarteole":
            if farm.source.hub_height_m is None:
                raise ValueError("smarteole's files give no hub height; set source.hub_height_m")
            return read_smarteole(path, farm.source.hub_height_m)
