"""the wind frame: one turbine's upstream neighbors, placed by their distance along and across the
wind in rotor diameters, inside the shared correction's window, which turns with the wind."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon

from windfold import config
from windfold.geometry import downwind, relations, upstream
from windfold.report.style import MM_PER_INCH, Style

FARM = "penmanshiel"
RECEIVER = "T12"
SOURCE = "T01"
DIRECTION = 243.0
WIDTH_MM = 170.0
PAD_D = 1.5
WIND_D = 4.0
POINTS_PER_MM = 72 / MM_PER_INCH


@dataclass(frozen=True)
class Frame:
    """one receiver at one wind direction: every turbine in rotor diameters, each source's along
    and across distance from the receiver and which sources the kernel window takes."""

    names: tuple[str, ...]
    east: np.ndarray
    north: np.ndarray
    receiver: int
    direction: float
    along: np.ndarray
    across: np.ndarray
    inside: np.ndarray
    along_max: float
    across_max: float

    @property
    def upwind(self) -> np.ndarray:
        to_east, to_north = downwind(self.direction)
        return -np.array([float(to_east), float(to_north)])

    @property
    def left(self) -> np.ndarray:
        """the unit vector to the left of an observer facing downwind."""
        up_east, up_north = self.upwind
        return np.array([up_north, -up_east])

    def at(self, turbine: int) -> np.ndarray:
        return np.array([self.east[turbine], self.north[turbine]])

    @property
    def far(self) -> np.ndarray:
        return self.at(self.receiver) + self.along_max * self.upwind

    def window(self) -> np.ndarray:
        """the window's corners: across_max either side, from the receiver to along_max upwind."""
        near, side = self.at(self.receiver), self.across_max * self.left
        return np.array([near + side, self.far + side, self.far - side, near - side])

    def foot(self, source: int) -> np.ndarray:
        """the point on the window's center line level with the source."""
        return self.at(self.receiver) + self.along[source] * self.upwind


def frame(
    names: list[str],
    east_m: np.ndarray,
    north_m: np.ndarray,
    diameter_m: float,
    receiver: int,
    direction: float,
    along_max: float,
    across_max: float,
) -> Frame:
    along, across = relations(east_m, north_m, direction, diameter_m)
    return Frame(
        names=tuple(names),
        east=east_m / diameter_m,
        north=north_m / diameter_m,
        receiver=receiver,
        direction=direction,
        along=along[receiver],
        across=across[receiver],
        inside=upstream(along, across, along_max, across_max)[receiver],
        along_max=along_max,
        across_max=across_max,
    )


def offset(vector: np.ndarray, mm: float) -> tuple[float, float]:
    unit = vector / np.linalg.norm(vector)
    return float(unit[0] * mm * POINTS_PER_MM), float(unit[1] * mm * POINTS_PER_MM)


def label(
    ax: Axes, text: str, at: np.ndarray, towards: np.ndarray, mm: float, style: Style
) -> None:
    ax.annotate(
        text,
        (float(at[0]), float(at[1])),
        xytext=offset(towards, mm),
        textcoords="offset points",
        ha="center",
        va="center",
        color=style.role("text"),
    )


def wind(ax: Axes, tail: np.ndarray, found: Frame, style: Style) -> None:
    """the wind arrow from tail, downwind, with its direction written below it."""
    head = tail - WIND_D * found.upwind
    ax.add_patch(
        FancyArrowPatch(
            (float(tail[0]), float(tail[1])),
            (float(head[0]), float(head[1])),
            arrowstyle="-|>",
            mutation_scale=10,
            color=style.role("text"),
            lw=style.report.lines.data_pt,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.annotate(
        f"wind from {found.direction:g}°",
        (float(tail[0]), float(tail[1])),
        xytext=(0.0, -2.5 * POINTS_PER_MM),
        textcoords="offset points",
        ha="left",
        va="top",
        color=style.role("text"),
    )


def bounds(found: Frame) -> tuple[float, float, float, float]:
    """the drawing's extent in rotor diameters: every turbine and the window, padded."""
    points = np.r_[np.c_[found.east, found.north], found.window()]
    (x0, y0), (x1, y1) = points.min(axis=0) - PAD_D, points.max(axis=0) + PAD_D
    return float(x0), float(y0), float(x1), float(y1)


def panel(ax: Axes, found: Frame, style: Style, xlim: tuple[float, float] | None = None) -> None:
    """the frame on axes in rotor diameters, north up; xlim widens the view around the drawing."""
    lines = style.report.lines
    blue, ink, grey = style.role("windfold"), style.role("text"), style.role("axes")
    near = found.at(found.receiver)
    corners = found.window()
    x0, y0, x1, y1 = bounds(found)
    source = found.names.index(SOURCE)
    foot, position = found.foot(source), found.at(source)
    others = ~found.inside
    others[found.receiver] = False
    ax.add_patch(Polygon(corners, fill=False, ec=blue, lw=lines.thin_pt))
    ax.scatter(
        found.east[others],
        found.north[others],
        s=16,
        fc="white",
        ec=grey,
        lw=lines.thin_pt,
        zorder=3,
    )
    ax.scatter(
        found.east[found.inside], found.north[found.inside], s=22, color=blue, lw=0, zorder=3
    )
    ax.plot(*np.c_[near, foot, position], color=ink, lw=lines.data_pt)
    ax.scatter(*near, s=56, fc="white", ec=blue, lw=lines.thick_pt, zorder=3)
    label(ax, "i", near, -found.upwind, 3.8, style)
    label(ax, "j", position, found.left - found.upwind, 4.4, style)
    label(ax, "a", (near + foot) / 2, foot - position, 5.0, style)
    label(ax, "c", (foot + position) / 2, found.upwind, 3.6, style)
    label(ax, f"{found.along_max:g} D", found.far, -found.upwind, 7.0, style)
    wind(ax, np.array([x0 + PAD_D, y1 - 2 * PAD_D]), found, style)
    ax.set(xlim=xlim or (x0, x1), ylim=(y0, y1), aspect="equal")
    ax.set_axis_off()


def draw(found: Frame, style: Style) -> Figure:
    x0, y0, x1, y1 = bounds(found)
    with style.context():
        figure = style.figure(WIDTH_MM, WIDTH_MM * (y1 - y0) / (x1 - x0))
        figure.set_layout_engine("none")
        panel(figure.add_axes((0.0, 0.0, 1.0, 1.0)), found, style)
    return figure


def frame_of(data: Path, configs: Path | None) -> Frame:
    farm = config.load_farm(FARM, configs)
    diameter = config.load_turbine(farm.turbine_type, configs).rotor_diameter_m
    kernel = config.load_protocol(configs).kernel
    layout = pl.read_parquet(data / "prepared" / FARM / "layout.parquet")
    names = layout["turbine"].to_list()
    return frame(
        names,
        layout["x_m"].to_numpy(),
        layout["y_m"].to_numpy(),
        diameter,
        names.index(RECEIVER),
        DIRECTION,
        kernel.along_max_d,
        kernel.across_max_d,
    )


def figure(data: Path, configs: Path | None, style: Style) -> Figure:
    return draw(frame_of(data, configs), style)
