"""the four farms. layouts: each farm's turbines in rotor diameters, north up. sectors: each
farm's wind rose, the share of its development window's timestamps with the wind from each 10
degree bin of farm direction, with spokes bounding the twelve 30 degree sectors held out in turn.
farms: the paper's figure, each farm's layout above its rose. periods: each farm's calibration
window, development window and confirmation period."""

from collections.abc import Mapping, Sequence
from typing import cast

import matplotlib.dates as mdates
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.projections.polar import PolarAxes

from windfold import config
from windfold.config import FarmConfig, TurbineConfig
from windfold.report.sources import Sources
from windfold.report.style import Style

PERIODS = ("calibration", "development", "confirmatory")
ROSE_PERIOD = "development"
BIN_DEG = 10.0


def nice_bar(extent_d: float) -> float:
    """a scale bar of 2, 5, 10 or 20 diameters, about a fifth of the layout."""
    for length in (2.0, 5.0, 10.0, 20.0):
        if length >= extent_d / 5:
            return length
    return 20.0


def layout_panel(ax: Axes, layout: pl.DataFrame, diameter: float, style: Style) -> None:
    x = (layout["x_m"].to_numpy() - layout["x_m"].mean()) / diameter
    y = (layout["y_m"].to_numpy() - layout["y_m"].mean()) / diameter
    ax.scatter(x, y, s=style.report.lines.marker_pt**2, color=style.role("text"), lw=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)
    ax.set_axis_off()


def scale_bar(ax: Axes, style: Style) -> None:
    """a bar of a round number of rotor diameters under the panel, clear of every turbine."""
    x0, x1 = ax.get_xlim()
    length = nice_bar(x1 - x0)
    right, below = x1 - 0.04 * (x1 - x0), -0.15
    ax.plot(
        [right - length, right],
        [below, below],
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        color=style.role("axes"),
        lw=1.5,
    )
    ax.annotate(
        f"{length:g} D",
        xy=(right - length, below),
        xycoords=("data", "axes fraction"),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=style.role("text"),
    )


def rose(directions: np.ndarray) -> np.ndarray:
    """the percent of timestamps in each 10 degree bin of direction, the first from north."""
    index = np.floor(np.mod(directions, 360.0) / BIN_DEG).astype(np.int64)
    bins = int(360 / BIN_DEG)
    return np.bincount(index, minlength=bins) / max(len(index), 1) * 100


def rose_panel(ax: PolarAxes, share: np.ndarray, width: float, top: float, style: Style) -> None:
    """the rose, and spokes where one held-out sector ends and the next begins."""
    centres = np.deg2rad((np.arange(len(share)) + 0.5) * BIN_DEG)
    ax.bar(centres, share, width=np.deg2rad(BIN_DEG) * 0.86, color=style.role("axes"), lw=0)
    for edge in np.arange(0.0, 360.0, width):
        ax.plot(
            [np.deg2rad(edge)] * 2,
            [0, top],
            color=style.role("context"),
            lw=style.report.lines.axes_pt,
            zorder=0,
        )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([0.0], ["N"])
    ax.set_ylim(0, top)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_color(style.role("grid"))
    ax.spines["polar"].set_linewidth(style.report.lines.axes_pt)


def timeline_panel(ax: Axes, farms: Sequence[FarmConfig], style: Style) -> list[Patch]:
    shades = {period: style.role(period) for period in PERIODS}
    for row, farm in enumerate(farms):
        for period in PERIODS:
            window = getattr(farm.windows, period)
            if window is None:
                continue
            start, end = (mdates.date2num(t) for t in (window.start, window.end))
            ax.barh(row, end - start, left=start, height=0.5, color=shades[period], lw=0)
        for gap in farm.exclusions:
            start, end = (mdates.date2num(t) for t in (gap.start, gap.end))
            ax.barh(row, max(end - start, 2.0), left=start, height=0.5, color="white", lw=0)
    first = min(w.start for f in farms for p in PERIODS if (w := getattr(f.windows, p)))
    last = max(w.end for f in farms for p in PERIODS if (w := getattr(f.windows, p)))
    ax.set_xlim(float(mdates.date2num(first)) - 60, float(mdates.date2num(last)) + 60)
    style.rows(ax, [style.farm_name(f.name) for f in farms], len(farms))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    names = {
        "calibration": "calibration window",
        "development": "development window",
        "confirmatory": "confirmation period",
    }
    return [Patch(color=shades[p], label=names[p]) for p in PERIODS] + [
        Patch(facecolor="white", edgecolor=style.role("axes"), lw=0.5, label="excluded")
    ]


def layouts(
    farms: Sequence[FarmConfig],
    turbines: Mapping[str, TurbineConfig],
    found: Mapping[str, pl.DataFrame],
    style: Style,
) -> Figure:
    with style.context():
        figure = style.figure(170.0, 62.0)
        axes = figure.subplots(1, len(farms))
        for ax, farm, letter in zip(axes, farms, "abcd", strict=False):
            diameter = turbines[farm.name].rotor_diameter_m
            layout_panel(ax, found[farm.name], diameter, style)
            style.panel_letter(ax, letter, style.farm_name(farm.name))
        figure.draw_without_rendering()
        for ax in axes:
            scale_bar(ax, style)
    return figure


def sectors(
    farms: Sequence[FarmConfig], roses: Mapping[str, np.ndarray], width: float, style: Style
) -> Figure:
    top = float(np.ceil(max(r.max() for r in roses.values())))
    with style.context():
        figure = style.figure(170.0, 64.0)
        for column, farm in enumerate(farms):
            ax = cast(PolarAxes, figure.add_subplot(1, len(farms), column + 1, projection="polar"))
            rose_panel(ax, roses[farm.name], width, top, style)
            style.panel_letter(ax, "abcd"[column], style.farm_name(farm.name))
        figure.supxlabel(f"share of time from each {BIN_DEG:g}° direction (outer circle {top:g}%)")
    return figure


def composite(
    farms: Sequence[FarmConfig],
    turbines: Mapping[str, TurbineConfig],
    found: Mapping[str, pl.DataFrame],
    roses: Mapping[str, np.ndarray],
    width: float,
    style: Style,
) -> Figure:
    """a column per farm: its layout above its rose, every rose on one scale."""
    top = float(np.ceil(max(r.max() for r in roses.values())))
    with style.context():
        figure = style.figure(170.0, 100.0)
        grid = figure.add_gridspec(2, len(farms), height_ratios=(1.0, 1.15))
        plans = []
        for column, farm in enumerate(farms):
            plan = figure.add_subplot(grid[0, column])
            layout_panel(plan, found[farm.name], turbines[farm.name].rotor_diameter_m, style)
            style.panel_letter(plan, "abcd"[column], style.farm_name(farm.name))
            plans.append(plan)
            ax = cast(PolarAxes, figure.add_subplot(grid[1, column], projection="polar"))
            rose_panel(ax, roses[farm.name], width, top, style)
        figure.supxlabel(f"share of time from each {BIN_DEG:g}° direction (outer circle {top:g}%)")
        figure.draw_without_rendering()
        for plan in plans:
            scale_bar(plan, style)
    return figure


def periods(farms: Sequence[FarmConfig], style: Style) -> Figure:
    with style.context():
        figure = style.figure(170.0, 54.0)
        ax = figure.add_subplot()
        handles = timeline_panel(ax, farms, style)
        figure.legend(handles=handles, loc="outside lower center", ncols=2)
    return figure


def farms_of(sources: Sources, style: Style) -> list[FarmConfig]:
    return [config.load_farm(name, sources.configs) for name in style.report.farms.order]


def layouts_figure(sources: Sources, style: Style) -> Figure:
    farms = farms_of(sources, style)
    return layouts(farms, *layouts_of(sources, farms), style)


def roses_of(sources: Sources, farms: Sequence[FarmConfig]) -> dict[str, np.ndarray]:
    """each farm's rose over the timestamps of its development window, each counted once; a
    timestamp has one farm direction for all its turbines."""
    roses = {}
    for farm in farms:
        rows = pl.read_parquet(
            sources.data / "prepared" / farm.name / "rows.parquet",
            columns=["timestamp", "farm_direction", "period"],
        ).filter(pl.col("period") == ROSE_PERIOD)
        if rows.is_empty():
            raise ValueError(f"{farm.name} has no {ROSE_PERIOD} rows to draw a rose from")
        directions = rows.unique("timestamp", keep="first", maintain_order=True)["farm_direction"]
        roses[farm.name] = rose(directions.to_numpy())
    return roses


def layouts_of(
    sources: Sources, farms: Sequence[FarmConfig]
) -> tuple[dict[str, TurbineConfig], dict[str, pl.DataFrame]]:
    turbines = {f.name: config.load_turbine(f.turbine_type, sources.configs) for f in farms}
    found = {
        f.name: pl.read_parquet(sources.data / "prepared" / f.name / "layout.parquet")
        for f in farms
    }
    return turbines, found


def sectors_figure(sources: Sources, style: Style) -> Figure:
    farms = farms_of(sources, style)
    width = sources.protocol().splits.arc_width_deg
    return sectors(farms, roses_of(sources, farms), width, style)


def farms_figure(sources: Sources, style: Style) -> Figure:
    farms = farms_of(sources, style)
    turbines, found = layouts_of(sources, farms)
    roses = roses_of(sources, farms)
    width = sources.protocol().splits.arc_width_deg
    return composite(farms, turbines, found, roses, width, style)


def periods_figure(sources: Sources, style: Style) -> Figure:
    return periods(farms_of(sources, style), style)
