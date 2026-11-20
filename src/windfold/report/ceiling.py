"""where the error of calibrated physics on held-out sectors goes, from the exploratory
transfer-ceiling analysis of the development sectors, per farm and as the mean of the farms. the
waked-row error PyWake at measured inflow leaves on held-out sectors, in points of the no-wake
error, splits into four parts: what no fit explains even on the sector's own rows, what
turbine-by-direction cells add there, what the shared terms fit on the sector's own rows beyond
what they learn on the other sectors and what they learn on the other sectors with the penalty
chosen nested. transfer, the paper's figure: that split for the mean of the farms, and its last
part enlarged beside the skill windfold realizes over the same physics, each with its 95%
interval. ceiling: the split per farm. realised: the last part beside windfold's skill, per
farm."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from windfold.report.check import LEADER
from windfold.report.labels import settle
from windfold.report.sources import Sources, bounded
from windfold.report.style import Style

ANALYSIS = "transfer_ceiling"
ERROR = "pywake_inflow_error"
OWN_ROWS = "own_rows_shared_terms"
NESTED = "shared_terms_nested"
PARTS = (
    ("unexplained", "not explained on the sector's own rows"),
    ("turbine_by_direction", "turbine-by-direction cells, own rows"),
    ("own_rows_only", "shared terms, only with own rows"),
    ("other_sectors", "shared terms learned on other sectors"),
)
GAINS = ((NESTED, PARTS[-1][1], "D", "axes"), ("windfold", "windfold", "o", "windfold"))
MEAN = "mean of farms"
OFFSET = 0.16
TINT = 0.3
BAR = 0.5
LEVEL = 1.0
GAP_PT = 3.0
MISSED = "held-out error of PyWake at measured inflow (% of no-wake error)"
OVER = "skill points over PyWake at measured inflow"


def shade(weight: float, style: Style) -> tuple[float, float, float]:
    """the axes neutral laid over the grid neutral at weight."""
    light, dark = (np.array(to_rgb(style.role(r))) for r in ("grid", "axes"))
    red, green, blue = ((1 - weight) * light + weight * dark).tolist()
    return red, green, blue


def fills(style: Style) -> dict[str, tuple[float, float, float]]:
    """each part's fill, stepping in lightness between neighbors so they stay apart in gray; the
    part learned on the other sectors is a light tint of windfold's color."""
    red, green, blue = (TINT * c + 1 - TINT for c in to_rgb(style.role("windfold")))
    return {
        "unexplained": shade(0.0, style),
        "turbine_by_direction": shade(0.85, style),
        "own_rows_only": shade(0.45, style),
        "other_sectors": (red, green, blue),
    }


def split(points: Mapping[str, Any]) -> dict[str, float]:
    """the four parts of PyWake's held-out error, in points, which add up to it."""
    own, nested = bounded(points[OWN_ROWS])[0], bounded(points[NESTED])[0]
    found = {
        "unexplained": bounded(points["unexplained"])[0],
        "turbine_by_direction": bounded(points["turbine_by_direction"])[0],
        "own_rows_only": own - nested,
        "other_sectors": nested,
    }
    total, added = bounded(points[ERROR])[0], sum(found.values())
    if not np.isclose(added, total, rtol=0.0, atol=1e-6):
        raise ValueError(f"the parts add up to {added:.6f} points, not the error {total:.6f}")
    return found


def rows_of(found: Mapping[str, Any], style: Style) -> tuple[list[str], list[Mapping[str, Any]]]:
    """the farms in the report's order, then the mean of the farms, as names and points."""
    farms = [farm for farm in style.report.farms.order if farm in found["farms"]]
    names = [style.farm_name(farm) for farm in farms] + [MEAN]
    points = [found["farms"][farm]["points"] for farm in farms] + [found["equal_farm"]["points"]]
    return names, points


def stack(ax: Axes, row: float, points: Mapping[str, Any], style: Style, height: float) -> None:
    left, colours = 0.0, fills(style)
    for key, width in split(points).items():
        ax.barh(row, width, left=left, height=height, color=colours[key], lw=0)
        left += width


def parts(names: Sequence[str], points: Sequence[Mapping[str, Any]], style: Style) -> Figure:
    colours = fills(style)
    with style.context():
        figure = style.figure(170.0, 36.0 + 9.0 * len(names))
        ax = figure.add_subplot()
        for row, found in enumerate(points):
            stack(ax, row, found, style, 0.6)
        style.rows(ax, names, len(names))
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel(MISSED)
        figure.legend(
            handles=[Patch(color=colours[key], label=name) for key, name in PARTS],
            loc="outside upper center",
            ncols=2,
            borderaxespad=0.6,
            borderpad=0.1,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.6,
        )
    return figure


def realised(names: Sequence[str], points: Sequence[Mapping[str, Any]], style: Style) -> Figure:
    with style.context():
        figure = style.figure(85.0, 36.0 + 10.0 * len(names))
        ax = figure.add_subplot()
        style.zero_line(ax)
        for row, found in enumerate(points):
            for index, (key, _, marker, role) in enumerate(GAINS):
                at = row + OFFSET * (2 * index - 1)
                style.estimate(ax, at, *bounded(found[key]), style.role(role), marker=marker)
        style.rows(ax, names, len(names))
        ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel(OVER)
        width = style.report.lines.interval_pt
        keys = [
            Line2D([], [], marker=marker, color=style.role(role), mew=width, ls="none")
            for _, _, marker, role in GAINS
        ]
        figure.legend(
            keys,
            [name for _, name, _, _ in GAINS],
            loc="outside upper center",
            ncols=1,
            borderaxespad=0.6,
            borderpad=0.1,
            handletextpad=0.4,
        )
    return figure


def name_parts(ax: Axes, found: Mapping[str, float], style: Style, figure: Figure) -> int:
    """each part named inside its bar where it fits on a light fill, the rest above the bar in
    steps rising to the right, each set left of a leader that drops to its part; a leader passes
    only the steps below it, which end further left. returns the number of steps."""
    renderer = settle(figure)
    colours = fills(style)
    points_per_unit = ax.get_window_extent(renderer).width / np.ptp(ax.get_xlim()) * 72
    points_per_unit /= figure.dpi
    gap = GAP_PT / points_per_unit
    left, above = 0.0, []
    for key, name in PARTS:
        width = found[key]
        centre = left + width / 2
        note = style.note(ax, centre, 0.0, name, ha="center", va="center")
        text = note.get_window_extent(renderer).width / figure.dpi * 72 / points_per_unit
        light = float(np.dot(colours[key], (0.2126, 0.7152, 0.0722))) > 0.5
        if not (light and text + 2 * gap <= width):
            note.remove()
            above.append((centre, name))
        left += width
    for level, (centre, name) in enumerate(above, start=1):
        height = BAR / 2 + LEVEL * level
        style.note(ax, centre - gap, height, name, ha="right", va="center")
        ax.plot(
            [centre, centre],
            [BAR / 2, height],
            color=style.role("axes"),
            lw=style.report.lines.min_pt,
            gid=LEADER,
            zorder=1,
        )
    return len(above)


def enlarged(ax: Axes, points: Mapping[str, Any], style: Style) -> None:
    """the last part as a band from zero, its estimate and interval on the first row and
    windfold's on the second, each named beside its interval."""
    nested = bounded(points[NESTED])
    ax.add_patch(
        Rectangle((0.0, -0.5), nested[0], 2.0, color=fills(style)["other_sectors"], lw=0, zorder=0)
    )
    for row, (key, name, marker, role) in enumerate(GAINS):
        centre, low, high = bounded(points[key])
        style.estimate(ax, row, centre, low, high, style.role(role), marker=marker)
        if row == 0:
            style.note(ax, low, row, name, dx=-GAP_PT, ha="right", va="center")
        else:
            style.note(ax, high, row, name, dx=GAP_PT, ha="left", va="center")
    ax.set_ylim(1.7, -0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.0, None)
    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel(OVER)


def decomposition(points: Mapping[str, Any], style: Style) -> Figure:
    """the mean of the farms: (a) the split of PyWake's error, (b) its last part enlarged with
    windfold's skill over the same physics."""
    found = split(points)
    total = sum(found.values())
    with style.context():
        figure = style.figure(170.0, 92.0)
        whole, part = figure.subplots(2, 1, height_ratios=(1.6, 1.0))
        stack(whole, 0.0, points, style, BAR)
        whole.set_xlim(0.0, 1.03 * total)
        whole.set_yticks([])
        whole.spines["left"].set_visible(False)
        whole.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
        whole.set_xlabel(MISSED)
        style.panel_letter(whole, "a")
        enlarged(part, points, style)
        style.panel_letter(part, "b")
        steps = name_parts(whole, found, style, figure)
        whole.set_ylim(-BAR / 2 - 0.3, BAR / 2 + LEVEL * steps + 0.55)
    return figure


def figure(sources: Sources, style: Style) -> Figure:
    return parts(*rows_of(sources.analysis(ANALYSIS), style), style)


def realised_figure(sources: Sources, style: Style) -> Figure:
    return realised(*rows_of(sources.analysis(ANALYSIS), style), style)


def transfer_figure(sources: Sources, style: Style) -> Figure:
    return decomposition(sources.analysis(ANALYSIS)["equal_farm"]["points"], style)
