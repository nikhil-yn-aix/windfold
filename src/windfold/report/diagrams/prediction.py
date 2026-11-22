"""one prediction: windfold adds each turbine's offset, calibrated physics, a correction shared by
every turbine pair and small local cells, and its gaussian posterior gives each prediction a mean
and a variance."""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge

from windfold import config
from windfold.config import Processing
from windfold.report.style import Style

WIDTH_MM = 170.0
HEIGHT_MM = 37.0
CENTRES_MM = (16.5, 54.0, 91.0, 123.0, 154.5)
GLYPH_Y_MM = 26.0
NAME_Y_MM = 10.5
DETAIL_Y_MM = 4.5
CELL_DEG = 250.0
PLUS_MM = 1.6
OFFSETS_MM = (2.6, -1.8, 1.2, -3.0)
REACH_MM = ((9.0, 9.0), (8.75, 10.0), (10.0, 11.45), (9.0, 9.0))
PARTS = (
    ("turbine offsets", "per turbine"),
    ("calibrated physics", "measured inflow"),
    ("shared correction", "wind's frame"),
    ("local cells", "shrunk to zero"),
    ("prediction", "mean, variance"),
)


def turbine(ax: Axes, x: float, y: float, colour: str, style: Style, ring: bool = False) -> None:
    lines = style.report.lines
    if ring:
        ax.add_patch(Circle((x, y), 1.1, fc="white", ec=colour, lw=lines.thick_pt, zorder=3))
    else:
        ax.add_patch(Circle((x, y), 0.75, fc=colour, lw=0, zorder=3))


def offsets(ax: Axes, x: float, y: float, style: Style) -> None:
    """a level for each of four turbines, up or down from the farm's mean."""
    grey, lines = style.role("axes"), style.report.lines
    ax.plot([x - 9.0, x + 9.0], [y, y], color=grey, lw=lines.axes_pt)
    for k, level in enumerate(OFFSETS_MM):
        at = x - 6.0 + 4.0 * k
        ax.add_patch(Rectangle((at - 0.9, min(y, y + level)), 1.8, abs(level), fc=grey, lw=0))


def physics(ax: Axes, x: float, y: float, style: Style) -> None:
    orange = style.role("physics")
    ax.add_patch(
        Polygon(
            [(x - 8, y - 1.3), (x + 10, y - 4.5), (x + 10, y + 4.5), (x - 8, y + 1.3)],
            fc=orange,
            alpha=0.25,
            lw=0,
        )
    )
    turbine(ax, x - 8, y, style.role("axes"), style)
    turbine(ax, x + 4, y, orange, style, ring=True)


def shared(ax: Axes, x: float, y: float, style: Style) -> None:
    blue = style.role("windfold")
    lines = style.report.lines
    ax.add_patch(Rectangle((x - 10, y - 3.5), 20, 7, fill=False, ec=blue, lw=lines.data_pt))
    for dx, dy in ((-4.5, 1.4), (2.0, -1.3)):
        turbine(ax, x + dx, y + dy, blue, style)
    turbine(ax, x + 10, y, blue, style, ring=True)


def local(ax: Axes, x: float, y: float, cells: Processing, style: Style) -> None:
    """one ring per speed bin around the turbine, cut into direction buckets, one cell dark."""
    lines = style.report.lines
    edge = style.role("axes")
    bucket = cells.direction_bucket_deg
    radii = np.linspace(3.5, 9.0, len(cells.speed_bin_edges_ms))
    for radius in radii:
        ax.add_patch(Circle((x, y), radius, fill=False, ec=edge, alpha=0.3, lw=lines.min_pt))
    for angle in np.deg2rad(np.arange(0.0, 360.0, bucket)):
        ax.plot(
            x + radii[[0, -1]] * np.sin(angle),
            y + radii[[0, -1]] * np.cos(angle),
            color=edge,
            alpha=0.3,
            lw=lines.min_pt,
        )
    ax.add_patch(
        Wedge(
            (x, y),
            radii[2],
            90.0 - CELL_DEG - bucket,
            90.0 - CELL_DEG,
            width=radii[2] - radii[1],
            fc=edge,
            lw=0,
        )
    )
    turbine(ax, x, y, edge, style)


def prediction(ax: Axes, x: float, y: float, style: Style) -> None:
    blue = style.role("windfold")
    lines = style.report.lines
    spread = 3.0
    grid = np.linspace(-3.3 * spread, 3.3 * spread, 200)
    curve = 12.0 * np.exp(-0.5 * (grid / spread) ** 2)
    base = y - 6.0
    ax.fill_between(x + grid, base, base + curve, color=blue, alpha=0.15, lw=0)
    ax.plot(x + grid, base + curve, color=blue, lw=lines.data_pt)
    ax.plot([x, x], [base, base + 12.0], color=blue, lw=lines.thin_pt)
    ax.plot(x + grid[[0, -1]], [base, base], color=style.role("axes"), lw=lines.axes_pt)


def plus(ax: Axes, x: float, y: float, style: Style) -> None:
    for dx, dy in ((PLUS_MM, 0.0), (0.0, PLUS_MM)):
        ax.plot(
            [x - dx, x + dx],
            [y - dy, y + dy],
            color=style.role("text"),
            lw=style.report.lines.data_pt,
            solid_capstyle="butt",
        )


def panel(ax: Axes, cells: Processing, style: Style) -> None:
    """the sum of windfold's terms and its prediction, on axes measured in millimeters."""
    weight = style.report.type.panel_letter_weight
    offsets(ax, CENTRES_MM[0], GLYPH_Y_MM, style)
    physics(ax, CENTRES_MM[1], GLYPH_Y_MM, style)
    shared(ax, CENTRES_MM[2], GLYPH_Y_MM, style)
    local(ax, CENTRES_MM[3], GLYPH_Y_MM, cells, style)
    prediction(ax, CENTRES_MM[4], GLYPH_Y_MM, style)
    for k in range(len(REACH_MM) - 1):
        right = CENTRES_MM[k] + REACH_MM[k][1]
        left = CENTRES_MM[k + 1] - REACH_MM[k + 1][0]
        plus(ax, (right + left) / 2, GLYPH_Y_MM, style)
    start, end = CENTRES_MM[3] + 10.5, CENTRES_MM[4] - 11.5
    ax.add_patch(
        FancyArrowPatch(
            (start, GLYPH_Y_MM),
            (end, GLYPH_Y_MM),
            arrowstyle="-|>",
            mutation_scale=10,
            color=style.role("text"),
            lw=style.report.lines.data_pt,
        )
    )
    for centre, (name, detail) in zip(CENTRES_MM, PARTS, strict=True):
        style.note(ax, centre, NAME_Y_MM, name, ha="center", va="center", fontweight=weight)
        style.note(ax, centre, DETAIL_Y_MM, detail, ha="center", va="center")
    ax.set(xlim=(0.0, WIDTH_MM), ylim=(0.0, HEIGHT_MM), aspect="equal")
    ax.set_axis_off()


def figure(style: Style, configs: Path | None = None) -> Figure:
    cells = config.load_protocol(configs).processing
    with style.context():
        figure = style.figure(WIDTH_MM, HEIGHT_MM)
        figure.set_layout_engine("none")
        panel(figure.add_axes((0.0, 0.0, 1.0, 1.0)), cells, style)
    return figure
