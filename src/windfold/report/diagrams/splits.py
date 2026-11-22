"""what each of the three validations holds out, on discs read like a wind rose: the angle is the
wind direction, north up and clockwise, and the radius is time. random and time-blocked
validation test every direction that training also sees; a held-out sector tests directions
training never saw."""

from pathlib import Path

import matplotlib.path as mpath
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch, Rectangle, Wedge

from windfold import config
from windfold.config import Splits
from windfold.report.style import Style

WIDTH_MM = 170.0
HEIGHT_MM = 64.0
CENTRES_MM = (28.0, 85.0, 142.0)
DISC_Y_MM = 33.0
OUTER_MM = 15.0
INNER_MM = 4.5
TITLE_Y_MM = 11.5
DETAIL_Y_MM = 5.0
KEY_Y_MM = 59.5
KEY_STEP_MM = 26.0
ARC_START_DEG = 240.0
TIME_FOLD = 2
BUFFER_MM = 0.5
SPECKLE_DEG = 2.5
SPECKLE_MM = 0.5
SEED = 20261116
REGIMES = ("random", "time_blocked", "arc_single")


def ring(ax: Axes, x: float, y: float, inner: float, outer: float, colour: str) -> None:
    ax.add_patch(Wedge((x, y), outer, 0.0, 360.0, width=outer - inner, fc=colour, lw=0))


def sector(ax: Axes, x: float, y: float, start: float, width: float, colour: str) -> None:
    """a wedge of the disc from compass bearing start, clockwise over width degrees."""
    ax.add_patch(
        Wedge(
            (x, y),
            OUTER_MM,
            90.0 - start - width,
            90.0 - start,
            width=OUTER_MM - INNER_MM,
            fc=colour,
            lw=0,
        )
    )


def outline(ax: Axes, x: float, y: float, style: Style) -> None:
    for radius in (INNER_MM, OUTER_MM):
        ax.add_patch(
            Circle((x, y), radius, fill=False, ec=style.role("axes"), lw=style.report.lines.thin_pt)
        )


def random_disc(ax: Axes, x: float, y: float, splits: Splits, style: Style) -> None:
    """a fold's share of small cells, drawn at random, as one compound path."""
    rng = np.random.default_rng(SEED)
    cells = []
    for inner in np.arange(INNER_MM, OUTER_MM - SPECKLE_MM / 2, SPECKLE_MM):
        for start in np.deg2rad(np.arange(0.0, 360.0, SPECKLE_DEG)):
            if rng.random() < 1 / splits.random_folds:
                angles = np.array([start, start + np.deg2rad(SPECKLE_DEG)])
                radii = np.array([inner, inner + SPECKLE_MM])
                ring_x = x + np.outer(radii, np.sin(angles)).ravel()[[0, 1, 3, 2, 0]]
                ring_y = y + np.outer(radii, np.cos(angles)).ravel()[[0, 1, 3, 2, 0]]
                cells.append(mpath.Path(np.c_[ring_x, ring_y], closed=True))
    ax.add_patch(PathPatch(mpath.Path.make_compound_path(*cells), fc=style.role("text"), lw=0))


def time_disc(ax: Axes, x: float, y: float, splits: Splits, style: Style) -> None:
    fold = (OUTER_MM - INNER_MM) / splits.time_folds
    inner = INNER_MM + TIME_FOLD * fold
    ring(ax, x, y, inner - BUFFER_MM, inner + fold + BUFFER_MM, "white")
    ring(ax, x, y, inner, inner + fold, style.role("text"))


def arc_roles(splits: Splits, start: float, opposite: bool) -> list[tuple[float, float, str]]:
    """the arcs a sector split leaves out of training, as (compass bearing, width, role): the test
    arc with an unused buffer either side and, for the opposite variant, the opposite arc and its
    buffers unused too."""
    width, buffer = splits.arc_width_deg, splits.arc_buffer_deg
    held = [(start - buffer, width + 2 * buffer, "unused"), (start, width, "test")]
    if opposite:
        held.append((start + 180.0 - buffer, width + 2 * buffer, "unused"))
    return held


def role_at(held: list[tuple[float, float, str]], direction: float) -> str:
    """the role of a direction: the last held arc that covers it, else training."""
    found = "training"
    for start, width, role in held:
        if (direction - start) % 360.0 < width:
            found = role
    return found


def arc_disc(ax: Axes, x: float, y: float, splits: Splits, style: Style, opposite: bool) -> None:
    colours = {"unused": "white", "test": style.role("text")}
    for start, width, role in arc_roles(splits, ARC_START_DEG, opposite):
        sector(ax, x, y, start, width, colours[role])


def details(splits: Splits) -> dict[str, str]:
    return {
        "random": f"{splits.random_folds} folds of timestamps",
        "time_blocked": f"{splits.time_folds} blocks, {splits.time_buffer_days}-day buffers",
        "arc_single": f"{splits.arc_count} sectors of {splits.arc_width_deg:g}°, "
        f"{splits.arc_buffer_deg:g}° buffers",
    }


def legend(ax: Axes, style: Style) -> None:
    states = (
        ("training", style.role("context")),
        ("test", style.role("text")),
        ("unused", "white"),
    )
    for k, (name, colour) in enumerate(states):
        x = CENTRES_MM[0] - OUTER_MM + k * KEY_STEP_MM
        ax.add_patch(
            Rectangle(
                (x, KEY_Y_MM - 1.5),
                3.0,
                3.0,
                fc=colour,
                ec=style.role("axes"),
                lw=style.report.lines.thin_pt,
            )
        )
        style.note(ax, x + 4.5, KEY_Y_MM, name, ha="left", va="center")
    style.note(
        ax,
        CENTRES_MM[-1] + OUTER_MM,
        KEY_Y_MM,
        "angle: wind direction    radius: time",
        ha="right",
        va="center",
    )


def draw(splits: Splits, style: Style) -> Figure:
    with style.context():
        figure = style.figure(WIDTH_MM, HEIGHT_MM)
        figure.set_layout_engine("none")
        ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        notes = details(splits)
        for regime, x in zip(REGIMES, CENTRES_MM, strict=True):
            ring(ax, x, DISC_Y_MM, INNER_MM, OUTER_MM, style.role("context"))
            if regime == "random":
                random_disc(ax, x, DISC_Y_MM, splits, style)
            elif regime == "time_blocked":
                time_disc(ax, x, DISC_Y_MM, splits, style)
            else:
                arc_disc(ax, x, DISC_Y_MM, splits, style, opposite=regime == "arc_opposite")
            outline(ax, x, DISC_Y_MM, style)
            style.note(
                ax,
                x,
                TITLE_Y_MM,
                style.regime_name(regime),
                ha="center",
                va="center",
                fontweight=style.report.type.panel_letter_weight,
            )
            style.note(ax, x, DETAIL_Y_MM, notes[regime], ha="center", va="center")
        style.note(ax, CENTRES_MM[2], DISC_Y_MM + OUTER_MM + 2.0, "N", ha="center", va="bottom")
        legend(ax, style)
        ax.set(xlim=(0.0, WIDTH_MM), ylim=(0.0, HEIGHT_MM), aspect="equal")
        ax.set_axis_off()
    return figure


def figure(style: Style, configs: Path | None = None) -> Figure:
    return draw(config.load_protocol(configs).splits, style)
