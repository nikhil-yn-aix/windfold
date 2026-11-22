"""direct labels set clear of the data. each label tries spots around its point, nearest first,
and takes the first that keeps the check's clearance from every mark, line and label of its panel
and from the panel's edge. a spot beside the point must leave no other mark about as near the
label; a spot farther out gets a thin leader, which crosses no other label or leader. place labels
only after settle(), since the spots are measured on the drawn figure."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox

from windfold.report.check import LEADER, NO_MARK
from windfold.report.style import MM_PER_INCH, Style

COMPASS = {
    name: 22.5 * k
    for k, name in enumerate(
        (
            "e",
            "ene",
            "ne",
            "nne",
            "n",
            "nnw",
            "nw",
            "wnw",
            "w",
            "wsw",
            "sw",
            "ssw",
            "s",
            "sse",
            "se",
            "ese",
        )
    )
}
ORDER = (
    "e",
    "w",
    "ne",
    "nw",
    "se",
    "sw",
    "n",
    "s",
    "ene",
    "wnw",
    "ese",
    "wsw",
    "nne",
    "nnw",
    "sse",
    "ssw",
)
REACH_PT = (2.5, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 46.0, 58.0)
LEADER_PT = 9.0
AMBIGUOUS = 2.0
APART_DEG = 30.0


@dataclass(frozen=True)
class Label:
    """a label for the point x, y in data units, trying its preferred directions first."""

    x: float
    y: float
    text: str
    prefer: tuple[str, ...] = ()


@dataclass(frozen=True)
class Disc:
    x: float
    y: float
    radius: float


@dataclass
class Room:
    """what a new label must keep clear of, in display units: the panel's area, its marks and
    lines, the labels and leaders already placed and the gap to keep."""

    area: Bbox
    marks: list[Disc]
    lines: list[Path]
    taken: list[Bbox]
    leaders: list[tuple[np.ndarray, np.ndarray]]
    gap: float


def settle(figure: Figure) -> RendererBase:
    """lay the figure out once and keep that layout, so labels placed now stay where measured."""
    canvas = FigureCanvasAgg(figure)
    figure.draw_without_rendering()
    figure.set_layout_engine("none")
    return canvas.get_renderer()


def discs(ax: Axes) -> list[Disc]:
    """every drawn mark of the panel as a disc in display units."""
    scale = ax.figure.dpi / 72
    found = []
    for line in ax.lines:
        if line.get_visible() and line.get_marker() not in NO_MARK:
            radius = line.get_markersize() / 2 * scale
            points = line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))
            found += [Disc(float(x), float(y), radius) for x, y in points]
    for collection in ax.collections:
        if isinstance(collection, PathCollection) and collection.get_visible():
            offsets = np.asarray(collection.get_offsets(), dtype=float).reshape(-1, 2)
            points = collection.get_offset_transform().transform(offsets)
            sizes = np.resize(np.asarray(collection.get_sizes(), dtype=float), len(points))
            found += [
                Disc(float(x), float(y), float(np.sqrt(s) / 2 * scale))
                for (x, y), s in zip(points, sizes, strict=True)
            ]
    return found


def room(ax: Axes, style: Style, renderer: RendererBase) -> Room:
    mm = ax.figure.dpi / MM_PER_INCH
    gap = 1.3 * style.report.page.clearance_mm * mm
    lines = [
        line.get_transform().transform_path(line.get_path())
        for line in ax.lines
        if line.get_visible() and line.get_linestyle() not in NO_MARK and line.get_linewidth() > 0
    ]
    texts = [t.get_window_extent(renderer) for t in ax.texts if t.get_visible() and t.get_text()]
    return Room(grown(ax.get_window_extent(renderer), -gap), discs(ax), lines, texts, [], gap)


def grown(box: Bbox, by: float) -> Bbox:
    return Bbox.from_extents(box.x0 - by, box.y0 - by, box.x1 + by, box.y1 + by)


def nearest(box: Bbox, point: np.ndarray) -> np.ndarray:
    return np.array([np.clip(point[0], box.x0, box.x1), np.clip(point[1], box.y0, box.y1)])


def alignment(direction: str) -> tuple[float, float, str, str]:
    """the unit step toward a compass direction and the alignment that keeps the label on the
    far side of its point."""
    angle = np.deg2rad(COMPASS[direction])
    ux, uy = float(np.cos(angle)), float(np.sin(angle))
    ha = "left" if ux > 0.3 else "right" if ux < -0.3 else "center"
    va = "bottom" if uy > 0.3 else "top" if uy < -0.3 else "center"
    return ux, uy, ha, va


def crosses(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    """whether two segments cross."""

    def side(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    (p, q), (r, s) = a, b
    return side(p, q, r) * side(p, q, s) < 0 and side(r, s, p) * side(r, s, q) < 0


def along_line(start: np.ndarray, along: np.ndarray, rim: float, line: Path) -> bool:
    """whether a leader leaving start would run within APART_DEG of a line through its mark,
    such as the mark's own interval, and so read as part of it."""
    vertices = np.asarray(line.vertices, dtype=float)
    for a, b in zip(vertices[:-1], vertices[1:], strict=True):
        step = b - a
        size = float(step @ step)
        if size == 0:
            continue
        t = float(np.clip((start - a) @ step / size, 0.0, 1.0))
        if np.hypot(*(a + t * step - start)) > max(rim, 1.0):
            continue
        cosine = abs(float(step @ along)) / np.sqrt(size * float(along @ along) + 1e-12)
        if cosine > np.cos(np.deg2rad(APART_DEG)):
            return True
    return False


def clear(box: Bbox, start: np.ndarray, own: Disc | None, space: Room, led: bool) -> bool:
    """whether a label box keeps clear of everything in space and, when it is led, whether its
    leader passes every mark but those touching its own and crosses no label or leader."""
    if not (
        space.area.x0 <= box.x0
        and space.area.y0 <= box.y0
        and box.x1 <= space.area.x1
        and box.y1 <= space.area.y1
    ):
        return False
    wide = grown(box, space.gap)
    if any(line.intersects_bbox(wide, filled=False) for line in space.lines):
        return False
    if any(wide.overlaps(other) for other in space.taken):
        return False
    end = nearest(box, start)
    along = end - start
    if led:
        rim = 0.0 if own is None else own.radius
        length = max(float(np.hypot(*along)), 1e-9)
        path = Path(np.array([start + along / length * (rim + space.gap / 2), end]))
        if (
            any(crosses((start, end), other) for other in space.leaders)
            or any(path.intersects_bbox(other) for other in space.taken)
            or any(path.intersects_path(line, filled=False) for line in space.lines)
            or any(along_line(start, along, rim, line) for line in space.lines)
        ):
            return False
    for mark in space.marks:
        if mark is own:
            continue
        centre = np.array([mark.x, mark.y])
        if np.hypot(*(nearest(box, centre) - centre)) < mark.radius + space.gap:
            return False
        touching = own is not None and np.hypot(*(centre - start)) < own.radius + mark.radius
        t = float(np.clip((centre - start) @ along / max(along @ along, 1e-9), 0.0, 1.0))
        if not touching and np.hypot(*(start + t * along - centre)) < mark.radius + space.gap / 2:
            return False
    return True


def closer(box: Bbox, point: np.ndarray, own: Disc | None, marks: Sequence[Disc]) -> bool:
    """whether another mark lies about as near the label as its own, within AMBIGUOUS times its
    distance, which would leave the label ambiguous without a leader."""
    reach = np.hypot(*(nearest(box, point) - point))
    for mark in marks:
        if mark is own or (own is not None and np.hypot(mark.x - own.x, mark.y - own.y) < 1e-6):
            continue
        centre = np.array([mark.x, mark.y])
        if np.hypot(*(nearest(box, centre) - centre)) < AMBIGUOUS * reach:
            return True
    return False


class NoRoom(ValueError):
    def __init__(self, label: Label):
        super().__init__(f"no clear spot for the label {label.text!r}")
        self.label = label


def place(
    ax: Axes,
    labels: Sequence[Label],
    style: Style,
    renderer: RendererBase,
    reach: Sequence[float] = REACH_PT,
) -> list[Annotation]:
    """set each label at its first clear spot, in order; a label left without one goes first in
    the next try, and a label no order can place is refused by name."""
    order = list(labels)
    for tries in range(len(order)):
        drawn: list[Artist] = []
        try:
            return attempt(ax, order, style, renderer, reach, drawn)
        except NoRoom as failure:
            for artist in drawn:
                artist.remove()
            if tries == len(order) - 1:
                raise
            order.remove(failure.label)
            order.insert(0, failure.label)
    return []


def attempt(
    ax: Axes,
    labels: Sequence[Label],
    style: Style,
    renderer: RendererBase,
    reach: Sequence[float],
    drawn: list[Artist],
) -> list[Annotation]:
    """one pass over the labels in order, keeping in drawn every artist it adds."""
    space = room(ax, style, renderer)
    placed = []
    for label in labels:
        point = ax.transData.transform((label.x, label.y))
        own = min(
            (m for m in space.marks if np.hypot(m.x - point[0], m.y - point[1]) < 1e-6),
            key=lambda m: -m.radius,
            default=None,
        )
        edge = 0.0 if own is None else own.radius * 72 / ax.figure.dpi
        directions = [*label.prefer, *(d for d in ORDER if d not in label.prefer)]
        found = None
        for distance in reach:
            for direction in directions:
                ux, uy, ha, va = alignment(direction)
                offset = distance + edge
                note = style.note(
                    ax, label.x, label.y, label.text, dx=ux * offset, dy=uy * offset, ha=ha, va=va
                )
                note.set_in_layout(False)
                box = note.get_window_extent(renderer)
                led = distance >= LEADER_PT
                if (led or not closer(box, point, own, space.marks)) and clear(
                    box, point, own, space, led
                ):
                    found = (note, box, led)
                    break
                note.remove()
            if found:
                break
        if found is None:
            raise NoRoom(label)
        note, box, led = found
        drawn.append(note)
        space.taken.append(box)
        if led:
            line, path = leader(ax, point, box, edge, style)
            drawn.append(line)
            space.lines.append(path)
            ends = np.asarray(path.vertices, dtype=float)
            space.leaders.append((ends[0], ends[1]))
        placed.append(note)
    return placed


def leader(
    ax: Axes, point: np.ndarray, box: Bbox, edge_pt: float, style: Style
) -> tuple[Line2D, Path]:
    """a thin line from the mark's edge toward the label, which the check reads as a leader
    rather than data, and its path in display units."""
    mm = ax.figure.dpi / MM_PER_INCH
    end = nearest(grown(box, 0.5 * mm), point)
    along = end - point
    start = point + along / np.hypot(*along) * (edge_pt * ax.figure.dpi / 72 + 0.4 * mm)
    inverse = ax.transData.inverted()
    (x0, y0), (x1, y1) = inverse.transform(start), inverse.transform(end)
    (line,) = ax.plot(
        [x0, x1], [y0, y1], color=style.role("axes"), lw=style.report.lines.min_pt, zorder=1
    )
    line.set_gid(LEADER)
    return line, Path(np.array([start, end]))
