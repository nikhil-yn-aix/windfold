"""the automatic figure check: page size, text closer than the clearance to other text, to a
panel's edge or to the canvas edge, text or keys drawn over data, colored text, type and line
sizes and the outer margin. a figure that fails is never written."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from windfold.report.paths import PREVIEW_PX
from windfold.report.style import MM_PER_INCH, Style

TOLERANCE_MM = 0.01
NO_MARK = (None, "None", "none", "", " ")
LEADER = "leader"
RECORDED = "recorded-"


class FigureRefused(ValueError):
    pass


@dataclass
class Drawn:
    texts: dict[int, tuple[Text, float]] = field(default_factory=dict)
    strokes: set[tuple[float, str]] = field(default_factory=set)


class Recorder(RendererBase):
    """a renderer that draws nothing and records every text and every stroke width. matplotlib
    draws each line of a multi-line text without naming the text, inside a group carrying the
    text's gid, so names maps those gids back to their texts."""

    def __init__(self, figure: Figure, names: Mapping[str, Text] | None = None):
        super().__init__()
        self.dpi = figure.dpi
        self.width, self.height = figure.bbox.size
        self.measure = RendererAgg(int(self.width) + 1, int(self.height) + 1, self.dpi)
        self.drawn = Drawn()
        self.names = dict(names or {})
        self.groups: list[str | None] = []

    def open_group(self, s, gid=None) -> None:
        self.groups.append(gid)

    def close_group(self, s) -> None:
        if self.groups:
            self.groups.pop()

    def owner(self) -> Text | None:
        """the text whose lines are being drawn: the innermost group that names one."""
        return next((self.names[g] for g in reversed(self.groups) if g in self.names), None)

    def get_canvas_width_height(self) -> tuple[float, float]:
        return self.width, self.height

    def points_to_pixels(self, points):
        return np.asarray(points, dtype=float) * self.dpi / 72

    def get_text_width_height_descent(self, s, prop, ismath):
        return self.measure.get_text_width_height_descent(s, prop, ismath)

    def flipy(self) -> bool:
        return False

    def stroke(self, width: float, colour: Any) -> None:
        rgba = np.asarray(colour, dtype=float).ravel()
        if width > 0 and (rgba.size < 4 or rgba[3] > 0):
            red, green, blue = (float(v) for v in rgba[:3])
            self.drawn.strokes.add((round(float(width), 4), to_hex((red, green, blue))))

    def draw_path(self, gc: GraphicsContextBase, path, transform, rgbFace=None) -> None:
        self.stroke(gc.get_linewidth(), gc.get_rgb())
        if gc.get_hatch() is not None:
            self.stroke(gc.get_hatch_linewidth(), gc.get_hatch_color())

    def draw_path_collection(
        self,
        gc,
        master_transform,
        paths,
        all_transforms,
        offsets,
        offset_trans,
        facecolors,
        edgecolors,
        linewidths,
        linestyles,
        antialiaseds,
        urls,
        offset_position,
        *,
        hatchcolors=None,
    ) -> None:
        if len(paths) == 0:
            return
        widths = np.unique(np.atleast_1d(np.asarray(linewidths, dtype=float)))
        colours = np.asarray(edgecolors, dtype=float).reshape(-1, 4)
        for colour in np.unique(colours, axis=0):
            for width in widths if widths.size else [gc.get_linewidth()]:
                self.stroke(float(width), colour)
        if gc.get_hatch() is not None:
            self.stroke(gc.get_hatch_linewidth(), gc.get_hatch_color())

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None) -> None:
        text = mtext if mtext is not None else self.owner()
        if text is not None and s.strip():
            self.drawn.texts[id(text)] = (text, prop.get_size_in_points())

    def draw_image(self, gc, x, y, im) -> None:
        return None

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform) -> None:
        return None


@contextmanager
def named_texts(figure: Figure) -> Iterator[dict[str, Text]]:
    """every text under a unique gid while the figure is recorded; the gids are put back after."""
    texts = figure.findobj(Text)
    saved = [text.get_gid() for text in texts]
    names = {f"{RECORDED}{k}": text for k, text in enumerate(texts)}
    for name, text in names.items():
        text.set_gid(name)
    try:
        yield names
    finally:
        for text, gid in zip(texts, saved, strict=True):
            text.set_gid(gid)


def record(figure: Figure) -> tuple[Recorder, Drawn]:
    """draw once to settle the layout, which moves titles above polar tick labels, then record
    the drawing as it will be saved."""
    FigureCanvasAgg(figure)
    figure.draw_without_rendering()
    with named_texts(figure) as names:
        recorder = Recorder(figure, names)
        figure.draw(recorder)
    return recorder, recorder.drawn


def to_mm(box: Bbox, dpi: float) -> np.ndarray:
    return np.array([box.x0, box.y0, box.x1, box.y1]) / dpi * MM_PER_INCH


def panel_of(axes: Any, panels: Sequence[Axes]) -> Axes | None:
    """the panel an axes belongs to: itself, or the axes it is an inset of."""
    if axes is None:
        return None
    for panel in panels:
        if axes in panel.child_axes:
            return panel
    return axes


def label(text: Text) -> str:
    content = " ".join(text.get_text().split())
    return f"'{content[:40]}'"


def within(points: np.ndarray, box: Bbox, reach: np.ndarray | float) -> bool:
    """whether any point, a marker of radius reach in display units, touches box."""
    if points.size == 0:
        return False
    x, y = points[:, 0], points[:, 1]
    dx = np.maximum(np.maximum(box.x0 - x, x - box.x1), 0.0)
    dy = np.maximum(np.maximum(box.y0 - y, y - box.y1), 0.0)
    return bool(np.any(np.hypot(dx, dy) < reach))


def line_touches(line: Line2D, box: Bbox, dpi: float) -> bool:
    transform = line.get_transform()
    if line.get_linestyle() not in NO_MARK and line.get_linewidth() > 0:
        if transform.transform_path(line.get_path()).intersects_bbox(box, filled=False):
            return True
    if line.get_marker() in NO_MARK:
        return False
    reach = line.get_markersize() / 2 * dpi / 72
    return within(transform.transform(np.asarray(line.get_xydata(), dtype=float)), box, reach)


def collection_touches(collection: Any, box: Bbox, dpi: float) -> bool:
    if isinstance(collection, PathCollection):
        offsets = np.asarray(collection.get_offsets(), dtype=float).reshape(-1, 2)
        points = collection.get_offset_transform().transform(offsets)
        sizes = np.atleast_1d(np.asarray(collection.get_sizes(), dtype=float))
        reach = np.sqrt(np.resize(sizes, len(points))) / 2 * dpi / 72
        return within(points, box, reach)
    if isinstance(collection, LineCollection):
        transform = collection.get_transform()
        return any(
            transform.transform_path(path).intersects_bbox(box, filled=False)
            for path in collection.get_paths()
        )
    return False


def covers_data(ax: Axes, box: Bbox, area: Bbox, dpi: float) -> bool:
    """whether box, in display units, lies over a line, marker or point of ax's data; leader
    lines, drawn to join a label to its mark, do not count."""
    inside = Bbox.intersection(box, area)
    if inside is None:
        return False
    lines = [a for a in ax.lines if a.get_visible() and a.get_gid() != LEADER]
    if any(line_touches(line, inside, dpi) for line in lines):
        return True
    return any(collection_touches(c, inside, dpi) for c in ax.collections if c.get_visible())


def crowded(boxes: np.ndarray, clearance: float) -> list[tuple[int, int, float]]:
    """pairs of boxes, in mm, closer than clearance, with the gap between them; overlapping boxes
    have no gap."""
    x0, y0, x1, y1 = boxes.T
    across = np.maximum(x0[None, :] - x1[:, None], x0[:, None] - x1[None, :])
    up = np.maximum(y0[None, :] - y1[:, None], y0[:, None] - y1[None, :])
    gap = np.maximum(np.maximum(across, up), 0.0)
    first, second = np.nonzero(np.triu(gap < clearance - TOLERANCE_MM, k=1))
    return [(int(i), int(j), float(gap[i, j])) for i, j in zip(first, second, strict=True)]


def near_edge(box: np.ndarray, area: np.ndarray, clearance: float) -> bool:
    """whether box comes within clearance of the edge of area, from inside or outside."""
    grown = box + np.array([-clearance, -clearance, clearance, clearance]) * (1 - 1e-9)
    meets = grown[0] < area[2] and grown[2] > area[0] and grown[1] < area[3] and grown[3] > area[1]
    inside = (
        grown[0] >= area[0] and grown[1] >= area[1] and grown[2] <= area[2] and grown[3] <= area[3]
    )
    return meets and not inside


def problems(figure: Figure, style: Style) -> list[str]:
    """every reason the figure breaks the report's rules; empty when it may be written."""
    page, report = style.report.page, style.report
    found = []
    width_mm, height_mm = (float(v) * MM_PER_INCH for v in figure.get_size_inches())
    if not any(abs(width_mm - w) <= TOLERANCE_MM for w in page.widths_mm):
        allowed = " or ".join(f"{w:g}" for w in page.widths_mm)
        found.append(f"width is {width_mm:.2f} mm, not {allowed} mm")
    if height_mm > page.max_height_mm + TOLERANCE_MM:
        found.append(f"height is {height_mm:.2f} mm, over {page.max_height_mm:g} mm")
    recorder, drawn = record(figure)
    dpi = figure.dpi
    texts = [text for text, _ in drawn.texts.values()]
    neutrals = (report.neutrals.text, report.neutrals.axes)
    printed = page.print_scale(width_mm)
    for text, size in drawn.texts.values():
        if size * printed < report.type.min_pt - 1e-9:
            found.append(
                f"text {label(text)} prints at {size * printed:.3g} pt, "
                f"under {report.type.min_pt:g} pt"
            )
        if to_hex(text.get_color()) not in neutrals:
            found.append(f"text {label(text)} is coloured {to_hex(text.get_color())}")
    for width, colour in sorted(drawn.strokes):
        if width < report.lines.min_pt - 1e-9:
            found.append(f"a {colour} line is {width:g} pt, under {report.lines.min_pt:g} pt")
    boxes = np.array([to_mm(t.get_window_extent(recorder), dpi) for t in texts]).reshape(-1, 4)
    clearance = page.clearance_mm
    for i, j, gap in crowded(boxes, clearance):
        if gap <= 0:
            found.append(f"text {label(texts[i])} overlaps text {label(texts[j])}")
        else:
            found.append(
                f"text {label(texts[i])} is {gap:.2f} mm from text {label(texts[j])}, "
                f"under {clearance:g} mm"
            )
    for text, box in zip(texts, boxes, strict=True):
        if (
            box[0] < clearance - TOLERANCE_MM
            or box[1] < clearance - TOLERANCE_MM
            or box[2] > width_mm - clearance + TOLERANCE_MM
            or box[3] > height_mm - clearance + TOLERANCE_MM
        ):
            found.append(f"text {label(text)} is within {clearance:g} mm of the canvas edge")
    panels = [ax for ax in figure.axes if ax.get_visible()]
    areas = {id(ax): to_mm(ax.get_window_extent(recorder), dpi) for ax in panels}
    framed = [ax for ax in panels if ax.name == "rectilinear"]
    for text, box in zip(texts, boxes, strict=True):
        if any(near_edge(box, areas[id(ax)], clearance) for ax in framed):
            found.append(f"text {label(text)} is within {clearance:g} mm of a panel's edge")
    owners = {id(artist): ax for ax in panels for artist in ax.findobj(Text)}
    for text, box in zip(texts, boxes, strict=True):
        own = panel_of(text.axes or owners.get(id(text)), panels)
        for ax in panels:
            if own is not None and panel_of(ax, panels) is own:
                continue
            area = areas[id(ax)]
            crossed = (
                min(box[2], area[2]) - max(box[0], area[0]) > TOLERANCE_MM
                and min(box[3], area[3]) - max(box[1], area[1]) > TOLERANCE_MM
            )
            if crossed:
                found.append(f"text {label(text)} crosses into another panel")
                break
    keys: list[tuple[Text | Legend, list[Axes]]] = [(key, panels) for key in figure.legends]
    for ax in panels:
        keys += [(text, [ax]) for text in ax.texts if text.get_visible() and text.get_text()]
        if isinstance(legend := ax.get_legend(), Legend):
            keys.append((legend, [ax]))
    for key, under in keys:
        box = key.get_window_extent(recorder)
        if any(covers_data(ax, box, ax.get_window_extent(recorder), dpi) for ax in under):
            name = f"text {label(key)}" if isinstance(key, Text) else "a key"
            found.append(f"{name} lies over data")
    content = to_mm(figure.get_tightbbox(recorder), 1.0)
    margins = {
        "left": content[0],
        "bottom": content[1],
        "right": width_mm - content[2],
        "top": height_mm - content[3],
    }
    for side, margin in margins.items():
        if margin >= page.max_margin_mm:
            found.append(f"{side} margin is {margin:.2f} mm, not under {page.max_margin_mm:g} mm")
    return found


def check(figure: Figure, style: Style) -> None:
    found = problems(figure, style)
    if found:
        raise FigureRefused("figure refused: " + "; ".join(found))


def preview_dpi(figure: Figure, pixels: int) -> float:
    return pixels / float(figure.get_figwidth())


def save(
    figure: Figure, files: Sequence[Path], style: Style, metadata: dict[str, str] | None = None
) -> None:
    """check the figure, then write the vector pdf and its previews, PREVIEW_PX pixels wide."""
    check(figure, style)
    pdf, *previews = files
    if len(previews) != len(PREVIEW_PX):
        raise ValueError(f"expected {len(PREVIEW_PX)} preview files, got {len(previews)}")
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
    info = {"CreationDate": None, "Creator": "windfold"} | (metadata or {})
    with style.context():
        figure.savefig(pdf, format="pdf", dpi=style.report.files.raster_dpi, metadata=info)
        for path, pixels in zip(previews, PREVIEW_PX, strict=True):
            figure.savefig(
                path,
                format="png",
                dpi=preview_dpi(figure, pixels),
                metadata={"Software": "windfold"} | (metadata or {}),
            )
