"""the diagrams' spacing rule: every text keeps a clear gap to every line, shape and other text.

the check traces what the figure actually draws, so hidden axes and ticks outside the view never
count, and it names the artist each clash comes from."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from matplotlib.artist import Artist
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from windfold.report.style import MM_PER_INCH

GAP_MM = 1.0


@dataclass
class Stroke:
    owner: str | None
    path: Path
    filled: bool
    half_width: float


@dataclass
class Traced:
    strokes: list[Stroke] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)


class Tracer(RendererBase):
    """a renderer that draws nothing and keeps every path, in pixels, with the artist that drew
    it, and every text, a multi-line one found through the group its lines are drawn in."""

    def __init__(self, figure: Figure, names: Mapping[str, Artist] | None = None):
        super().__init__()
        self.names = dict(names or {})
        self.dpi = figure.dpi
        self.width, self.height = figure.bbox.size
        self.measure = RendererAgg(int(self.width) + 1, int(self.height) + 1, self.dpi)
        self.owners: list[str | None] = []
        self.traced = Traced()

    def get_canvas_width_height(self) -> tuple[float, float]:
        return self.width, self.height

    def points_to_pixels(self, points):
        return np.asarray(points, dtype=float) * self.dpi / 72

    def get_text_width_height_descent(self, s, prop, ismath):
        return self.measure.get_text_width_height_descent(s, prop, ismath)

    def flipy(self) -> bool:
        return False

    def open_group(self, s, gid=None) -> None:
        self.owners.append(gid)

    def close_group(self, s) -> None:
        self.owners.pop()

    def draw_path(self, gc: GraphicsContextBase, path, transform, rgbFace=None) -> None:
        width = float(gc.get_linewidth())
        filled = rgbFace is not None and to_rgba(rgbFace)[3] > 0
        stroked = width > 0 and to_rgba(gc.get_rgb())[3] > 0
        if filled or stroked:
            owner = self.owners[-1] if self.owners else None
            half = width * self.dpi / 72 / 2 if stroked else 0.0
            for part in transform.transform_path(path).to_polygons(closed_only=False):
                self.traced.strokes.append(Stroke(owner, Path(part), filled, half))

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None) -> None:
        owner = self.names.get(self.owners[-1] or "") if self.owners else None
        text = mtext if mtext is not None else owner
        seen = any(text is other for other in self.traced.texts)
        if isinstance(text, Text) and s.strip() and not seen:
            self.traced.texts.append(text)

    def draw_image(self, gc, x, y, im) -> None:
        return None


@contextmanager
def named(figure: Figure) -> Iterator[dict[str, Artist]]:
    """every artist under a unique gid while the figure is traced; the gids are put back after."""
    artists = figure.findobj()
    saved = [artist.get_gid() for artist in artists]
    names = {f"traced-{k}": artist for k, artist in enumerate(artists)}
    for name, artist in names.items():
        artist.set_gid(name)
    try:
        yield names
    finally:
        for artist, gid in zip(artists, saved, strict=True):
            artist.set_gid(gid)


def trace(figure: Figure) -> tuple[Tracer, dict[str, Artist]]:
    FigureCanvasAgg(figure)
    with named(figure) as names:
        tracer = Tracer(figure, names)
        figure.draw(tracer)
    return tracer, names


def furniture(figure: Figure) -> set[int]:
    """spines, ticks and the labels an axis sets beside them, close together by design."""
    found: set[int] = set()
    for ax in figure.axes:
        found |= {id(spine) for spine in ax.spines.values()}
        for axis in (ax.xaxis, ax.yaxis):
            found |= {id(artist) for artist in axis.findobj()}
    return found


def describe(text: Text) -> str:
    return "'" + " ".join(text.get_text().split())[:40] + "'"


def crowded(figure: Figure, gap_mm: float = GAP_MM) -> list[str]:
    """every text that comes closer than gap_mm, at print size, to a line, a shape or another
    text; axis furniture is only checked against everything else."""
    tracer, names = trace(figure)
    backgrounds = {id(figure.patch)} | {id(ax.patch) for ax in figure.axes}
    axis_parts = furniture(figure)
    gap = gap_mm / MM_PER_INCH * figure.dpi
    texts = tracer.traced.texts
    boxes = [text.get_window_extent(tracer) for text in texts]
    found = []
    for k, (text, box) in enumerate(zip(texts, boxes, strict=True)):
        grown = Bbox.from_extents(box.x0 - gap, box.y0 - gap, box.x1 + gap, box.y1 + gap)
        own = id(text) in axis_parts
        for stroke in tracer.traced.strokes:
            artist = names.get(stroke.owner or "")
            if artist is None or id(artist) in backgrounds:
                continue
            if own and id(artist) in axis_parts:
                continue
            reach = gap + stroke.half_width
            near = Bbox.from_extents(box.x0 - reach, box.y0 - reach, box.x1 + reach, box.y1 + reach)
            if stroke.path.intersects_bbox(near, filled=stroke.filled):
                kind = type(artist).__name__
                found.append(f"text {describe(text)} is within {gap_mm:g} mm of a {kind}")
                break
        for other, other_box in zip(texts[k + 1 :], boxes[k + 1 :], strict=True):
            if grown.overlaps(other_box):
                found.append(f"text {describe(text)} is within {gap_mm:g} mm of {describe(other)}")
    return found
