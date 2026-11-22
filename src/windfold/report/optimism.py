"""how optimistic time-blocked validation is, model by model: for each development farm, every
compared model's skill under time-blocked validation and on held-out single sectors, on the waked
rows held-out validation tests and against the no-wake model's time-blocked error on those rows
(the same_row_skill analysis), with the line of equal error. families differ by marker and by a
colour-blind-safe hue; a held-out skill below the floor is drawn at the floor, whose tick reads
"≤ floor"."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from windfold.config import ModelSpec
from windfold.report.labels import Label, place, settle
from windfold.report.sources import Sources
from windfold.report.style import Style, wrap

ANALYSIS = "same_row_skill"
FLOOR = -60.0
X_LIMITS = (-12.0, 112.0)
Y_LIMITS = (-70.0, 88.0)
X_TICKS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
Y_TICKS = (FLOOR, -30.0, 0.0, 30.0, 60.0)
ROW_MM = 62.0
NAMED = (("identity_gbm", ("w", "nw", "sw")),)
SAME = "same error under both"
SAME_LOOK = {"ls": (0, (4.0, 2.5)), "zorder": 1}


@dataclass(frozen=True)
class Family:
    """how one family's models are drawn: its name in the key, its marker, the marker's size as a
    share of the report's, its own colour where the report's palette has none for it and whether
    it is drawn open."""

    name: str
    marker: str
    scale: float
    colour: str | None = None
    open: bool = False


FAMILIES = {
    "windfold": Family("windfold", "o", 1.35),
    "physics": Family("calibrated physics", "s", 1.1),
    "hybrid": Family("hybrid models", "D", 0.95, "#DDCC77"),
    "wind_relative": Family("wind-frame models", "^", 1.15, "#44AA99"),
    "identity": Family("identity models", "X", 1.15, "#882255"),
    "absolute": Family("absolute graph", "h", 1.2, "#AA4499"),
    "peer": Family("peer AutoML", "P", 1.2, "#88CCEE"),
    "reference": Family("no wake", "o", 0.9, open=True),
}
LAYERS = (
    "reference",
    "identity",
    "absolute",
    "peer",
    "wind_relative",
    "hybrid",
    "physics",
    "windfold",
)


def points(result: Mapping[str, Any], farm: str) -> dict[str, tuple[float, float]]:
    """each compared model's time-blocked and held-out skill on the farm, in percent."""
    return {
        model: (skill["time_blocked"], skill["held_out"])
        for model, skill in sorted(result[farm].items())
    }


def family_of(spec: ModelSpec) -> str:
    """the key's family of a model: its representation, with the absolute graph apart as in the
    paper's comparison table."""
    return "absolute" if spec.edges == "absolute" else spec.representation


def look(name: str, style: Style) -> dict[str, Any]:
    """the marker options of a family; vivid families get a white rim and the others a dark one,
    so overlapping marks stay apart."""
    family = FAMILIES[name]
    vivid = name in ("windfold", "physics")
    colour = family.colour or style.representation_colour(name)
    return {
        "marker": family.marker,
        "ms": family.scale * style.report.lines.marker_pt,
        "mfc": "white" if family.open else colour,
        "mec": "white" if vivid else style.role("axes"),
        "mew": style.report.lines.min_pt,
        "ls": "none",
    }


def panel(
    ax: Axes,
    found: Mapping[str, tuple[float, float]],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> None:
    ax.plot(
        X_LIMITS, X_LIMITS, **SAME_LOOK, color=style.role("axes"), lw=style.report.lines.axes_pt
    )
    order = sorted(found, key=lambda m: LAYERS.index(family_of(specs[m])))
    for layer, model in enumerate(order):
        x, y = found[model]
        ax.plot(x, max(y, FLOOR), zorder=2 + layer / 100, **look(family_of(specs[model]), style))
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xticks(X_TICKS)
    ticks = [f"{v:g}".replace("-", "\N{MINUS SIGN}") for v in Y_TICKS]
    ax.set_yticks(Y_TICKS, [f"≤\N{THIN SPACE}{ticks[0]}", *ticks[1:]])


def key(figure: Figure, shown: Sequence[str], style: Style) -> None:
    handles: list[Line2D] = []
    names = []
    for name in shown:
        handles.append(Line2D([], [], **look(name, style)))
        names.append(FAMILIES[name].name)
    handles.append(
        Line2D([], [], **SAME_LOOK, color=style.role("axes"), lw=style.report.lines.axes_pt)
    )
    names.append(SAME)
    figure.legend(
        handles,
        names,
        loc="outside upper center",
        ncols=3,
        borderaxespad=0.8,
        borderpad=0.2,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=2.2,
        labelspacing=0.7,
    )


def draw(
    found: Mapping[str, Mapping[str, tuple[float, float]]],
    farms: Sequence[str],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> Figure:
    """found holds each farm's skills by model, time-blocked then held-out, in percent."""
    rows = (len(farms) + 1) // 2
    with style.context():
        figure = style.figure(170.0, 26.0 + ROW_MM * rows)
        axes = figure.subplots(rows, 2, sharex=True, sharey=True, squeeze=False).ravel()
        for index, (ax, farm) in enumerate(zip(axes, farms, strict=False)):
            panel(ax, found[farm], specs, style)
            style.panel_letter(ax, "abcd"[index], style.farm_name(farm))
            ax.tick_params(labelbottom=index + 2 >= len(farms), labelleft=index % 2 == 0)
        for ax in axes[len(farms) :]:
            ax.set_visible(False)
        figure.supxlabel("skill under time-blocked validation (%)")
        figure.supylabel("skill on held-out sectors (%)")
        shown = {family_of(specs[m]) for points in found.values() for m in points}
        key(figure, [r for r in FAMILIES if r in shown], style)
        renderer = settle(figure)
        for ax, farm in zip(axes, farms, strict=False):
            labels = [
                Label(*found[farm][model], wrap(style.label(specs[model])), prefer)
                for model, prefer in NAMED
                if model in found[farm] and found[farm][model][1] >= FLOOR
            ]
            place(ax, labels, style, renderer)
    return figure


def figure(sources: Sources, style: Style) -> Figure:
    result = sources.analysis(ANALYSIS)
    farms = [f for f in style.report.farms.order if f in result]
    return draw({farm: points(result, farm) for farm in farms}, farms, sources.models(), style)
