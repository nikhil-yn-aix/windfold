"""how optimistic time-blocked validation is, model by model: for each development farm, every
compared model's waked-row skill under time-blocked validation against its skill on held-out
single sectors, from the official accuracy table, with the line where the two agree. the compared
models are every model but windfold's variants that has both scores. families are told apart by
marker as well as color, so the figure reads in gray; held-out skill below the floor is drawn at
the floor as a downward triangle, the floor's tick reading "≤ floor"."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from windfold.config import ModelSpec
from windfold.report.labels import Label, place, settle
from windfold.report.sources import Sources
from windfold.report.style import Style, wrap

REGIMES = ("time_blocked", "arc_single")
VARIANTS = "windfold_"
FLOOR = -40.0
X_LIMITS = (-12.0, 112.0)
Y_LIMITS = (-47.0, 88.0)
X_TICKS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
Y_TICKS = (FLOOR, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0)
BELOW = "v"
ROW_MM = 62.0
NAMED = (
    ("windfold", ("e", "se", "s")),
    ("identity_gbm", ("w", "nw", "sw")),
    ("peer_automl", ("e", "n", "w", "nw", "sw")),
)
SAME = "same skill under both"
SAME_LOOK = {"ls": (0, (4.0, 2.5)), "zorder": 1}


@dataclass(frozen=True)
class Family:
    """how one family's models are drawn: its name in the key, its marker, the marker's size as a
    share of the report's and whether it is drawn open."""

    name: str
    marker: str
    scale: float
    open: bool = False


FAMILIES = {
    "windfold": Family("windfold", "o", 1.35),
    "physics": Family("calibrated physics", "s", 1.1),
    "hybrid": Family("hybrid models", "D", 0.95),
    "wind_relative": Family("wind-frame models", "^", 1.15),
    "identity": Family("identity models", "X", 1.15),
    "absolute": Family("absolute graph", "h", 1.2),
    "peer": Family("peer AutoML", "P", 1.2),
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


def compared(models: Sequence[str]) -> list[str]:
    return [m for m in models if not m.startswith(VARIANTS)]


def skills(accuracy: pl.DataFrame, farm: str) -> dict[str, tuple[float, float]]:
    """each compared model's waked-row skill in percent, time-blocked then held-out, for the
    models with both."""
    rows = accuracy.filter(
        pl.col("farm") == farm,
        pl.col("metric") == "skill",
        pl.col("rows") == "waked",
        pl.col("horizon_days").is_null(),
        pl.col("regime").is_in(REGIMES),
        pl.col("model").is_in(compared(accuracy["model"].unique().to_list())),
        pl.col("estimate").is_not_null(),
    )
    found: dict[str, dict[str, float]] = {}
    for model, regime, estimate in rows.select("model", "regime", "estimate").iter_rows():
        found.setdefault(model, {})[regime] = 100.0 * estimate
    return {
        model: (values[REGIMES[0]], values[REGIMES[1]])
        for model, values in sorted(found.items())
        if set(values) == set(REGIMES)
    }


def family_of(spec: ModelSpec) -> str:
    """the key's family of a model: its representation, with the absolute graph apart as in the
    paper's comparison table, drawn in the identity models' colour."""
    return "absolute" if spec.edges == "absolute" else spec.representation


def look(name: str, style: Style) -> dict[str, Any]:
    """the marker options of a family; vivid families get a white rim and muted ones a dark one,
    so overlapping marks stay apart."""
    family = FAMILIES[name]
    vivid = name in ("windfold", "physics")
    colour = style.representation_colour("identity" if name == "absolute" else name)
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
    points: Mapping[str, tuple[float, float]],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> None:
    ax.plot(
        X_LIMITS, X_LIMITS, **SAME_LOOK, color=style.role("axes"), lw=style.report.lines.axes_pt
    )
    order = sorted(points, key=lambda m: LAYERS.index(family_of(specs[m])))
    for layer, model in enumerate(order):
        x, y = points[model]
        options = look(family_of(specs[model]), style)
        if y < FLOOR:
            options["marker"] = BELOW
        ax.plot(x, max(y, FLOOR), zorder=2 + layer / 100, **options)
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
    accuracy = sources.results(sources.main, "accuracy")
    farms = [f for f in style.report.farms.order if f in accuracy["farm"].unique().to_list()]
    return draw({farm: skills(accuracy, farm) for farm in farms}, farms, sources.models(), style)
