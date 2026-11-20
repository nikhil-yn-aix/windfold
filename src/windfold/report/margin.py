"""windfold's waked-row skill on held-out sectors minus other models', in skill points with 95%
intervals, per farm and on the confirmation period; positive means windfold is better. figure:
against PyWake at measured inflow and against each dataset's strongest learned model, the comparison
model of least mse that learns from data, named beside it, from the practical-scale analysis.
physics, learned and ablations, from the planned comparisons: against each calibrated physics of
report.toml's margin list, each learned model at measured inflow and peer AutoML, and windfold's own
variants."""

from collections.abc import Callable, Mapping, Sequence
from textwrap import fill
from typing import Any

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from windfold.config import ModelSpec
from windfold.report.sources import Sources, bounded, point
from windfold.report.style import Dataset, Style, wrap

PHYSICS = "pywake_inflow"
LEARNED = (
    "physics_guided_gbm_inflow",
    "wind_frame_graph_inflow",
    "geometry_gbm_inflow",
    "identity_gbm_inflow",
    "absolute_graph_inflow",
    "peer_automl",
)
VARIANTS = (
    "windfold_constant_inflow_evidence_scales",
    "windfold_constant_inflow",
    "windfold_no_kernel",
    "windfold_permuted_layout",
    "windfold_no_local",
    "windfold_default_physics",
    "windfold_unit_physics",
)
REGIME = "arc_single"
SCALE = "practical_scale"
OFFSET = 0.16
XLABEL = "windfold minus model (skill points)"
TITLE_WIDTH = 9
NAME_WIDTH = 16


def held_out(table: pl.DataFrame, farm: str) -> pl.DataFrame:
    return table.filter(
        pl.col("farm") == farm,
        pl.col("against") == "windfold",
        pl.col("regime") == REGIME,
        pl.col("horizon_days").is_null(),
    )


def windfold_minus(table: pl.DataFrame, model: str) -> tuple[float, float, float] | None:
    """windfold's held-out skill minus the model's, in points, with its interval."""
    return point(table.filter(pl.col("model") == model), 100.0)


def ticks(ax: Axes, count: int = 4) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(count, steps=[1, 2, 2.5, 5, 10]))


def headline(
    found: Mapping[str, Mapping[str, Any]],
    shown: Sequence[Dataset],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> Figure:
    """found holds each study's practical-scale result; each row's strongest learned model is
    named at the right."""
    looks = ((style.colour(specs[PHYSICS]), "o"), (style.role("axes"), "D"))
    with style.context():
        figure = style.figure(170.0, 30.0 + 10.0 * len(shown))
        ax = figure.add_subplot()
        style.zero_line(ax)
        for row, dataset in enumerate(shown):
            farm = found[dataset.study][dataset.farm]
            strongest = farm["strongest_learned"]
            for index, (model, (colour, marker)) in enumerate(
                zip((PHYSICS, strongest), looks, strict=True)
            ):
                gain = bounded(farm["against"][model]["skill_points"])
                style.estimate(ax, row + OFFSET * (2 * index - 1), *gain, colour, marker=marker)
            ax.annotate(
                wrap(style.label(specs[strongest]), 24),
                xy=(1.0, row + OFFSET),
                xycoords=("axes fraction", "data"),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=style.role("text"),
            )
        style.rows(ax, [d.label.replace(", ", ",\n") for d in shown], len(shown))
        ticks(ax, 5)
        ax.set_xlabel(XLABEL)
        width = style.report.lines.interval_pt
        keys = [
            Line2D([], [], marker=marker, color=colour, mew=width, ls="none")
            for colour, marker in looks
        ]
        ax.legend(
            keys,
            [style.label(specs[PHYSICS]), "strongest learned model"],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncols=2,
            borderaxespad=0.6,
            borderpad=0.1,
            handletextpad=0.2,
            columnspacing=1.6,
        )
    return figure


def grid(
    tables: Mapping[str, pl.DataFrame],
    shown: Sequence[Dataset],
    models: Sequence[str],
    names: Callable[[ModelSpec], str],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> Figure:
    """a row per model and a column per dataset, every column on one scale."""
    rows = len(models)
    with style.context():
        figure = style.figure(170.0, 40.0 + 15.0 * rows)
        axes = np.atleast_1d(figure.subplots(1, len(shown), sharex=True))
        for column, (ax, dataset) in enumerate(zip(axes, shown, strict=True)):
            table = held_out(tables[dataset.study], dataset.farm)
            style.zero_line(ax)
            for row, model in enumerate(models):
                found = windfold_minus(table, model)
                if found is not None:
                    style.estimate(ax, row, *found, style.colour(specs[model]))
            labels = [fill(names(specs[m]), NAME_WIDTH) for m in models]
            style.rows(ax, labels if column == 0 else None, rows)
            title = "\n".join(wrap(part, TITLE_WIDTH) for part in dataset.label.split(", "))
            ax.set_title(title, loc="center", fontweight="normal")
        ticks(axes[0], 4)
        figure.supxlabel(XLABEL)
    return figure


def variant_name(spec: ModelSpec, style: Style) -> str:
    """a variant's label without the windfold the figure is about."""
    name = style.label(spec)
    for prefix in ("windfold, ", "windfold with ", "windfold at "):
        name = name.removeprefix(prefix)
    return name


def inputs(sources: Sources, style: Style) -> tuple[dict[str, pl.DataFrame], list[Dataset]]:
    tables = {study: sources.results(study, "comparisons") for study in sources.studies}
    return tables, style.datasets(sources.farms("comparisons"), sources.confirmatory())


def figure(sources: Sources, style: Style) -> Figure:
    found = {study: sources.analysis(SCALE, study) for study in sources.studies}
    farms = {study: list(result) for study, result in found.items()}
    return headline(found, style.datasets(farms, sources.confirmatory()), sources.models(), style)


def physics_figure(sources: Sources, style: Style) -> Figure:
    tables, shown = inputs(sources, style)
    specs = sources.models()
    models = style.report.margin.models
    unknown = sorted(set(models) - set(specs))
    if unknown:
        raise ValueError(f"report.toml's margin names models that do not exist: {unknown}")
    return grid(tables, shown, models, style.label, specs, style)


def learned_figure(sources: Sources, style: Style) -> Figure:
    tables, shown = inputs(sources, style)
    return grid(tables, shown, LEARNED, style.label, sources.models(), style)


def ablations_figure(sources: Sources, style: Style) -> Figure:
    tables, shown = inputs(sources, style)
    return grid(
        tables, shown, VARIANTS, lambda spec: variant_name(spec, style), sources.models(), style
    )
