"""what the skill differences mean in power: per farm and on the confirmation period, the mean
absolute error on held-out waked rows, in kW, of the no-wake model, PyWake at measured inflow and
windfold, each with its 95% interval. the intervals are of each level, not of the paired
differences."""

from collections.abc import Mapping, Sequence

import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from windfold.config import ModelSpec
from windfold.report.sources import Sources, point
from windfold.report.style import Dataset, Style

MODELS = ("no_wake", "pywake_inflow", "windfold")
SPREAD = 0.2


def draw(
    tables: Mapping[str, pl.DataFrame],
    shown: Sequence[Dataset],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> Figure:
    with style.context():
        figure = style.figure(85.0, 58.0)
        ax = figure.add_subplot()
        for row, dataset in enumerate(shown):
            table = tables[dataset.study].filter(
                pl.col("farm") == dataset.farm,
                pl.col("regime") == "arc_single",
                pl.col("horizon_days").is_null(),
                pl.col("rows") == "waked",
                pl.col("metric") == "mae_kw",
            )
            for k, model in enumerate(MODELS):
                found = point(table.filter(pl.col("model") == model))
                if found is not None:
                    at = row + SPREAD * (k - (len(MODELS) - 1) / 2)
                    style.estimate(ax, at, *found, style.colour(specs[model]))
        style.rows(ax, [d.label.replace(", ", ",\n") for d in shown], len(shown))
        ax.set_xlabel("mean absolute error (kW)")
        width = style.report.lines.interval_pt
        keys = [
            Line2D([], [], marker="o", color=style.colour(specs[m]), mew=width, ls="none")
            for m in MODELS
        ]
        figure.legend(
            keys,
            [style.label(specs[m]) for m in MODELS],
            loc="outside upper center",
            ncols=3,
            borderaxespad=0.3,
            borderpad=0.1,
            handletextpad=0.2,
            columnspacing=0.8,
        )
    return figure


def figure(sources: Sources, style: Style) -> Figure:
    tables = {study: sources.results(study, "accuracy") for study in sources.studies}
    shown = style.datasets(sources.farms("accuracy"), sources.confirmatory())
    return draw(tables, shown, sources.models(), style)
