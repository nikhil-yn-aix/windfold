"""monitoring on the confirmation period with 30 days of history, from the secondary test of a
threshold that sees day-scale error. false_alarms: the share of monitored turbine-days without
an injected drop on which each monitor alarms, at the standard and the recalibrated threshold,
inside or outside the band where the calibration check passes, half to twice the nominal rate;
real underperformance is not labeled, so the figure does not call these alarms false. recall:
the share of injected 5% drops each finds within 14 days at both thresholds. detection: the
advantage check at the recalibrated threshold, windfold's alarm-rate margin with its half-point
allowance and its recall gain over the two planned comparators, each with its 95% interval; the
check needs all four above zero."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from windfold.config import ModelSpec
from windfold.report.sources import Sources, point
from windfold.report.style import Style, wrap

TEST = "monitoring_threshold"
FARM = "penmanshiel"
HORIZON = 30
RULES = (("registered", "standard threshold"), ("ct", "recalibrated threshold"))
EXTRA = ("direction_lookup",)
COMPARATORS = ("cyclic_gam", "identity_gbm")
PARTS = (("exceedance_margin", "alarm-rate margin"), ("recall_gain", "recall gain"))
ROW_MM = 10.0
OFFSET = 0.16


def rates(table: pl.DataFrame, model: str, rule: str, metric: str, drop: float | None):
    """a monitor's rate under a rule at the 30-day history, in percent, with its interval."""
    rows = table.filter(
        pl.col("farm") == FARM,
        pl.col("model") == model,
        pl.col("candidate") == rule,
        pl.col("horizon") == HORIZON,
        pl.col("metric") == metric,
        pl.col("drop").is_null() if drop is None else pl.col("drop") == drop,
    )
    return point(rows, 100.0)


def pairs(
    ax: Axes,
    table: pl.DataFrame,
    monitors: Sequence[str],
    metric: str,
    drop: float | None,
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> None:
    """one row per monitor, the standard threshold open above the recalibrated one filled."""
    for row, model in enumerate(monitors):
        for index, (rule, _) in enumerate(RULES):
            value = rates(table, model, rule, metric, drop)
            if value is not None:
                at = row + OFFSET * (2 * index - 1)
                style.estimate(ax, at, *value, style.colour(specs[model]), rule == "ct")
    style.rows(ax, [wrap(style.label(specs[m])) for m in monitors], len(monitors))


def key(ax: Axes, style: Style, nominal: bool = False) -> None:
    """the two thresholds' marks, and the nominal rate's line and the calibration check's band
    where they are drawn, above the panel's right edge."""
    neutral, width = style.role("axes"), style.report.lines.interval_pt
    keys: list[Artist] = [
        Line2D([], [], marker="o", color=neutral, mfc=fill, mew=width, ls="none")
        for fill in ("white", neutral)
    ]
    names = [name for _, name in RULES]
    if nominal:
        keys.append(Line2D([], [], color=neutral, lw=style.report.lines.axes_pt, ls="dashed"))
        keys.append(
            Patch(fc=style.role("band"), ec=style.role("grid"), lw=style.report.lines.grid_pt)
        )
        names += ["nominal rate", "half to twice nominal"]
    ax.legend(
        keys,
        names,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=2,
        borderaxespad=0.6,
        borderpad=0.1,
        handlelength=1.4,
        handletextpad=0.3,
        columnspacing=1.2,
    )


def single(rows: int) -> tuple[float, float]:
    return 85.0, 26.0 + ROW_MM * rows


def false_alarms(
    table: pl.DataFrame, nominal: float, specs: Mapping[str, ModelSpec], style: Style
) -> Figure:
    monitors = [*style.report.monitoring.models, *EXTRA]
    with style.context():
        figure = style.figure(*single(len(monitors)))
        ax = figure.add_subplot()
        ax.axvspan(nominal / 2, 2 * nominal, color=style.role("band"), lw=0, zorder=0)
        ax.axvline(
            nominal, color=style.role("axes"), lw=style.report.lines.axes_pt, ls="dashed", zorder=1
        )
        pairs(ax, table, monitors, "exceedance", None, specs, style)
        ax.set_xlim(0, None)
        ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel("alarms (% of monitored turbine-days)")
        key(ax, style, nominal=True)
    return figure


def recall(table: pl.DataFrame, specs: Mapping[str, ModelSpec], style: Style) -> Figure:
    monitors = [*style.report.monitoring.models, *EXTRA]
    drop = style.report.monitoring.drop
    with style.context():
        figure = style.figure(*single(len(monitors)))
        ax = figure.add_subplot()
        pairs(ax, table, monitors, "recall", drop, specs, style)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(f"{drop:.0%} drops found within 14 days (%)")
        key(ax, style)
    return figure


def detection(components: pl.DataFrame, specs: Mapping[str, ModelSpec], style: Style) -> Figure:
    colour = style.role("windfold")
    with style.context():
        figure = style.figure(170.0, 26.0 + ROW_MM * len(COMPARATORS))
        axes = figure.subplots(1, len(PARTS))
        for column, (ax, (prefix, title)) in enumerate(zip(axes, PARTS, strict=True)):
            style.zero_line(ax)
            for row, comparator in enumerate(COMPARATORS):
                rows = components.filter(
                    pl.col("farm") == FARM,
                    pl.col("candidate") == "ct",
                    pl.col("horizon") == HORIZON,
                    pl.col("component") == f"{prefix}_{comparator}",
                )
                found = point(rows, 100.0)
                if found is not None:
                    style.estimate(ax, row, *found, colour)
            names = [f"over {style.label(specs[c])}" for c in COMPARATORS]
            style.rows(ax, names if column == 0 else None, len(COMPARATORS))
            ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
            style.panel_letter(ax, "ab"[column], title)
        figure.supxlabel("windfold's advantage (percentage points)")
    return figure


def folder_of(sources: Sources) -> Path:
    confirmatory = sources.confirmatory()
    if not confirmatory:
        raise ValueError("the monitoring figures read the confirmation period's secondary test")
    folder = sources.secondary(confirmatory[0], TEST) / FARM
    for name in ("rates.parquet", "components.parquet", "meta.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"{folder / name} is missing; run the {TEST} test first")
    return folder


def false_alarms_figure(sources: Sources, style: Style) -> Figure:
    folder = folder_of(sources)
    nominal = float(json.loads((folder / "meta.json").read_text(encoding="utf-8"))["nominal"])
    table = pl.read_parquet(folder / "rates.parquet")
    return false_alarms(table, 100 * nominal, sources.models(), style)


def recall_figure(sources: Sources, style: Style) -> Figure:
    table = pl.read_parquet(folder_of(sources) / "rates.parquet")
    return recall(table, sources.models(), style)


def detection_figure(sources: Sources, style: Style) -> Figure:
    components = pl.read_parquet(folder_of(sources) / "components.parquet")
    return detection(components, sources.models(), style)
