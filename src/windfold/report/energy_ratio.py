"""what failure in an unseen sector looks like. the energy ratio of each farm's most waked
turbine against wind direction, from SCADA with its 95% interval and as each model of
report.toml's failure list predicts it; every 5 degree bucket comes from fits that never saw its
30 degree sector, the shaded bands mark alternate sectors and lines break where a bucket has too
few rows."""

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from windfold.config import ModelSpec
from windfold.report.sources import Sources
from windfold.report.style import Style

BUCKET_DEG = 5.0


def most_waked(table: pl.DataFrame, min_rows: int) -> str:
    """the turbine with the lowest observed energy ratio over buckets with enough rows."""
    observed = (
        table.filter(pl.col("rows") >= min_rows).group_by("turbine").agg(pl.col("observed").min())
    )
    return observed.sort("observed", "turbine")["turbine"][0]


def on_grid(table: pl.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """a column on every 5 degree bucket's center, nan where a bucket is missing, so that lines
    break there instead of bridging directions without rows."""
    count = int(round(360.0 / BUCKET_DEG))
    values = np.full(count, np.nan)
    values[table["direction_bucket"].to_numpy()] = table[column].to_numpy()
    return (np.arange(count) + 0.5) * BUCKET_DEG, values


def panel(
    ax: Axes,
    table: pl.DataFrame,
    turbine: str,
    models: Sequence[str],
    specs: Mapping[str, ModelSpec],
    arc_width: float,
    min_rows: int,
    style: Style,
) -> None:
    lines = style.report.lines
    here = table.filter(pl.col("turbine") == turbine, pl.col("rows") >= min_rows)
    for start in np.arange(arc_width, 360.0, 2 * arc_width):
        ax.axvspan(start, start + arc_width, color=style.role("band"), lw=0, zorder=0)
    observed = here.filter(pl.col("model") == models[-1]).sort("direction_bucket")
    centre, lower = on_grid(observed, "lower")
    ax.fill_between(
        centre, lower, on_grid(observed, "upper")[1], color=style.role("grid"), lw=0, zorder=1
    )
    ax.plot(*on_grid(observed, "observed"), color=style.role("text"), lw=lines.data_pt, zorder=2)
    for name in models:
        own = here.filter(pl.col("model") == name).sort("direction_bucket")
        spec = specs[name]
        width = lines.thick_pt if name == models[-1] else lines.data_pt
        ax.plot(
            *on_grid(own, "predicted"),
            color=style.colour(spec),
            ls=style.line(spec),
            lw=width,
            zorder=3,
        )
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.tick_params(axis="x", pad=style.report.lines.tick_length_pt + 3.0)


def draw(
    tables: Mapping[str, pl.DataFrame],
    models: Sequence[str],
    specs: Mapping[str, ModelSpec],
    arc_width: float,
    style: Style,
) -> Figure:
    min_rows = style.report.failure.min_rows
    with style.context():
        figure = style.figure(170.0, 112.0)
        axes = figure.subplots(2, 2, sharex=True, sharey=True)
        for ax, (farm, table), letter in zip(axes.flat, tables.items(), "abcd", strict=False):
            turbine = most_waked(table, min_rows)
            panel(ax, table, turbine, models, specs, arc_width, min_rows, style)
            style.panel_letter(ax, letter)
            ax.set_title(f"{style.farm_name(farm)}, {turbine}", loc="center", fontweight="normal")
        for ax in axes[:, 0]:
            ax.set_ylabel("energy ratio")
        for ax in axes[-1, :]:
            ax.set_xlabel("wind direction (°)")
        for ax in axes.flat[len(tables) :]:
            ax.set_visible(False)
        handles = [
            axes.flat[0].plot([], [], color=style.role("text"), label="SCADA")[0],
            *(
                axes.flat[0].plot(
                    [],
                    [],
                    color=style.colour(specs[name]),
                    ls=style.line(specs[name]),
                    lw=style.report.lines.thick_pt
                    if name == models[-1]
                    else style.report.lines.data_pt,
                    label=style.label(specs[name]),
                )[0]
                for name in models
            ),
        ]
        figure.legend(
            handles=handles, loc="outside upper center", ncols=len(handles), columnspacing=2.0
        )
    return figure


def figure(sources: Sources, style: Style) -> Figure:
    protocol = sources.protocol()
    tables = {}
    for farm in style.report.farms.order:
        path = sources.runs / sources.main / "results" / farm / "energy_ratio.parquet"
        if path.is_file():
            tables[farm] = pl.read_parquet(path)
    models = list(style.report.failure.models)
    return draw(tables, models, sources.models(), protocol.splits.arc_width_deg, style)
