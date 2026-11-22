"""how much history each view needs, per farm. coverage: how much of the common test window a
history of a given length has already seen, with no outcome used: the test rows' own turbine,
direction and speed cells, their upstream relations and those relations weighted by the default
physics wake energy. skill: the waked-row skill of windfold and the identity comparators on that
test window, in points, floored at -50."""

from collections.abc import Mapping, Sequence
from datetime import timedelta

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from windfold import config, coverage
from windfold.config import ModelSpec
from windfold.evaluation.reductions import coverage_physics, positions, turbine_order
from windfold.models.registry import modified
from windfold.report.sources import Sources
from windfold.report.style import Style

COLUMNS = [
    "turbine",
    "timestamp",
    "turbine_index",
    "farm_direction",
    "operating_mask",
    "speed_bin",
    "direction_bucket",
    "waked",
]
MEASURES = ("identity", "relational", "wake-weighted")
NAMES = {
    "identity": "turbine and direction",
    "relational": "wind frame",
    "wake-weighted": "wind frame, wake-weighted",
}
UNLABELLED_DAYS = (60, 182)
SKILL_FLOOR = -50.0


def relation_keys(rows: pl.DataFrame, east, north, diameter, along_max, across_max, step):
    row, source, along, across = coverage.pairs(rows, east, north, diameter, along_max, across_max)
    speed = rows["speed_bin"].to_numpy().astype(np.int64)[row]
    a = np.floor(along / step).astype(np.int64)
    c = np.floor((across + across_max) / step).astype(np.int64)
    return row, source, (a * 1000 + c) * 10 + speed


def identity_keys(rows: pl.DataFrame) -> np.ndarray:
    turbine = rows["turbine_index"].to_numpy().astype(np.int64)
    bucket = rows["direction_bucket"].to_numpy().astype(np.int64)
    speed = rows["speed_bin"].to_numpy().astype(np.int64)
    return (turbine * 1000 + bucket) * 10 + speed


def shares(
    train: pl.DataFrame,
    test: pl.DataFrame,
    east: np.ndarray,
    north: np.ndarray,
    diameter: float,
    along_max: float,
    across_max: float,
    step: float,
    rings: np.ndarray,
) -> dict[str, float]:
    """the share of waked test rows whose own cell the history saw, the mean share of their
    relations it saw and the same share weighted by k_phys squared."""
    window = (east, north, diameter, along_max, across_max, step)
    _, _, seen_relations = relation_keys(train, *window)
    row, source, needed = relation_keys(test, *window)
    seen = np.isin(needed, seen_relations)
    weight = coverage.kernel_weights(test, row, source, rings)
    pairs_per_row = np.bincount(row, minlength=test.height)
    has = pairs_per_row > 0
    relational = np.bincount(row, seen, minlength=test.height)[has] / pairs_per_row[has]
    total = np.bincount(row, weight, minlength=test.height)[has]
    weighted = np.bincount(row, weight * seen, minlength=test.height)[has]
    own = np.isin(identity_keys(test), identity_keys(train))[has]
    return {
        "identity": float(own.mean()),
        "relational": float(relational.mean()),
        "wake-weighted": float(np.sum(weighted) / np.sum(total)),
    }


def readiness(
    rows: pl.DataFrame,
    farm: config.FarmConfig,
    horizons: Sequence[int],
    east: np.ndarray,
    north: np.ndarray,
    diameter: float,
    protocol: config.Protocol,
    step: float,
    rings: np.ndarray,
) -> pl.DataFrame:
    """shares by history length, averaged over the farm's starts."""
    window = farm.histories.test
    test = rows.filter(
        pl.col("timestamp") >= window.start, pl.col("timestamp") < window.end, pl.col("waked")
    )
    records = []
    for days in horizons:
        found = []
        for start in farm.histories.starts:
            train = rows.filter(
                pl.col("timestamp") >= start, pl.col("timestamp") < start + timedelta(days=days)
            )
            if train.is_empty():
                continue
            found.append(
                shares(
                    train,
                    test,
                    east,
                    north,
                    diameter,
                    protocol.kernel.along_max_d,
                    protocol.kernel.across_max_d,
                    step,
                    rings,
                )
            )
        if found:
            records += [(days, m, float(np.mean([f[m] for f in found]))) for m in MEASURES]
    return pl.DataFrame(records, schema=["horizon_days", "measure", "share"], orient="row")


def history_axes(ax: Axes, days: Sequence[int]) -> None:
    labelled = [d for d in days if d not in UNLABELLED_DAYS or d == days[-1]]
    ax.set_xscale("log")
    ax.set_xticks(days, [str(d) if d in labelled else "" for d in days])
    ax.minorticks_off()


def coverage_draw(support: Mapping[str, pl.DataFrame], style: Style) -> Figure:
    farms = list(support)
    lines = style.report.lines
    looks = {
        "identity": (style.role("identity"), "solid", lines.thick_pt),
        "relational": (style.role("windfold"), "solid", lines.thick_pt),
        "wake-weighted": (style.role("windfold"), "dashed", lines.data_pt),
    }
    with style.context():
        figure = style.figure(170.0, 66.0)
        axes = figure.subplots(1, len(farms), sharey=True, squeeze=False)[0]
        for ax, farm in zip(axes, farms, strict=True):
            for measure, (colour, line, width) in looks.items():
                own = support[farm].filter(pl.col("measure") == measure).sort("horizon_days")
                ax.plot(
                    own["horizon_days"],
                    own["share"] * 100,
                    color=colour,
                    ls=line,
                    lw=width,
                    label=NAMES[measure],
                )
            history_axes(ax, support[farm]["horizon_days"].unique().sort().to_list())
            ax.set_title(style.farm_name(farm), loc="center", fontweight="normal")
        axes[0].set_ylim(0, 100)
        axes[0].set_yticks([0, 25, 50, 75, 100])
        axes[0].set_ylabel("test needs already seen (%)")
        figure.supxlabel("days of history")
        figure.legend(
            handles=axes[0].get_legend_handles_labels()[0],
            loc="outside upper center",
            ncols=len(looks),
            columnspacing=2.0,
        )
    return figure


def skill_draw(
    farms: Sequence[str],
    accuracy: pl.DataFrame,
    models: Sequence[str],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> Figure:
    lines = style.report.lines
    with style.context():
        figure = style.figure(170.0, 66.0)
        axes = figure.subplots(1, len(farms), sharey=True, squeeze=False)[0]
        for ax, farm in zip(axes, farms, strict=True):
            here = accuracy.filter(
                pl.col("farm") == farm,
                pl.col("regime") == "history",
                pl.col("metric") == "skill",
                pl.col("rows") == "waked",
            )
            for name in models:
                spec = specs[name]
                own = here.filter(pl.col("model") == name).sort("horizon_days")
                if own.is_empty():
                    continue
                width = lines.thick_pt if spec.representation == "windfold" else lines.data_pt
                ax.plot(
                    own["horizon_days"],
                    own["estimate"] * 100,
                    color=style.colour(spec),
                    ls=style.line(spec),
                    lw=width,
                    marker=style.marker(spec),
                    label=style.label(spec),
                )
            history_axes(ax, here["horizon_days"].drop_nulls().unique().sort().to_list())
            ax.set_title(style.farm_name(farm), loc="center", fontweight="normal")
        axes[0].set_ylim(SKILL_FLOOR, None)
        axes[0].yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
        axes[0].set_ylabel("waked-row skill (points)")
        figure.supxlabel("days of history")
        handles = {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                handles.setdefault(label, handle)
        figure.legend(
            handles=list(handles.values()),
            loc="outside upper center",
            ncols=3,
            columnspacing=2.0,
        )
    return figure


def coverage_figure(sources: Sources, style: Style) -> Figure:
    protocol = sources.protocol()
    support = {}
    for name in style.report.farms.order:
        farm = config.load_farm(name, sources.configs)
        turbine = config.load_turbine(farm.turbine_type, sources.configs)
        prepared = sources.data / "prepared" / name
        rows = pl.read_parquet(prepared / "rows.parquet", columns=COLUMNS)
        layout_path = prepared / "layout.parquet"
        east, north = positions(turbine_order(rows), pl.read_parquet(layout_path))
        rings = coverage.default_wake_rings(
            layout_path,
            modified(layout_path),
            turbine.model_dump_json(),
            coverage_physics(protocol),
        )
        horizons = farm.histories.horizons_days or protocol.splits.history_horizons_days
        support[name] = readiness(
            rows,
            farm,
            horizons,
            east,
            north,
            turbine.rotor_diameter_m,
            protocol,
            style.report.idea.relation_bin_d,
            rings,
        )
    return coverage_draw(support, style)


def skill_figure(sources: Sources, style: Style) -> Figure:
    endpoint = sources.protocol().endpoints.early_history_advantage
    models = [endpoint.model, *endpoint.accuracy_comparators]
    accuracy = sources.results(sources.main, "accuracy")
    held = set(accuracy["farm"].unique().to_list())
    farms = [farm for farm in style.report.farms.order if farm in held]
    return skill_draw(farms, accuracy, models, sources.models(), style)
