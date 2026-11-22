"""why the gradient-boosting pair of the wind-frame test failed where it did, per farm of the
development data: the wind-frame GBM's waked-row skill minus the identity GBM's, in skill points,
positive when the wind frame is better, under time-blocked validation and on held-out sectors,
from the official accuracy table, and on held-out sectors once each turbine's mean residual is
removed from both, from the exploratory turbine-levels analysis: the level a model without
turbine identity cannot transfer to a sector it has not seen. each difference of mse is divided by
the no-wake model's mse on the same rows, under the same validation."""

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from windfold.report.sources import Sources
from windfold.report.style import Style

ANALYSIS = "turbine_levels"
PAIR = ("identity_gbm", "geometry_gbm")
REFERENCE = "no_wake"
HELD_OUT = "arc_single"
MARKS = (
    ("time_blocked", "time-blocked", "s", False),
    (HELD_OUT, "held-out sector", "o", False),
    ("levels_removed", "held-out sector, turbine mean errors removed", "o", True),
)


def mse(accuracy: pl.DataFrame, farm: str, regime: str, model: str) -> float:
    rows = accuracy.filter(
        pl.col("farm") == farm,
        pl.col("model") == model,
        pl.col("regime") == regime,
        pl.col("rows") == "waked",
        pl.col("metric") == "mse",
        pl.col("horizon_days").is_null(),
    )
    return float(rows["estimate"][0]) if rows.height == 1 else np.nan


def points(difference: float, reference: float) -> float:
    """a difference of mse, identity minus wind frame, in skill points of the no-wake mse."""
    return 100.0 * difference / reference


def gains(accuracy: pl.DataFrame, farm: str, levels_removed: float) -> dict[str, float]:
    """the wind-frame GBM's skill minus the identity GBM's under each validation, and on held-out
    sectors with turbine levels removed, whose difference of mse levels_removed gives."""
    found = {}
    for regime in ("time_blocked", HELD_OUT):
        identity, wind_frame = (mse(accuracy, farm, regime, model) for model in PAIR)
        found[regime] = points(identity - wind_frame, mse(accuracy, farm, regime, REFERENCE))
    found["levels_removed"] = points(levels_removed, mse(accuracy, farm, HELD_OUT, REFERENCE))
    return found


def draw(gaps: Mapping[str, Mapping[str, float]], farms: Sequence[str], style: Style) -> Figure:
    neutral = style.role("axes")
    with style.context():
        figure = style.figure(85.0, 36.0 + 10.0 * len(farms))
        ax = figure.add_subplot()
        style.zero_line(ax)
        for row, farm in enumerate(farms):
            values = gaps[farm]
            ax.plot(
                [values[HELD_OUT], values["levels_removed"]],
                [row, row],
                color=style.role("grid"),
                lw=style.report.lines.thick_pt,
                zorder=1,
            )
            for key, _, marker, filled in MARKS:
                style.estimate(ax, row, values[key], np.nan, np.nan, neutral, filled, marker)
        style.rows(ax, [style.farm_name(farm) for farm in farms], len(farms))
        ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel("wind-frame GBM minus identity GBM\n(skill points)")
        width = style.report.lines.interval_pt
        keys = [
            Line2D(
                [], [], marker=m, color=neutral, mfc=neutral if f else "white", mew=width, ls="none"
            )
            for _, _, m, f in MARKS
        ]
        figure.legend(
            keys,
            [name for _, name, _, _ in MARKS],
            loc="outside upper center",
            ncols=1,
            borderaxespad=0.6,
            borderpad=0.1,
            handletextpad=0.4,
            labelspacing=0.5,
        )
    return figure


def figure(sources: Sources, style: Style) -> Figure:
    found = sources.analysis(ANALYSIS)
    accuracy = sources.results(sources.main, "accuracy")
    farms = [farm for farm in style.report.farms.order if farm in found]
    removed = "identity_minus_geometry_levels_removed"
    values = {farm: gains(accuracy, farm, float(found[farm][removed])) for farm in farms}
    return draw(values, farms, style)
