from datetime import UTC, datetime, timedelta

import matplotlib.dates as mdates
import numpy as np
import polars as pl
import pytest
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.projections.polar import PolarAxes
from matplotlib.text import Annotation
from typer.testing import CliRunner

from windfold.cli import app
from windfold.config import load_farm, load_models, load_turbine
from windfold.evaluation.schemas import CONTRASTS
from windfold.evaluation.summaries import ACCURACY, COMPARISONS
from windfold.report import (
    ceiling,
    confirmation,
    energy_ratio,
    farms,
    history,
    idea,
    kernel,
    margin,
    monitoring,
    optimism,
    practical,
    sector_gains,
    turbine_levels,
    validation,
)
from windfold.report.check import LEADER, problems
from windfold.report.sources import Sources, point
from windfold.report.style import Dataset, Style, load_style

NO_MARK = (None, "None", "none", "", " ")
KELMARSH = Dataset("development", "kelmarsh", "Kelmarsh")
CONFIRMATORY = Dataset("confirmatory", "penmanshiel", "Penmanshiel, confirmation period")


@pytest.fixture(scope="module")
def style() -> Style:
    return load_style()


@pytest.fixture(scope="module")
def specs():
    return load_models()


def xs(line) -> np.ndarray:
    return np.atleast_1d(np.asarray(line.get_xdata(), dtype=float))


def ys(line) -> np.ndarray:
    return np.atleast_1d(np.asarray(line.get_ydata(), dtype=float))


def r(value) -> float:
    return round(float(value), 9)


def marks(ax) -> list[tuple[float, float, str, bool]]:
    """every single-point mark as x, y, marker and whether it is filled."""
    found = []
    for line in ax.lines:
        x, y = xs(line), ys(line)
        if len(x) == 1 and line.get_marker() not in NO_MARK:
            filled = line.get_markerfacecolor() != "white"
            found.append((r(x[0]), r(y[0]), str(line.get_marker()), filled))
    return sorted(found)


def spans(ax) -> list[tuple[float, float, float]]:
    """every horizontal two-point line without a mark, as low, high and y."""
    found = []
    for line in ax.lines:
        x, y = xs(line), ys(line)
        if len(x) == 2 and line.get_marker() in NO_MARK and y[0] == y[1]:
            found.append((r(min(x)), r(max(x)), r(y[0])))
    return sorted(found)


def bars(ax) -> list[tuple[float, float, float]]:
    """every bar as its left edge, width and center height."""
    shapes = [p for p in ax.patches if isinstance(p, Rectangle)]
    return sorted(
        (r(p.get_x()), r(p.get_width()), r(p.get_y() + p.get_height() / 2)) for p in shapes
    )


def traced(ax) -> dict[str, tuple[list[float], list[float]]]:
    """every labeled line's x and y."""
    return {
        line.get_label(): ([r(v) for v in xs(line)], [r(v) for v in ys(line)])
        for line in ax.lines
        if not line.get_label().startswith("_")
    }


def legend(figure) -> list[str]:
    found = list(figure.legends) + [ax.get_legend() for ax in figure.axes if ax.get_legend()]
    return [t.get_text() for key in found for t in key.get_texts()]


def accuracy(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=pl.Schema(dict(ACCURACY) | {"farm": pl.String}), orient="row")


def comparisons(
    farm: str, values: dict[str, float], unbounded: tuple[str, ...] = ()
) -> pl.DataFrame:
    rows = [
        {
            "model": model,
            "against": "windfold",
            "regime": "arc_single",
            "horizon_days": None,
            "estimate": value,
            "lower": None if model in unbounded else value - 0.002,
            "upper": None if model in unbounded else value + 0.003,
            "p": 0.01,
            "farm": farm,
        }
        for model, value in values.items()
    ]
    return pl.DataFrame(rows, schema=pl.Schema(dict(COMPARISONS) | {"farm": pl.String}))


def contrasts(farm: str, values: dict[str, float], unbounded: tuple[str, ...] = ()) -> pl.DataFrame:
    rows = [
        (
            name,
            "estimated",
            value,
            None if name in unbounded else value - 0.0002,
            None if name in unbounded else value + 0.0003,
            0.001,
            10000,
        )
        for name, value in values.items()
    ]
    return pl.DataFrame(rows, schema=CONTRASTS, orient="row").with_columns(farm=pl.lit(farm))


def gain(estimate: float, low: float | None, high: float | None) -> dict:
    return {"skill_points": {"estimate": estimate, "low": low, "high": high, "p": 0.01}}


def test_the_margin_draws_both_comparisons_of_every_row_and_names_its_strongest_model(
    style: Style, specs
):
    found = {
        "development": {
            "kelmarsh": {
                "strongest_learned": "kernel_linear",
                "against": {
                    "pywake_inflow": gain(0.5, 0.2, 0.8),
                    "kernel_linear": gain(-1.3, -2.1, -0.5),
                },
            }
        },
        "confirmatory": {
            "penmanshiel": {
                "strongest_learned": "pywake_discrepancy_inflow",
                "against": {
                    "pywake_inflow": gain(-0.4, -0.6, -0.2),
                    "pywake_discrepancy_inflow": gain(3.8, None, None),
                },
            }
        },
    }
    figure = margin.headline(found, [KELMARSH, CONFIRMATORY], specs, style)
    assert problems(figure, style) == []
    ax, o = figure.axes[0], margin.OFFSET
    assert marks(ax) == sorted(
        [
            (0.5, -o, "o", True),
            (-1.3, o, "D", True),
            (-0.4, 1 - o, "o", True),
            (3.8, 1 + o, "D", True),
        ]
    )
    assert spans(ax) == sorted([(0.2, 0.8, -o), (-2.1, -0.5, o), (-0.6, -0.2, 1 - o)])
    notes = sorted((t for t in ax.texts if isinstance(t, Annotation)), key=lambda t: t.xy[1])
    named = [t.get_text() for t in notes]
    assert named == ["wind-frame linear", "PyWake with discrepancy\nat measured inflow"]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "Kelmarsh",
        "Penmanshiel,\nconfirmation period",
    ]


def test_a_margin_grid_draws_every_model_of_every_dataset_on_its_row(style: Style, specs):
    models = ("pywake", "pywake_inflow", "pywake_discrepancy_inflow")
    tables = {
        "development": comparisons(
            "kelmarsh", dict(zip(models, (0.02, -0.005, 0.03), strict=True)), ("pywake_inflow",)
        ),
        "confirmatory": comparisons(
            "penmanshiel", dict(zip(models, (-0.01, 0.004, 0.012), strict=True))
        ),
    }
    figure = margin.grid(tables, [KELMARSH, CONFIRMATORY], models, style.label, specs, style)
    assert problems(figure, style) == []
    first, second = figure.axes
    assert marks(first) == [(-0.5, 1.0, "o", True), (2.0, 0.0, "o", True), (3.0, 2.0, "o", True)]
    assert spans(first) == [(1.8, 2.3, 0.0), (2.8, 3.3, 2.0)]
    assert marks(second) == [(-1.0, 0.0, "o", True), (0.4, 1.0, "o", True), (1.2, 2.0, "o", True)]
    assert spans(second) == [(-1.2, -0.7, 0.0), (0.2, 0.7, 1.0), (1.0, 1.5, 2.0)]
    assert [t.get_text().replace("\n", " ") for t in first.get_yticklabels()] == [
        style.label(specs[m]) for m in models
    ]
    assert {t.get_text() for t in second.get_yticklabels()} == {""}
    assert [ax.get_title(loc="center") for ax in figure.axes] == [
        "Kelmarsh",
        "Penmanshiel\nconfirmation\nperiod",
    ]


def test_ablation_rows_name_each_variant_without_the_windfold_it_varies(style: Style, specs):
    variants = ("windfold_no_kernel", "windfold_constant_inflow_evidence_scales")
    tables = {
        "development": comparisons("kelmarsh", dict(zip(variants, (0.01, -0.02), strict=True)))
    }
    figure = margin.grid(
        tables, [KELMARSH], variants, lambda spec: margin.variant_name(spec, style), specs, style
    )
    assert problems(figure, style) == []
    ax = figure.axes[0]
    assert [t.get_text().replace("\n", " ") for t in ax.get_yticklabels()] == [
        "no shared correction",
        "first configuration",
    ]
    assert marks(ax) == [(-2.0, 1.0, "o", True), (1.0, 0.0, "o", True)]


def test_the_contrasts_draw_development_open_and_confirmation_filled_on_every_row(style: Style):
    names = [c for _, c in confirmation.CONTRASTS.rows if c is not None]
    tables = {
        "development": contrasts(
            "penmanshiel", dict(zip(names, (0.004, -0.002, 0.006, 0.001), strict=True))
        ),
        "confirmatory": contrasts(
            "penmanshiel", dict(zip(names, (0.005, 0.003, -0.001, 0.002), strict=True)), (names[2],)
        ),
    }
    figure = confirmation.draw(confirmation.CONTRASTS, tables, ["confirmatory"], style)
    assert problems(figure, style) == []
    ax, o = figure.axes[0], confirmation.OFFSET
    developed = [(0.4, -o), (-0.2, 1 - o), (0.6, 2 - o), (0.1, 3 - o)]
    closed = [(0.5, o), (0.3, 1 + o), (-0.1, 2 + o), (0.2, 3 + o)]
    assert marks(ax) == sorted(
        [(x, r(y), "o", False) for x, y in developed] + [(x, r(y), "o", True) for x, y in closed]
    )
    expected = [(x - 0.02, x + 0.03, r(y)) for x, y in developed + closed if (x, y) != closed[2]]
    assert spans(ax) == sorted((r(a), r(b), y) for a, b, y in expected)
    assert legend(figure) == ["development window", "confirmation period"]


def skill_table(values: dict[str, float]) -> pl.DataFrame:
    """a secondary test's contrasts in mse and skill points, pooled over every seed and arc, then
    a seed's and an arc's rows that the figure must leave out."""
    rows = []
    for seed, split, shift in ((None, None, 0.0), (0, None, 5.0), (None, "arc_000_single", 9.0)):
        for name, value in values.items():
            skill = value + shift
            rows.append((name, 0.001 * skill, skill, skill - 0.4, skill + 0.6, 0.11, seed, split))
    names = ["component", "estimate", "skill", "skill_lower", "skill_upper", "no_wake_mse"]
    table = pl.DataFrame(rows, schema=[*names, "seed", "split"], orient="row")
    return table.with_columns(farm=pl.lit("penmanshiel"), lower=0.0, upper=0.0)


@pytest.mark.parametrize(
    "block", [confirmation.HYBRID, confirmation.RESTRICTED], ids=["hybrid", "restricted"]
)
def test_a_secondary_test_draws_its_pooled_skill_points_and_leaves_untested_rows_empty(
    block: confirmation.Block, style: Style
):
    names = [c for _, c in block.rows if c is not None]
    values = dict(zip(names, (38.5, -9.7, 2.1, -8.4)[: len(names)], strict=True))
    tables = {"confirmatory": skill_table(values)}
    figure = confirmation.draw(block, tables, ["confirmatory"], style)
    assert problems(figure, style) == []
    ax = figure.axes[0]
    rows = [k for k, (_, c) in enumerate(block.rows) if c is not None]
    drawn = list(zip(values.values(), rows, strict=True))
    untested = [name for name, _ in block.rows].index(confirmation.UNTESTED)
    divider = (0.0, 1.0, untested - 0.5)
    assert marks(ax) == sorted((r(v), float(k), "o", True) for v, k in drawn)
    assert spans(ax) == sorted([divider, *((r(v - 0.4), r(v + 0.6), float(k)) for v, k in drawn)])
    assert ax.get_yticklabels()[untested].get_color() == style.role("axes")


def test_a_secondary_table_without_skill_points_is_refused_by_name(style: Style):
    names = [c for _, c in confirmation.HYBRID.rows if c is not None]
    table = contrasts("penmanshiel", dict.fromkeys(names, 0.004))
    with pytest.raises(ValueError, match="hybrid_gbm table has no skill, skill_lower, skill_upper"):
        confirmation.draw(confirmation.HYBRID, {"confirmatory": table}, ["confirmatory"], style)


def test_a_secondary_figure_names_the_table_it_cannot_find(tmp_path, style: Style):
    sources = Sources(tmp_path, tmp_path, None, ("development",))
    with pytest.raises(FileNotFoundError, match="no hybrid_gbm table"):
        confirmation.hybrid_figure(sources, style)


FILLERS = ("power_curve", "direction_lookup", "identity_linear", "kernel_linear", "geometry_gbm")


def ranked_errors(farm: str) -> pl.DataFrame:
    """four ranked models, five fillers ranked below them in every regime and a model without a
    held-out score, which is not ranked."""
    errors = {
        "peer_automl": (0.01, 0.01, 0.05),
        "windfold": (0.02, 0.02, 0.02),
        "pywake": (0.03, 0.03, 0.03),
        "no_wake": (0.10, 0.10, 0.10),
        "cyclic_gam": (0.001, 0.001, None),
    } | {name: (0.2 + k / 100,) * 3 for k, name in enumerate(FILLERS)}
    rows = [
        (model, regime, None, "waked", "mse", value, value, value, farm)
        for model, values in errors.items()
        for regime, value in zip(validation.RANKED, values, strict=True)
        if value is not None
    ]
    return accuracy(rows)


@pytest.mark.parametrize(
    "shown", [[KELMARSH, CONFIRMATORY], [CONFIRMATORY]], ids=["development", "confirmatory"]
)
def test_rank_lines_follow_each_complete_models_ranks_and_name_the_highlighted(
    shown: list[Dataset], style: Style, specs
):
    tables = {
        "development": ranked_errors("kelmarsh"),
        "confirmatory": ranked_errors("penmanshiel"),
    }
    baselines = ("no_wake", "pywake", "peer_automl", "cyclic_gam", *FILLERS)
    figure = validation.draw(tables, shown, specs, baselines, "pywake", "windfold", style)
    assert problems(figure, style) == []
    assert len(figure.axes) == len(shown)
    assert figure.get_size_inches()[0] * 25.4 == pytest.approx(85.0 if len(shown) == 1 else 170.0)
    expected = {
        (1.0, 1.0, 3.0): style.role("text"),
        (2.0, 2.0, 1.0): style.role("windfold"),
        (3.0, 3.0, 2.0): style.role("physics"),
        (4.0, 4.0, 4.0): style.role("context"),
    } | {(float(k),) * 3: style.role("context") for k in range(5, 5 + len(FILLERS))}
    for ax, dataset in zip(figure.axes, shown, strict=True):
        drawn = {
            tuple(r(v) for v in ys(line)): line.get_color()
            for line in ax.lines
            if len(ys(line)) == 3 and line.get_linestyle() not in NO_MARK
        }
        assert drawn == expected
        assert {t.get_text() for t in ax.texts} == {
            "peer AutoML",
            "windfold",
            "PyWake at\nconstant inflow",
        }
        titles = {ax.get_title(loc=side).replace("\n", ", ") for side in ("center", "left")}
        assert dataset.label in titles


def ceiling_points(scale: float, windfold: float) -> dict:
    """an analysis's points: PyWake's error of 53.3 split 40 + 0.8 + 9 + 3.5, the last learned
    on the other sectors, with decoys the figures must not read."""
    values = {
        "pywake_inflow_error": 53.3,
        "unexplained": 40.0,
        "turbine_by_direction": 0.8,
        "own_rows_shared_terms": 12.5,
        "sector_specific": 1.0,
        "transferable": 11.5,
    }
    found = {k: {"estimate": scale * v, "low": None, "high": None} for k, v in values.items()}
    return found | {
        "shared_terms_nested": {"estimate": 3.5 * scale, "low": 2.5 * scale, "high": 4.0 * scale},
        "windfold": {"estimate": windfold, "low": windfold - 0.2, "high": None},
    }


def test_the_ceiling_stacks_the_four_parts_of_every_row_in_order(style: Style):
    found = {
        "farms": {
            "smarteole": {"points": ceiling_points(1.0, 1.5)},
            "kelmarsh": {"points": ceiling_points(0.5, -0.5)},
        },
        "equal_farm": {"points": ceiling_points(0.75, 0.5)},
    }
    names, rows = ceiling.rows_of(found, style)
    assert names == ["Kelmarsh", "SMARTEOLE", ceiling.MEAN]
    figure = ceiling.parts(names, rows, style)
    assert problems(figure, style) == []
    expected = []
    for row, scale in enumerate((0.5, 1.0, 0.75)):
        left = 0.0
        for width in (40.0, 0.8, 9.0, 3.5):
            expected.append((r(left), r(scale * width), float(row)))
            left += scale * width
    assert bars(figure.axes[0]) == sorted(expected)
    assert legend(figure) == [name for _, name in ceiling.PARTS]


def test_the_realised_gain_sits_beside_the_part_learned_on_other_sectors(style: Style):
    found = {
        "farms": {"kelmarsh": {"points": ceiling_points(0.5, -0.5)}},
        "equal_farm": {"points": ceiling_points(1.0, 1.5)},
    }
    figure = ceiling.realised(*ceiling.rows_of(found, style), style)
    assert problems(figure, style) == []
    ax, o = figure.axes[0], ceiling.OFFSET
    assert marks(ax) == sorted(
        [
            (1.75, -o, "D", True),
            (-0.5, o, "o", True),
            (3.5, 1 - o, "D", True),
            (1.5, 1 + o, "o", True),
        ]
    )
    assert spans(ax) == [(1.25, 2.0, -o), (2.5, 4.0, r(1 - o))]
    assert legend(figure) == ["shared terms learned on other sectors", "windfold"]


def test_parts_that_do_not_add_up_to_the_error_are_refused():
    points = ceiling_points(1.0, 1.5)
    points["own_rows_shared_terms"] = {"estimate": 13.0, "low": None, "high": None}
    with pytest.raises(ValueError, match="add up to 53.8"):
        ceiling.split(points)


def test_the_transfer_splits_the_error_and_enlarges_the_part_learned_elsewhere(style: Style):
    points = ceiling_points(1.0, 1.56)
    points["windfold"] = {"estimate": 1.56, "low": 0.47, "high": 2.41}
    figure = ceiling.decomposition(points, style)
    assert problems(figure, style) == []
    whole, part = figure.axes
    widths = [(0.0, 40.0), (40.0, 0.8), (40.8, 9.0), (49.8, 3.5)]
    assert bars(whole) == [(r(x), r(w), 0.0) for x, w in widths]
    notes = [t for t in whole.texts if isinstance(t, Annotation)]
    inside = {t.get_text() for t in notes if t.xy[1] == 0.0}
    above = sorted((t.xy[1], t.get_text()) for t in notes if t.xy[1] > 0.0)
    assert inside == {"not explained on the sector's own rows"}
    assert [name for _, name in above] == [name for _, name in ceiling.PARTS[1:]]
    leaders = sorted(r(xs(line)[0]) for line in whole.lines if line.get_gid() == LEADER)
    assert leaders == [r(40.4), r(45.3), r(51.55)]
    (band,) = [p for p in part.patches if isinstance(p, Rectangle)]
    assert (band.get_x(), band.get_width()) == (0.0, 3.5)
    assert marks(part) == [(1.56, 1.0, "o", True), (3.5, 0.0, "D", True)]
    assert spans(part) == [(0.47, 2.41, 1.0), (2.5, 4.0, 0.0)]
    assert whole.get_xlabel() == "held-out error of PyWake at measured inflow (% of no-wake error)"
    assert part.get_xlabel() == "skill points over PyWake at measured inflow"
    words = " ".join(t.get_text() for ax in figure.axes for t in ax.texts).lower()
    for banned in ("specific", "transferable", "ceiling", "recur"):
        assert banned not in words


def monitor_rates() -> pl.DataFrame:
    values = {
        "windfold": 0.04,
        "cyclic_gam": 0.05,
        "identity_gbm": 0.06,
        "pywake": 0.07,
        "direction_lookup": 0.08,
    }
    rows = []
    for k, (model, value) in enumerate(values.items()):
        for rule, shift in (("registered", 0.1), ("ct", 0.0), ("c", 0.2)):
            rows.append(
                (
                    "penmanshiel",
                    rule,
                    model,
                    30,
                    "exceedance",
                    None,
                    4,
                    value + shift,
                    value + shift - 0.01,
                    value + shift + 0.02,
                )
            )
            rows.append(("penmanshiel", rule, model, 60, "exceedance", None, 4, 0.9, 0.8, 0.95))
            for drop, base in ((0.05, 0.5), (0.02, 0.9)):
                found = base + 0.01 * k + shift
                rows.append(
                    (
                        "penmanshiel",
                        rule,
                        model,
                        30,
                        "recall",
                        drop,
                        4,
                        found,
                        found - 0.05,
                        found + 0.04,
                    )
                )
    names = ["farm", "candidate", "model", "horizon", "metric", "drop", "starts"]
    return pl.DataFrame(rows, schema=[*names, "estimate", "lower", "upper"], orient="row")


def test_false_alarms_draw_each_monitor_standard_open_above_recalibrated_filled(
    style: Style, specs
):
    figure = monitoring.false_alarms(monitor_rates(), 3.0, specs, style)
    assert problems(figure, style) == []
    ax, o = figure.axes[0], monitoring.OFFSET
    expected = []
    for row, value in enumerate((4.0, 5.0, 6.0, 7.0, 8.0)):
        expected += [(r(value + 10), r(row - o), "o", False), (r(value), r(row + o), "o", True)]
    assert marks(ax) == sorted(expected)
    assert spans(ax) == sorted((r(x - 1), r(x + 2), y) for x, y, _, _ in expected)
    assert [(x, width) for x, width, _ in bars(ax)] == [(1.5, 4.5)]
    assert [r(xs(line)[0]) for line in ax.lines if line.get_linestyle() == "--"] == [3.0]
    assert ax.get_xlabel() == "alarms (% of monitored turbine-days)"
    assert legend(figure) == [
        "standard threshold",
        "recalibrated threshold",
        "nominal rate",
        "half to twice nominal",
    ]


def test_recall_draws_the_planned_drop_of_each_monitor_at_both_thresholds(style: Style, specs):
    figure = monitoring.recall(monitor_rates(), specs, style)
    assert problems(figure, style) == []
    ax, o = figure.axes[0], monitoring.OFFSET
    expected = []
    for row in range(5):
        found = 50.0 + row
        expected += [(r(found + 10), r(row - o), "o", False), (r(found), r(row + o), "o", True)]
    assert marks(ax) == sorted(expected)
    assert spans(ax) == sorted((r(x - 5), r(x + 4), y) for x, y, _, _ in expected)
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "windfold",
        "cyclic GAM",
        "identity GBM",
        "PyWake at\nconstant inflow",
        "direction lookup",
    ]


def test_detection_draws_each_component_where_it_lies_even_outside_its_interval(
    style: Style, specs
):
    values = {
        "exceedance_margin_cyclic_gam": (0.02, 0.013, 0.028),
        "exceedance_margin_identity_gbm": (-0.004, -0.01, 0.002),
        "recall_gain_cyclic_gam": (0.015, -0.1, 0.065),
        "recall_gain_identity_gbm": (0.0967, -0.05, 0.0883),
    }
    rows = [
        ("penmanshiel", "ct", 30, name, name.split("_", 2)[-1], *v, 0.1)
        for name, v in values.items()
    ]
    rows.append(
        (
            "penmanshiel",
            "registered",
            30,
            "recall_gain_identity_gbm",
            "identity_gbm",
            0.5,
            0.4,
            0.6,
            0.1,
        )
    )
    names = ["farm", "candidate", "horizon", "component", "registered_comparator"]
    table = pl.DataFrame(rows, schema=[*names, "estimate", "lower", "upper", "p"], orient="row")
    figure = monitoring.detection(table, specs, style)
    assert problems(figure, style) == []
    margins, recalls = figure.axes
    assert [margins.get_title(), recalls.get_title()] == ["alarm-rate margin", "recall gain"]
    assert marks(margins) == [(-0.4, 1.0, "o", True), (2.0, 0.0, "o", True)]
    assert spans(margins) == [(-1.0, 0.2, 1.0), (1.3, 2.8, 0.0)]
    assert marks(recalls) == [(1.5, 0.0, "o", True), (9.67, 1.0, "o", True)]
    assert spans(recalls) == [(-10.0, 6.5, 0.0), (-5.0, 8.83, 1.0)]


def test_turbine_levels_draw_the_wind_frame_gain_in_skill_points_on_each_farms_row(style: Style):
    errors = {
        ("time_blocked", "identity_gbm"): 0.046,
        ("time_blocked", "geometry_gbm"): 0.048,
        ("time_blocked", "no_wake"): 0.16,
        ("arc_single", "identity_gbm"): 0.23,
        ("arc_single", "geometry_gbm"): 0.094,
        ("arc_single", "no_wake"): 0.17,
        ("random", "no_wake"): 0.5,
    }
    rows = [
        (model, regime, None, "waked", "mse", value, None, None, "kelmarsh")
        for (regime, model), value in errors.items()
    ]
    rows.append(("no_wake", "arc_single", None, "all", "mse", 0.9, None, None, "kelmarsh"))
    gains = turbine_levels.gains(accuracy(rows), "kelmarsh", 0.143)
    assert gains == pytest.approx(
        {
            "time_blocked": 100 * (0.046 - 0.048) / 0.16,
            "arc_single": 100 * (0.23 - 0.094) / 0.17,
            "levels_removed": 100 * 0.143 / 0.17,
        }
    )
    found = {"kelmarsh": gains, "hill_of_towie": {k: -v / 4 for k, v in gains.items()}}
    figure = turbine_levels.draw(found, ["kelmarsh", "hill_of_towie"], style)
    assert problems(figure, style) == []
    ax = figure.axes[0]
    expected = []
    for row, farm in enumerate(("kelmarsh", "hill_of_towie")):
        values = found[farm]
        expected += [
            (r(values["time_blocked"]), float(row), "s", False),
            (r(values["arc_single"]), float(row), "o", False),
            (r(values["levels_removed"]), float(row), "o", True),
        ]
    assert marks(ax) == sorted(expected)
    assert ax.get_xlabel() == "wind-frame GBM minus identity GBM\n(skill points)"
    assert legend(figure) == [
        "time-blocked",
        "held-out sector",
        "held-out sector, turbine mean errors removed",
    ]


def test_practical_draws_each_models_error_level_on_its_own_line_of_each_row(style: Style, specs):
    def table(farm: str, levels: tuple[float, float, float]) -> pl.DataFrame:
        rows = [
            (model, "arc_single", None, "waked", "mae_kw", level, level - 2.0, level + 3.0, farm)
            for model, level in zip(practical.MODELS, levels, strict=True)
        ]
        rows.append(("windfold", "random", None, "waked", "mae_kw", 1.0, 0.0, 2.0, farm))
        return accuracy(rows)

    tables = {
        "development": table("kelmarsh", (180.0, 98.0, 97.0)),
        "confirmatory": table("penmanshiel", (120.0, 99.0, 95.0)),
    }
    figure = practical.draw(tables, [KELMARSH, CONFIRMATORY], specs, style)
    assert problems(figure, style) == []
    ax, s = figure.axes[0], practical.SPREAD
    expected = [
        (r(level), r(row + s * (k - 1)), "o", True)
        for row, levels in enumerate(((180.0, 98.0, 97.0), (120.0, 99.0, 95.0)))
        for k, level in enumerate(levels)
    ]
    assert marks(ax) == sorted(expected)
    assert spans(ax) == sorted((r(x - 2), r(x + 3), y) for x, y, _, _ in expected)
    assert legend(figure) == [style.label(specs[m]) for m in practical.MODELS]


def test_energy_ratios_follow_the_most_waked_turbine_and_break_at_thin_buckets(style: Style, specs):
    models = style.report.failure.models
    observed = {
        "A": ((0, 0.8, 40), (1, 0.7, 40), (2, 0.6, 40), (3, 0.1, 10), (5, 0.65, 40)),
        "B": ((0, 0.9, 40), (1, 0.6, 40), (2, 0.5, 40), (3, 0.4, 10), (5, 0.7, 40)),
    }
    rows = [
        (turbine, bucket, count, model, value, value - 0.05, value + 0.05, value + 0.1 * (k - 1))
        for turbine, buckets in observed.items()
        for bucket, value, count in buckets
        for k, model in enumerate(models)
    ]
    names = [
        "turbine",
        "direction_bucket",
        "rows",
        "model",
        "observed",
        "lower",
        "upper",
        "predicted",
    ]
    table = pl.DataFrame(rows, schema=names, orient="row")
    tables = dict.fromkeys(style.report.farms.order, table)
    figure = energy_ratio.draw(tables, models, specs, 30.0, style)
    assert problems(figure, style) == []
    ax = figure.axes[0]
    assert [a.get_title(loc="center") for a in figure.axes] == [
        f"{style.farm_name(farm)}, B" for farm in style.report.farms.order
    ]
    lines = [line for line in ax.lines if len(ys(line)) == 72]
    nan = float("nan")
    grid = np.full(72, nan)
    grid[[0, 1, 2, 5]] = [0.9, 0.6, 0.5, 0.7]
    np.testing.assert_allclose(ys(lines[0]), grid)
    for k, line in enumerate(lines[1:]):
        np.testing.assert_allclose(ys(line), grid + 0.1 * (k - 1))
    assert len(lines) == 1 + len(models)
    assert legend(figure) == ["SCADA", *(style.label(specs[m]) for m in models)]


def test_history_skill_draws_each_models_skill_at_each_history_length(style: Style, specs):
    models = ["windfold", "power_curve", "cyclic_gam"]
    skills = {"windfold": (0.6, 0.7), "power_curve": (0.0, 0.01), "cyclic_gam": (-0.3, 0.2)}
    rows = [
        (model, "history", days, "waked", "skill", farm_shift + value, None, None, farm)
        for farm, farm_shift in (("kelmarsh", 0.0), ("smarteole", -0.1))
        for model, values in skills.items()
        for days, value in zip((7, 30), values, strict=True)
    ]
    figure = history.skill_draw(["kelmarsh", "smarteole"], accuracy(rows), models, specs, style)
    assert problems(figure, style) == []
    for ax, shift in zip(figure.axes, (0.0, -0.1), strict=True):
        expected = {
            style.label(specs[m]): ([7.0, 30.0], [r(100 * (shift + v)) for v in skills[m]])
            for m in models
        }
        assert traced(ax) == expected
    assert legend(figure) == [style.label(specs[m]) for m in models]


def test_history_coverage_draws_each_view_at_each_history_length(style: Style):
    def support(shift: float) -> pl.DataFrame:
        rows = [
            (days, measure, share - shift)
            for measure, shares in zip(
                history.MEASURES, ((0.5, 0.9), (0.9, 1.0), (0.95, 1.0)), strict=True
            )
            for days, share in zip((7, 30), shares, strict=True)
        ]
        return pl.DataFrame(rows, schema=["horizon_days", "measure", "share"], orient="row")

    figure = history.coverage_draw({"kelmarsh": support(0.0), "smarteole": support(0.2)}, style)
    assert problems(figure, style) == []
    for ax, shift in zip(figure.axes, (0.0, 0.2), strict=True):
        assert traced(ax) == {
            history.NAMES["identity"]: (
                [7.0, 30.0],
                [r(100 * (0.5 - shift)), r(100 * (0.9 - shift))],
            ),
            history.NAMES["relational"]: (
                [7.0, 30.0],
                [r(100 * (0.9 - shift)), r(100 * (1.0 - shift))],
            ),
            history.NAMES["wake-weighted"]: (
                [7.0, 30.0],
                [r(100 * (0.95 - shift)), r(100 * (1.0 - shift))],
            ),
        }
    assert legend(figure) == [history.NAMES[m] for m in history.MEASURES]


def test_the_kernel_draws_its_three_profiles_up_to_the_widest_spacing(style: Style):
    along = kernel.ALONG_D
    found = kernel.Learned(
        physics=-0.5 * np.exp(-along / 10),
        whole=-0.5 * np.exp(-along / 10) + 0.05,
        lower=-0.5 * np.exp(-along / 10),
        upper=-0.5 * np.exp(-along / 10) + 0.1,
        kernel=0.05 * np.sin(along / 5),
        spacings=np.array([3.0, 6.0, 12.0]),
    )
    figure = kernel.draw({"kelmarsh": found, "smarteole": found}, style)
    assert problems(figure, style) == []
    shown = along <= 12.0
    for ax in figure.axes:
        curves = [line for line in ax.lines if len(xs(line)) == shown.sum()]
        assert [line.get_linestyle() for line in curves] == ["--", "-", ":"]
        for line, values in zip(curves, (found.physics, found.whole, found.kernel), strict=True):
            np.testing.assert_allclose(xs(line), along[shown])
            np.testing.assert_allclose(ys(line), 100 * np.expm1(values[shown]))
        rug = [line for line in ax.lines if line.get_marker() == "|"]
        assert len(rug) == 1 and list(xs(rug[0])) == [3.0, 6.0, 12.0]
    assert legend(figure) == ["PyWake × weight", "windfold", "shared correction"]


def test_the_idea_draws_seen_cells_relations_and_shares_by_direction(style: Style):
    centres = (np.arange(36) + 0.5) * 10.0
    identity = np.ones((3, 36))
    identity[:, 4:10] = 0.0
    shares = np.ones(36)
    shares[4:10] = 0.0
    shares[20] = np.nan
    relational = np.zeros((4, 2))
    relational[1, 0] = 3.0
    wind = np.ones(36)
    wind[5] = 0.5
    wind[20] = np.nan
    found = idea.Support(
        identity,
        relational,
        np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        np.array([-0.5, 0.0, 0.5]),
        centres,
        shares,
        wind,
        ["T1", "T2", "T3"],
        np.array([[0.7, -0.2], [1.2, 0.3]]),
    )
    figure = idea.draw(found, style)
    assert problems(figure, style) == []
    cells, frame, by_direction = figure.axes
    np.testing.assert_array_equal(cells.get_images()[0].get_array(), (identity > 0).astype(float))
    np.testing.assert_array_equal(
        frame.get_images()[0].get_array(), (relational.T > 0).astype(float)
    )
    np.testing.assert_allclose(np.asarray(frame.collections[0].get_offsets()), found.needed)
    assert bars(cells) == [] and bars(by_direction) == []
    assert by_direction.get_ylabel() == "seen in training (%)"
    assert legend(figure) == [
        "turbine and direction seen in training",
        "relation seen in training",
        "relation needed by unseen directions",
    ]
    assert {cells.get_title(loc="center"), frame.get_title(loc="center")} == {""}
    kept = np.isfinite(shares)
    own, seen = [line for line in by_direction.lines if len(xs(line)) == kept.sum()]
    np.testing.assert_allclose(xs(own), centres[kept])
    np.testing.assert_allclose(ys(own), 100 * shares[kept])
    np.testing.assert_allclose(ys(seen), 100 * wind[kept])
    assert (own.get_color(), seen.get_color()) == (style.role("identity"), style.role("windfold"))


def test_layouts_draw_every_turbine_north_up_in_rotor_diameters(style: Style):
    chosen = [load_farm("kelmarsh"), load_farm("smarteole")]
    turbines = {farm.name: load_turbine(farm.turbine_type) for farm in chosen}
    found = {
        "kelmarsh": pl.DataFrame({"x_m": [0.0, 920.0, 0.0], "y_m": [0.0, 0.0, 920.0]}),
        "smarteole": pl.DataFrame({"x_m": [0.0, 0.0], "y_m": [0.0, -820.0]}),
    }
    figure = farms.layouts(chosen, turbines, found, style)
    assert problems(figure, style) == []
    for ax, farm in zip(figure.axes, chosen, strict=True):
        diameter = turbines[farm.name].rotor_diameter_m
        x, y = (found[farm.name][c].to_numpy() for c in ("x_m", "y_m"))
        expected = np.column_stack([(x - x.mean()) / diameter, (y - y.mean()) / diameter])
        np.testing.assert_allclose(np.asarray(ax.collections[0].get_offsets()), expected)
        (bar,) = [line for line in ax.lines if len(xs(line)) == 2]
        length = float(np.ptp(xs(bar)))
        assert [t.get_text() for t in ax.texts] == [f"{length:g} D"]
        assert length == farms.nice_bar(float(np.ptp(ax.get_xlim())))


def test_a_rose_draws_each_bin_clockwise_from_north_with_the_sector_spokes(style: Style):
    chosen = [load_farm(name) for name in style.report.farms.order]
    north_east = np.zeros(36)
    north_east[[0, 9]] = (60.0, 40.0)
    west = np.zeros(36)
    west[27] = 100.0
    roses = [north_east, west, west, north_east]
    figure = farms.sectors(
        chosen, {f.name: s for f, s in zip(chosen, roses, strict=True)}, 30.0, style
    )
    assert problems(figure, style) == []
    for ax, share in zip(figure.axes, roses, strict=True):
        assert isinstance(ax, PolarAxes)
        assert ax.get_theta_direction() == -1 and ax.get_theta_offset() == pytest.approx(np.pi / 2)
        drawn = sorted(
            (p.get_x() + p.get_width() / 2, p.get_height())
            for p in ax.patches
            if isinstance(p, Rectangle)
        )
        np.testing.assert_allclose([h for _, h in drawn], share)
        np.testing.assert_allclose([c for c, _ in drawn], np.deg2rad((np.arange(36) + 0.5) * 10))
        spokes = sorted(r(xs(line)[0]) for line in ax.lines if len(xs(line)) == 2)
        assert spokes == [r(v) for v in np.deg2rad(np.arange(0, 360, 30))]
        assert ax.get_ylim()[1] == 100.0
    assert figure.get_supxlabel() == "share of time from each 10° direction (outer circle 100%)"


def test_the_periods_draw_every_farms_windows_and_exclusions(style: Style):
    chosen = [load_farm(name) for name in style.report.farms.order]
    figure = farms.periods(chosen, style)
    assert problems(figure, style) == []
    ax = figure.axes[0]
    shapes = [p for p in ax.patches if isinstance(p, Rectangle)]
    for row, farm in enumerate(chosen):
        for period in farms.PERIODS:
            window = getattr(farm.windows, period)
            if window is None:
                continue
            start, end = (mdates.date2num(t) for t in (window.start, window.end))
            hit = [
                p
                for p in shapes
                if p.get_y() + p.get_height() / 2 == row
                and p.get_x() == pytest.approx(start)
                and p.get_width() == pytest.approx(end - start)
                and p.get_facecolor()[:3] == pytest.approx(to_rgb(style.role(period)))
            ]
            assert len(hit) == 1, (farm.name, period)
        gaps = [
            p
            for p in shapes
            if p.get_y() + p.get_height() / 2 == row and p.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        ]
        assert len(gaps) == len(farm.exclusions)
    assert legend(figure) == [
        "calibration window",
        "development window",
        "confirmation period",
        "excluded",
    ]


def test_a_rose_counts_each_development_timestamp_once(tmp_path):
    folder = tmp_path / "prepared" / "kelmarsh"
    folder.mkdir(parents=True)
    stamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=10 * k) for k in range(4)]
    rows = {
        "timestamp": [stamps[0], stamps[0], stamps[1], stamps[2], stamps[3]],
        "farm_direction": [5.0, 5.0, 95.0, 185.0, 275.0],
        "period": ["development", "development", "development", "calibration", "confirmatory"],
    }
    pl.DataFrame(rows).write_parquet(folder / "rows.parquet")
    sources = Sources(tmp_path, tmp_path, None, ("development",))
    (rose,) = farms.roses_of(sources, [load_farm("kelmarsh")]).values()
    expected = np.zeros(36)
    expected[[0, 9]] = 50.0
    np.testing.assert_allclose(rose, expected)
    pl.DataFrame(rows).filter(pl.col("period") != "development").write_parquet(
        folder / "rows.parquet"
    )
    with pytest.raises(ValueError, match="kelmarsh has no development rows"):
        farms.roses_of(sources, [load_farm("kelmarsh")])


def test_a_rose_bins_directions_from_north_and_wraps_at_360():
    share = farms.rose(np.array([0.0, 9.9, 355.0, 360.0]))
    assert share.sum() == pytest.approx(100.0)
    assert share[0] == pytest.approx(75.0) and share[-1] == pytest.approx(25.0)


def test_a_result_whose_bootstrap_left_no_bounds_keeps_its_estimate(tmp_path):
    rows = pl.DataFrame({"estimate": [0.01], "lower": [None], "upper": [None]})
    found = point(rows.cast(pl.Float64), 100.0)
    assert found is not None
    centre, low, high = found
    assert centre == pytest.approx(1.0) and np.isnan(low) and np.isnan(high)
    assert point(rows.clear()) is None
    sources = Sources(tmp_path, tmp_path, None, ("development", "confirmatory"))
    with pytest.raises(FileNotFoundError, match="confirmatory/analyses/practical_scale.json"):
        sources.analysis("practical_scale", "confirmatory")


def test_the_figures_command_refuses_a_figure_the_runner_draws():
    result = CliRunner().invoke(app, ["figures", "validation"])
    assert result.exit_code != 0 and "no figure validation" in result.output


def skill_rows(farm: str, skills: dict[str, tuple[float | None, float]]) -> list[tuple]:
    rows = []
    for model, (blocked, held_out) in skills.items():
        for regime, value in (("time_blocked", blocked), ("arc_single", held_out)):
            if value is not None:
                rows.append((model, regime, None, "waked", "skill", value, None, None, farm))
        rows.append((model, "random", None, "waked", "skill", 0.99, None, None, farm))
        rows.append((model, "arc_single", None, "all", "skill", 0.98, None, None, farm))
    return rows


def test_optimism_places_each_compared_model_by_its_two_skills_and_floors_the_worst(
    style: Style, specs
):
    skills = {
        "windfold": (0.73, 0.70),
        "pywake_inflow": (0.69, 0.70),
        "peer_automl": (0.78, -0.07),
        "identity_gbm": (0.71, -0.36),
        "identity_linear": (0.64, -66.2),
        "kernel_linear": (0.68, 0.58),
        "pywake_discrepancy": (0.68, 0.64),
        "no_wake": (0.0, 0.0),
        "windfold_no_kernel": (0.70, 0.65),
        "pretrained_graph": (None, 0.51),
    }
    table = accuracy(skill_rows("kelmarsh", skills) + skill_rows("smarteole", skills))
    found = {farm: optimism.skills(table, farm) for farm in ("kelmarsh", "smarteole")}
    shown = set(skills) - {"windfold_no_kernel", "pretrained_graph"}
    assert set(found["kelmarsh"]) == shown
    assert found["kelmarsh"]["peer_automl"] == pytest.approx((78.0, -7.0))
    figure = optimism.draw(found, ["kelmarsh", "smarteole"], specs, style)
    assert problems(figure, style) == []
    markers = {
        "windfold": "o",
        "pywake_inflow": "s",
        "peer_automl": "P",
        "identity_gbm": "X",
        "identity_linear": "v",
        "kernel_linear": "^",
        "pywake_discrepancy": "D",
        "no_wake": "o",
    }
    expected = sorted(
        (r(100 * skills[m][0]), r(max(100 * skills[m][1], optimism.FLOOR)), mark, m != "no_wake")
        for m, mark in markers.items()
    )
    assert [ax.get_visible() for ax in figure.axes] == [True, True]
    for ax in figure.axes:
        assert marks(ax) == expected
        named = {t.get_text() for t in ax.texts}
        assert named == {"windfold", "peer AutoML", "identity GBM"}
        diagonal = [line for line in ax.lines if len(xs(line)) == 2 and line.get_gid() != LEADER]
        assert [list(xs(line)) for line in diagonal] == [list(ys(line)) for line in diagonal]
    first = figure.axes[0]
    assert first.get_yticklabels()[0].get_text().startswith("≤")
    assert first.get_ylim()[0] < optimism.FLOOR
    families = ["windfold", "calibrated physics", "hybrid models", "wind-frame models"]
    assert legend(figure) == [*families, "identity models", "peer AutoML", "no wake", optimism.SAME]


def per_arc(splits: dict[str, tuple[int, float, float, float]]) -> list[dict]:
    return [
        {
            "split": split,
            "waked_rows": rows,
            "against_pywake_inflow": {
                "skill_points": {"estimate": gain, "low": low, "high": high, "p": 0.01}
            },
        }
        for split, (rows, gain, low, high) in splits.items()
    ]


def test_sector_gains_draw_each_sector_at_its_share_named_by_its_centre(style: Style):
    found = sector_gains.sectors(
        per_arc(
            {
                "arc_330_single": (100, 10.0, 8.0, 12.0),
                "arc_000_single": (300, 2.0, 1.0, 3.0),
                "arc_240_single": (600, -1.0, -1.5, -0.5),
            }
        ),
        30.0,
    )
    assert [(s["centre"], s["share"]) for s in found] == [
        (15.0, 30.0),
        (255.0, 60.0),
        (345.0, 10.0),
    ]
    figure = sector_gains.draw(found, style)
    assert problems(figure, style) == []
    ax = figure.axes[0]
    assert marks(ax) == [(10.0, 10.0, "o", True), (30.0, 2.0, "o", True), (60.0, -1.0, "o", True)]
    upright = sorted(
        (r(xs(line)[0]), r(min(ys(line))), r(max(ys(line))))
        for line in ax.lines
        if len(xs(line)) == 2 and xs(line)[0] == xs(line)[1] and line.get_gid() != LEADER
    )
    assert upright == [(10.0, 8.0, 12.0), (30.0, 1.0, 3.0), (60.0, -1.5, -0.5)]
    assert sorted(t.get_text() for t in ax.texts) == ["15°", "255°", "345°"]
    with pytest.raises(ValueError, match="not a single held-out sector"):
        sector_gains.start_of("arc_030_opposite")


def test_sectors_with_near_shares_and_overlapping_intervals_are_set_apart():
    near = [
        {"share": 3.90, "low": 2.7, "high": 6.4},
        {"share": 3.98, "low": 4.3, "high": 8.1},
        {"share": 13.25, "low": -0.4, "high": 0.8},
        {"share": 13.39, "low": -1.4, "high": -0.6},
    ]
    drawn = sector_gains.positions(near)
    middle = (3.90 + 3.98) / 2
    apart = sector_gains.APART / 2
    assert drawn == pytest.approx([middle - apart, middle + apart, 13.25, 13.39])


def test_the_farms_figure_sets_each_layout_above_its_rose(style: Style):
    chosen = [load_farm(name) for name in style.report.farms.order]
    turbines = {farm.name: load_turbine(farm.turbine_type) for farm in chosen}
    found = {
        farm.name: pl.DataFrame({"x_m": [0.0, 900.0, 300.0], "y_m": [0.0, 200.0, 800.0]})
        for farm in chosen
    }
    roses = {farm.name: np.roll(np.linspace(0.0, 7.0, 36), k) for k, farm in enumerate(chosen)}
    figure = farms.composite(chosen, turbines, found, roses, 30.0, style)
    assert problems(figure, style) == []
    plans = [ax for ax in figure.axes if not isinstance(ax, PolarAxes)]
    circles = [ax for ax in figure.axes if isinstance(ax, PolarAxes)]
    assert len(plans) == len(circles) == len(chosen)
    for plan, rose, farm, letter in zip(plans, circles, chosen, "abcd", strict=True):
        above, below = plan.get_window_extent(), rose.get_window_extent()
        assert above.y0 > below.y1 and above.x0 < below.x1 and below.x0 < above.x1
        assert (plan.get_title(loc="left"), plan.get_title(loc="center")) == (
            letter,
            style.farm_name(farm.name),
        )
        heights = sorted(
            (p.get_x() + p.get_width() / 2, p.get_height())
            for p in rose.patches
            if isinstance(p, Rectangle)
        )
        np.testing.assert_allclose([h for _, h in heights], roses[farm.name])
        assert np.asarray(plan.collections[0].get_offsets()).shape == (3, 2)
        assert [t.get_text() for t in plan.texts][0].endswith(" D")


def ranked_skills(farm: str) -> pl.DataFrame:
    errors = {
        "peer_automl": (0.01, 0.01, 0.09),
        "windfold": (0.02, 0.02, 0.02),
        "pywake_inflow": (0.04, 0.04, 0.01),
        "pywake": (0.05, 0.05, 0.03),
        "no_wake": (0.10, 0.10, 0.10),
        "windfold_no_kernel": (0.015, 0.015, 0.015),
        "pretrained_graph": (None, None, 0.05),
    }
    rows = [
        (model, regime, None, "waked", "mse", value, value, value, farm)
        for model, values in errors.items()
        for regime, value in zip(validation.RANKED, values, strict=True)
        if value is not None
    ]
    return accuracy(rows)


def test_the_paper_ranks_rank_every_compared_model_and_name_three(style: Style, specs):
    figure = validation.paper({"development": ranked_skills("kelmarsh")}, [KELMARSH], specs, style)
    assert problems(figure, style) == []
    ax = figure.axes[0]
    drawn = {
        tuple(r(v) for v in ys(line)): line.get_color()
        for line in ax.lines
        if len(ys(line)) == 3 and line.get_linestyle() not in NO_MARK
    }
    assert drawn == {
        (1.0, 1.0, 4.0): style.role("text"),
        (2.0, 2.0, 2.0): style.role("windfold"),
        (3.0, 3.0, 1.0): style.role("physics"),
        (4.0, 4.0, 3.0): style.role("context"),
        (5.0, 5.0, 5.0): style.role("context"),
    }
    assert {t.get_text() for t in ax.texts} == {
        "windfold",
        "PyWake at\nmeasured inflow",
        "peer AutoML",
    }
    assert list(ax.get_yticks()) == [1, 5]
    assert ax.get_title(loc="center") == "Kelmarsh"
