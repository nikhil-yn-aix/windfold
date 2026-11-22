"""rankings change when a sector is held out. validation and validation_confirmatory, for the
planned test on the development farms and on the confirmation period: each of the test's baselines'
and windfold's rank by waked-row mse under random and time-blocked validation and on held-out
sectors; the baseline time-blocked validation would pick, the test's physics and windfold are drawn
thick and named, every other model thin and light. ranks, the paper's figure: the same on the
development farms for every model scored under all three validations but windfold's variants, with
windfold, PyWake at measured inflow and peer AutoML named."""

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure

from windfold.config import ModelSpec
from windfold.report.labels import settle
from windfold.report.sources import Sources
from windfold.report.style import MM_PER_INCH, Dataset, Style, wrap

RANKED = ("random", "time_blocked", "arc_single")
NAMES = ("random", "time-\nblocked", "held-out\nsector")
VARIANTS = "windfold_"
NAMED = (("windfold", "windfold"), ("pywake_inflow", "physics"), ("peer_automl", "text"))


def ranks(accuracy: pl.DataFrame, farm: str, names: Sequence[str]) -> pl.DataFrame:
    """rank 1 is the lowest waked-row mse in each regime, over the models with all three."""
    here = accuracy.filter(
        pl.col("farm") == farm,
        pl.col("metric") == "mse",
        pl.col("rows") == "waked",
        pl.col("regime").is_in(RANKED),
        pl.col("horizon_days").is_null(),
        pl.col("model").is_in(list(names)),
    )
    complete = here.group_by("model").len().filter(pl.col("len") == len(RANKED))["model"]
    return (
        here.filter(pl.col("model").is_in(complete.to_list()))
        .with_columns(rank=pl.col("estimate").rank("ordinal").over("regime"))
        .select("model", "regime", "estimate", "rank")
    )


def spread(
    positions: Sequence[float], heights: Sequence[float], gap: float, low: float, high: float
) -> list[float]:
    """centers for labels of the given heights, as near positions as they can be while each stays
    gap from its neighbors and from low and high; all in display units, up positive."""
    order = np.argsort(positions)
    placed = np.asarray(positions, dtype=float)[order]
    half = np.asarray(heights, dtype=float)[order] / 2
    if len(placed):
        placed[0] = max(placed[0], low + half[0])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + half[i - 1] + gap + half[i])
    overflow = placed[-1] + half[-1] - high if len(placed) else 0.0
    if overflow > 0:
        placed[-1] -= overflow
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - half[i + 1] - gap - half[i])
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def panel(
    ax: Axes,
    table: pl.DataFrame,
    looks: Mapping[str, str],
    style: Style,
    last: bool = False,
) -> dict[str, float]:
    """each model's rank under the three validations, the highlighted ones in the colors of
    looks; returns the rank where each highlighted line ends. last also marks the last rank."""
    lines = style.report.lines
    x = np.arange(len(RANKED))
    count = table["model"].n_unique()
    position = {regime: i for i, regime in enumerate(RANKED)}
    ends: dict[str, float] = {}
    for model in sorted(table["model"].unique().to_list(), key=lambda m: m in looks):
        own = table.filter(pl.col("model") == model).sort(pl.col("regime").replace_strict(position))
        y = own["rank"].to_numpy()
        if model in looks:
            ax.plot(x, y, color=looks[model], lw=lines.thick_pt, zorder=3, solid_capstyle="round")
            ax.plot(x, y, ls="none", marker="o", color=looks[model], zorder=4)
            ends[model] = float(y[-1])
        else:
            ax.plot(
                x,
                y,
                color=style.role("context"),
                lw=lines.thin_pt,
                zorder=1,
                solid_capstyle="round",
            )
    ax.set_xticks(x, NAMES)
    ax.set_xlim(-0.2, len(RANKED) - 1 + 2.2)
    ax.set_ylim(count + (1.2 if last else 0.6), -0.5 if last else 0.0)
    steps = [v for v in range(5, count + 1, 5) if count - v >= 3]
    ax.set_yticks(sorted({1, *steps} | ({count} if last else set())))
    ax.spines["bottom"].set_bounds(0, len(RANKED) - 1)
    ax.tick_params(axis="x", length=0, pad=4.0)
    return {model: ends[model] for model in looks if model in ends}


def name_ends(
    ax: Axes,
    ends: Mapping[str, float],
    names: Mapping[str, str],
    style: Style,
    renderer: RendererBase,
) -> None:
    """each highlighted line's name beside its end, a long name on two lines, spread in display
    units so the names keep 1.3 times the check's clearance from each other and from the panel's
    edges; a name moved off its line's end gets a leader. call after settle()."""
    end_x = len(RANKED) - 1
    notes = {
        model: style.note(ax, end_x + 0.22, end, wrap(names[model]), va="center", ha="left")
        for model, end in ends.items()
    }
    for note in notes.values():
        note.set_in_layout(False)
    gap = 1.3 * style.report.page.clearance_mm / MM_PER_INCH * ax.figure.dpi
    area = ax.get_window_extent(renderer)
    centres = [float(ax.transData.transform((end_x, end))[1]) for end in ends.values()]
    heights = [note.get_window_extent(renderer).height for note in notes.values()]
    placed = spread(centres, heights, gap, area.y0 + gap, area.y1 - gap)
    inverse = ax.transData.inverted()
    for end, centre, note in zip(ends.values(), placed, notes.values(), strict=True):
        height = float(inverse.transform((0.0, centre))[1])
        note.xy = (end_x + 0.22, height)
        if abs(height - end) > 0.3:
            ax.plot(
                [end_x + 0.06, end_x + 0.19],
                [end, height],
                color=style.report.neutrals.axes,
                lw=style.report.lines.min_pt,
                gid="leader",
            )


def draw(
    tables: Mapping[str, pl.DataFrame],
    shown: Sequence[Dataset],
    specs: Mapping[str, ModelSpec],
    baselines: Sequence[str],
    physics: str,
    model: str,
    style: Style,
) -> Figure:
    """a panel per dataset, two to a row; rank 1 is the best."""
    colours = {model: style.role("windfold"), physics: style.role("physics")}
    single = len(shown) == 1
    with style.context():
        figure = style.figure(85.0 if single else 170.0, 72.0 if single else 140.0)
        axes = figure.subplots(1 if single else (len(shown) + 1) // 2, 1 if single else 2)
        flat = np.atleast_1d(axes).ravel()
        ends: list[dict[str, float]] = []
        for ax, dataset, letter in zip(flat, shown, "abcd", strict=False):
            table = ranks(tables[dataset.study], dataset.farm, [*baselines, model])
            time = table.filter(pl.col("regime") == "time_blocked", pl.col("model") != model)
            chosen = time.sort("estimate")["model"][0]
            looks = dict(colours)
            looks.setdefault(chosen, style.role("text"))
            ends.append(panel(ax, table, looks, style))
            if single:
                ax.set_title(dataset.label, loc="center", fontweight="normal")
            else:
                style.panel_letter(ax, letter, dataset.label.replace(", ", "\n"))
        figure.supylabel("rank by waked-row error")
        named(figure, flat, ends, specs, style)
    return figure


def compared(accuracy: pl.DataFrame) -> list[str]:
    """every model of the table but windfold's variants; ranks keeps those scored under all three
    validations."""
    return sorted(m for m in accuracy["model"].unique().to_list() if not m.startswith(VARIANTS))


def paper(
    tables: Mapping[str, pl.DataFrame],
    shown: Sequence[Dataset],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> Figure:
    """a panel per development farm, two to a row, ranking every compared model; rank 1 is the
    lowest waked-row error."""
    looks = {model: style.role(role) for model, role in NAMED}
    single = len(shown) == 1
    rows = (len(shown) + 1) // 2
    with style.context():
        figure = style.figure(85.0 if single else 170.0, 70.0 * rows)
        axes = np.atleast_1d(figure.subplots(rows, 1 if single else 2)).ravel()
        ends: list[dict[str, float]] = []
        for ax, dataset, letter in zip(axes, shown, "abcd", strict=False):
            accuracy = tables[dataset.study]
            table = ranks(accuracy, dataset.farm, compared(accuracy))
            ends.append(panel(ax, table, looks, style, last=True))
            if single:
                ax.set_title(dataset.label, loc="center", fontweight="normal")
            else:
                style.panel_letter(ax, letter, dataset.label.replace(", ", "\n"))
        for ax in axes[len(shown) :]:
            ax.set_visible(False)
        figure.supylabel("rank by waked-row error")
        named(figure, axes, ends, specs, style)
    return figure


def named(
    figure: Figure,
    axes: Sequence[Axes],
    ends: Sequence[Mapping[str, float]],
    specs: Mapping[str, ModelSpec],
    style: Style,
) -> None:
    """settle the layout, then name the highlighted lines of every panel."""
    renderer = settle(figure)
    for ax, own in zip(axes, ends, strict=False):
        name_ends(ax, own, {model: style.label(specs[model]) for model in own}, style, renderer)


def drawn(sources: Sources, style: Style, confirmatory: bool) -> Figure:
    protocol = sources.protocol()
    endpoint = protocol.endpoints.validation_misleads
    tables = {study: sources.results(study, "accuracy") for study in sources.studies}
    every = style.datasets(sources.farms("accuracy"), sources.confirmatory())
    shown = [d for d in every if (d.study in sources.confirmatory()) == confirmatory]
    model = protocol.endpoints.early_history_advantage.model
    return draw(tables, shown, sources.models(), endpoint.baselines, endpoint.physics, model, style)


def figure(sources: Sources, style: Style) -> Figure:
    return drawn(sources, style, confirmatory=False)


def confirmatory_figure(sources: Sources, style: Style) -> Figure:
    return drawn(sources, style, confirmatory=True)


def ranks_figure(sources: Sources, style: Style) -> Figure:
    tables = {study: sources.results(study, "accuracy") for study in sources.studies}
    every = style.datasets(sources.farms("accuracy"), sources.confirmatory())
    shown = [d for d in every if d.study == sources.main]
    return paper(tables, shown, sources.models(), style)
