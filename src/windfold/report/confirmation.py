"""the confirmation period, Penmanshiel 2020 to 2023, in skill points: the tested model's
waked-row skill on held-out sectors minus each comparator's, with 95% intervals; positive favors
the tested model. the contrasts figure: windfold's planned confirmation contrasts, beside
Penmanshiel's development years. hybrid and restricted: the secondary tests of the hybrid GBM
and of windfold restricted, over every seed and arc, in the skill points their tables give,
no_wake's mse on the same rows dividing each contrast inside every bootstrap replicate. rows
under "not tested" were reported beside a test and decide nothing."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from windfold.report.sources import Sources, point
from windfold.report.style import Style

FARM = "penmanshiel"
UNTESTED = "not tested"
ROW_MM = 8.0


@dataclass(frozen=True)
class Block:
    """one test: its table, the component of each row, the axis it reads on and the columns of
    its estimate and 95% bounds with the scale that puts them in skill points."""

    table: str
    rows: tuple[tuple[str, str | None], ...]
    xlabel: str
    columns: tuple[str, str, str]
    scale: float


SKILL = ("skill", "skill_lower", "skill_upper")
CONTRASTS = Block(
    "repair",
    (
        ("first configuration", "windfold_minus_windfold_constant_inflow_evidence_scales"),
        ("windfold at constant inflow", "windfold_minus_windfold_constant_inflow"),
        ("PyWake at measured inflow", "windfold_minus_pywake_inflow"),
        ("diurnal-inflow control", "windfold_minus_windfold_diurnal_inflow"),
    ),
    "windfold minus model (skill points)",
    ("estimate", "lower", "upper"),
    100.0,
)
HYBRID = Block(
    "hybrid_gbm",
    (
        ("identity GBM", "identity_gbm_minus_hybrid_gbm"),
        ("wind-frame GBM", "geometry_gbm_minus_hybrid_gbm"),
        (UNTESTED, None),
        ("windfold", "windfold_minus_hybrid_gbm"),
        ("PyWake at measured inflow", "pywake_inflow_minus_hybrid_gbm"),
    ),
    "combined GBM minus model (skill points)",
    SKILL,
    1.0,
)
RESTRICTED = Block(
    "windfold_restricted",
    (
        ("windfold", "windfold_minus_windfold_restricted"),
        ("PyWake at measured inflow", "pywake_inflow_minus_windfold_restricted"),
        (UNTESTED, None),
        ("windfold without local cells", "windfold_no_local_minus_windfold_restricted"),
    ),
    "tighter variant minus model (skill points)",
    SKILL,
    1.0,
)
OFFSET = 0.17


def tables_of(sources: Sources, block: Block) -> dict[str, pl.DataFrame]:
    """the confirmation contrasts' planned results of every study, development beside
    confirmatory; a secondary test's outputs on the confirmation period alone."""
    if block is CONTRASTS:
        return {study: sources.results(study, block.table) for study in sources.studies}
    found = {}
    for study in sources.confirmatory():
        path = sources.secondary(study, block.table) / f"{FARM}.parquet"
        if path.is_file():
            found[study] = pl.read_parquet(path)
    if not found:
        raise FileNotFoundError(f"no {block.table} table for {FARM} in {sources.runs}")
    return found


def pooled(table: pl.DataFrame, block: Block, component: str) -> pl.DataFrame:
    """a component's estimate and bounds over every seed and arc, named as point reads them."""
    missing = [c for c in block.columns if c not in table.columns]
    if missing:
        raise ValueError(f"the {block.table} table has no {', '.join(missing)}; score it again")
    breakdown = [pl.col(c).is_null() for c in ("seed", "split") if c in table.columns]
    estimate, lower, upper = block.columns
    return table.filter(
        pl.col("farm") == FARM, pl.col("component") == component, *breakdown
    ).select(estimate=estimate, lower=lower, upper=upper)


def panel(
    ax: Axes,
    block: Block,
    tables: Mapping[str, pl.DataFrame],
    confirmatory: Sequence[str],
    style: Style,
) -> None:
    style.zero_line(ax)
    studies = [s for s in tables if s not in confirmatory] + [
        s for s in tables if s in confirmatory
    ]
    for row, (_, component) in enumerate(block.rows):
        if component is None:
            continue
        for study in studies:
            closed = study in confirmatory
            found = point(pooled(tables[study], block, component), block.scale)
            if found is None:
                continue
            at = row + (OFFSET if closed else -OFFSET) * (len(studies) > 1)
            colour = style.role("confirmatory" if closed else "development")
            style.estimate(ax, at, *found, colour, filled=closed)
    style.rows(ax, [name for name, _ in block.rows], len(block.rows))
    for row, (label, (_, component)) in enumerate(
        zip(ax.get_yticklabels(), block.rows, strict=True)
    ):
        if component is None:
            label.set_color(style.role("axes"))
            ax.axhline(row - 0.5, color=style.role("grid"), lw=style.report.lines.grid_pt, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel(block.xlabel.replace(" (", "\n("))


def draw(
    block: Block, tables: Mapping[str, pl.DataFrame], confirmatory: Sequence[str], style: Style
) -> Figure:
    with style.context():
        figure = style.figure(85.0, 24.0 + ROW_MM * len(block.rows))
        ax = figure.add_subplot()
        panel(ax, block, tables, confirmatory, style)
        periods = {s in confirmatory for s in tables}
        if len(periods) > 1:
            width = style.report.lines.interval_pt
            keys = [
                Line2D([], [], marker="o", color=style.role(role), mfc=fill, mew=width, ls="none")
                for role, fill in (
                    ("development", "white"),
                    ("confirmatory", style.role("confirmatory")),
                )
            ]
            ax.legend(
                keys,
                ["development window", "confirmation period"],
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                ncols=2,
                borderaxespad=0.6,
                borderpad=0.1,
                handletextpad=0.2,
                columnspacing=1.2,
            )
    return figure


def drawn(sources: Sources, style: Style, block: Block) -> Figure:
    return draw(block, tables_of(sources, block), sources.confirmatory(), style)


def contrasts_figure(sources: Sources, style: Style) -> Figure:
    return drawn(sources, style, CONTRASTS)


def hybrid_figure(sources: Sources, style: Style) -> Figure:
    return drawn(sources, style, HYBRID)


def restricted_figure(sources: Sources, style: Style) -> Figure:
    return drawn(sources, style, RESTRICTED)
