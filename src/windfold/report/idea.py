"""the idea on a real layout, after the first 30 days of one farm's history. (a) the turbine and
direction cells an identity model has rows for; (b) the same rows as relations to their upstream
turbines in rotor diameters, pooled over turbines and directions, the view the shared correction
learns in, with the relations needed by the directions where no waked turbine has rows of its own;
(c) for each direction, the share of waked turbines with rows of their own beside the share of
their upstream relations already seen in training. a pair seen at θ lands where the reversed pair
lands at θ + 180°, so seen directions inform their opposites."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from windfold import config
from windfold.geometry import relations
from windfold.report.sources import Sources
from windfold.report.style import Style


@dataclass(frozen=True)
class Support:
    identity: np.ndarray
    relational: np.ndarray
    along_edges: np.ndarray
    across_edges: np.ndarray
    centres_deg: np.ndarray
    identity_share: np.ndarray
    relational_share: np.ndarray
    turbines: Sequence[str]
    needed: np.ndarray


def window_mask(along: np.ndarray, across: np.ndarray, along_max: float, across_max: float):
    return (along > 0) & (along <= along_max) & (np.abs(across) <= across_max)


def support(
    rows: pl.DataFrame,
    east: np.ndarray,
    north: np.ndarray,
    diameter: float,
    along_max: float,
    across_max: float,
    direction_bin: float,
    relation_bin: float,
    turbines: Sequence[str],
) -> Support:
    """counts of training rows in the identity and relational views, coverage by direction and
    the relations needed by the directions none of whose waked turbines has rows of its own."""
    n = len(east)
    degrees = np.arange(360.0)
    along_all, across_all = relations(east, north, degrees, diameter)
    bins = int(round(360 / direction_bin))
    turbine = rows["turbine_index"].to_numpy().astype(np.int64)
    degree = np.mod(np.rint(rows["farm_direction"].to_numpy()), 360).astype(np.int64)
    mask = rows["operating_mask"].to_numpy().astype(np.int64)
    identity = np.zeros((n, bins))
    np.add.at(identity, (turbine, np.minimum(degree // int(direction_bin), bins - 1)), 1.0)
    along = along_all[degree, turbine, :]
    across = across_all[degree, turbine, :]
    operating = (mask[:, None] >> np.arange(n)[None, :]) & 1 == 1
    kept = window_mask(along, across, along_max, across_max) & operating
    along_edges = np.arange(0.0, along_max + relation_bin / 2, relation_bin)
    across_edges = np.arange(-across_max, across_max + relation_bin / 2, relation_bin)
    relational, _, _ = np.histogram2d(along[kept], across[kept], bins=[along_edges, across_edges])
    centres = (np.arange(bins) + 0.5) * direction_bin
    identity_share, relational_share = np.full(bins, np.nan), np.full(bins, np.nan)
    needed = [np.empty((0, 2))]
    for k, centre in enumerate(centres):
        a, c = relations(east, north, centre, diameter)
        inside = window_mask(a, c, along_max, across_max)
        waked = inside.any(axis=1)
        if not waked.any():
            continue
        p = np.clip(np.digitize(a[inside], along_edges) - 1, 0, len(along_edges) - 2)
        q = np.clip(np.digitize(c[inside], across_edges) - 1, 0, len(across_edges) - 2)
        seen = relational[p, q] > 0
        identity_share[k] = float(np.mean(identity[waked, k] > 0))
        relational_share[k] = float(np.mean(seen))
        if identity_share[k] == 0:
            needed.append(np.column_stack([a[inside], c[inside]]))
    return Support(
        identity,
        relational,
        along_edges,
        across_edges,
        centres,
        identity_share,
        relational_share,
        turbines,
        np.vstack(needed),
    )


def unseen(found: Support) -> list[tuple[float, float]]:
    """the spans of direction bins none of whose waked turbines has rows of its own."""
    half = (found.centres_deg[1] - found.centres_deg[0]) / 2
    empty = [c for c, s in zip(found.centres_deg, found.identity_share, strict=True) if s == 0]
    spans: list[tuple[float, float]] = []
    for centre in empty:
        if spans and np.isclose(spans[-1][1], centre - half):
            spans[-1] = (spans[-1][0], centre + half)
        else:
            spans.append((centre - half, centre + half))
    return spans


def beside(ax: Axes, spans: Sequence[tuple[float, float]], style: Style) -> None:
    """name the turbine-and-direction line beside its first gap, on the side with more room, or
    under the line's end when it has none."""
    name = "turbine and\ndirection"
    if not spans:
        style.note(ax, 360, 100, name, dx=-4, dy=-6, ha="right", va="top")
    elif spans[0][1] <= 180:
        style.note(ax, spans[0][1], 45, name, dx=4, ha="left", va="center")
    else:
        style.note(ax, spans[0][0], 45, name, dx=-4, ha="right", va="center")


def draw(found: Support, style: Style) -> Figure:
    lines = style.report.lines
    with style.context():
        figure = style.figure(170.0, 72.0)
        identity, shared, shares = figure.subplots(1, 3, width_ratios=[1.0, 1.0, 1.15])
        identity.imshow(
            (found.identity > 0).astype(float),
            aspect="auto",
            cmap=ListedColormap(["#ffffff00", style.role("identity")]),
            vmin=0,
            vmax=1,
            extent=(0, 360, len(found.turbines) - 0.5, -0.5),
            interpolation="nearest",
            zorder=1,
        )
        identity.set_xticks([0, 90, 180, 270, 360])
        identity.set_yticks([0, len(found.turbines) - 1], [found.turbines[0], found.turbines[-1]])
        identity.set_xlabel("wind direction (°)")
        identity.set_ylabel("turbine")
        style.panel_letter(identity, "a")
        shared.imshow(
            (found.relational.T > 0).astype(float),
            aspect="auto",
            cmap=ListedColormap(["white", style.role("grid")]),
            vmin=0,
            vmax=1,
            origin="lower",
            extent=(
                found.along_edges[0],
                found.along_edges[-1],
                found.across_edges[0],
                found.across_edges[-1],
            ),
            interpolation="nearest",
        )
        shared.scatter(
            found.needed[:, 0],
            found.needed[:, 1],
            s=(lines.marker_pt * 0.6) ** 2,
            color=style.role("windfold"),
            lw=0,
            zorder=2,
        )
        shared.set_xlabel("distance along the wind (D)")
        shared.set_ylabel("across the wind (D)")
        style.panel_letter(shared, "b")
        for share, role in (
            (found.identity_share, "identity"),
            (found.relational_share, "windfold"),
        ):
            shown = np.isfinite(share)
            shares.plot(
                found.centres_deg[shown],
                100 * share[shown],
                color=style.role(role),
                lw=lines.thick_pt,
                solid_capstyle="butt",
            )
        style.note(shares, 360, 100, "wind frame", dx=-4, dy=4, ha="right", va="bottom")
        beside(shares, unseen(found), style)
        shares.set_xlim(0, 360)
        shares.set_xticks([0, 90, 180, 270, 360])
        shares.set_ylim(-6, 140)
        shares.set_yticks([0, 50, 100])
        shares.set_xlabel("wind direction (°)")
        shares.set_ylabel("seen in training (%)")
        style.panel_letter(shares, "c")
        for ax in (identity, shared, shares):
            ax.tick_params(axis="x", pad=lines.tick_length_pt + 3.0)
        key(figure, style)
    return figure


def key(figure: Figure, style: Style) -> None:
    """what the cells and dots of panels a and b stand for."""
    size = style.report.lines.marker_pt * 0.6
    handles = [
        Patch(color=style.role("identity")),
        Patch(color=style.role("grid")),
        Line2D([], [], marker="o", ms=size, color=style.role("windfold"), mew=0, ls="none"),
    ]
    names = [
        "turbine and direction seen in training",
        "relation seen in training",
        "relation needed by unseen directions",
    ]
    figure.legend(
        handles,
        names,
        loc="outside upper center",
        ncols=2,
        borderaxespad=0.6,
        borderpad=0.2,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=2.0,
    )


def figure(sources: Sources, style: Style) -> Figure:
    idea = style.report.idea
    protocol = sources.protocol()
    farm = config.load_farm(idea.farm, sources.configs)
    turbine = config.load_turbine(farm.turbine_type, sources.configs)
    prepared = sources.data / "prepared" / idea.farm
    layout = pl.read_parquet(prepared / "layout.parquet")
    start = farm.histories.starts[0]
    rows = pl.read_parquet(
        prepared / "rows.parquet",
        columns=["timestamp", "turbine_index", "farm_direction", "operating_mask"],
    ).filter(
        pl.col("timestamp") >= start,
        pl.col("timestamp") < start + timedelta(days=idea.history_days),
    )
    found = support(
        rows,
        layout["x_m"].to_numpy(),
        layout["y_m"].to_numpy(),
        turbine.rotor_diameter_m,
        protocol.kernel.along_max_d,
        protocol.kernel.across_max_d,
        idea.direction_bin_deg,
        idea.relation_bin_d,
        layout["turbine"].to_list(),
    )
    return draw(found, style)
