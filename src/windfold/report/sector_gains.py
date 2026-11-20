"""where windfold gains on the confirmation period: for each held-out sector of Penmanshiel 2020 to
2023, windfold's waked-row skill minus that of PyWake at measured inflow, in skill points with its
95% interval, against the sector's share of the waked rows, from the practical-scale analysis.
each point is named by its sector's center direction; the sector arc_<start>_single spans start
to start plus the sector width."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from windfold.report.labels import Label, place, settle
from windfold.report.sources import Sources, bounded
from windfold.report.style import Style

ANALYSIS = "practical_scale"
FARM = "penmanshiel"
AGAINST = "against_pywake_inflow"
APART = 0.3


def start_of(split: str) -> float:
    """the sector's first direction, from a split named arc_<start>_single."""
    parts = split.split("_")
    if len(parts) != 3 or parts[0] != "arc" or parts[2] != "single":
        raise ValueError(f"{split} is not a single held-out sector")
    return float(parts[1])


def sectors(per_arc: Sequence[Mapping[str, Any]], width: float) -> list[dict[str, float]]:
    """each sector's center direction, share of the waked rows in percent and gain in skill
    points with its bounds."""
    total = sum(int(entry["waked_rows"]) for entry in per_arc)
    found = []
    for entry in per_arc:
        centre = (start_of(entry["split"]) + width / 2) % 360.0
        gain, low, high = bounded(entry[AGAINST]["skill_points"])
        share = 100.0 * int(entry["waked_rows"]) / total
        found.append({"centre": centre, "share": share, "gain": gain, "low": low, "high": high})
    return sorted(found, key=lambda s: s["centre"])


def positions(found: Sequence[Mapping[str, float]], apart: float = APART) -> list[float]:
    """where each sector is drawn along the share axis: at its share, except that two sectors
    closer than apart whose intervals overlap are set apart symmetrically, so each interval
    stays readable."""
    drawn = [s["share"] for s in found]
    for i, first in enumerate(found):
        for j in range(i + 1, len(found)):
            second = found[j]
            near = abs(first["share"] - second["share"]) < apart
            overlap = min(first["high"], second["high"]) > max(first["low"], second["low"])
            if near and overlap:
                middle = (first["share"] + second["share"]) / 2
                left, right = (i, j) if first["share"] <= second["share"] else (j, i)
                drawn[left], drawn[right] = middle - apart / 2, middle + apart / 2
    return drawn


def draw(found: Sequence[Mapping[str, float]], style: Style) -> Figure:
    colour, width = style.role("windfold"), style.report.lines.interval_pt
    with style.context():
        figure = style.figure(85.0, 72.0)
        ax = figure.add_subplot()
        style.zero_line(ax, vertical=False)
        for sector, share in zip(found, positions(found), strict=True):
            low, high = sector["low"], sector["high"]
            if np.isfinite([low, high]).all():
                ax.plot([share, share], [low, high], color=colour, lw=width, zorder=2)
            ax.plot(share, sector["gain"], marker="o", color=colour, mew=width, ls="none", zorder=3)
        shares = [s["share"] for s in found]
        bottom = min(0.0, *(s["low"] for s in found if np.isfinite(s["low"])))
        top = max(s["high"] for s in found if np.isfinite(s["high"]))
        ax.set_xlim(0.0, 1.15 * max(shares))
        ax.set_ylim(bottom - 0.2 * (top - bottom), top + 0.08 * (top - bottom))
        ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel("sector's share of waked rows (%)")
        ax.set_ylabel("windfold minus PyWake at\nmeasured inflow (skill points)")
        renderer = settle(figure)
        drawn = dict(zip((s["centre"] for s in found), positions(found), strict=True))
        labels = [
            Label(drawn[s["centre"]], s["gain"], f"{s['centre']:g}°", ("e", "w", "ne", "nw"))
            for s in sorted(found, key=lambda s: -s["gain"])
        ]
        place(ax, labels, style, renderer)
    return figure


def figure(sources: Sources, style: Style) -> Figure:
    confirmatory = sources.confirmatory()
    if not confirmatory:
        raise ValueError("the sector gains read the confirmation period's practical-scale analysis")
    found = sources.analysis(ANALYSIS, confirmatory[0])
    if FARM not in found:
        raise ValueError(f"the {ANALYSIS} analysis has no {FARM}")
    width = sources.protocol().splits.arc_width_deg
    return draw(sectors(found[FARM]["per_arc"], width), style)
