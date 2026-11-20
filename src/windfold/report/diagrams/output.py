"""where a diagram is written and how: its checks, then the pdf, the svg and the previews."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

from windfold.report.check import FigureRefused, problems, save
from windfold.report.diagrams import spacing
from windfold.report.paths import PREVIEW_PX
from windfold.report.style import Style


def files(out: Path, name: str) -> tuple[Path, Path, list[Path]]:
    """the pdf, the svg and the previews of one diagram."""
    previews = [out / "previews" / f"{name}_{px}px.png" for px in PREVIEW_PX]
    return out / f"{name}.pdf", out / f"{name}.svg", previews


def write(figure: Figure, name: str, out: Path, style: Style) -> list[Path]:
    found = problems(figure, style) + spacing.crowded(figure)
    if found:
        raise FigureRefused(f"{name} refused: " + "; ".join(found))
    pdf, svg, previews = files(out, name)
    save(figure, [pdf, *previews], style)
    with style.context(), mpl.rc_context({"svg.hashsalt": name}):
        figure.savefig(
            svg,
            format="svg",
            facecolor=style.rc["savefig.facecolor"],
            transparent=False,
            metadata={"Date": None},
        )
    return [pdf, svg, *previews]
