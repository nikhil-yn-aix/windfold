"""windfold in one figure. (a) the wind frame: one turbine's upstream neighbors, placed by their
distance along (a) and across (c) the wind inside the shared correction's window; (b) the sum of
windfold's terms and its prediction."""

from pathlib import Path

from matplotlib.figure import Figure

from windfold import config
from windfold.config import Processing
from windfold.report.diagrams import prediction, wind_frame
from windfold.report.style import Style

WIDTH_MM = 170.0
FRAME_MM = 74.0
TOP_MM = 8.0
GAP_MM = 9.0
LETTER_MM = 1.5


def draw(found: wind_frame.Frame, cells: Processing, style: Style) -> Figure:
    x0, y0, x1, y1 = wind_frame.bounds(found)
    span = (y1 - y0) * WIDTH_MM / FRAME_MM
    if span < x1 - x0:
        raise ValueError(f"the wind frame needs more than {FRAME_MM:g} mm of height")
    middle = (x0 + x1) / 2
    terms = prediction.HEIGHT_MM
    height = TOP_MM + FRAME_MM + GAP_MM + terms
    with style.context():
        figure = style.figure(WIDTH_MM, height)
        letter = {
            "fontsize": style.letter_pt,
            "fontweight": style.report.type.panel_letter_weight,
            "color": style.role("text"),
            "ha": "left",
            "va": "center",
        }
        figure.set_layout_engine("none")
        frame = figure.add_axes((0.0, (terms + GAP_MM) / height, 1.0, FRAME_MM / height))
        wind_frame.panel(frame, found, style, xlim=(middle - span / 2, middle + span / 2))
        sums = figure.add_axes((0.0, 0.0, 1.0, terms / height))
        prediction.panel(sums, cells, style)
        for name, middle_mm in (("a", height - TOP_MM / 2), ("b", terms + GAP_MM / 2)):
            figure.text(LETTER_MM / WIDTH_MM, middle_mm / height, name, **letter)
    return figure


def figure(data: Path, configs: Path | None, style: Style) -> Figure:
    cells = config.load_protocol(configs).processing
    return draw(wind_frame.frame_of(data, configs), cells, style)
