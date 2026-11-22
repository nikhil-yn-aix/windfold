from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from windfold.config import load_farm
from windfold.report import check, farms
from windfold.report.style import Style, load_style

MM = 1 / 25.4


@pytest.fixture(scope="module")
def style() -> Style:
    return load_style()


def clean(style: Style, width_mm: float = 85.0) -> Figure:
    figure = style.figure(width_mm, 50.0)
    ax = figure.add_subplot()
    x = np.linspace(0, 360, 73)
    ax.plot(x, np.cos(np.deg2rad(x)), color="#e42536")
    ax.set_xlabel("direction (°)")
    ax.set_ylabel("contrast")
    style.panel_letter(ax, "a")
    return figure


def overlapping_labels(style: Style) -> Figure:
    figure = clean(style)
    ax = figure.axes[0]
    ax.text(100, 0.2, "windfold")
    ax.text(110, 0.2, "physics")
    return figure


def small_label(style: Style) -> Figure:
    figure = clean(style)
    figure.axes[0].text(100, 0.2, "windfold", fontsize=6)
    return figure


def unscaled_label(style: Style) -> Figure:
    figure = clean(style, 170.0)
    figure.axes[0].text(100, 0.2, "windfold", fontsize=style.report.type.size_pt)
    return figure


def wide(style: Style) -> Figure:
    figure = clean(style)
    figure.set_size_inches(90 * MM, 50 * MM)
    return figure


def thin_line(style: Style) -> Figure:
    figure = clean(style)
    figure.axes[0].axhline(0, linewidth=0.3)
    return figure


def tall(style: Style) -> Figure:
    figure = clean(style)
    figure.set_size_inches(85 * MM, 230 * MM)
    return figure


def text_off_canvas(style: Style) -> Figure:
    figure = clean(style)
    figure.text(0.97, 0.5, "beyond the edge")
    return figure


def text_into_panel(style: Style) -> Figure:
    figure = style.figure(170.0, 50.0)
    left, right = figure.subplots(1, 2)
    for ax in (left, right):
        ax.plot([0, 1], [0, 1])
    text = left.text(1.05, 0.5, "a label that runs into the next panel", transform=left.transAxes)
    text.set_in_layout(False)
    return figure


def crowded_labels(style: Style) -> Figure:
    figure = clean(style)
    ax = figure.axes[0]
    first = ax.text(100, 0.2, "windfold", va="bottom")
    ax.annotate("physics", xy=(100, 0.2), xytext=(0, -1.5), textcoords="offset points", va="top")
    first.set_in_layout(False)
    return figure


def overlapping_two_line_label(style: Style) -> Figure:
    figure = clean(style)
    ax = figure.axes[0]
    ax.text(100, 0.2, "windfold")
    ax.text(110, 0.2, "PyWake at\nconstant inflow", va="top")
    return figure


def two_line_text_off_canvas(style: Style) -> Figure:
    figure = clean(style)
    figure.text(0.95, 0.5, "beyond\nthe edge")
    return figure


def label_at_the_panel_edge(style: Style) -> Figure:
    figure = clean(style)
    figure.axes[0].text(0.5, 0.995, "near the top", transform=figure.axes[0].transAxes, va="top")
    return figure


def coloured_label(style: Style) -> Figure:
    figure = clean(style)
    figure.axes[0].text(100, 0.6, "windfold", color=style.role("windfold"))
    return figure


def label_over_line(style: Style) -> Figure:
    figure = clean(style)
    figure.axes[0].text(170, -1.0, "minimum", va="center")
    return figure


def key_over_data(style: Style) -> Figure:
    figure = clean(style)
    figure.axes[0].legend(["cosine"], loc="lower center")
    return figure


def loose(style: Style) -> Figure:
    figure = clean(style)
    figure.get_layout_engine().set(w_pad=5 * MM, h_pad=5 * MM)  # type: ignore[union-attr]
    return figure


REFUSED: dict[str, tuple[Callable[[Style], Figure], str]] = {
    "two overlapping labels": (overlapping_labels, "overlaps text"),
    "a two-line label over another": (overlapping_two_line_label, "'PyWake at constant inflow'"),
    "a two-line text past the canvas edge": (
        two_line_text_off_canvas,
        "'beyond the edge' is within",
    ),
    "a 6 pt label": (small_label, "prints at 6 pt, under 8 pt"),
    "8 pt type on a full-width canvas": (unscaled_label, r"prints at 6\.15 pt, under 8 pt"),
    "a coloured label": (coloured_label, "is coloured #2552e6"),
    "a label over a line": (label_over_line, "'minimum' lies over data"),
    "a key over the data": (key_over_data, "a key lies over data"),
    "a 90 mm width": (wide, "width is 90.00 mm, not 85 or 170 mm"),
    "a 0.3 pt line": (thin_line, "line is 0.3 pt, under 0.5 pt"),
    "a 230 mm height": (tall, "height is 230.00 mm, over 225 mm"),
    "text past the canvas edge": (text_off_canvas, "within 1 mm of the canvas edge"),
    "two labels 0.5 mm apart": (crowded_labels, r"is 0\.5\d mm from text 'physics', under 1 mm"),
    "a label at a panel's edge": (label_at_the_panel_edge, "within 1 mm of a panel's edge"),
    "text into another panel": (text_into_panel, "crosses into another panel"),
    "a 5 mm margin": (loose, "margin is"),
}


@pytest.mark.parametrize("case", REFUSED)
def test_a_broken_figure_is_refused_and_not_written(case: str, style: Style, tmp_path: Path):
    build, reason = REFUSED[case]
    files = [tmp_path / "f.pdf", tmp_path / "f_600px.png", tmp_path / "f_1200px.png"]
    with style.context():
        figure = build(style)
        with pytest.raises(check.FigureRefused, match=reason):
            check.save(figure, files, style)
    assert not any(path.exists() for path in files)


@pytest.mark.parametrize("width_mm", [85.0, 170.0])
def test_a_clean_figure_is_written_with_embedded_truetype_and_exact_previews(
    width_mm: float, style: Style, tmp_path: Path
):
    files = [tmp_path / "f.pdf", tmp_path / "f_600px.png", tmp_path / "f_1200px.png"]
    with style.context():
        figure = clean(style, width_mm)
        assert check.problems(figure, style) == []
        check.save(figure, files, style, {"Subject": "results abc123"})
    pdf = files[0].read_bytes()
    assert b"/FontFile2" in pdf and b"Inter" in pdf
    assert b"/FontFile3" not in pdf and b"DejaVu" not in pdf
    assert [Image.open(path).size[0] for path in files[1:]] == [600, 1200]


def test_a_label_beside_its_mark_with_a_leader_line_passes(style: Style):
    with style.context():
        figure = clean(style)
        ax = figure.axes[0]
        ax.set_ylim(-1.5, 1.2)
        ax.plot([180, 228], [-1.0, -1.0], color=style.role("axes"), lw=0.5, gid="leader")
        ax.text(230, -1.0, "minimum", va="center", color=style.role("text"))
        assert check.problems(figure, style) == []


def test_a_line_drawn_exactly_at_the_floor_passes(style: Style):
    with style.context():
        figure = clean(style)
        figure.axes[0].axhline(0, linewidth=0.5)
        assert check.problems(figure, style) == []


def test_invisible_edges_and_hidden_ticks_are_not_lines_or_texts(style: Style):
    with style.context():
        figure = clean(style)
        ax = figure.axes[0]
        ax.scatter([10, 20], [0, 0.5], linewidths=0.1, edgecolors="none")
        ax.bar([50], [0.5], width=10, linewidth=0.2, edgecolor="none")
        ax.set_xlim(0, 360)
        assert check.problems(figure, style) == []


def test_radial_labels_of_side_by_side_polar_panels_stay_in_their_own_panel(style: Style):
    with style.context():
        figure = style.figure(85.0, 45.0)
        for column in range(2):
            ax = figure.add_subplot(1, 2, column + 1, projection="polar")
            ax.bar(np.linspace(0, 2 * np.pi, 8, endpoint=False), np.arange(1, 9), width=0.6)
            ax.set_yticks([4.0, 8.0], ["4%", ""])
            ax.set_xticks([0.0], ["N"])
    assert not [p for p in check.problems(figure, style) if "another panel" in p]


def test_a_colourbar_without_drawn_edges_has_no_hairline(style: Style):
    with style.context():
        figure = style.figure(85.0, 45.0)
        ax = figure.add_subplot()
        image = ax.imshow(np.arange(12.0).reshape(3, 4))
        bar = figure.colorbar(image, ax=ax)
        bar.outline.set_linewidth(style.report.lines.axes_pt)
        bar.ax.tick_params(width=style.report.lines.axes_pt)
    assert not [p for p in check.problems(figure, style) if " pt, under " in p]


def test_titles_are_checked_where_the_layout_settles_them_above_polar_ticks(style: Style):
    found = [load_farm(name) for name in style.report.farms.order]
    roses = {farm.name: np.linspace(1.0, 9.0, 36) for farm in found}
    figure = farms.sectors(found, roses, 30.0, style)
    assert check.problems(figure, style) == []
