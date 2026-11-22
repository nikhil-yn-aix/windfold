import re
from datetime import UTC, date, datetime, timedelta
from xml.etree import ElementTree

import numpy as np
import polars as pl
import pytest
from matplotlib.patches import Rectangle
from matplotlib.path import Path as Outline

from windfold import splits as manifests
from windfold.config import load_protocol
from windfold.report.check import problems
from windfold.report.diagrams import (
    output,
    prediction,
    spacing,
    splits,
    wind_frame,
    windfold,
)
from windfold.report.style import Style, load_style

MM = 1 / 25.4
PENMANSHIEL = {
    "T01": (-902, -131),
    "T02": (-582, -408),
    "T04": (-671, 252),
    "T05": (-401, -43),
    "T06": (-125, -303),
    "T07": (132, -550),
    "T08": (-335, 472),
    "T09": (10, 146),
    "T10": (274, -72),
    "T11": (600, -314),
    "T12": (61, 560),
    "T13": (380, 373),
    "T14": (645, 153),
    "T15": (915, -135),
}


@pytest.fixture(scope="module")
def style() -> Style:
    return load_style()


def above_a_line(style: Style, gap_mm: float):
    """a label gap_mm above a line, on axes measured in millimeters."""
    figure = style.figure(85.0, 40.0)
    figure.set_layout_engine("none")
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set(xlim=(0.0, 85.0), ylim=(0.0, 40.0))
    ax.set_axis_off()
    ax.plot([10.0, 70.0], [20.0, 20.0], color="black", lw=0.5)
    ax.text(40.0, 20.0 + gap_mm, "label", ha="center", va="bottom")
    return figure


def beside_a_tick_label(style: Style, hidden: bool):
    """a label just above where the axis writes its tick label."""
    figure = style.figure(85.0, 40.0)
    figure.set_layout_engine("none")
    ax = figure.add_axes((0.5, 0.2, 0.4, 0.6))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xticks=[], yticks=[0.5], yticklabels=["tick"])
    ax.annotate("label", (0.0, 0.5), xytext=(-12.0, 5.0), textcoords="offset points")
    if hidden:
        ax.set_axis_off()
    return figure


@pytest.mark.parametrize(
    ("gap_mm", "expected"), [(0.3, ["text 'label' is within 1 mm of a Line2D"]), (1.5, [])]
)
def test_crowded_finds_text_close_to_a_line(style, gap_mm, expected):
    assert spacing.crowded(above_a_line(style, gap_mm)) == expected


def test_crowded_finds_a_two_line_text_close_to_a_line(style):
    figure = above_a_line(style, 1.5)
    figure.axes[0].text(40.0, 20.3, "two\nlines", ha="center", va="bottom")
    found = spacing.crowded(figure)
    assert "text 'two lines' is within 1 mm of a Line2D" in found


def test_crowded_ignores_an_axis_that_is_not_drawn(style):
    found = spacing.crowded(beside_a_tick_label(style, hidden=False))
    assert any("'label'" in line and "'tick'" in line for line in found)
    assert spacing.crowded(beside_a_tick_label(style, hidden=True)) == []


def test_wind_frame_draws_the_window_the_kernel_uses():
    east, north = np.meshgrid(np.arange(4) * 500.0, np.arange(3) * 400.0)
    found = wind_frame.frame(
        [f"T{k}" for k in range(12)], east.ravel(), north.ravel(), 100.0, 11, 250.0, 30.0, 3.0
    )
    window = Outline(found.window())
    others = [k for k in range(12) if k != found.receiver]
    drawn = [window.contains_point((found.east[k], found.north[k])) for k in others]
    assert drawn == [bool(found.inside[k]) for k in others]
    assert 0 < found.inside.sum() < len(others)
    for source in map(int, np.flatnonzero(found.inside)):
        leg = found.at(source) - found.foot(source)
        assert np.linalg.norm(found.foot(source) - found.at(found.receiver)) == pytest.approx(
            found.along[source]
        )
        assert leg @ found.left == pytest.approx(-found.across[source])


def test_sector_discs_hold_out_what_the_split_manifest_holds_out():
    rules = load_protocol().splits
    stamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=10 * k) for k in range(720)]
    table = pl.DataFrame(
        {
            "timestamp": stamps,
            "direction": np.arange(720) * 0.5,
            "rows": [1] * 720,
            "waked_rows": [1] * 720,
            "speed_bin": [0] * 720,
            "day": [date(2020, 1, 1) + timedelta(days=k % 5) for k in range(720)],
        }
    )
    manifest, _ = manifests.arcs(
        table,
        count=rules.arc_count,
        width=rules.arc_width_deg,
        buffer=rules.arc_buffer_deg,
        min_rows=1,
        min_days=1,
        tolerance=rules.control_tolerance,
        seed=3,
    )
    names = {"train": "training", "test": "test", "buffer": "unused"}
    for variant, opposite in (("single", False), ("opposite", True)):
        start = splits.ARC_START_DEG
        chosen = manifest.filter(pl.col("split") == f"arc_{start:03.0f}_{variant}")
        role_of = dict(zip(chosen["timestamp"], chosen["role"], strict=True))
        held = splits.arc_roles(rules, start, opposite)
        for stamp, direction in zip(stamps, table["direction"], strict=True):
            expected = names[role_of[stamp]] if stamp in role_of else "unused"
            assert splits.role_at(held, direction) == expected, (variant, direction)


def test_diagrams_from_the_configs_pass_the_figure_and_spacing_checks(style):
    for figure in (prediction.figure(style), splits.figure(style)):
        assert problems(figure, style) == []
        assert spacing.crowded(figure) == []


def test_the_svg_starts_with_a_solid_white_canvas(style, tmp_path):
    output.write(prediction.figure(style), "prediction", tmp_path, style)
    root = ElementTree.parse(tmp_path / "prediction.svg").getroot()
    *_, width, height = (float(v) for v in root.attrib["viewBox"].split())
    canvas = next(root.iter("{http://www.w3.org/2000/svg}path"))
    corners = np.array([float(v) for v in re.findall(r"-?[\d.]+", canvas.attrib["d"])])
    assert canvas.attrib["style"] == "fill: #ffffff"
    assert (corners[0::2].min(), corners[0::2].max()) == (0.0, pytest.approx(width))
    assert (corners[1::2].min(), corners[1::2].max()) == (0.0, pytest.approx(height))


def test_the_prediction_sums_offsets_physics_correction_and_cells(style):
    figure = prediction.figure(style)
    (ax,) = figure.axes
    texts = [t.get_text() for t in ax.texts]
    assert list(zip(texts[::2], texts[1::2], strict=True)) == list(prediction.PARTS)
    assert texts[:4] == [
        "turbine offsets",
        "per turbine",
        "calibrated physics",
        "measured inflow",
    ]
    left = prediction.CENTRES_MM[0]
    levels = sorted(
        (p.get_x(), p.get_height() * (1 if p.get_y() >= prediction.GLYPH_Y_MM else -1))
        for p in ax.patches
        if isinstance(p, Rectangle) and abs(p.get_x() - left) < 10.0
    )
    assert [h for _, h in levels] == pytest.approx(list(prediction.OFFSETS_MM))


def test_the_splits_show_the_three_validations_in_the_papers_words(style):
    figure = splits.figure(style)
    texts = [t.get_text() for t in figure.axes[0].texts]
    rules = load_protocol().splits
    for name in ("random", "time-blocked", "held-out sector"):
        assert name in texts
    assert (
        f"{rules.arc_count} sectors of {rules.arc_width_deg:g}°, {rules.arc_buffer_deg:g}° buffers"
        in texts
    )
    assert not any("arc" in text for text in texts)


def test_the_windfold_figure_sets_the_wind_frame_above_the_sum_of_terms(style):
    found = wind_frame.frame(
        list(PENMANSHIEL),
        np.array([east for east, _ in PENMANSHIEL.values()], dtype=float),
        np.array([north for _, north in PENMANSHIEL.values()], dtype=float),
        82.0,
        list(PENMANSHIEL).index(wind_frame.RECEIVER),
        wind_frame.DIRECTION,
        30.0,
        3.0,
    )
    figure = windfold.draw(found, load_protocol().processing, style)
    assert problems(figure, style) == []
    assert spacing.crowded(figure) == []
    frame, terms = figure.axes
    assert frame.get_window_extent().y0 > terms.get_window_extent().y1
    assert [t.get_text() for t in figure.texts] == ["a", "b"]
    assert {"i", "j", "a", "c"} <= {t.get_text() for t in frame.texts}
    assert [t.get_text() for t in terms.texts][::2] == [name for name, _ in prediction.PARTS]
