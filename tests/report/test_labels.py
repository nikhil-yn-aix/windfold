from collections.abc import Sequence

import numpy as np
import pytest

from windfold.report.check import LEADER, problems
from windfold.report.labels import Label, NoRoom, place, settle
from windfold.report.style import Style, load_style


@pytest.fixture(scope="module")
def style() -> Style:
    return load_style()


def scatter(style: Style, points: Sequence[tuple[float, float]], width: float = 85.0):
    figure = style.figure(width, 60.0)
    ax = figure.add_subplot()
    for x, y in points:
        ax.plot(x, y, marker="o", ls="none", color=style.role("axes"))
    ax.set(xlim=(0.0, 10.0), ylim=(0.0, 10.0), xticks=[], yticks=[])
    return figure, ax


def leaders(ax) -> list:
    return [line for line in ax.lines if line.get_gid() == LEADER]


def test_a_lone_point_is_labelled_beside_it_without_a_leader(style: Style):
    with style.context():
        figure, ax = scatter(style, [(5.0, 5.0)])
        (note,) = place(ax, [Label(5.0, 5.0, "lone")], style, settle(figure))
    assert note.get_text() == "lone" and note.xy == (5.0, 5.0)
    assert np.hypot(*note.xyann) < 10.0
    assert leaders(ax) == []
    assert problems(figure, style) == []


def test_a_label_near_a_neighbour_moves_out_on_a_leader_and_keeps_clear(style: Style):
    points = [(5.0, 5.0), (5.4, 5.0), (4.6, 5.0), (5.0, 5.5), (5.0, 4.5)]
    with style.context():
        figure, ax = scatter(style, points)
        place(ax, [Label(5.0, 5.0, "middle")], style, settle(figure))
    (leader,) = leaders(ax)
    start = np.array([leader.get_xdata()[0], leader.get_ydata()[0]], dtype=float)
    assert np.hypot(*(start - 5.0)) < 0.5
    assert problems(figure, style) == []


def test_labels_keep_clear_of_each_other_and_of_lines(style: Style):
    with style.context():
        figure, ax = scatter(style, [(3.0, 3.0), (3.0, 3.6), (7.0, 7.0)])
        ax.plot([0.0, 10.0], [0.0, 10.0], color=style.role("axes"))
        labels = [Label(3.0, 3.0, "first"), Label(3.0, 3.6, "second"), Label(7.0, 7.0, "third")]
        place(ax, labels, style, settle(figure))
    assert sorted(t.get_text() for t in ax.texts) == ["first", "second", "third"]
    assert problems(figure, style) == []


def test_a_label_with_no_room_is_refused_by_name(style: Style):
    points = [(x, y) for x in np.linspace(0.3, 9.7, 30) for y in np.linspace(0.3, 9.7, 30)]
    with style.context():
        figure, ax = scatter(style, points)
        with pytest.raises(NoRoom, match="'crowded'"):
            place(ax, [Label(5.0, 5.0, "crowded")], style, settle(figure))
    assert len(ax.texts) == 0
