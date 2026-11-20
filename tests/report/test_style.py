import numpy as np
import pytest

from windfold.report.style import Dataset, Style, load_style


@pytest.fixture(scope="module")
def style() -> Style:
    return load_style()


def test_every_colour_role_resolves_by_name_and_an_unknown_one_is_refused(style: Style):
    report = style.report
    assert style.role("windfold") == report.colours["windfold"][0]
    assert style.role("text") == report.neutrals.text
    assert style.role("confirmatory") == report.periods.confirmatory
    with pytest.raises(ValueError, match="no colour role unknown"):
        style.role("unknown")


def test_datasets_follow_the_farm_order_and_name_the_confirmation_period(style: Style):
    farms = {"development": ["smarteole", "kelmarsh"], "confirmatory": ["penmanshiel"]}
    assert style.datasets(farms, ["confirmatory"]) == [
        Dataset("development", "kelmarsh", "Kelmarsh"),
        Dataset("development", "smarteole", "SMARTEOLE"),
        Dataset("confirmatory", "penmanshiel", "Penmanshiel, confirmation period"),
    ]


def test_an_estimate_without_finite_bounds_draws_only_its_point(style: Style):
    with style.context():
        figure = style.figure(85.0, 40.0)
        ax = figure.add_subplot()
        style.estimate(ax, 1.0, 2.0, np.nan, np.nan, style.role("windfold"))
        style.estimate(ax, 2.0, 2.0, 1.0, 3.0, style.role("windfold"), filled=False)
    shapes = sorted(np.atleast_1d(line.get_xdata()).size for line in ax.lines)
    assert shapes == [1, 1, 2]
    assert ax.lines[-1].get_markerfacecolor() == "white"


@pytest.mark.parametrize("width_mm", [85.0, 170.0])
def test_type_prints_at_its_set_size_once_the_figure_fits_the_text_width(
    style: Style, width_mm: float
):
    page, size = style.report.page, style.report.type.size_pt
    with style.context():
        style.figure(width_mm, 50.0)
        assert style.size_pt * page.print_scale(width_mm) == pytest.approx(size)
