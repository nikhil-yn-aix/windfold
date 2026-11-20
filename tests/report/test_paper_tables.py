import polars as pl
import pytest

from windfold.evaluation.results import DECISIONS
from windfold.report import paper_tables
from windfold.report.style import load_report

HYPOTHESES = (
    "validation_misleads",
    "wind_frame_transfers",
    "coverage_predicts_failure",
    "early_history_advantage",
)


def decision(cells: dict[str, tuple[int, int, str]]) -> pl.DataFrame:
    rows = [
        (name, *cells[name], None, None, None, None)
        if name in cells
        else (name, None, None, None, None, None, None, None)
        for name in HYPOTHESES
    ]
    return pl.DataFrame(rows, schema=DECISIONS, orient="row")


def test_each_analysis_shows_farms_succeeded_of_farms_judged_or_why_there_are_none():
    report = load_report()
    main = decision(
        {
            "validation_misleads": (4, 3, "success"),
            "wind_frame_transfers": (4, 1, "failure"),
            "coverage_predicts_failure": (0, 0, "not estimable"),
        }
    )
    shrinkage = decision({"validation_misleads": (4, 2, "failure")})
    table = paper_tables.sensitivities(
        report,
        {"development": main},
        {"development": {"fixed_shrinkage": shrinkage}},
        {"development": "development"},
    )
    (group,) = table.groups
    assert group.label == "development data"
    first, second = group.rows
    assert first == {
        "analysis": "main analysis",
        "validation_misleads": "3 of 4",
        "wind_frame_transfers": "1 of 4",
        "coverage_predicts_failure": "not estimable",
        "early_history_advantage": "pending",
    }
    assert second["analysis"] == "fixed shrinkage" and second["validation_misleads"] == "2 of 4"


def test_a_sensitivity_without_a_display_name_is_refused():
    with pytest.raises(ValueError, match="names no sensitivities"):
        paper_tables.sensitivities(
            load_report(),
            {"development": decision({})},
            {"development": {"unnamed": decision({})}},
            {"development": "development"},
        )
