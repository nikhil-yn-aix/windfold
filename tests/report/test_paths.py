import subprocess
import sys

import pytest

from windfold.report import paths


def test_figure_files_are_a_pdf_and_two_previews():
    assert paths.figure_files("development", "idea") == (
        "development/figures/idea.pdf",
        "development/figures/previews/idea_600px.png",
        "development/figures/previews/idea_1200px.png",
    )


def test_table_files_are_tex_and_csv_under_tables():
    assert paths.table_files("confirmatory", "hypotheses") == (
        "confirmatory/tables/hypotheses.tex",
        "confirmatory/tables/hypotheses.csv",
    )


def test_an_unknown_name_is_refused():
    with pytest.raises(ValueError, match="unknown figure"):
        paths.figure_files("development", "ideas")
    with pytest.raises(ValueError, match="unknown table"):
        paths.table_files("development", "results")


def test_the_planner_can_name_outputs_without_loading_plotting_or_tables():
    code = (
        "import sys, windfold.report.paths; "
        "print(sorted(m for m in ('matplotlib', 'numpy', 'polars') if m in sys.modules))"
    )
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert loaded.stdout.strip() == "[]", loaded.stderr
