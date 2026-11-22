import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest

from windfold.report import tables
from windfold.report.tables import Column, Group, Span, Table

FIGURES = 3

TABLE = Table(
    columns=(
        Column("model", "model"),
        Column("mse", "MSE", "estimate", best="min"),
        Column("skill", "skill", "estimate", best="max"),
        Column("mae", "MAE", "number", unit="kW", best="min"),
        Column("rows", "rows", "count"),
    ),
    groups=(
        Group(
            "Kelmarsh",
            (
                {
                    "model": "power curve",
                    "mse": (0.1234, 0.11, 0.14),
                    "skill": (0.12, 0.05, 0.2),
                    "mae": 45.67,
                    "rows": 1274311,
                },
                {
                    "model": "windfold",
                    "mse": (0.12309, 0.1, 0.13),
                    "skill": (0.4, 0.3, 0.5),
                    "mae": 40.01,
                    "rows": 12,
                },
                {"model": "peer_automl", "mse": None, "skill": (-0.011, -0.05, 0.03), "mae": None},
            ),
        ),
        Group(
            "Penmanshiel",
            ({"model": "windfold", "mse": (0.2, 0.1, 0.3), "skill": (0.0, -0.012, 0.011)},),
        ),
    ),
    spans=(Span("waked rows", "mse", "skill"),),
)


def body_rows(latex: str) -> list[str]:
    return [line for line in latex.splitlines() if line.startswith(("power", "windfold", "peer"))]


def test_bold_follows_each_column_direction_and_marks_printed_ties_together():
    first, second, third, fourth = body_rows(tables.to_latex(TABLE, FIGURES, "r1"))
    assert first.startswith("power curve & \\bfseries 0.123 & {[}0.110{,} & 0.140{]} & 0.120 &")
    assert second.startswith(
        "windfold & \\bfseries 0.123 & {[}0.100{,} & 0.130{]} & \\bfseries 0.400"
    )
    assert "& \\bfseries 40.0 &" in second and "& 45.7 &" in first
    assert "\\bfseries" not in first.split("&")[-1] and "1274311" in first
    assert third.startswith("peer\\_automl & \\multicolumn{3}{c}{--} & -0.0110 & {[}-0.0500{,}")
    assert fourth.startswith(
        "windfold & \\bfseries 0.200 & {[}0.100{,} & 0.300{]} & \\bfseries 0.0000"
    )


def test_estimates_sit_beside_their_intervals_in_aligned_columns_without_vertical_rules():
    latex = tables.to_latex(TABLE, FIGURES, "r1")
    preamble = next(line for line in latex.splitlines() if line.startswith("\\begin{tabular}"))
    assert "|" not in preamble
    assert (
        "S[table-format=1.3]@{\\,}S[table-format=1.3, table-space-text-pre={[}, "
        "table-space-text-post={,}]@{\\,}S[table-format=1.3, table-space-text-post={]}]"
    ) in preamble
    assert "S[table-format=-1.4]" in preamble and "S[table-format=7.0]" in preamble
    assert "\\multicolumn{6}{c}{waked rows}" in latex and "\\cmidrule(lr){2-7}" in latex
    assert latex.splitlines()[0] == "% windfold table from results r1"


def test_the_csv_keeps_every_number_at_full_precision(tmp_path: Path):
    tables.write_table(TABLE, tmp_path / "t.tex", tmp_path / "t.csv", FIGURES, "r1")
    frame = pl.read_csv(tmp_path / "t.csv")
    assert frame.columns[:5] == ["group", "model", "mse", "mse_lower", "mse_upper"]
    assert frame["mse"][1] == pytest.approx(0.12309) and frame["rows"][0] == 1274311


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not on the path")
def test_the_table_compiles_in_a_minimal_document_at_the_page_width(tmp_path: Path):
    (tmp_path / "t.tex").write_text(tables.to_latex(TABLE, FIGURES, "r1"))
    (tmp_path / "doc.tex").write_text(
        "\\documentclass{article}\\usepackage[textwidth=170mm]{geometry}"
        "\\usepackage{booktabs}\\usepackage{siunitx}"
        "\\begin{document}\\input{t}\\end{document}"
    )
    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout[-2000:]
    assert "Overfull" not in run.stdout
