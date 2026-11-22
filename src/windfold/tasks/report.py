"""report jobs: the paper's tables and figures, drawn from the results of its studies.

the development and smoke reports read their own study; the paper's report reads the
development study and adds the confirmatory one.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import polars as pl
from matplotlib.figure import Figure

from windfold import config
from windfold.config import folder_of
from windfold.evaluation.paths import results_folder
from windfold.report import (
    confirmation,
    energy_ratio,
    farms,
    history,
    idea,
    kernel,
    margin,
    paper_tables,
    practical,
    validation,
)
from windfold.report.check import save
from windfold.report.paths import DRAWN, figure_files, table_files
from windfold.report.sources import Sources
from windfold.report.style import Style, load_report, load_style
from windfold.report.tables import write_table

FIGURES: dict[str, Callable[[Sources, Style], Figure]] = {
    "ranks": validation.ranks_figure,
    "farms": farms.farms_figure,
    "idea": idea.figure,
    "layouts": farms.layouts_figure,
    "sectors": farms.sectors_figure,
    "periods": farms.periods_figure,
    "validation": validation.figure,
    "validation_confirmatory": validation.confirmatory_figure,
    "kernel": kernel.figure,
    "repair": confirmation.contrasts_figure,
    "physics": margin.physics_figure,
    "learned": margin.learned_figure,
    "ablations": margin.ablations_figure,
    "energy_ratio": energy_ratio.figure,
    "history_coverage": history.coverage_figure,
    "history_skill": history.skill_figure,
    "practical": practical.figure,
}
if set(FIGURES) != set(DRAWN):
    raise RuntimeError(f"drawn figures {sorted(FIGURES)} differ from the planned {sorted(DRAWN)}")


def sources_of(studies: list[str], data_dir: str, runs_dir: str, configs: str | None) -> Sources:
    return Sources(Path(data_dir), Path(runs_dir), folder_of(configs), tuple(studies))


def revision(sources: Sources) -> str:
    """the first twelve hex digits of the sha256 over the studies' decision tables."""
    digest = hashlib.sha256()
    for study in sources.studies:
        digest.update((sources.runs / study / "results" / "decision.parquet").read_bytes())
    return digest.hexdigest()[:12]


def write_tables(
    folder: str,
    studies: list[str],
    sensitivities: dict[str, list[str]],
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """the tables as LaTeX and csv; sensitivities names, for each study, the sensitivities whose
    decisions the sensitivity table reads."""
    sources = sources_of(studies, data_dir, runs_dir, configs)
    report = load_report(sources.configs)
    names = list(config.load_study(sources.main, sources.configs).models)
    accuracy = {study: sources.results(study, "accuracy") for study in sources.studies}
    comparisons = {study: sources.results(study, "comparisons") for study in sources.studies}
    confirmatory = sources.studies[1] if len(sources.studies) > 1 else None
    decision = sources.results(confirmatory or sources.main, "decision")
    tables = {
        "datasets": paper_tables.datasets(report, sources.data, sources.configs),
        "rows_kept": paper_tables.rows_kept(report, sources.data),
        "models": paper_tables.models(report, names, sources.configs),
        "main_results": paper_tables.main_results(report, accuracy, names, sources.configs),
        "ablations": paper_tables.ablations(report, comparisons, sources.configs),
        "hypotheses": paper_tables.hypotheses(
            report,
            sources.results(sources.main, "hypotheses"),
            None if confirmatory is None else sources.results(confirmatory, "hypotheses"),
            decision,
        ),
        "sensitivities": paper_tables.sensitivities(
            report,
            {study: sources.results(study, "decision") for study in sources.studies},
            {
                study: {
                    name: pl.read_parquet(
                        sources.runs / results_folder(study, name) / "decision.parquet"
                    )
                    for name in sensitivities.get(study, [])
                }
                for study in sources.studies
            },
            {study: config.load_study(study, sources.configs).period for study in sources.studies},
        ),
    }
    stamp = revision(sources)
    for name, table in tables.items():
        tex, csv = table_files(folder, name)
        write_table(table, out_dir / tex, out_dir / csv, report.tables.significant_figures, stamp)


def draw_figure(
    folder: str,
    figure: str,
    studies: list[str],
    data_dir: str,
    runs_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    """one figure as a vector pdf with two png previews, refused if it breaks the style rules."""
    sources = sources_of(studies, data_dir, runs_dir, configs)
    style = load_style(sources.configs)
    drawn = FIGURES[figure](sources, style)
    save(drawn, [out_dir / path for path in figure_files(folder, figure)], style)
