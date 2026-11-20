"""names of every report output, relative to the runs folder. FIGURES are in the order the paper
first mentions them, then the figures the paper does not show; the runner draws DRAWN from the
studies' own results, CONFIRMATION_ONLY only for the paper's report, and `windfold figures` draws
COMMANDED, which read the exploratory analyses and the secondary tests."""

FIGURES = (
    "ranks",
    "farms",
    "idea",
    "turbine_levels",
    "transfer",
    "margin",
    "sector_gains",
    "false_alarms",
    "optimism",
    "hybrid",
    "repair",
    "layouts",
    "sectors",
    "periods",
    "validation",
    "ceiling",
    "realised",
    "kernel",
    "restricted",
    "detection",
    "energy_ratio",
    "history_coverage",
    "history_skill",
    "physics",
    "learned",
    "ablations",
    "practical",
    "validation_confirmatory",
    "recall",
)
COMMANDED = (
    "margin",
    "ceiling",
    "realised",
    "transfer",
    "hybrid",
    "restricted",
    "false_alarms",
    "detection",
    "recall",
    "turbine_levels",
    "sector_gains",
)
DRAWN = tuple(name for name in FIGURES if name not in COMMANDED)
CONFIRMATION_ONLY = ("validation_confirmatory",)
TABLES = (
    "datasets",
    "rows_kept",
    "models",
    "main_results",
    "ablations",
    "hypotheses",
    "sensitivities",
)
PREVIEW_PX = (600, 1200)


def figure_files(study: str, figure: str) -> tuple[str, str, str]:
    """the vector pdf and its 600 and 1200 pixel previews."""
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure}; known: {list(FIGURES)}")
    small, large = (f"{study}/figures/previews/{figure}_{px}px.png" for px in PREVIEW_PX)
    return f"{study}/figures/{figure}.pdf", small, large


def table_files(study: str, table: str) -> tuple[str, str]:
    """the LaTeX table and its csv with every number at full precision."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table}; known: {list(TABLES)}")
    return f"{study}/tables/{table}.tex", f"{study}/tables/{table}.csv"
