"""tables for the paper: booktabs rules, siunitx columns aligned on the decimal point, each estimate
beside its interval and a csv with every number at full precision.

numbers show three significant figures (report.toml), counts in full. bold marks each column's
best value inside each group of rows (lowest for "min" columns, highest for "max"), comparing
the numbers as printed, so ties are bold together. bold is a reading aid, never a test.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

Kind = Literal["text", "count", "number", "estimate"]
Best = Literal["min", "max"]
SIUNITX = (
    "\\sisetup{reset-text-series=false, text-series-to-math=true, mode=match, "
    "propagate-math-font=true, table-align-text-before=false, table-align-text-after=false, "
    "group-digits=integer}"
)
ESCAPES = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
    "\u2009": "\\,",
}


@dataclass(frozen=True)
class Column:
    """decimals fixes the places of a number or estimate column instead of significant figures;
    width_mm wraps a text column at that width."""

    key: str
    header: str
    kind: Kind = "text"
    unit: str | None = None
    best: Best | None = None
    decimals: int | None = None
    width_mm: float | None = None


@dataclass(frozen=True)
class Span:
    header: str
    first: str
    last: str


@dataclass(frozen=True)
class Group:
    label: str | None
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Table:
    """rows hold a str for text, an int for counts, a float for numbers and a (estimate, lower,
    upper) tuple for estimates; none for a value that does not exist."""

    columns: tuple[Column, ...]
    groups: tuple[Group, ...]
    spans: tuple[Span, ...] = ()


def escape(text: str) -> str:
    return "".join(ESCAPES.get(c, c) for c in text)


def places(value: float, figures: int) -> int:
    """decimals that show value to figures significant figures, after rounding."""
    if value == 0 or not math.isfinite(value):
        return 0
    rounded = float(f"{value:.{figures}g}")
    return max(0, figures - 1 - math.floor(math.log10(abs(rounded))))


def significant(value: float, figures: int, decimals: int | None = None) -> str:
    """value to figures significant figures, or with exactly decimals places when given."""
    shown = float(f"{value:.{figures}g}") if decimals is None else value
    text = f"{shown:.{places(value, figures) if decimals is None else decimals}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def cell_numbers(column: Column, value: Any, figures: int) -> tuple[str, ...] | None:
    """the printed numbers of one cell: one for counts and numbers, three for an estimate, the
    interval at the estimate's decimals."""
    if value is None or column.kind == "text":
        return None
    if column.kind == "count":
        return (str(int(value)),)
    fixed = column.decimals
    if column.kind == "number":
        if not math.isfinite(value):
            return None
        return (significant(float(value), figures, fixed),)
    estimate, lower, upper = value
    if estimate is None or not math.isfinite(estimate):
        return None
    if lower is None or upper is None:
        return (significant(float(estimate), figures, fixed),)
    decimals = fixed
    if decimals is None:
        shown = (estimate,) if estimate else (lower, upper)
        decimals = max(places(float(v), figures) for v in shown)
    return tuple(significant(float(v), figures, decimals) for v in (estimate, lower, upper))


def table_format(numbers: Sequence[str]) -> str:
    if not numbers:
        return "1.0"
    integers = max(len(n.lstrip("-").split(".")[0]) for n in numbers)
    places = max(len(n.split(".")[1]) if "." in n else 0 for n in numbers)
    sign = "-" if any(n.startswith("-") for n in numbers) else ""
    return f"{sign}{integers}.{places}"


def best_rows(column: Column, rows: Sequence[Mapping[str, Any]], figures: int) -> set[int]:
    if column.best is None:
        return set()
    printed = [cell_numbers(column, row.get(column.key), figures) for row in rows]
    values = {i: float(p[0]) for i, p in enumerate(printed) if p is not None}
    if not values:
        return set()
    target = min(values.values()) if column.best == "min" else max(values.values())
    return {i for i, v in values.items() if v == target}


def header(column: Column) -> str:
    text = escape(column.header)
    return f"{text} ({column.unit})" if column.unit else text


def to_latex(table: Table, figures: int, revision: str) -> str:
    columns = table.columns
    printed: dict[str, list[str]] = {c.key: [] for c in columns}
    width: dict[str, int] = {}
    for column in columns:
        parts = 3 if column.kind == "estimate" else 1
        width[column.key] = parts if column.kind != "text" else 1
        for group in table.groups:
            for row in group.rows:
                numbers = cell_numbers(column, row.get(column.key), figures)
                printed[column.key] += list(numbers or ())
    specs = []
    for column in columns:
        if column.kind == "text":
            if column.width_mm:
                specs.append(f">{{\\raggedright\\arraybackslash}}p{{{column.width_mm:g}mm}}")
            else:
                specs.append("l")
            continue
        fmt = f"table-format={table_format(printed[column.key])}"
        if column.kind == "estimate":
            lower = f"{fmt}, table-space-text-pre={{[}}, table-space-text-post={{,}}"
            upper = f"{fmt}, table-space-text-post={{]}}"
            specs.append(f"S[{fmt}]@{{\\,}}S[{lower}]@{{\\,}}S[{upper}]")
        else:
            specs.append(f"S[{fmt}]")
    total = sum(width.values())
    lines = [
        f"% windfold table from results {revision}",
        SIUNITX,
        f"\\begin{{tabular}}{{@{{}}{' '.join(specs)}@{{}}}}",
        "\\toprule",
    ]
    position = {}
    start = 1
    for column in columns:
        position[column.key] = (start, start + width[column.key] - 1)
        start += width[column.key]
    if table.spans:
        row, rules, at = [], [], 1
        for span in table.spans:
            first, last = position[span.first][0], position[span.last][1]
            row += [""] * (first - at)
            row.append(f"\\multicolumn{{{last - first + 1}}}{{c}}{{{escape(span.header)}}}")
            rules.append(f"\\cmidrule(lr){{{first}-{last}}}")
            at = last + 1
        row += [""] * (total - at + 1)
        lines += [" & ".join(row) + " \\\\", " ".join(rules)]
    heads = []
    for column in columns:
        if column.kind == "text":
            heads.append(header(column))
        else:
            span = width[column.key]
            heads.append(f"\\multicolumn{{{span}}}{{c}}{{{header(column)}}}")
    lines += [" & ".join(heads) + " \\\\", "\\midrule"]
    for index, group in enumerate(table.groups):
        if group.label is not None:
            if index:
                lines.append("\\midrule")
            lines.append(
                f"\\multicolumn{{{total}}}{{@{{}}l}}{{\\itshape {escape(group.label)}}} \\\\"
            )
        bold = {c.key: best_rows(c, group.rows, figures) for c in columns}
        for i, row in enumerate(group.rows):
            cells = []
            for column in columns:
                value = row.get(column.key)
                if column.kind == "text":
                    cells.append(escape(str(value)) if value is not None else "")
                    continue
                numbers = cell_numbers(column, value, figures)
                mark = "\\bfseries " if i in bold[column.key] else ""
                if numbers is None:
                    cells.append(f"\\multicolumn{{{width[column.key]}}}{{c}}{{--}}")
                elif column.kind == "estimate" and len(numbers) == 1:
                    cells.append(f"{mark}{numbers[0]} & & ")
                elif column.kind == "estimate":
                    cells.append(f"{mark}{numbers[0]} & {{[}}{numbers[1]}{{,}} & {numbers[2]}{{]}}")
                else:
                    cells.append(mark + numbers[0])
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def to_frame(table: Table) -> pl.DataFrame:
    records = []
    for group in table.groups:
        for row in group.rows:
            record: dict[str, Any] = {} if group.label is None else {"group": group.label}
            for column in table.columns:
                value = row.get(column.key)
                if column.kind == "estimate":
                    estimate, lower, upper = value if value is not None else (None, None, None)
                    record |= {
                        column.key: estimate,
                        f"{column.key}_lower": lower,
                        f"{column.key}_upper": upper,
                    }
                else:
                    record[column.key] = value
            records.append(record)
    return pl.DataFrame(records, infer_schema_length=None)


def write_table(table: Table, tex: Path, csv: Path, figures: int, revision: str) -> None:
    for path in (tex, csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text(to_latex(table, figures, revision), encoding="utf-8")
    to_frame(table).write_csv(csv)
