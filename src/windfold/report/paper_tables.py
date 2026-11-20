"""the paper's tables, from the configurations, the prepared rows and the results."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path

import polars as pl

from windfold import config
from windfold.config import FarmConfig, ModelSpec, Window
from windfold.report.style import Report, learner_of
from windfold.report.tables import Column, Group, Span, Table

MAIN_REGIMES = ("random", "time_blocked", "arc_single")
ABLATIONS = (
    "windfold_no_physics",
    "windfold_no_kernel",
    "windfold_no_local",
    "windfold_permuted_layout",
    "windfold_default_physics",
    "windfold_constant_inflow",
    "windfold_evidence_scales",
    "windfold_constant_inflow_evidence_scales",
    "windfold_average_inflow",
    "windfold_climatology_inflow",
    "windfold_diurnal_inflow",
    "windfold_unit_physics",
)
ABLATION_REGIMES = ("time_blocked", "arc_single")
MINUTE_RULES = ("derated", "control", "minutes_per_window")
MINUTES = "minutes"
LEARNER_NAMES = {
    "linear": "linear Gaussian",
    "lookup": "lookup table",
    "gbm": "LightGBM",
    "graph": "graph network",
    "automl": "AutoGluon",
}


def months(window: Window | None) -> str:
    if window is None:
        return "none"
    last = window.end - timedelta(microseconds=1)
    return f"{window.start:%b %Y} to {last:%b %Y}"


def period_rows(data: Path, farm: str) -> dict[str, int]:
    rows = pl.read_parquet(data / "prepared" / farm / "rows.parquet", columns=["period"])
    return dict(rows.group_by("period").len().iter_rows())


def datasets(report: Report, data: Path, configs: Path | None) -> Table:
    """one column per farm: where the data come from, the turbines and the periods used."""
    farms = [config.load_farm(name, configs) for name in report.farms.order]
    turbines = {f.name: config.load_turbine(f.turbine_type, configs) for f in farms}
    counts = {f.name: period_rows(data, f.name) for f in farms}
    layouts = {
        f.name: pl.read_parquet(data / "prepared" / f.name / "layout.parquet").height for f in farms
    }

    def row(label: str, value) -> dict[str, str]:
        return {"item": label} | {f.name: value(f) for f in farms}

    def count(f: FarmConfig, period: str) -> str:
        found = counts[f.name].get(period)
        return f"{found:,}".replace(",", "\u2009") if found else "none"

    rows = (
        row("source", lambda f: f.dataset.citation.split(" (")[0]),
        row("licence", lambda f: f.dataset.licence),
        row("turbine", lambda f: report.turbines[f.turbine_type]),
        row("turbines", lambda f: str(layouts[f.name])),
        row("rotor diameter (m)", lambda f: f"{turbines[f.name].rotor_diameter_m:g}"),
        row("rated power (MW)", lambda f: f"{turbines[f.name].rated_power_kw / 1000:g}"),
        row("calibration", lambda f: months(f.windows.calibration)),
        row("development", lambda f: months(f.windows.development)),
        row("confirmatory", lambda f: months(f.windows.confirmatory)),
        row("development rows", lambda f: count(f, "development")),
        row("confirmatory rows", lambda f: count(f, "confirmatory")),
    )
    columns = (
        Column("item", ""),
        *(Column(f.name, report.farms.names[f.name]) for f in farms),
    )
    return Table(columns, (Group(None, rows),))


def rows_kept(report: Report, data: Path) -> Table:
    """rows left after each processing step, one column per farm. 1-minute data are first
    checked minute by minute and grouped into 10-minute windows, a row of its own."""
    farms = report.farms.order
    kept: dict[str, dict[str, int]] = {}
    for farm in farms:
        ledger = pl.read_csv(data / "prepared" / farm / "ledger.csv")
        ledger = ledger.filter(pl.col("step") <= max(report.steps))
        key = (
            pl.when(pl.col("rule").is_in(MINUTE_RULES))
            .then(pl.lit(MINUTES))
            .otherwise(pl.col("step").cast(pl.String))
        )
        last = ledger.with_columns(key=key).group_by("key", maintain_order=True).last()
        kept[farm] = dict(last.select("key", "rows_kept").iter_rows())
    names = [(str(step), f"{step}. {name}") for step, name in sorted(report.steps.items())]
    names.insert(2, (MINUTES, "1-minute checks, grouped into 10-minute windows"))
    rows = []
    for key, name in names:
        values = {farm: kept[farm].get(key) for farm in farms}
        if any(value is not None for value in values.values()):
            rows.append({"step": name} | values)
    columns = (
        Column("step", "after step"),
        *(Column(farm, report.farms.names[farm], "count") for farm in farms),
    )
    return Table(columns, (Group(None, tuple(rows)),))


def inputs_of(spec: ModelSpec, report: Report) -> str:
    names = [report.inputs[b] for b in spec.blocks]
    names += [report.inputs[f] for f in spec.features]
    if spec.edges is not None:
        names.append(report.inputs[spec.edges])
    if spec.pretrained:
        names.append(report.inputs["pretrained"])
    if spec.kernel_layout == "permuted":
        names.append("positions permuted")
    if spec.physics_parameters == "default":
        names.append("default physics parameters")
    if spec.inflow != "constant":
        names.append(report.inputs[f"{spec.inflow}_inflow"])
    return ", ".join(names)


def tuning_of(spec: ModelSpec, menus: Mapping[str, int]) -> str:
    if spec.learner == "automl":
        return "time budget chosen on the calibration window"
    if spec.learner is not None:
        return f"{menus[spec.learner]} configurations on the calibration window"
    if "turbine_direction_table" in spec.blocks:
        return "none"
    if spec.physics_scale == "fixed":
        return "physics fitted on the calibration window"
    if spec.scales == "fixed":
        return "declared prior scales"
    return "scales by marginal likelihood"


def models(report: Report, names: Sequence[str], configs: Path | None) -> Table:
    """each model's inputs, what it can know about an unseen direction and its tuning, grouped
    by representation."""
    specs = config.load_models(configs)
    menus = {learner: len(config.load_menu(learner, configs)) for learner in ("gbm", "graph")}
    groups = []
    for representation in report.knows:
        rows = tuple(
            {
                "model": specs[name].display_name,
                "learner": LEARNER_NAMES[learner_of(specs[name])],
                "inputs": inputs_of(specs[name], report),
                "tuning": tuning_of(specs[name], menus),
            }
            for name in names
            if specs[name].representation == representation
        )
        if rows:
            label = f"{representation.replace('_', '-')}: {report.knows[representation]}"
            groups.append(Group(label, rows))
    columns = (
        Column("model", "model", width_mm=40),
        Column("learner", "learner", width_mm=22),
        Column("inputs", "inputs", width_mm=60),
        Column("tuning", "tuning", width_mm=36),
    )
    return Table(columns, tuple(groups))


def estimate(
    table: pl.DataFrame, scale: float = 1.0, **keys
) -> tuple[float | None, float | None, float | None] | None:
    chosen = table
    for name, value in keys.items():
        chosen = chosen.filter(pl.col(name).is_null() if value is None else pl.col(name) == value)
    if chosen.height != 1:
        return None
    found = chosen.row(0, named=True)

    def scaled(key: str) -> float | None:
        return None if found[key] is None else found[key] * scale

    return scaled("estimate"), scaled("lower"), scaled("upper")


def main_results(
    report: Report,
    accuracy: Mapping[str, pl.DataFrame],
    names: Sequence[str],
    configs: Path | None,
) -> Table:
    """skill against the no-wake reference with 95% intervals, by study and farm: waked rows in
    each regime, and all rows when sectors are held out."""
    specs = config.load_models(configs)
    shown = [n for n in names if specs[n].representation != "reference"]
    groups = []
    for study, table in accuracy.items():
        for farm in report.farms.order:
            here = table.filter(pl.col("farm") == farm, pl.col("metric") == "skill")
            if here.is_empty():
                continue
            rows = []
            for name in shown:
                cells = {
                    regime: estimate(
                        here, 100.0, model=name, regime=regime, rows="waked", horizon_days=None
                    )
                    for regime in MAIN_REGIMES
                }
                cells["all"] = estimate(
                    here, 100.0, model=name, regime="arc_single", rows="all", horizon_days=None
                )
                if any(value is not None for value in cells.values()):
                    rows.append({"model": specs[name].display_name} | cells)
            label = f"{report.farms.names[farm]}, {study} data"
            groups.append(Group(label, tuple(rows)))
    columns = (
        Column("model", "model"),
        *(Column(r, report.regimes[r], "estimate", best="max", decimals=1) for r in MAIN_REGIMES),
        Column("all", report.regimes["arc_single"], "estimate", best="max", decimals=1),
    )
    spans = (
        Span("skill on waked rows (%)", MAIN_REGIMES[0], MAIN_REGIMES[-1]),
        Span("all rows (%)", "all", "all"),
    )
    return Table(columns, tuple(groups), spans)


def ablations(
    report: Report, comparisons: Mapping[str, pl.DataFrame], configs: Path | None
) -> Table:
    """the waked-row skill windfold loses when each part is removed, in points, paired by day
    blocks, with 95% intervals; positive means the removed part helped."""
    specs = config.load_models(configs)
    keys = [
        (study, farm)
        for study, t in comparisons.items()
        for farm in report.farms.order
        if farm in t["farm"].to_list()
    ]
    columns = [Column("ablation", "skill lost (points)")]
    columns += [
        Column(f"{study}:{farm}", report.farms.names[farm], "estimate", decimals=1)
        for study, farm in keys
    ]
    groups = []
    for regime in ABLATION_REGIMES:
        rows = []
        for name in ABLATIONS:
            cells = {
                f"{study}:{farm}": estimate(
                    comparisons[study].filter(pl.col("farm") == farm),
                    100.0,
                    model=name,
                    against="windfold",
                    regime=regime,
                    horizon_days=None,
                )
                for study, farm in keys
            }
            rows.append({"ablation": specs[name].display_name.removeprefix("windfold ")} | cells)
        groups.append(Group(report.regimes[regime], tuple(rows)))
    spans = []
    for study in comparisons:
        farms = [f"{s}:{farm}" for s, farm in keys if s == study]
        if farms:
            spans.append(Span(f"{study} data", farms[0], farms[-1]))
    return Table(tuple(columns), tuple(groups), tuple(spans))


def hypotheses(
    report: Report,
    development: pl.DataFrame,
    confirmatory: pl.DataFrame | None,
    decision: pl.DataFrame,
) -> Table:
    """each hypothesis's outcome on every farm, the confirmatory data and the verdict."""
    farms = [f for f in report.farms.order if f in development["farm"].to_list()]

    def outcome(table: pl.DataFrame | None, farm: str, hypothesis: str) -> str | None:
        if table is None:
            return None
        found = table.filter(pl.col("farm") == farm, pl.col("hypothesis") == hypothesis)
        return found["outcome"][0] if found.height else None

    confirmed = (
        []
        if confirmatory is None
        else [f for f in report.farms.order if f in confirmatory["farm"].to_list()]
    )
    rows = []
    for hypothesis, name in report.hypotheses.items():
        verdict = decision.filter(pl.col("hypothesis") == hypothesis)
        rows.append(
            {"hypothesis": name}
            | {f"development:{f}": outcome(development, f, hypothesis) for f in farms}
            | {f"confirmatory:{f}": outcome(confirmatory, f, hypothesis) for f in confirmed}
            | {
                "verdict": verdict["verdict"][0]
                if verdict.height and verdict["verdict"][0]
                else "pending"
            }
        )
    columns = (
        Column("hypothesis", "hypothesis"),
        *(Column(f"development:{f}", report.farms.names[f]) for f in farms),
        *(Column(f"confirmatory:{f}", report.farms.names[f]) for f in confirmed),
        Column("verdict", "verdict"),
    )
    spans = [Span("development data", f"development:{farms[0]}", f"development:{farms[-1]}")]
    if confirmed:
        spans.append(
            Span(
                "confirmatory data", f"confirmatory:{confirmed[0]}", f"confirmatory:{confirmed[-1]}"
            )
        )
    return Table(columns, (Group(None, tuple(rows)),), tuple(spans))


def sensitivity_outcome(decision: pl.DataFrame, hypothesis: str, period: str) -> str:
    """farms where the hypothesis succeeded, of farms where it was judged, or why there are none."""
    found = decision.filter(pl.col("hypothesis") == hypothesis)
    if found.is_empty() or found[period][0] is None:
        return "pending"
    judged, successes = found[f"{period}_judged"][0], found[f"{period}_successes"][0]
    if not judged:
        return found[period][0]
    return f"{successes} of {judged}"


def sensitivities(
    report: Report,
    main: Mapping[str, pl.DataFrame],
    chosen: Mapping[str, Mapping[str, pl.DataFrame]],
    periods: Mapping[str, str],
) -> Table:
    """each hypothesis in the main analysis and every sensitivity of each study: farms where it
    succeeded of farms where it was judged."""
    columns = (
        Column("analysis", "analysis"),
        *(Column(key, name) for key, name in report.hypotheses.items()),
    )
    groups = []
    for study, decision in main.items():
        missing = sorted(set(chosen[study]) - set(report.sensitivities))
        if missing:
            raise ValueError(f"report.toml names no sensitivities {missing}")
        analyses = {"main analysis": decision} | {
            report.sensitivities[name]: frame for name, frame in chosen[study].items()
        }
        rows = tuple(
            {"analysis": label}
            | {key: sensitivity_outcome(frame, key, periods[study]) for key in report.hypotheses}
            for label, frame in analyses.items()
        )
        groups.append(Group(f"{periods[study]} data", rows))
    return Table(columns, tuple(groups))
