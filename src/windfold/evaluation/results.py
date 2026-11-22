"""one farm's results from its reductions: one bootstrap, the four hypotheses and every table.

a sensitivity that refits only some models computes each hypothesis with those models alone; a
component that needs a model it does not refit is not in this sensitivity. a split sensitivity
reads the main study's reductions for the regimes it does not refit.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from windfold.config import FarmConfig, Protocol, Statistics, Study
from windfold.evaluation import bootstrap, hypotheses, summaries
from windfold.evaluation.hypotheses import (
    OUT_OF_SCOPE,
    Component,
    CoverageConfirmation,
    NotEstimable,
)
from windfold.evaluation.paths import REDUCTIONS

CV_REGIMES = ("random", "time_blocked", "arc_single", "arc_opposite", "arc_control")
RANKED_REGIMES = ("random", "time_blocked", "arc_single")
SECONDARY = (
    "peer_automl",
    "pretrained_graph",
    "pywake",
    "pywake_inflow",
    "pywake_discrepancy",
    "pywake_discrepancy_inflow",
    "identity_gbm",
    "identity_gbm_inflow",
    "geometry_gbm",
    "geometry_gbm_inflow",
    "physics_guided_gbm",
    "physics_guided_gbm_inflow",
    "absolute_graph",
    "absolute_graph_inflow",
    "wind_frame_graph",
    "wind_frame_graph_inflow",
)
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
SECONDARY_AGAINST = "windfold"
INFLOW_PAIRS = (
    ("identity_gbm_inflow", "geometry_gbm_inflow"),
    ("absolute_graph_inflow", "wind_frame_graph_inflow"),
)
CONFIRMATION_COMPARATORS = (
    "windfold_constant_inflow_evidence_scales",
    "windfold_constant_inflow",
    "pywake_inflow",
    "windfold_diurnal_inflow",
)
CONFIRMATION_CELLS = (
    "windfold_constant_inflow",
    "windfold_evidence_scales",
    "windfold_constant_inflow_evidence_scales",
)
REFERENCE = "no_wake"
KEYS = ["model", "split", "seed", "waked"]
QUANTITIES = ["sse", "rows", "abs_error_kw", "crps"]
DECISIONS = pl.Schema(
    {
        "hypothesis": pl.String,
        "development_judged": pl.Int64,
        "development_successes": pl.Int64,
        "development": pl.String,
        "confirmatory_judged": pl.Int64,
        "confirmatory_successes": pl.Int64,
        "confirmatory": pl.String,
        "verdict": pl.String,
    }
)


@dataclass(frozen=True)
class Scope:
    """the models a sensitivity refits, or every model (none)."""

    models: frozenset[str] | None = None

    def has(self, *names: str) -> bool:
        return self.models is None or all(name in self.models for name in names)

    def keep(self, names: Sequence[str]) -> list[str]:
        return [name for name in names if self.has(name)]


def scope_of(fits: pl.DataFrame, sensitivity: str | None, statistics_only: bool) -> Scope:
    """a sensitivity's scope is every model it plans a fit for; the main study and a
    statistics-only sensitivity cover every model."""
    if sensitivity is None or statistics_only:
        return Scope()
    return Scope(frozenset(fits["model"].to_list()))


def combined(
    own: Mapping[str, pl.DataFrame],
    main: Mapping[str, pl.DataFrame],
    borrowed: Sequence[str],
    scope: Scope,
    main_arc_width: float,
    arc_width: float,
) -> dict[str, pl.DataFrame]:
    """a split sensitivity's reductions with the main study's for the regimes it does not
    refit, kept to the models in its scope. the main study's time-blocked arc breakdown is
    regrouped into the sensitivity's arcs, which must each hold whole main arcs."""
    ratio = arc_width / main_arc_width
    if not float(ratio).is_integer():
        raise ValueError(f"{arc_width}-degree arcs do not hold whole {main_arc_width}-degree arcs")
    lent = main["splits"].filter(pl.col("regime").is_in(list(borrowed)))["split"]
    out = {}
    for name in REDUCTIONS:
        if name in ("arcs", "turbines"):
            out[name] = own[name]
            continue
        extra = main[name].filter(pl.col("split").is_in(lent.implode()))
        if "model" in extra.columns and scope.models is not None:
            extra = extra.filter(pl.col("model").is_in(sorted(scope.models)))
        if name == "breakdown":
            extra = extra.with_columns(
                level=pl.when(pl.col("factor") == "arc")
                .then(pl.col("level") // int(ratio))
                .otherwise(pl.col("level"))
            )
        kept = own[name].filter(~pl.col("split").is_in(lent.implode()))
        out[name] = pl.concat([kept, extra.select(kept.columns)])
    return out


@dataclass(frozen=True)
class Windows:
    """the bootstrap of a farm: per window, its first day and days, block count and draw."""

    origins: dict[str, tuple[date, int]]
    blocks: dict[str, int]
    drawn: dict[str, np.ndarray]
    draws: dict[str, summaries.Draw]
    seeds: np.ndarray


def study_windows(farm: FarmConfig, period: str) -> dict[str, tuple[datetime, datetime]]:
    """the study period (cross-validation) and the histories' common test window."""
    window = getattr(farm.windows, period)
    plan = farm.histories if period == "development" else farm.confirmatory_histories
    if window is None or plan is None:
        raise ValueError(f"{farm.name} has no {period} window and histories")
    return {"period": (window.start, window.end), "test": (plan.test.start, plan.test.end)}


def bootstrap_windows(
    spans: Mapping[str, tuple[datetime, datetime]],
    block_days: int,
    replicates: int,
    model_seeds: int,
    level: float,
    min_finite: float,
    seed: int,
) -> Windows:
    origins = {name: bootstrap.window_days(*span) for name, span in spans.items()}
    blocks = {
        name: bootstrap.block_count(first, first + timedelta(days=days), block_days)
        for name, (first, days) in origins.items()
    }
    drawn = bootstrap.draw_blocks(blocks, replicates, seed)
    seeds = bootstrap.draw_seeds(replicates, model_seeds, seed)
    draws = {
        name: summaries.Draw(
            bootstrap.multiplicity(drawn[name], blocks[name]), seeds, level, min_finite
        )
        for name in blocks
    }
    return Windows(origins, blocks, drawn, draws, seeds)


def horizon_groups(history_splits: Sequence[str]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for split in sorted(history_splits):
        groups[int(split.split("_")[1].removesuffix("d"))].append(split)
    return dict(sorted(groups.items()))


def validation_components(
    cv: bootstrap.Totals,
    draw: summaries.Draw,
    splits_of: Mapping[str, list[str]],
    protocol: Protocol,
    scope: Scope,
) -> dict[str, Component]:
    endpoint = protocol.endpoints.validation_misleads
    name = "time_blocked_choice_minus_physics"
    baselines = scope.keep(endpoint.baselines)
    if not baselines or not scope.has(endpoint.physics):
        return {name: OUT_OF_SCOPE}
    return {
        name: hypotheses.attempt(
            lambda: hypotheses.estimate(
                lambda c, s: hypotheses.validation_misleads(
                    cv,
                    c,
                    s,
                    baselines=baselines,
                    physics=endpoint.physics,
                    time_splits=splits_of["time_blocked"],
                    arc_splits=splits_of["arc_single"],
                ),
                draw.counts,
                draw.seeds,
            )
        )
    }


def transfer_components(
    cv: bootstrap.Totals,
    draw: summaries.Draw,
    splits_of: Mapping[str, list[str]],
    pairs: Sequence[Sequence[str]],
    scope: Scope,
) -> dict[str, Component]:
    components: dict[str, Component] = {}
    for identity, relative in pairs:
        name = f"{identity}_minus_{relative}"
        if not scope.has(identity, relative):
            components[name] = OUT_OF_SCOPE
            continue
        components[name] = hypotheses.attempt(
            lambda a=identity, b=relative: hypotheses.estimate(
                lambda c, s: hypotheses.wind_frame_transfers(
                    cv, c, s, identity=a, relative=b, arc_splits=splits_of["arc_single"]
                ),
                draw.counts,
                draw.seeds,
            )
        )
    return components


def coverage_components(
    cv: bootstrap.Totals,
    time_arcs: bootstrap.Totals,
    draw: summaries.Draw,
    arcs: pl.DataFrame,
    splits_of: Mapping[str, list[str]],
    confirmation: CoverageConfirmation | None,
    protocol: Protocol,
    scope: Scope,
) -> dict[str, Component]:
    endpoint = protocol.endpoints.coverage_predicts_failure
    if not scope.has(endpoint.relational, endpoint.physics):
        return dict.fromkeys(("partial_spearman", "auc_gain"), OUT_OF_SCOPE)
    return hypotheses.coverage_predicts_failure(
        cv,
        time_arcs,
        draw.counts,
        design=arcs,
        covariates=endpoint.covariates,
        confirmation=confirmation,
        model=endpoint.relational,
        physics=endpoint.physics,
        time_splits=splits_of["time_blocked"],
        min_class=endpoint.min_class,
    )


def coverage_null(
    cv: bootstrap.Totals,
    time_arcs: bootstrap.Totals,
    draw: summaries.Draw,
    arcs: pl.DataFrame,
    splits_of: Mapping[str, list[str]],
    protocol: Protocol,
    scope: Scope,
) -> pl.DataFrame:
    endpoint = protocol.endpoints.coverage_predicts_failure
    if not scope.has(endpoint.relational, endpoint.physics):
        return pl.DataFrame(schema=summaries.COVERAGE_NULL)
    return summaries.coverage_null(
        cv,
        time_arcs,
        draw,
        arcs,
        covariates=endpoint.covariates,
        model=endpoint.relational,
        physics=endpoint.physics,
        time_splits=splits_of["time_blocked"],
    )


def early_history(
    reduced: Mapping[str, pl.DataFrame],
    history: bootstrap.Totals,
    windows: Windows,
    horizons: dict[int, list[str]],
    turbines: Sequence[str],
    protocol: Protocol,
    seed: int,
    scope: Scope,
) -> dict[str, Component]:
    endpoint = protocol.endpoints.early_history_advantage
    accuracy = [f"{name}_minus_{endpoint.model}" for name in endpoint.accuracy_comparators]
    monitors = [
        f"{kind}_{name}"
        for name in endpoint.monitor_comparators
        for kind in summaries.MONITOR_PARTS
    ]
    names = [*accuracy, *monitors]
    if not scope.has(endpoint.model):
        return dict.fromkeys(names, OUT_OF_SCOPE)
    found = summaries.early_history_components(
        dict(reduced),
        history,
        windows.draws["test"],
        windows.drawn["test"],
        horizons,
        windows.origins["test"],
        turbines,
        protocol,
        seed,
        accuracy_comparators=scope.keep(endpoint.accuracy_comparators),
        monitor_comparators=scope.keep(endpoint.monitor_comparators),
        monitors=protocol.monitoring.models,
    )
    return {name: found.get(name, OUT_OF_SCOPE) for name in names}


def matched_monitoring(
    reduced: Mapping[str, pl.DataFrame],
    history: bootstrap.Totals,
    windows: Windows,
    horizons: dict[int, list[str]],
    turbines: Sequence[str],
    protocol: Protocol,
    seed: int,
    scope: Scope,
) -> dict[str, Component]:
    """windfold's monitor against each matched monitor on copies every monitor can watch,
    outside the decision rule."""
    model = protocol.endpoints.early_history_advantage.model
    comparators = scope.keep(protocol.matched_monitoring.models)
    names = [f"{kind}_{name}" for name in comparators for kind in summaries.MONITOR_PARTS]
    if not scope.has(model):
        return dict.fromkeys(names, OUT_OF_SCOPE)
    found = summaries.early_history_components(
        dict(reduced),
        history,
        windows.draws["test"],
        windows.drawn["test"],
        horizons,
        windows.origins["test"],
        turbines,
        protocol,
        seed,
        accuracy_comparators=[],
        monitor_comparators=comparators,
        monitors=(*protocol.monitoring.models, *comparators),
    )
    return {name: found.get(name, OUT_OF_SCOPE) for name in names}


def confirmation_contrast_rows(
    cv: bootstrap.Totals,
    draw: summaries.Draw,
    splits_of: Mapping[str, list[str]],
    statistics: Statistics,
    scope: Scope,
) -> pl.DataFrame:
    """windfold's single-arc skill against each comparator of the confirmation contrasts,
    holm-adjusted over them, and the interaction of measured inflow and declared scales,
    unadjusted."""
    names = [f"{SECONDARY_AGAINST}_minus_{other}" for other in CONFIRMATION_COMPARATORS]
    if not scope.has(SECONDARY_AGAINST, *CONFIRMATION_COMPARATORS, *CONFIRMATION_CELLS):
        components: dict[str, Component] = dict.fromkeys([*names, "interaction"], OUT_OF_SCOPE)
    else:
        components = summaries.confirmation_contrasts(
            cv,
            draw,
            splits_of["arc_single"],
            SECONDARY_AGAINST,
            CONFIRMATION_COMPARATORS,
            CONFIRMATION_CELLS,
            REFERENCE,
        )
    rows = hypotheses.contrast_rows(components, statistics.interval, statistics.min_finite)
    found = dict(zip(rows["component"], rows["p"], strict=True))
    adjusted = hypotheses.holm({name: found[name] for name in names})
    return rows.with_columns(
        holm_p=pl.col("component").replace_strict(adjusted, default=None, return_dtype=pl.Float64)
    )


def confirmation_from_arcs(
    arcs: Sequence[pl.DataFrame], protocol: Protocol
) -> CoverageConfirmation | None:
    """the coverage map and simple-predictor choice fitted on other farms' arc tables, or none
    when they cannot fit one."""
    covariates = protocol.endpoints.coverage_predicts_failure.covariates
    if not arcs:
        return None
    table = pl.concat(arcs)
    labels = (table["relational_mse"] > table["physics_mse"]).to_numpy()
    predictors = {name: table[name].to_numpy() for name in covariates}
    try:
        return hypotheses.fit_confirmation(table["score"].to_numpy(), predictors, labels)
    except NotEstimable:
        return None


def farm_tables(
    reduced: Mapping[str, pl.DataFrame],
    *,
    farm: str,
    protocol: Protocol,
    farm_config: FarmConfig,
    study: Study,
    scope: Scope,
    confirmation: CoverageConfirmation | None,
) -> dict[str, pl.DataFrame]:
    """every result table of one farm from its reductions and one synchronized bootstrap."""
    statistics = protocol.statistics
    fits = reduced["fits"]
    done = fits.filter(pl.col("status") == "done")
    regime_of = dict(reduced["splits"].select("split", "regime").iter_rows())
    planned = fits.select("split", "regime").unique()
    splits_of = {
        regime: sorted(planned.filter(pl.col("regime") == regime)["split"].to_list())
        for regime in (*CV_REGIMES, "history")
    }
    limits = study.limits
    replicates = limits.replicates if limits and limits.replicates else statistics.replicates
    model_seeds = len(protocol.seeds.model)
    if limits and limits.seeds:
        model_seeds = min(model_seeds, limits.seeds)
    seed = bootstrap.farm_seed(protocol.seeds.study, farm)
    windows = bootstrap_windows(
        study_windows(farm_config, study.period),
        statistics.block_days,
        replicates,
        model_seeds,
        statistics.interval,
        statistics.min_finite,
        seed,
    )
    period, test = windows.draws["period"], windows.draws["test"]

    def totals(
        frame: pl.DataFrame, window: str, extra: Sequence[str], amounts: Sequence[str]
    ) -> bootstrap.Totals:
        return bootstrap.block_totals(
            frame,
            [*KEYS, *extra],
            amounts,
            windows.origins[window][0],
            statistics.block_days,
            windows.blocks[window],
        )

    def in_regimes(frame: pl.DataFrame, regimes: Sequence[str]) -> pl.DataFrame:
        chosen = [s for s, r in regime_of.items() if r in regimes]
        return frame.filter(pl.col("split").is_in(pl.Series(chosen, dtype=pl.String).implode()))

    errors = reduced["errors"]
    cv = totals(in_regimes(errors, CV_REGIMES), "period", [], QUANTITIES)
    history = totals(in_regimes(errors, ["history"]), "test", [], QUANTITIES)
    time_arcs = totals(
        in_regimes(reduced["breakdown"], ["time_blocked"]).filter(pl.col("factor") == "arc"),
        "period",
        ["factor", "level"],
        ["sse", "rows"],
    )
    horizons = horizon_groups(splits_of["history"])
    turbines = reduced["turbines"].sort("turbine_index")["turbine"].to_list()
    components = {
        "validation_misleads": validation_components(cv, period, splits_of, protocol, scope),
        "wind_frame_transfers": transfer_components(
            cv, period, splits_of, protocol.endpoints.wind_frame_transfers.pairs, scope
        ),
        "coverage_predicts_failure": coverage_components(
            cv, time_arcs, period, reduced["arcs"], splits_of, confirmation, protocol, scope
        ),
        "early_history_advantage": early_history(
            reduced, history, windows, horizons, turbines, protocol, seed, scope
        ),
    }
    models = sorted(done["model"].unique().to_list())
    groups: dict[tuple[str, int | None], list[str]] = {
        (regime, None): splits_of[regime] for regime in CV_REGIMES if splits_of[regime]
    }
    history_groups: dict[tuple[str, int | None], list[str]] = {
        ("history", days): names for days, names in horizons.items()
    }
    secondary = scope.keep((*SECONDARY, *ABLATIONS))
    gaussian = sorted(errors.filter(pl.col("crps").is_not_null())["model"].unique().to_list())
    interval_totals = totals(
        in_regimes(reduced["intervals"], CV_REGIMES),
        "period",
        ["level"],
        ["rows", "covered", "width"],
    )
    longest = max(protocol.splits.history_horizons_days)
    tables = {
        "hypotheses": hypotheses.hypothesis_rows(
            farm, components, statistics.interval, statistics.alpha, statistics.min_finite
        ),
        "accuracy": pl.concat(
            [
                summaries.accuracy(cv, period, groups, models, REFERENCE),
                summaries.accuracy(history, test, history_groups, models, REFERENCE),
            ]
        ),
        "arcs": summaries.arc_results(cv, period, models, reduced["splits"]),
        "rank_agreement": summaries.rank_agreement(
            cv, period, {r: splits_of[r] for r in RANKED_REGIMES}, models
        ),
        "comparisons": pl.concat(
            [
                summaries.comparisons(cv, period, groups, secondary, SECONDARY_AGAINST, REFERENCE),
                summaries.comparisons(
                    history, test, history_groups, secondary, SECONDARY_AGAINST, REFERENCE
                ),
            ]
        ),
        "mechanism": summaries.deletion_contrasts(
            cv, period, splits_of["arc_opposite"], splits_of["arc_control"], models
        ),
        "coverage_null": coverage_null(
            cv, time_arcs, period, reduced["arcs"], splits_of, protocol, scope
        ),
        "noninferiority": summaries.noninferiority(
            history,
            test,
            SECONDARY_AGAINST,
            scope.keep(protocol.endpoints.validation_misleads.baselines),
            horizons.get(longest, []),
            longest,
            protocol.decision.noninferiority_margin,
        ),
        "calibration": summaries.interval_calibration(
            interval_totals,
            period,
            {r: s for (r, _), s in groups.items()},
            gaussian,
            statistics.coverage_levels,
        ),
        "speed_bin_variance": summaries.speed_bin_variance(errors, regime_of),
        "inflow_pairs": hypotheses.contrast_rows(
            transfer_components(cv, period, splits_of, INFLOW_PAIRS, scope),
            statistics.interval,
            statistics.min_finite,
        ),
        "repair": confirmation_contrast_rows(cv, period, splits_of, statistics, scope),
        "matched_monitoring": hypotheses.contrast_rows(
            matched_monitoring(
                reduced, history, windows, horizons, turbines, protocol, seed, scope
            ),
            statistics.interval,
            statistics.min_finite,
        ),
        "energy_ratio": summaries.energy_ratios(
            reduced["energy_ratio"],
            period,
            windows.origins["period"][0],
            statistics.block_days,
            windows.blocks["period"],
        ),
    }
    return tables | summaries.monitoring_results(
        dict(reduced),
        test,
        windows.drawn["test"],
        windows.origins["test"],
        turbines,
        protocol,
        seed,
    )


def outcomes_of(table: pl.DataFrame) -> dict[str, list[hypotheses.Outcome]]:
    """each hypothesis's farm outcomes, in farm order."""
    per_farm = table.select("farm", "hypothesis", "outcome").unique(["farm", "hypothesis"])
    outcomes: dict[str, list[hypotheses.Outcome]] = defaultdict(list)
    for _, hypothesis, outcome in per_farm.sort("farm").iter_rows():
        outcomes[hypothesis].append(outcome)
    return dict(outcomes)


def decision_table(
    names: Sequence[str],
    development: Mapping[str, Sequence[hypotheses.Outcome]],
    confirmatory: Mapping[str, Sequence[hypotheses.Outcome]] | None,
    protocol: Protocol,
) -> pl.DataFrame:
    """each hypothesis's development part, and with confirmatory outcomes its confirmatory part
    and verdict."""
    rule = protocol.decision
    records = []
    for name in names:
        dev = list(development.get(name, []))
        row = [
            name,
            sum(o in ("success", "failure") for o in dev),
            dev.count("success"),
            hypotheses.development_part(dev, rule.min_development_farms),
        ]
        if confirmatory is None:
            row += [None, None, None, None]
        else:
            conf = list(confirmatory.get(name, []))
            row += [
                sum(o in ("success", "failure") for o in conf),
                conf.count("success"),
                hypotheses.confirmatory_part(conf, rule.min_confirmatory_datasets),
                hypotheses.decision(
                    dev,
                    conf,
                    min_development=rule.min_development_farms,
                    min_confirmatory=rule.min_confirmatory_datasets,
                ),
            ]
        records.append(tuple(row))
    return pl.DataFrame(records, schema=DECISIONS, orient="row")
