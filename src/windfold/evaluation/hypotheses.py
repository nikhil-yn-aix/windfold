"""hypothesis tests: bootstrap p-values, intersection-union, holm and the decision rule.

every component is a benefit, larger is better, so "above zero" reads the same everywhere.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from windfold.evaluation.bootstrap import Totals, seed_mean
from windfold.evaluation.schemas import CONTRASTS, HYPOTHESES
from windfold.monitoring import Grid, recall_and_exceedance

type Outcome = Literal["success", "failure", "not estimable", "not in this sensitivity"]
type Verdict = Literal["supported", "not supported", "not estimable"]
type Status = Literal["not estimable", "not in this sensitivity"]

NOT_ESTIMABLE = "not estimable"
OUT_OF_SCOPE = "not in this sensitivity"
ESTIMATED = "estimated"


def exceedance_benefit[T: (float, np.ndarray)](
    windfold_rate: T, comparator_rate: T, margin: float
) -> T:
    """the exceedance component as a benefit: margin minus windfold's excess exceedance rate."""
    return margin - (windfold_rate - comparator_rate)


def finite(replicates: np.ndarray) -> np.ndarray:
    values = np.asarray(replicates, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap replicates must be finite and not empty")
    return values


def kept(replicates: np.ndarray, min_finite: float) -> np.ndarray:
    """the finite replicates. a replicate whose blocks hold no row of a quantity's window leaves
    the quantity undefined; when fewer than min_finite of all replicates are finite, the estimate
    is not estimable."""
    values = np.asarray(replicates, dtype=float)
    used = values[np.isfinite(values)]
    if used.size == 0 or used.size < min_finite * values.size:
        raise NotEstimable(f"{used.size} of {values.size} replicates are finite")
    return used


def one_sided_p(replicates: np.ndarray) -> float:
    """percentile-bootstrap inversion: (1 + replicates at or below zero) / (replicates + 1)."""
    values = finite(replicates)
    return (1 + int(np.count_nonzero(values <= 0))) / (values.size + 1)


def percentile_interval(replicates: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    lower, upper = np.quantile(finite(replicates), [tail, 1 - tail])
    return float(lower), float(upper)


def intersection_union(p_values: Sequence[float | None]) -> float | None:
    """a hypothesis's p on a farm: its largest component p, none if one is not estimable."""
    if not p_values:
        raise ValueError("a hypothesis needs at least one component")
    if any(p is None for p in p_values):
        return None
    return max(p for p in p_values if p is not None)


def holm(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """holm-adjusted p-values over one farm's hypotheses.

    a hypothesis that is not estimable (none) stays in the family with p = 1, so the family is
    always every hypothesis given, and it stays none in the result.
    """
    names = list(p_values)
    raw = np.array([1.0 if (p := p_values[name]) is None else p for name in names])
    order = np.argsort(raw, kind="stable")
    steps = np.maximum.accumulate((raw.size - np.arange(raw.size)) * raw[order])
    adjusted = np.empty(raw.size)
    adjusted[order] = np.minimum(steps, 1.0)
    return {
        name: None if p_values[name] is None else float(value)
        for name, value in zip(names, adjusted, strict=True)
    }


def farm_outcome(
    lower_bounds: Sequence[float | None], adjusted_p: float | None, alpha: float
) -> Outcome:
    """success when every component interval lies above zero and holm's p is below alpha."""
    if adjusted_p is None or any(bound is None for bound in lower_bounds):
        return "not estimable"
    above = all(bound is not None and bound > 0 for bound in lower_bounds)
    return "success" if above and adjusted_p < alpha else "failure"


def development_part(outcomes: Sequence[Outcome], min_development: int) -> Outcome:
    """success needs at least min_development judged farms with that many successes; a
    hypothesis outside a sensitivity on every farm is not in this sensitivity."""
    if outcomes and all(o == OUT_OF_SCOPE for o in outcomes):
        return OUT_OF_SCOPE
    judged = [o for o in outcomes if o in ("success", "failure")]
    if len(judged) < min_development:
        return NOT_ESTIMABLE
    return "success" if judged.count("success") >= min_development else "failure"


def confirmatory_part(outcomes: Sequence[Outcome], min_confirmatory: int) -> Outcome:
    """success needs success on every judged confirmatory dataset, at least min_confirmatory."""
    if outcomes and all(o == OUT_OF_SCOPE for o in outcomes):
        return OUT_OF_SCOPE
    judged = [o for o in outcomes if o in ("success", "failure")]
    if len(judged) < min_confirmatory:
        return NOT_ESTIMABLE
    return "success" if all(o == "success" for o in judged) else "failure"


def decision(
    development: Sequence[Outcome],
    confirmatory: Sequence[Outcome],
    *,
    min_development: int,
    min_confirmatory: int,
) -> Verdict:
    """one hypothesis across farms: supported when both parts succeed. a part that can be
    judged and fails makes it not supported; otherwise a part that cannot be judged makes it
    not estimable. sensitivities have no verdict, so neither part may be out of scope."""
    parts = (
        development_part(development, min_development),
        confirmatory_part(confirmatory, min_confirmatory),
    )
    if OUT_OF_SCOPE in parts:
        raise ValueError("a verdict needs both parts in scope")
    if "failure" in parts:
        return "not supported"
    return "not estimable" if NOT_ESTIMABLE in parts else "supported"


class NotEstimable(ValueError):
    pass


type Estimate = tuple[float, np.ndarray]
type Component = Estimate | Status


def estimate(
    endpoint: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    counts: np.ndarray,
    seeds: np.ndarray,
) -> Estimate:
    """the observed value (every block once, the mean over seeds) and the bootstrap replicates."""
    observed = endpoint(np.ones((1, counts.shape[1])), None)
    return float(observed[0]), endpoint(counts, seeds)


def total(totals: Totals, counts: np.ndarray, quantity: str, **keys: object) -> np.ndarray:
    columns = totals.columns(quantity, **keys)
    if columns.size == 0:
        raise NotEstimable(f"no {quantity} for {keys}")
    return counts @ totals.values[:, columns].sum(axis=1)


def pooled_mse(
    totals: Totals,
    counts: np.ndarray,
    seeds: np.ndarray | None,
    model: str,
    splits: Sequence[str],
    waked: bool | None = True,
    quantity: str = "sse",
    **keys: object,
) -> np.ndarray:
    """row-weighted mean of a quantity (squared error by default) of a model over splits, on
    waked rows (all rows when waked is none), as the mean over the model's seeds or the
    replicate's drawn seeds."""
    labels = totals.labels.filter(pl.col("model") == model)
    seed_values = labels["seed"].unique().sort(nulls_last=True).to_list()
    if not seed_values:
        raise NotEstimable(f"{model} has no results here")
    if seeds is not None and seed_values != [None] and len(seed_values) != seeds.shape[1]:
        raise NotEstimable(f"{model} has seeds {seed_values}")
    chosen = dict(keys) | ({} if waked is None else {"waked": waked})
    per_seed = []
    for seed in seed_values:
        present = set(labels.filter(pl.col("seed").eq_missing(seed))["split"].to_list())
        if not set(splits) <= present:
            raise NotEstimable(f"{model} seed {seed} lacks {sorted(set(splits) - present)}")
        where = chosen | {"model": model, "split": list(splits), "seed": seed}
        per_seed.append(
            total(totals, counts, quantity, **where) / total(totals, counts, "rows", **where)
        )
    return seed_mean(np.column_stack(per_seed), seeds)


def validation_misleads(
    totals: Totals,
    counts: np.ndarray,
    seeds: np.ndarray | None,
    *,
    baselines: Sequence[str],
    physics: str,
    time_splits: Sequence[str],
    arc_splits: Sequence[str],
) -> np.ndarray:
    """single-arc mse of the baseline with the lowest time-blocked mse, chosen afresh in each
    replicate, minus the tuned physics model's single-arc mse."""
    time = np.column_stack([pooled_mse(totals, counts, seeds, m, time_splits) for m in baselines])
    arc = np.column_stack([pooled_mse(totals, counts, seeds, m, arc_splits) for m in baselines])
    chosen = arc[np.arange(arc.shape[0]), np.argmin(time, axis=1)]
    return chosen - pooled_mse(totals, counts, seeds, physics, arc_splits)


def wind_frame_transfers(
    totals: Totals,
    counts: np.ndarray,
    seeds: np.ndarray | None,
    *,
    identity: str,
    relative: str,
    arc_splits: Sequence[str],
) -> np.ndarray:
    """single-arc mse of the identity side minus the wind-relative side of one matched pair."""
    return pooled_mse(totals, counts, seeds, identity, arc_splits) - pooled_mse(
        totals, counts, seeds, relative, arc_splits
    )


def early_history_accuracy(
    totals: Totals,
    counts: np.ndarray,
    seeds: np.ndarray | None,
    *,
    model: str,
    comparator: str,
    splits: Sequence[str],
) -> np.ndarray:
    """mse on the common test window, each start weighted equally: comparator minus model."""

    def averaged(name: str) -> np.ndarray:
        return np.mean([pooled_mse(totals, counts, seeds, name, [s]) for s in splits], axis=0)

    return averaged(comparator) - averaged(model)


def excess_errors(
    sector: Totals,
    time: Totals,
    counts: np.ndarray,
    *,
    model: str,
    arcs: Sequence[tuple[str, int]],
    time_splits: Sequence[str],
) -> np.ndarray:
    """replicates by arcs: the model's sector mse on each (split, arc) minus its cross-fitted
    time-blocked mse on the same rows (the time totals' arc factor)."""
    columns = []
    for split, arc in arcs:
        sector_mse = pooled_mse(sector, counts, None, model, [split])
        where = {"model": model, "split": list(time_splits), "factor": "arc", "level": arc}
        time_mse = total(time, counts, "sse", waked=True, **where) / total(
            time, counts, "rows", waked=True, **where
        )
        columns.append(sector_mse - time_mse)
    return np.column_stack(columns)


def losses(
    sector: Totals,
    counts: np.ndarray,
    *,
    model: str,
    physics: str,
    splits: Sequence[str],
) -> np.ndarray:
    """replicates by splits: 1 where the model's mse exceeds the physics model's on the split,
    0 where it does not, nan where a replicate drew none of the split's rows."""
    columns = []
    for s in splits:
        relational = pooled_mse(sector, counts, None, model, [s])
        reference = pooled_mse(sector, counts, None, physics, [s])
        defined = np.isfinite(relational) & np.isfinite(reference)
        columns.append(np.where(defined, (relational > reference).astype(float), np.nan))
    return np.column_stack(columns)


def partial_spearman(score: np.ndarray, outcome: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """per replicate (rows of outcome): pearson correlation of the ranks of score and outcome,
    each residualized by least squares with an intercept on the ranks of the covariates.
    raises NotEstimable when the covariates' ranks explain the score's ranks entirely."""
    design = np.column_stack([np.ones(score.size), rankdata(covariates, axis=0)])
    residualise = np.eye(score.size) - design @ np.linalg.pinv(design)
    ranks = rankdata(score)
    fixed = residualise @ ranks
    if np.linalg.norm(fixed) <= 1e-9 * np.linalg.norm(ranks - ranks.mean()):
        raise NotEstimable("the score's ranks are explained by the covariates' ranks")
    moving = rankdata(np.atleast_2d(outcome), axis=1) @ residualise.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return (moving @ fixed) / (np.linalg.norm(moving, axis=1) * np.linalg.norm(fixed))


def auc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """area under the roc curve of scores for each row of labels; nan when a class is empty."""
    above = (scores[:, None] > scores[None, :]) + 0.5 * (scores[:, None] == scores[None, :])
    positive = np.atleast_2d(labels).astype(float)
    negative = 1.0 - positive
    pairs = positive.sum(axis=1) * negative.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.einsum("ri,ij,rj->r", positive, above, negative) / pairs


@dataclass(frozen=True)
class CoverageConfirmation:
    """the coverage logistic map and the simple-predictor choice, fitted on other farms only.

    the map scores an arc by intercept + slope * (score - centre) / spread; the chosen simple
    predictor scores it by direction * its value.
    """

    centre: float
    spread: float
    intercept: float
    slope: float
    predictor: str
    direction: float


def fit_confirmation(
    scores: np.ndarray, predictors: Mapping[str, np.ndarray], labels: np.ndarray
) -> CoverageConfirmation:
    """fit the map by logistic regression on the standardized score, and choose the simple
    predictor (and its sign) with the highest auc, the first listed on a tie."""
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        raise NotEstimable("the fitting farms have arcs of one class only")
    centre, spread = float(scores.mean()), float(scores.std()) or 1.0
    fit = LogisticRegression().fit(((scores - centre) / spread)[:, None], labels)
    best = ("", 1.0, -1.0)
    for name, values in predictors.items():
        area = float(auc(values, labels[None, :])[0])
        direction = 1.0 if area >= 0.5 else -1.0
        if max(area, 1 - area) > best[2]:
            best = (name, direction, max(area, 1 - area))
    return CoverageConfirmation(
        centre, spread, float(fit.intercept_[0]), float(fit.coef_[0, 0]), best[0], best[1]
    )


def auc_gain(
    confirmation: CoverageConfirmation,
    scores: np.ndarray,
    predictors: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    """per replicate: auc of the map minus auc of the chosen simple predictor; zero where a
    replicate has arcs of one class only, nan where it leaves an arc's label undefined."""
    mapped = confirmation.intercept + confirmation.slope * (
        (scores - confirmation.centre) / confirmation.spread
    )
    simple = confirmation.direction * predictors[confirmation.predictor]
    gain = auc(mapped, labels) - auc(simple, labels)
    defined = np.isfinite(np.atleast_2d(labels).astype(float)).all(axis=1)
    return np.where(defined & np.isnan(gain), 0.0, gain)


def coverage_arcs(
    sector: Totals,
    blocks: int,
    *,
    design: pl.DataFrame,
    model: str,
    physics: str,
) -> tuple[list[tuple[str, int]], np.ndarray] | None:
    """the (split, arc) rows of design that coverage predicts failure judges and their observed
    losses to physics, or none when there are no rows or the models have no results on them."""
    if design.is_empty():
        return None
    arcs = list(zip(design["split"].to_list(), design["arc"].to_list(), strict=True))
    splits = [split for split, _ in arcs]
    try:
        observed = losses(sector, np.ones((1, blocks)), model=model, physics=physics, splits=splits)
    except NotEstimable:
        return None
    return arcs, observed[0]


def coverage_predicts_failure(
    sector: Totals,
    time: Totals,
    counts: np.ndarray,
    *,
    design: pl.DataFrame,
    covariates: Sequence[str],
    confirmation: CoverageConfirmation | None,
    model: str,
    physics: str,
    time_splits: Sequence[str],
    min_class: int,
) -> dict[str, Component]:
    """partial spearman of score and excess error, and the auc gain of the coverage map over
    the simple predictor, over the farm's (split, arc) rows of design.

    design holds split, arc, score and the covariates. the gain is not estimable when the
    observed arcs have fewer than min_class in either class, or without a map; either leaves the
    hypothesis not estimable on the farm while the partial spearman is still reported.
    """
    judged = coverage_arcs(sector, counts.shape[1], design=design, model=model, physics=physics)
    if judged is None:
        return dict.fromkeys(("partial_spearman", "auc_gain"), NOT_ESTIMABLE)
    arcs, observed = judged
    both_classes = (
        min(np.count_nonzero(observed == 1), np.count_nonzero(observed == 0)) >= min_class
    )
    splits = [split for split, _ in arcs]
    score = design["score"].to_numpy()
    covariate = design.select(covariates).to_numpy()
    predictors = {name: design[name].to_numpy() for name in covariates}
    fixed = np.zeros((counts.shape[0], 1), int)

    def spearman(c: np.ndarray, _: np.ndarray | None) -> np.ndarray:
        excess = excess_errors(sector, time, c, model=model, arcs=arcs, time_splits=time_splits)
        return partial_spearman(score, excess, covariate)

    def gain(c: np.ndarray, _: np.ndarray | None) -> np.ndarray:
        if not both_classes:
            raise NotEstimable(f"fewer than {min_class} arcs in a class")
        if confirmation is None:
            raise NotEstimable("no coverage map")
        lost = losses(sector, c, model=model, physics=physics, splits=splits)
        return auc_gain(confirmation, score, predictors, lost)

    return {
        "partial_spearman": attempt(lambda: estimate(spearman, counts, fixed)),
        "auc_gain": attempt(lambda: estimate(gain, counts, fixed)),
    }


def attempt(compute: Callable[[], Estimate]) -> Component:
    try:
        return compute()
    except NotEstimable:
        return NOT_ESTIMABLE


def contrast_rows(
    components: Mapping[str, Component], level: float, min_finite: float
) -> pl.DataFrame:
    """one row per contrast outside the decision rule, with its interval, one-sided p and the
    finite replicates behind them, computed as the hypotheses' components are."""
    records = []
    for name, part in components.items():
        if isinstance(part, str):
            records.append((name, part, None, None, None, None, None))
            continue
        value, replicates = part
        try:
            used = kept(replicates, min_finite)
        except NotEstimable:
            records.append((name, NOT_ESTIMABLE, None, None, None, None, None))
            continue
        lower, upper = percentile_interval(used, level)
        records.append((name, ESTIMATED, value, lower, upper, one_sided_p(used), used.size))
    return pl.DataFrame(records, schema=CONTRASTS, orient="row")


def hypothesis_rows(
    farm: str,
    components: Mapping[str, Mapping[str, Component]],
    level: float,
    alpha: float,
    min_finite: float,
) -> pl.DataFrame:
    """one row per component with its status, interval, p and the finite replicates behind them;
    the hypothesis's intersection-union p over its components in scope, its holm-adjusted p and
    the farm outcome. a component with too few finite replicates is not estimable.

    a hypothesis with no component in scope is not in this sensitivity. like a not-estimable
    one, it stays in holm's family with p = 1, so every analysis adjusts over the same family.
    """
    rows: dict[str, dict[str, tuple]] = {}
    hypothesis_p: dict[str, float | None] = {}
    for hypothesis, parts in components.items():
        if not parts:
            raise ValueError(f"{hypothesis} names no components")
        rows[hypothesis] = {}
        for name, part in parts.items():
            if isinstance(part, str):
                rows[hypothesis][name] = (part, None, None, None, None, None)
                continue
            value, replicates = part
            try:
                used = kept(replicates, min_finite)
            except NotEstimable:
                rows[hypothesis][name] = (NOT_ESTIMABLE, None, None, None, None, None)
                continue
            lower, upper = percentile_interval(used, level)
            p = one_sided_p(used)
            rows[hypothesis][name] = (ESTIMATED, value, lower, upper, p, used.size)
        scoped = [row for row in rows[hypothesis].values() if row[0] != OUT_OF_SCOPE]
        hypothesis_p[hypothesis] = intersection_union([r[4] for r in scoped]) if scoped else None
    adjusted = holm(hypothesis_p)
    records = []
    for hypothesis, named in rows.items():
        scoped = [row for row in named.values() if row[0] != OUT_OF_SCOPE]
        outcome = (
            farm_outcome([r[2] for r in scoped], adjusted[hypothesis], alpha)
            if scoped
            else OUT_OF_SCOPE
        )
        records += [
            (farm, hypothesis, name, *row, hypothesis_p[hypothesis], adjusted[hypothesis], outcome)
            for name, row in named.items()
        ]
    return pl.DataFrame(records, schema=HYPOTHESES, orient="row")


def monitor_rates(
    grids: Sequence[Sequence[Grid]],
    copies: Sequence[pl.DataFrame],
    turbines: Sequence[str],
    rebuilt: np.ndarray,
    seeds: np.ndarray | None,
    quantile: float,
    drop: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """a monitor's recall and exceedance per replicate from its grids, one list of start grids
    per seed: each metric found per seed, then averaged over the seeds, or over each replicate's
    drawn seeds, as for accuracy."""
    rates = [
        recall_and_exceedance(own, copies, turbines, rebuilt, quantile, drop, horizon)
        for own in grids
    ]
    if seeds is not None and 1 < len(rates) != seeds.shape[1]:
        raise NotEstimable(f"a monitor has {len(rates)} seeds, the draw {seeds.shape[1]}")
    recall, exceedance = (np.column_stack([rate[part] for rate in rates]) for part in (0, 1))
    return seed_mean(recall, seeds), seed_mean(exceedance, seeds)


def early_history_monitoring(
    grids: Mapping[str, Sequence[Sequence[Grid]]],
    copies: Sequence[pl.DataFrame],
    turbines: Sequence[str],
    slots: np.ndarray,
    seeds: np.ndarray,
    *,
    model: str,
    comparators: Sequence[str],
    quantile: float,
    drop: float,
    horizon: int,
    margin: float,
) -> dict[str, Estimate]:
    """recall gain and exceedance margin of the model's monitor against each comparator's, on
    the same injected copies, the same rebuilt days and the same drawn seeds in every replicate.
    grids holds each monitor's start grids per seed."""
    identity = np.arange(grids[model][0][0][0].shape[1])[None, :]

    def measured(name: str, rebuilt: np.ndarray, drawn: np.ndarray | None):
        return monitor_rates(grids[name], copies, turbines, rebuilt, drawn, quantile, drop, horizon)

    results = {}
    for rebuilt, drawn, label in ((identity, None, "observed"), (slots, seeds, "replicates")):
        own = measured(model, rebuilt, drawn)
        for name in comparators:
            other = measured(name, rebuilt, drawn)
            results.setdefault(f"recall_gain_{name}", {})[label] = own[0] - other[0]
            results.setdefault(f"exceedance_margin_{name}", {})[label] = exceedance_benefit(
                own[1], other[1], margin
            )
    return {
        name: (float(values["observed"][0]), values["replicates"])
        for name, values in results.items()
    }
