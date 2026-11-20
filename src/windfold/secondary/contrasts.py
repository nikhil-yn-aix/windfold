"""what the tests of the confirmation period share: a study's single arcs, the refusal of
confirmatory data before the second freeze, the daily error sums of a model fitted outside the
runner and contrasts scored with the farm's planned bootstrap."""

from collections.abc import Sequence
from functools import partial
from pathlib import Path

import polars as pl

from windfold import config
from windfold.config import folder_of, for_farm
from windfold.evaluation import bootstrap, hypotheses, metrics, paths, reductions
from windfold.evaluation.results import KEYS, REFERENCE, bootstrap_windows, study_windows
from windfold.evaluation.summaries import Draw
from windfold.locations import manifest_file, prepared_folder
from windfold.schemas import SPLIT_SUMMARY
from windfold.tasks.evaluation import read_reductions, second_freeze


def output_folder(runs: Path, study: str, model: str) -> Path:
    return runs / study / "secondary" / model


def prediction_names(farm: str, split: str, seed: int | None) -> tuple[str, str]:
    stem = f"predictions/{farm}/{split}" + ("" if seed is None else f"_seed{seed}")
    return f"{stem}.parquet", f"{stem}.json"


def single_arcs(runs: Path, study: str, farm: str, protocol: config.Protocol) -> list[str]:
    """the study's included single-arc splits of the farm, from its split summary."""
    summary = pl.read_csv(
        runs / paths.summary_file(manifest_file(study, farm, None, protocol)),
        schema=SPLIT_SUMMARY,
    )
    chosen = summary.filter((pl.col("regime") == "arc_single") & (pl.col("status") == "included"))
    return sorted(chosen["split"].to_list())


def confirmatory_ready(runs: Path, study: str, configs: str | None) -> None:
    """refuses confirmatory data until the study's second freeze record exists."""
    if config.load_study(study, folder_of(configs)).period == "confirmatory":
        second_freeze(runs, study)


def model_errors(
    data: Path,
    runs: Path,
    study: str,
    farm: str,
    model: str,
    seeds: Sequence[int | None],
    arcs: Sequence[str],
    protocol: config.Protocol,
) -> pl.DataFrame:
    """a model's daily error sums on every single arc and seed, as the reductions sum the
    runner's models."""
    prepared = pl.read_parquet(
        data / "prepared" / prepared_folder(farm, None, protocol) / "rows.parquet"
    )
    manifest = pl.read_parquet(runs / manifest_file(study, farm, None, protocol))
    out = output_folder(runs, study, model)
    parts = []
    for split in arcs:
        here = reductions.window_rows(prepared, manifest, split, None)
        for seed in seeds:
            table, _ = prediction_names(farm, split, seed)
            if not (out / table).is_file():
                raise FileNotFoundError(f"{out / table} is missing; fit it first")
            scored = metrics.scored(pl.read_parquet(out / table), here)
            parts.append(metrics.errors(scored, farm, model, split, seed))
    return pl.concat(parts)


def mse_difference(cv: bootstrap.Totals, other: str, model: str, arcs: Sequence[str]):
    return lambda counts, drawn: hypotheses.wind_frame_transfers(
        cv, counts, drawn, identity=other, relative=model, arc_splits=arcs
    )


def skill_difference(cv: bootstrap.Totals, other: str, model: str, arcs: Sequence[str]):
    difference = mse_difference(cv, other, model, arcs)
    return lambda counts, drawn: (
        100 * difference(counts, drawn) / hypotheses.pooled_mse(cv, counts, drawn, REFERENCE, arcs)
    )


def fixed_seeds(compute):
    """compute with each replicate on the seeds the totals hold, none redrawn."""
    return lambda counts, _: compute(counts, None)


def contrast_table(
    cv: bootstrap.Totals,
    draw: Draw,
    model: str,
    others: Sequence[str],
    arcs: Sequence[str],
    statistics: config.Statistics,
    redraw: bool = True,
) -> pl.DataFrame:
    """each other model's waked-row mse over arcs minus the model's, with 95% percentile intervals
    and one-sided p-values from the draw, its seed indices redrawn or not; skill is the same
    contrast in points of no_wake's mse on those rows, converted in each replicate, and
    no_wake_mse that denominator."""
    found = {}
    for kind, measure in (("mse", mse_difference), ("skill", skill_difference)):
        components: dict[str, hypotheses.Component] = {}
        for other in others:
            compute = measure(cv, other, model, arcs)
            endpoint = compute if redraw else fixed_seeds(compute)
            components[f"{other}_minus_{model}"] = hypotheses.attempt(
                partial(hypotheses.estimate, endpoint, draw.counts, draw.seeds)
            )
        found[kind] = hypotheses.contrast_rows(
            components, statistics.interval, statistics.min_finite
        )
    reference = hypotheses.pooled_mse(cv, draw.observed(), None, REFERENCE, arcs)
    return (
        found["mse"]
        .join(
            found["skill"].select(
                "component", skill="estimate", skill_lower="lower", skill_upper="upper"
            ),
            on="component",
            how="left",
            maintain_order="left",
        )
        .with_columns(no_wake_mse=pl.lit(float(reference[0])))
    )


def contrasts(
    data: Path,
    runs: Path,
    study: str,
    farm: str,
    model: str,
    seeds: Sequence[int | None],
    tested: Sequence[str],
    described: Sequence[str],
    configs: str | None = None,
    breakdown: bool = False,
) -> pl.DataFrame:
    """each runner model's waked-row mse over the single arcs minus the model's, each seed's mse
    averaged, with 95% percentile intervals and one-sided p-values from the farm's planned
    bootstrap: the same day blocks and seed redraws that scored the planned hypotheses. holm
    runs over the tested contrasts, which are confirmed when their interval lies above zero and
    their adjusted p is below the protocol's alpha; the described ones decide nothing. with
    breakdown, rows follow for each model seed alone (seed) and for each arc with waked test rows
    (split), seeds averaged and alone, scored with the same blocks and tested by none."""
    confirmatory_ready(runs, study, configs)
    folder = folder_of(configs)
    protocol, farm_config, _ = for_farm(farm, None, folder)
    statistics = protocol.statistics
    arcs = single_arcs(runs, study, farm, protocol)
    others = (*tested, *described)
    official = read_reductions(runs, study, farm, None)["errors"].filter(
        pl.col("model").is_in([*others, REFERENCE]) & pl.col("split").is_in(arcs)
    )
    own = model_errors(data, runs, study, farm, model, seeds, arcs, protocol)
    windows = bootstrap_windows(
        study_windows(farm_config, config.load_study(study, folder).period),
        statistics.block_days,
        statistics.replicates,
        len(protocol.seeds.model),
        statistics.interval,
        statistics.min_finite,
        bootstrap.farm_seed(protocol.seeds.study, farm),
    )
    errors = pl.concat([official, own])
    draw = windows.draws["period"]

    def totals(frame: pl.DataFrame) -> bootstrap.Totals:
        return bootstrap.block_totals(
            frame,
            KEYS,
            ["sse", "rows"],
            windows.origins["period"][0],
            statistics.block_days,
            windows.blocks["period"],
        )

    cv = totals(errors)
    rows = contrast_table(cv, draw, model, others, arcs, statistics)
    names = [f"{other}_minus_{model}" for other in tested]
    adjusted = hypotheses.holm(
        {name: p for name, p in zip(rows["component"], rows["p"], strict=True) if name in names}
    )
    table = rows.with_columns(
        holm_p=pl.col("component").replace_strict(adjusted, default=None, return_dtype=pl.Float64),
        farm=pl.lit(farm),
    ).with_columns(confirmed=(pl.col("lower") > 0) & (pl.col("holm_p") < statistics.alpha))
    if not breakdown:
        return table
    waked = own.filter(pl.col("waked"))["split"].unique().sort().to_list()
    parts = [table]
    for seed in seeds:
        alone = totals(errors.filter(pl.col("seed").is_null() | (pl.col("seed") == seed)))
        for chosen in (arcs, *[[arc] for arc in waked]):
            part = contrast_table(alone, draw, model, others, chosen, statistics, redraw=False)
            split = chosen[0] if len(chosen) == 1 else None
            parts.append(part.with_columns(seed=pl.lit(seed, dtype=pl.Int64), split=pl.lit(split)))
    for arc in waked:
        part = contrast_table(cv, draw, model, others, [arc], statistics)
        parts.append(part.with_columns(split=pl.lit(arc)))
    return pl.concat(parts, how="diagonal_relaxed").with_columns(farm=pl.lit(farm))
