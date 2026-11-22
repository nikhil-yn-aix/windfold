"""where the evaluation jobs read and write, relative to the runs folder.

light enough for the planner: it imports only windfold.config and windfold.locations.
"""

from collections.abc import Sequence
from typing import get_args

from windfold.config import Protocol, Regime
from windfold.locations import (
    changes_processing,
    needs_own_manifest,
    prediction_files,
    study_folder,
)

REGIMES: tuple[str, ...] = get_args(Regime)
REDUCTIONS = (
    "errors",
    "intervals",
    "breakdown",
    "energy_ratio",
    "arcs",
    "monitoring_scales",
    "monitoring_days",
    "fits",
    "splits",
    "turbines",
)
RESULTS = (
    "hypotheses",
    "accuracy",
    "arcs",
    "rank_agreement",
    "comparisons",
    "mechanism",
    "coverage_null",
    "noninferiority",
    "calibration",
    "speed_bin_variance",
    "energy_ratio",
    "monitoring",
    "episodes",
    "inflow_pairs",
    "repair",
    "matched_monitoring",
)
DECISION = (
    "decision",
    "hypotheses",
    "accuracy",
    "rank_agreement",
    "monitoring",
    "comparisons",
    "inflow_pairs",
    "repair",
    "matched_monitoring",
)
DEVELOPMENT_STUDY = "development"
SECOND_RECORD = "freeze_second.json"


def statistics_only(protocol: Protocol, sensitivity: str | None) -> bool:
    """a sensitivity that changes only statistics reuses the main study's reductions."""
    if sensitivity is None:
        return False
    overrides = protocol.sensitivities[sensitivity].overrides
    return all(path.startswith("statistics.") for path in overrides)


def own_data(protocol: Protocol, sensitivity: str | None) -> bool:
    """a processing or split sensitivity: its own manifest or prepared rows, so every fit it
    reads is its own."""
    return changes_processing(protocol, sensitivity) or needs_own_manifest(protocol, sensitivity)


def reuses_main_fits(protocol: Protocol, sensitivity: str | None) -> bool:
    """a sensitivity that refits some models on the main data (a kernel or shrinkage change)
    reads every model it does not refit from the main study."""
    if sensitivity is None or statistics_only(protocol, sensitivity):
        return False
    return not own_data(protocol, sensitivity)


def transfer_sensitivities(protocol: Protocol, names: Sequence[str]) -> tuple[str, ...]:
    """the sensitivities that refit windfold on the main data, read beside the main analysis by
    the kernel transfer table."""
    return tuple(
        name
        for name in names
        if reuses_main_fits(protocol, name) and "windfold" in protocol.sensitivities[name].models
    )


def borrowed_regimes(protocol: Protocol, sensitivity: str | None) -> tuple[str, ...]:
    """the regimes whose reductions a split sensitivity's results read from the main study.

    a split sensitivity refits only its own regimes; its other splits are the main study's,
    so its results job also needs the main study's reduction of the same farm.
    """
    if sensitivity is None or changes_processing(protocol, sensitivity):
        return ()
    if not needs_own_manifest(protocol, sensitivity):
        return ()
    refit = set(protocol.sensitivities[sensitivity].regimes)
    return tuple(regime for regime in REGIMES if regime not in refit)


def reduction_source(protocol: Protocol, sensitivity: str | None) -> str | None:
    """the sensitivity whose reductions the results job reads: the main study's (none) for a
    statistics-only sensitivity."""
    return None if statistics_only(protocol, sensitivity) else sensitivity


def reductions_folder(study: str, farm: str, sensitivity: str | None) -> str:
    return f"{study_folder(study, sensitivity)}/reductions/{farm}"


def results_folder(study: str, sensitivity: str | None) -> str:
    return f"{study_folder(study, sensitivity)}/results"


def reduction_outputs(study: str, farm: str, sensitivity: str | None) -> tuple[str, ...]:
    folder = reductions_folder(study, farm, sensitivity)
    return tuple(f"{folder}/{name}.parquet" for name in REDUCTIONS)


def reduction_file(study: str, farm: str, sensitivity: str | None, name: str) -> str:
    if name not in REDUCTIONS:
        raise ValueError(f"{name} is not a reduction; known: {REDUCTIONS}")
    return f"{reductions_folder(study, farm, sensitivity)}/{name}.parquet"


def result_outputs(study: str, farm: str, sensitivity: str | None) -> tuple[str, ...]:
    folder = f"{results_folder(study, sensitivity)}/{farm}"
    return tuple(f"{folder}/{name}.parquet" for name in RESULTS)


def decision_outputs(study: str, sensitivity: str | None) -> tuple[str, ...]:
    folder = results_folder(study, sensitivity)
    return (
        *(f"{folder}/{name}.parquet" for name in DECISION),
        f"{folder}/decision.csv",
        f"{folder}/hypotheses.csv",
    )


def trial_outputs(study: str, farm: str) -> tuple[str, ...]:
    return (f"{results_folder(study, None)}/{farm}/trial_support.parquet",)


def kernel_transfer_outputs(study: str, farm: str) -> tuple[str, ...]:
    return (f"{results_folder(study, None)}/{farm}/kernel_transfer.parquet",)


def second_record_file(study: str) -> str:
    return f"{study}/{SECOND_RECORD}"


def fit_inputs(
    study: str,
    farm: str,
    sensitivity: str | None,
    model: str,
    split: str,
    seed: int | None,
    protocol: Protocol,
) -> tuple[str, str]:
    """the prediction and record files of one fit: a model a sensitivity does not refit on
    the main data is read from the main study."""
    own = sensitivity
    if reuses_main_fits(protocol, sensitivity) and sensitivity is not None:
        if model not in protocol.sensitivities[sensitivity].models:
            own = None
    return prediction_files(study, own, farm, model, split, seed)


def summary_file(manifest: str) -> str:
    return manifest.removesuffix(".parquet") + "_summary.csv"
