"""where each output lives, relative to the data or runs folder."""

from windfold.config import Protocol

INSPECT_FILES = (
    "raw_week.pdf",
    "status_mix.pdf",
    "power_curves.pdf",
    "direction_offsets.pdf",
    "speed_drift.pdf",
    "wake_dips.pdf",
    "rows_dropped.csv",
    "rows_kept.csv",
)


def inspect_files(farm: str) -> tuple[str, ...]:
    return tuple(f"inspect/{farm}/{name}" for name in INSPECT_FILES)


def prepared_name(farm: str, sensitivity: str | None) -> str:
    return f"{farm}/{sensitivity}" if sensitivity else farm


def changes_processing(protocol: Protocol, sensitivity: str | None) -> bool:
    return sensitivity is not None and protocol.sensitivities[sensitivity].changes_processing


def changes_kernel(protocol: Protocol, sensitivity: str | None) -> bool:
    if sensitivity is None:
        return False
    return any(path.startswith("kernel.") for path in protocol.sensitivities[sensitivity].overrides)


def needs_own_manifest(protocol: Protocol, sensitivity: str | None) -> bool:
    if sensitivity is None:
        return False
    prefixes = ("processing.", "splits.", "farms.")
    overrides = protocol.sensitivities[sensitivity].overrides
    return any(path.startswith(prefixes) for path in overrides)


def prepared_folder(farm: str, sensitivity: str | None, protocol: Protocol) -> str:
    """the prepared rows, physics and layout a sensitivity's fits read."""
    own = changes_processing(protocol, sensitivity)
    return prepared_name(farm, sensitivity if own else None)


def kernel_folder(farm: str, sensitivity: str | None, protocol: Protocol) -> str:
    """the shared-kernel design a sensitivity's fits read."""
    own = changes_processing(protocol, sensitivity) or changes_kernel(protocol, sensitivity)
    return f"{prepared_name(farm, sensitivity if own else None)}/kernel_design"


def study_folder(study: str, sensitivity: str | None) -> str:
    return f"{study}/sensitivities/{sensitivity}" if sensitivity else study


def manifest_file(study: str, farm: str, sensitivity: str | None, protocol: Protocol) -> str:
    own = needs_own_manifest(protocol, sensitivity)
    return f"{study_folder(study, sensitivity if own else None)}/manifests/{farm}.parquet"


def fit_name(model: str, split: str, seed: int | None) -> str:
    return f"{model}/{split}" if seed is None else f"{model}/{split}_seed{seed}"


def prediction_files(
    study: str, sensitivity: str | None, farm: str, model: str, split: str, seed: int | None
) -> tuple[str, str]:
    stem = f"{study_folder(study, sensitivity)}/predictions/{farm}/{fit_name(model, split, seed)}"
    return f"{stem}.parquet", f"{stem}.json"


def tuning_file(farm: str, model: str, index: int) -> str:
    return f"calibration/tuning/{farm}/{model}/{index:02d}.json"


def search_file(farm: str, model: str) -> str:
    return f"calibration/tuning/{farm}/{model}/search.parquet"


def chosen_file(farm: str, model: str) -> str:
    return f"calibration/tuning/{farm}/{model}/chosen.json"


def pretraining_files(farm: str) -> tuple[str, str, str]:
    folder = f"calibration/pretraining/{farm}"
    return f"{folder}/network.pt", f"{folder}/curve.parquet", f"{folder}/record.json"


def budget_file(farm: str) -> str:
    return f"calibration/automl_budget/{farm}/choice.json"


def curve_file(
    study: str, sensitivity: str | None, farm: str, model: str, split: str, seed: int | None
) -> str:
    return (
        f"{study_folder(study, sensitivity)}/curves/{farm}/{fit_name(model, split, seed)}.parquet"
    )


def automl_file(study: str, sensitivity: str | None, farm: str, split: str) -> str:
    return f"{study_folder(study, sensitivity)}/automl/{farm}/{split}.json"
