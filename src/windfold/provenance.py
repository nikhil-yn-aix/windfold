"""the provenance of a refit made outside the runner: a digest of its code, the settings it used
and its inputs, which its record carries, so a refit made otherwise is never reused, nor
overwritten without a word."""

from collections.abc import Callable, Iterable
from pathlib import Path

from windfold import inflow
from windfold.config import Protocol
from windfold.fingerprints import digest, function_hash
from windfold.locations import kernel_folder, manifest_file, prepared_folder
from windfold.runner.jobs import SIDECAR, proper_sidecar, read_json
from windfold.study import KERNEL_FILES, PREPARED_FILES

PREPARED = (
    *PREPARED_FILES,
    "physics_fit.json",
    "physics.parquet",
    *inflow.TABLES.values(),
    inflow.RECORD,
)
DESIGN = tuple(f"{sub}{file}" for sub in ("", "permuted/") for file in KERNEL_FILES)


def fit_inputs(data: Path, runs: Path, study: str, farm: str, protocol: Protocol) -> list[Path]:
    """the runner outputs a refit on a study's own data rests on, named whether or not their
    sidecars exist: the farm's manifest, its prepared rows, physics and inflow tables and its
    kernel designs."""
    prepared = data / "prepared" / prepared_folder(farm, None, protocol)
    design = data / "prepared" / kernel_folder(farm, None, protocol)
    return [
        runs / manifest_file(study, farm, None, protocol),
        *(prepared / name for name in PREPARED),
        *(design / name for name in DESIGN),
    ]


def provenance(fit: Callable[..., object], inputs: Iterable[Path], *settings: object) -> str:
    """the digest of a refit: the code of the function that makes it (fingerprints.function_hash),
    the configuration values, seed and split in settings and the sha256 of each input that its
    runner sidecar holds."""
    hashes = []
    for path in inputs:
        proof = read_json(path.with_name(path.name + SIDECAR))
        if proof is None or not proper_sidecar(proof):
            raise FileNotFoundError(f"{path} has no runner sidecar to say what it holds")
        hashes.append(proof["sha256"])
    target = f"{fit.__module__}:{fit.__name__}"
    return digest(target, function_hash(target), hashes, *settings)


def reusable(record: Path, made: Callable[[], str], refit: str, output: Path | None = None) -> bool:
    """whether a refit can be reused: false while output, by default its record, is missing, and
    true when the record carries the provenance made gives. a refit with another provenance or
    none raises instead, with refit saying how to make it again."""
    found = output or record
    if not found.is_file():
        return False
    held = (read_json(record) or {}).get("provenance")
    if held != made():
        why = "has no provenance" if held is None else "was made by other code, settings or inputs"
        raise ValueError(f"{found} {why}; {refit}")
    return True
