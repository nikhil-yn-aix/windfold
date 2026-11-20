"""one job inside its own process: pin it, call its target, then check and hash its outputs.

the runner starts `python -m windfold.runner.worker spec.json`. the worker first waits for one
byte on the spec's gate, which the runner sends once it has recorded the worker's lease; if the
runner dies first, the gate closes empty and the worker stops without running anything. only
the standard library is imported before the process pins itself, so every thread a job starts
inherits the pinning.
"""

import errno
import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def allotment() -> tuple[int, str | None]:
    """the thread count and device the runner gave this job; the device is "cuda:0" when the
    job has a gpu, else None."""
    threads = os.environ.get("OMP_NUM_THREADS")
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else None
    return (int(threads) if threads else len(os.sched_getaffinity(0))), device


def error_kind(error: BaseException) -> str:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(current, torch.OutOfMemoryError):
            return "gpu_out_of_memory"
        if "CUDA out of memory" in str(current) or "CUDA error: out of memory" in str(current):
            return "gpu_out_of_memory"
        if isinstance(current, MemoryError):
            return "out_of_memory"
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return "no_space"
        current = current.__cause__ or current.__context__
    return "error"


def nonfinite_columns(path: Path) -> list[str]:
    import polars as pl

    schema = pl.read_parquet_schema(path)
    floats = [name for name, dtype in schema.items() if dtype.is_float()]
    if not floats:
        return []
    counts = (
        pl.scan_parquet(path)
        .select([(pl.col(n).is_nan() | pl.col(n).is_infinite()).sum().alias(n) for n in floats])
        .collect()
        .row(0, named=True)
    )
    return [f"{name} ({count} values)" for name, count in counts.items() if count]


def outcome(out_dir: Path, outputs: list[str]) -> dict[str, Any]:
    missing = [name for name in outputs if not (out_dir / name).is_file()]
    if missing:
        return {"ok": False, "kind": "missing_outputs", "error": f"no output written: {missing}"}
    records = {}
    nonfinite = {}
    for name in outputs:
        path = out_dir / name
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        records[name] = {"sha256": digest, "bytes": path.stat().st_size}
        if path.suffix == ".parquet" and (columns := nonfinite_columns(path)):
            nonfinite[name] = columns
    if nonfinite:
        return {"ok": False, "kind": "nonfinite", "error": f"non-finite values: {nonfinite}"}
    return {"ok": True, "outputs": records}


def main(spec_file: str) -> int:
    spec = json.loads(Path(spec_file).read_text())
    if os.read(spec["gate"], 1) != b"1":
        print("the runner stopped before it recorded this worker; nothing ran", file=sys.stderr)
        return 1
    os.close(spec["gate"])
    os.sched_setaffinity(0, spec["cpus"])
    out_dir = Path(spec["out_dir"])
    try:
        module, _, name = spec["target"].partition(":")
        function = getattr(importlib.import_module(module), name)
        function(**spec["args"], out_dir=out_dir)
        result = outcome(out_dir, spec["outputs"])
    except Exception as error:
        traceback.print_exc()
        result = {"ok": False, "kind": error_kind(error), "error": traceback.format_exc()}
    if not result["ok"] and result["kind"] in ("missing_outputs", "nonfinite"):
        print(result["error"], file=sys.stderr)
    result_file = Path(spec["result"])
    partial = result_file.with_name(result_file.name + ".partial")
    partial.write_text(json.dumps(result))
    os.replace(partial, result_file)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
