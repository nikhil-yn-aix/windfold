"""tiny job targets for the runner tests, imported by worker processes through PYTHONPATH.

running this file as a module runs the runner on a plan file, so a test can kill it.
"""

import atexit
import errno
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

import polars as pl

from windfold.config import Resources
from windfold.runner import pools
from windfold.runner.jobs import Job
from windfold.runner.pools import RunOptions, run
from windfold.runner.resources import Gpu, Hardware, detect_gpus

FAKE_GPU = Gpu(9, "GPU-00000000-0000-0000-0000-000000000009", "test device", 1.0, "8.6")
HOLD_SECONDS = 30.0


def note(trace: str | None, job: str, what: str, **extra: object) -> None:
    if trace is None:
        return
    record = {"job": job, "what": what, "pid": os.getpid(), "time": time.time(), **extra}
    with open(Path(trace) / "trace.jsonl", "a") as handle:
        handle.write(json.dumps(record) + "\n")


def held(release: str | None, until: float) -> bool:
    """whether a job its test holds until the file release exists still waits; it waits at most
    until the time until."""
    return release is not None and not Path(release).exists() and time.time() < until


def exclusive(trace: str | None, job: str) -> int | None:
    if trace is None:
        return None
    fd = os.open(Path(trace) / f"{job.replace('/', '_')}.running", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        (Path(trace) / f"{job.replace('/', '_')}.concurrent").write_text("")
    return fd


def write(
    job: str,
    value: float,
    output: str,
    out_dir: Path,
    inputs: list[str] | None = None,
    sleep: float = 0.0,
    trace: str | None = None,
    release: str | None = None,
) -> None:
    lock = exclusive(trace, job)
    note(
        trace,
        job,
        "start",
        cpus=sorted(os.sched_getaffinity(0)),
        threads=os.environ.get("OMP_NUM_THREADS"),
        polars=os.environ.get("POLARS_MAX_THREADS"),
        cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    time.sleep(sleep)
    until = time.time() + HOLD_SECONDS
    while held(release, until):
        time.sleep(0.05)
    total = value + sum(pl.read_parquet(path)["mean"][0] for path in inputs or [])
    pl.DataFrame({"step": [0, 1, 2], "mean": [total, total + 1.0, total + 2.0]}).write_parquet(
        out_dir / output, compression="uncompressed"
    )
    note(trace, job, "end")
    if lock is not None:
        os.close(lock)


def fail_first(job: str, output: str, marker: str, out_dir: Path) -> None:
    if not Path(marker).exists():
        Path(marker).write_text("")
        raise RuntimeError(f"{job} fails on its first attempt")
    pl.DataFrame({"mean": [1.0]}).write_parquet(out_dir / output)


def changes_after_hashing(job: str, output: str, out_dir: Path) -> None:
    path = out_dir / output
    pl.DataFrame({"mean": [1.0, 2.0, 3.0]}).write_parquet(path, compression="uncompressed")

    def flip() -> None:
        with open(path, "r+b") as handle:
            handle.seek(path.stat().st_size // 2)
            byte = handle.read(1)
            handle.seek(-1, os.SEEK_CUR)
            handle.write(bytes([byte[0] ^ 0xFF]))

    atexit.register(flip)


def fail_always(job: str, output: str, out_dir: Path) -> None:
    raise ValueError(f"{job} always fails")


def no_space(job: str, output: str, out_dir: Path) -> None:
    with open(out_dir / output, "wb") as handle:
        handle.write(b"PAR1" + bytes(1000))
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(out_dir / output))


def nonfinite(job: str, output: str, out_dir: Path) -> None:
    pl.DataFrame({"mean": [1.0, float("nan")], "variance": [0.1, 0.2]}).write_parquet(
        out_dir / output
    )


def gpu_out_of_memory(job: str, output: str, marker: str, out_dir: Path, trace: str) -> None:
    import torch

    note(trace, job, "start", cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not Path(marker).exists():
        Path(marker).write_text("")
        torch.cuda.set_per_process_memory_fraction(0.01, 0)
        torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda:0")
    value = float(torch.ones(4, device="cuda:0").sum().item())
    pl.DataFrame({"mean": [value]}).write_parquet(out_dir / output)
    note(trace, job, "end")


def load_workers(runs_dir: str) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if "windfold.runner.worker" in command and f"{runs_dir}/load/" in command:
            found.append(int(entry.name))
    return found


def automl_probe(
    job: str,
    output: str,
    runs_dir: str,
    out_dir: Path,
    trace: str,
    sleep: float = 0.0,
    release: str | None = None,
) -> None:
    """the most load workers seen at once, sampled for sleep seconds and until release exists."""
    from windfold.models.automl import gpu_count
    from windfold.runner.worker import allotment

    cuda, gpus = os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_count(allotment()[1])
    note(trace, job, "start", cuda=cuda, gpus=gpus)
    deadline = time.time() + sleep
    until = time.time() + HOLD_SECONDS
    seen = 0
    while time.time() < deadline or held(release, until):
        seen = max(seen, len(load_workers(runs_dir)))
        time.sleep(0.1)
    seen = max(seen, len(load_workers(runs_dir)))
    pl.DataFrame({"mean": [float(seen)]}).write_parquet(out_dir / output)
    note(trace, job, "end", load_workers=seen)


SMALL_FITS = {
    "GBM": {"num_boost_round": 20},
    "CAT": {"iterations": 20},
    "XGB": {"n_estimators": 20},
    "NN_TORCH": {"num_epochs": 2},
    "FASTAI": {"epochs": 2},
}


def marked(base: Path) -> dict[int, tuple[str, int, int, int, str]]:
    """processes whose environment carries WINDFOLD_TEST_MARK=base: state, parent, process
    group, session and command."""
    mark = f"WINDFOLD_TEST_MARK={base}".encode()
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if mark not in (entry / "environ").read_bytes():
                continue
            stat = (entry / "stat").read_text()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()
        if fields[0] != "Z":
            found[int(entry.name)] = (
                fields[0],
                int(fields[1]),
                int(fields[2]),
                int(fields[3]),
                command,
            )
    return found


def autogluon_fit(job: str, output: str, out_dir: Path, trace: str, seconds: float) -> None:
    import numpy as np
    import pandas as pd
    from autogluon.tabular import TabularPredictor
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    from windfold.config import load_protocol
    from windfold.runner.worker import allotment

    settings = load_protocol().automl
    threads, _ = allotment()
    note(trace, job, "start")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 6))
    frame = pd.DataFrame({f"x{k}": x[:, k] for k in range(6)})
    frame["day"] = np.arange(128) % settings.bag_folds
    frame["y"] = x @ rng.standard_normal(6) + 0.1 * rng.standard_normal(128)
    portfolio = get_hyperparameter_config(settings.hyperparameters)
    first = {
        model: portfolio[model][0] if isinstance(portfolio[model], list) else portfolio[model]
        for model in settings.models
    }
    predictor = TabularPredictor(
        label="y",
        problem_type="regression",
        eval_metric="mean_squared_error",
        path=str(out_dir / "predictor"),
        groups="day",
        verbosity=1,
    )
    predictor.fit(
        frame,
        presets=settings.preset,
        hyperparameters={model: {**first[model], **SMALL_FITS[model]} for model in first},
        time_limit=seconds,
        num_bag_folds=settings.bag_folds,
        num_stack_levels=settings.stack_levels,
        dynamic_stacking=False,
        use_bag_holdout=False,
        num_cpus=threads,
        num_gpus=0,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    models = predictor.model_names()
    pl.DataFrame({"mean": [float(len(models))]}).write_parquet(out_dir / output)
    note(trace, job, "end", models=models)


def virtual_hardware(cores: list[list[int]], fake_gpu: bool) -> Hardware:
    """a 64 gb machine with the given cores, and a fake gpu or the visible ones, so the runner
    splits the same machine anywhere, however little memory the real one has."""
    gpus = (FAKE_GPU,) if fake_gpu else detect_gpus()
    return Hardware(tuple(tuple(core) for core in cores), 64.0, 60.0, gpus)


def main(plan_file: str, runs_dir: str, data_dir: str, settings: str) -> None:
    jobs = [Job.model_validate(item) for item in json.loads(Path(plan_file).read_text())]
    resources, options, cores, fake_gpu, die_before = json.loads(settings)
    probe = pools.torch_problem
    pools.detect = lambda: virtual_hardware(cores, fake_gpu)
    pools.torch_problem = lambda gpus: None if gpus == (FAKE_GPU,) else probe(gpus)
    if die_before:
        write = pools.write_json

        def write_or_die(path: Path, record: dict) -> None:
            if path.name.endswith(die_before):
                os.kill(os.getpid(), signal.SIGKILL)
            write(path, record)

        pools.write_json = write_or_die
    run(lambda: jobs, Path(runs_dir), Path(data_dir), Resources(**resources), RunOptions(**options))


if __name__ == "__main__":
    main(*sys.argv[1:])
