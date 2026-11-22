import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from windfold.config import Resources
from windfold.runner import pools
from windfold.runner.jobs import Job, read_events
from windfold.runner.pools import RunOptions, RunSummary, group_alive, run
from windfold.runner.resources import Gpu, detect_gpus, physical_cores

HERE = Path(__file__).parent
MARK = "WINDFOLD_TEST_MARK"
VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
RESOURCES = Resources(
    data_dir="data",
    runs_dir="runs",
    reserved_threads=2,
    cores_per_cpu_job=1,
    gpu_workers_per_device=2,
    automl_jobs_per_device=2,
    status_interval_seconds=60,
    stall_minutes=20,
    replan_seconds=120,
    failure_fraction=0.02,
    failures_before_pause=3,
    memory_margin=1.25,
    reserved_memory_gb=8,
    load_rows=20_000,
    load_features=12,
)


def virtual_cores(count: int = 8) -> list[list[int]]:
    """count cores for the runner to see on any machine: the last real cores this process may
    use, each reused in turn when there are fewer than count."""
    real = physical_cores(os.sched_getaffinity(0))[-count:]
    return [list(real[k % len(real)]) for k in range(count)]


class Bench:
    def __init__(self, base: Path) -> None:
        self.cores = virtual_cores()
        self.fake_gpu = True
        self.base = base
        self.runs = base / "runs"
        self.data = base / "data"
        self.trace = base / "trace"
        self.trace.mkdir(parents=True)
        self.made: dict[str, Job] = {}

    def folder(self, root: str) -> Path:
        return self.data if root == "data" else self.runs

    def output(self, job_id: str) -> Path:
        job = self.made[job_id]
        return self.folder(job.root) / job.outputs[0]

    def job(
        self,
        job_id: str,
        target: str = "write",
        *,
        root: str = "runs",
        needs: Sequence[str] = (),
        value: float = 1.0,
        sleep: float = 0.0,
        release: Path | None = None,
        args: dict[str, Any] | None = None,
        fingerprint: str | None = None,
        **fields: Any,
    ) -> Job:
        """a job of fixture_jobs; a write job held by release goes on once that file exists."""
        output = f"{job_id}.parquet"
        if target == "write":
            inputs = [str(self.output(need)) for need in needs]
            args = {"value": value, "sleep": sleep, "trace": str(self.trace), "inputs": inputs}
            args |= {"release": str(release)} if release else {}
        job = Job.model_validate(
            {
                "id": job_id,
                "target": f"fixture_jobs:{target}",
                "args": {"job": job_id, "output": output, **(args or {})},
                "needs": tuple(needs),
                "outputs": (output,),
                "root": root,
                "fingerprint": fingerprint or f"made-{job_id}",
                "cores": 1,
                "memory_gb": 0.5,
                "family": "fixture",
                **fields,
            }
        )
        self.made[job_id] = job
        return job

    def run(
        self,
        plan: Sequence[Job] | Callable[[], Sequence[Job]],
        resources: dict[str, Any] | None = None,
        **options: Any,
    ) -> RunSummary:
        planner = plan if callable(plan) else (lambda: plan)
        settings = RESOURCES.model_copy(update=resources or {})
        return run(
            planner, self.runs, self.data, settings, RunOptions(**{"cpu_jobs": 5, **options})
        )

    def runner(self, plan: Sequence[Job]) -> pools.Runner:
        """a runner over plan for a test to step through by hand."""
        self.runs.mkdir(parents=True, exist_ok=True)
        return pools.Runner(lambda: plan, self.runs, self.data, RESOURCES, RunOptions(cpu_jobs=5))

    def events(self, session: int = -1) -> list[dict[str, Any]]:
        records = read_events(self.runs / "jobs.jsonl")
        sessions = list(dict.fromkeys(record["session"] for record in records))
        return [record for record in records if record["session"] == sessions[session]]

    def statuses(self, job_id: str, session: int = -1) -> list[str]:
        return [e["status"] for e in self.events(session) if e["job"] == job_id]

    def traced(self) -> list[dict[str, Any]]:
        path = self.trace / "trace.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        return [json.loads(line) for line in lines]

    def run_log(self) -> str:
        return (self.runs / "logs" / "run.log").read_text()

    def start_runner(
        self,
        plan: Sequence[Job],
        runs: Path | None = None,
        resources: dict[str, Any] | None = None,
        die_before: str | None = None,
        **options: Any,
    ) -> subprocess.Popen[bytes]:
        """the runner in its own process, so a test can kill it; die_before makes it kill
        itself just before it writes a lease file whose name ends that way."""
        plan_file = self.base / "plan.json"
        plan_file.write_text(json.dumps([job.model_dump(mode="json") for job in plan]))
        chosen = RESOURCES.model_copy(update=resources or {}).model_dump()
        options = {"cpu_jobs": 5, **options}
        settings = json.dumps([chosen, options, self.cores, self.fake_gpu, die_before])
        command = [sys.executable, "-m", "fixture_jobs", str(plan_file), str(runs or self.runs)]
        return subprocess.Popen([*command, str(self.data), settings], start_new_session=True)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = [str(HERE), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in paths if p))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv(MARK, str(tmp_path))
    monkeypatch.syspath_prepend(str(HERE))
    from fixture_jobs import FAKE_GPU, virtual_hardware

    made = Bench(tmp_path)
    probe = pools.torch_problem
    monkeypatch.setattr(pools, "detect", lambda: virtual_hardware(made.cores, made.fake_gpu))
    monkeypatch.setattr(pools, "torch_problem", lambda g: None if g == (FAKE_GPU,) else probe(g))
    yield made
    strays = stray_groups(tmp_path)
    for group in strays:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    deadline = time.monotonic() + 30
    while any(group_alive(group) for group in strays) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not strays, f"process groups left running, now stopped: {sorted(strays)}"


def stray_groups(base: Path) -> set[int]:
    """process groups of live processes that a test under base started: their environment
    carries its mark or their command line names a path under base."""
    mark = f"{MARK}={base}".encode()
    groups = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
            command = (entry / "cmdline").read_bytes()
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()
        if (mark in environ or str(base).encode() in command) and fields[0] != "Z":
            groups.add(int(fields[2]))
    return groups - {os.getpgrp()}


@pytest.fixture
def real_gpu(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> Gpu:
    """the first gpu this process may use, the only device the runner then sees."""
    if shutil.which("nvidia-smi") is None:
        pytest.skip("needs a gpu: nvidia-smi is missing")
    import torch

    if torch.version.cuda is None:
        pytest.skip("needs a cuda build of torch")
    with monkeypatch.context() as scoped:
        if VISIBLE_DEVICES is None:
            scoped.delenv("CUDA_VISIBLE_DEVICES")
        else:
            scoped.setenv("CUDA_VISIBLE_DEVICES", VISIBLE_DEVICES)
        gpus = detect_gpus()
    if not gpus:
        pytest.skip("needs a gpu this process may use")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(gpus[0].index))
    bench.fake_gpu = False
    return gpus[0]
