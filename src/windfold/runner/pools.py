"""the runner: one per runs and data folder, a dependency-aware queue over cpu and gpu slots.

the planner is called at the start and again every replan interval. new jobs join the queue,
queued jobs that left the plan are dropped, running jobs are never touched and a job whose
fingerprint changed is a new job. every job runs in its own session, pinned, and writes into
a staging folder; its outputs move into place only after it exits cleanly with every output
hashed and finite, and a sidecar per output proves them. a lease per running job names its
process group, so a runner that starts after a kill stops every orphan before anything runs.

the runner's files in the runs folder: run.lock, jobs.jsonl, allocation.json, leases/, load/,
staging/, paused/<family> (delete it to resume that family), logs/run.log, logs/health.log,
logs/load.log and logs/jobs/<job>.log. the data folder holds its own run.lock, and data jobs
stage under its staging/.
leases and staging folders are named by job id with "/" as "+", so no job's folder holds
another's.
"""

import contextlib
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

from windfold.config import Resources
from windfold.runner import health
from windfold.runner.jobs import (
    Job,
    append_event,
    event,
    now,
    read_events,
    read_json,
    sha256,
    sidecar_path,
    start_ticks,
    stat_fields,
    unverified,
    write_json,
)
from windfold.runner.resources import (
    Allocation,
    Gpu,
    Hardware,
    allocate,
    detect,
    torch_problem,
    write_allocation,
)
from windfold.runner.worker import nonfinite_columns

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "POLARS_MAX_THREADS",
)
RESERVED = {
    "runs": {
        "run.lock",
        "jobs.jsonl",
        "allocation.json",
        "leases",
        "load",
        "staging",
        "paused",
        "logs",
    },
    "data": {"run.lock", "staging"},
}
LOAD_TARGET = "windfold.runner.load:fit_forever"
POLL_SECONDS = 0.2
KILL_WAIT_SECONDS = 30.0
VERIFY_THREADS = 8
LOG = logging.getLogger("windfold.runner")


class RunnerBusy(RuntimeError):
    pass


class PlanError(ValueError):
    pass


class OrphanSurvived(RuntimeError):
    pass


class Stopped(Exception):
    pass


@dataclass(frozen=True)
class RunOptions:
    """runner choices; none of them touches a scientific value.

    cpu_jobs caps the cpu slots (None: as many as whole cores allow). job runs that one job
    alone, even when it is done. keep_waiting keeps the runner replanning when nothing can
    run, instead of finishing.
    """

    cpu_jobs: int | None = None
    job: str | None = None
    keep_waiting: bool = False


@dataclass(frozen=True)
class RunSummary:
    done: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[str, ...]
    waiting: tuple[str, ...]


@dataclass
class Device:
    gpu: Gpu
    limit: int
    workers: int = 0
    automl: int = 0
    held_for: str | None = None


@dataclass
class Slot:
    """a cpu slot: the first thread of each of its cores, the job holding each core and the
    load job keeping each idle core busy. a wide job holds every core; narrow jobs hold one
    core each."""

    threads: tuple[int, ...]
    jobs: dict[int, str] = field(default_factory=dict)
    load: dict[int, subprocess.Popen[bytes]] = field(default_factory=dict)

    def open_cores(self) -> list[int]:
        """cores no job holds, idle ones before those that only hold load."""
        cores = [core for core in range(len(self.threads)) if core not in self.jobs]
        return sorted(cores, key=lambda core: core in self.load)


@dataclass
class Attempt:
    job: Job
    number: int
    process: subprocess.Popen[bytes]
    staging: Path
    started: float
    slot: int | None
    slot_cores: tuple[int, ...]
    device: Device | None
    alone: bool
    reservation_gb: float
    where: str
    unsettled: bool = False
    restarting: str | None = None


@dataclass
class Profile:
    seconds: list[float] = field(default_factory=list)
    peak_gb: float = 0.0


def group_alive(pgid: int) -> bool:
    """some thread of the process group is not a zombie. a process whose main thread has exited
    shows as a zombie while its other threads, which SIGKILL can leave stuck, run on."""
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            fields = stat_fields(int(entry.name))
            if fields and int(fields[2]) == pgid and (fields[0] != "Z" or threads_alive(entry)):
                return True
    return False


def threads_alive(process: Path) -> bool:
    try:
        threads = [int(task.name) for task in (process / "task").iterdir()]
    except OSError:
        return False
    for thread in threads:
        fields = stat_fields(thread)
        if fields and fields[0] != "Z":
            return True
    return False


def stop_group(pgid: int, leader_ticks: int | None) -> bool:
    """kill what is left of a job's process group and wait until it is gone; true if anything
    was alive. a leader pid now held by a newer process means the group is already gone."""
    if pgid <= 1 or pgid == os.getpgrp():
        return False
    ticks = start_ticks(pgid)
    if ticks is not None and ticks != leader_ticks:
        return False
    if not group_alive(pgid):
        return False
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + KILL_WAIT_SECONDS
    while group_alive(pgid):
        if time.monotonic() > deadline:
            raise OrphanSurvived(
                f"process group {pgid} is still alive {KILL_WAIT_SECONDS:.0f} s after SIGKILL; "
                f"look at it with ps -g {pgid} and stop it before starting the runner again"
            )
        time.sleep(POLL_SECONDS)
    return True


def thread_ranges(threads: Sequence[int]) -> str:
    ordered = sorted(threads)
    parts: list[str] = []
    start = previous = ordered[0]
    for thread in ordered[1:]:
        if thread != previous + 1:
            parts.append(f"{start}-{previous}" if previous > start else f"{start}")
            start = thread
        previous = thread
    parts.append(f"{start}-{previous}" if previous > start else f"{start}")
    return ",".join(parts)


def child_env(threads: int, device: Device | None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({name: str(threads) for name in THREAD_VARIABLES})
    env["CUDA_VISIBLE_DEVICES"] = device.gpu.uuid if device else ""
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def check_plan(jobs: Sequence[Job], cores_per_cpu_job: int | None = None) -> dict[str, Job]:
    """the jobs by id in plan order, once ids, needs, outputs and the dependency graph check.
    given cores_per_cpu_job, every job must ask for 1 core or that many."""
    plan: dict[str, Job] = {}
    for job in jobs:
        if job.id in plan:
            raise PlanError(f"job {job.id} is planned twice")
        if cores_per_cpu_job is not None and job.cores not in (1, cores_per_cpu_job):
            raise PlanError(
                f"job {job.id} asks for {job.cores} cores; jobs take 1 or {cores_per_cpu_job} cores"
            )
        plan[job.id] = job
    owners: dict[tuple[str, str], str] = {}
    paths = {(job.root, output) for job in plan.values() for output in job.outputs}
    for root, output in paths:
        for parent in PurePosixPath(output).parents[:-1]:
            if (root, str(parent)) in paths:
                raise PlanError(f"{root}/{parent} is planned as an output and as a folder")
    for job in plan.values():
        unknown = [need for need in job.needs if need not in plan]
        if unknown:
            raise PlanError(f"job {job.id} needs {unknown}, which are not in the plan")
        for output in job.outputs:
            if PurePosixPath(output).parts[0] in RESERVED[job.root]:
                raise PlanError(f"job {job.id} writes {output}, a name the runner keeps")
            owner = owners.setdefault((job.root, output), job.id)
            if owner != job.id:
                raise PlanError(f"jobs {owner} and {job.id} both write {job.root}/{output}")
    waiting = {job.id: len(set(job.needs)) for job in plan.values()}
    dependants: dict[str, list[str]] = defaultdict(list)
    for job in plan.values():
        for need in set(job.needs):
            dependants[need].append(job.id)
    free = [job_id for job_id, count in waiting.items() if count == 0]
    while free:
        for dependant in dependants[free.pop()]:
            waiting[dependant] -= 1
            if waiting[dependant] == 0:
                free.append(dependant)
    cycle = sorted(job_id for job_id, count in waiting.items() if count > 0)
    if cycle:
        raise PlanError(f"the needs of these jobs form a cycle: {cycle[:10]}")
    return plan


def detach(command: Sequence[str], log: Path) -> int:
    """start command in its own session with its output in log, so no terminal, chat or
    interrupt can stop it; returns its pid."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            list(command),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return process.pid


def lock_folder(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = read_json(path) or {}
        os.close(fd)
        raise RunnerBusy(
            f"a runner already works on {folder} (pid {holder.get('pid', 'unknown')}, since "
            f"{holder.get('since', 'unknown')}); follow it with windfold status, or stop it first"
        ) from None
    record = json.dumps({"pid": os.getpid(), "ticks": start_ticks(os.getpid()), "since": now()})
    os.ftruncate(fd, 0)
    os.pwrite(fd, record.encode(), 0)
    return fd


def name(job: Job) -> str:
    return job.id.replace("/", " ")


def flat(job_id: str) -> str:
    return job_id.replace("/", "+")


class Runner:
    def __init__(
        self,
        planner: Callable[[], Sequence[Job]],
        runs_dir: Path,
        data_dir: Path,
        resources: Resources,
        options: RunOptions,
    ) -> None:
        self.planner = planner
        self.runs_dir = runs_dir
        self.data_dir = data_dir
        self.resources = resources
        self.options = options
        self.session = now()
        self.log_file = runs_dir / "jobs.jsonl"
        self.plan: dict[str, Job] = {}
        self.order: list[str] = []
        self.done: dict[str, str] = {}
        self.failed: set[tuple[str, str]] = set()
        self.tries: Counter[tuple[str, str]] = Counter()
        self.held: dict[str, int] = {}
        self.running: dict[str, Attempt] = {}
        self.dying: list[subprocess.Popen[bytes]] = []
        self.load_broken = False
        self.load_stops = 0
        self.gpu_problem: str | None = None
        self.paused: set[str] = set()
        self.family_failures: Counter[str] = Counter()
        self.profiles: dict[str, Profile] = defaultdict(Profile)
        self.finished: list[str] = []
        self.skipped: list[str] = []
        self.notes: list[str] = []
        self.devices: list[Device] | None = None
        self.stop_signal: int | None = None
        self.changed = True
        self.hardware: Hardware = detect()
        self.allocation: Allocation = self.split(self.hardware.gpus)
        self.slots = [Slot(threads) for threads in self.allocation.cpu_slots]

    def split(self, gpus: tuple[Gpu, ...]) -> Allocation:
        return allocate(
            self.hardware,
            self.resources,
            cpu_jobs=self.options.cpu_jobs,
            reserved_memory_gb=self.resources.reserved_memory_gb,
            gpus=gpus,
        )

    def folder(self, job: Job) -> Path:
        return self.data_dir if job.root == "data" else self.runs_dir

    def lease(self, job_id: str) -> Path:
        return self.runs_dir / "leases" / f"{flat(job_id)}.lease"

    def staging(self, job: Job) -> Path:
        return self.folder(job) / "staging" / flat(job.id)

    def clear(self, job: Job) -> None:
        for output in job.outputs:
            (self.folder(job) / output).unlink(missing_ok=True)
            sidecar_path(self.folder(job) / output).unlink(missing_ok=True)

    def record(self, job: Job, status: str, **fields: Any) -> None:
        append_event(self.log_file, event(job, status, self.session, **fields))

    def run(self) -> RunSummary:
        self.reclaim()
        self.seed_profiles()
        if self.options.job is None:
            self.replan(strict=True)
        else:
            self.single(self.options.job)
        self.write_allocation()
        for note in self.allocation.notes:
            LOG.info(note)
        next_replan = time.monotonic() + self.resources.replan_seconds
        next_health = time.monotonic()
        while True:
            self.check_stop()
            self.reap()
            if self.options.job is None and time.monotonic() >= next_replan:
                self.replan(strict=False)
                next_replan = time.monotonic() + self.resources.replan_seconds
            self.read_pauses()
            if self.changed:
                self.changed = False
                self.fill()
            self.balance_load()
            if time.monotonic() >= next_health:
                self.heartbeat()
                next_health = time.monotonic() + self.resources.status_interval_seconds
            idle = not self.running and not self.changed
            if idle and (self.options.job is not None or not self.options.keep_waiting):
                if self.options.job is None:
                    self.replan(strict=False)
                    next_replan = time.monotonic() + self.resources.replan_seconds
                    if any(self.runnable(self.plan[job_id]) for job_id in self.order):
                        continue
                break
            time.sleep(POLL_SECONDS)
        self.stop_load()
        self.reap_dying(block=True)
        self.heartbeat()
        return self.summary()

    def check_stop(self) -> None:
        if self.stop_signal is not None:
            raise Stopped(signal.Signals(self.stop_signal).name)

    def reclaim(self) -> None:
        """stop every process group a killed runner left, its load jobs' first, so a job group
        that outlives the kill stops this runner without leaving a load job running; then clear
        the leases and the load and staging folders."""
        leases = self.runs_dir / "leases"
        load = self.runs_dir / "load"
        for path in [*sorted(load.glob("*/lease.json")), *sorted(leases.glob("*.lease"))]:
            record = read_json(path) or {}
            pgid = record.get("pgid")
            if isinstance(pgid, int) and stop_group(pgid, record.get("job_ticks")):
                LOG.warning(f"stopped what was left of {record.get('job')}, process group {pgid}")
        shutil.rmtree(leases, ignore_errors=True)
        shutil.rmtree(load, ignore_errors=True)
        for root in (self.runs_dir, self.data_dir):
            shutil.rmtree(root / "staging", ignore_errors=True)

    def seed_profiles(self) -> None:
        for record in read_events(self.log_file):
            if record.get("status") == "done" and "seconds" in record:
                key = (
                    f"{record['family']}/{record['farm']}" if "farm" in record else record["family"]
                )
                profile = self.profiles[key]
                profile.seconds.append(record["seconds"])
                profile.peak_gb = max(profile.peak_gb, record.get("peak_memory_gb", 0.0))

    def verify(self, jobs: list[Job]) -> list[str | None]:
        with ThreadPoolExecutor(VERIFY_THREADS) as pool:
            return list(pool.map(lambda job: unverified(job, self.folder(job)), jobs))

    def replan(self, strict: bool) -> None:
        started = time.monotonic()
        try:
            plan = check_plan(list(self.planner()), self.resources.cores_per_cpu_job)
        except Exception as error:
            if strict:
                raise
            LOG.error(f"the planner failed, so the current plan stays: {error}")
            self.notes.append(f"the planner failed at {now()}: {error}")
            return
        old = self.plan
        entering = [
            job
            for job in plan.values()
            if job.id not in old or old[job.id].fingerprint != job.fingerprint
        ]
        for job_id, job in old.items():
            if job_id not in plan and self.pending(job) and job_id not in self.running:
                self.record(job, "dropped")
                self.held.pop(job_id, None)
                LOG.info(f"{name(job)} left the plan before it ran")
        self.plan = plan
        index = {job_id: i for i, job_id in enumerate(plan)}
        self.order = sorted(plan, key=lambda job_id: (plan[job_id].priority, index[job_id]))
        for job, reason in zip(entering, self.verify(entering), strict=True):
            if reason is None:
                self.done[job.id] = job.fingerprint
                self.skipped.append(job.id)
                self.record(job, "skipped")
            else:
                self.record(job, "queued", reason=reason)
        if entering:
            LOG.info(
                f"plan of {len(plan)} jobs, {len(entering)} new or changed, "
                f"checked in {time.monotonic() - started:.1f} s"
            )
        self.changed = True

    def single(self, job_id: str) -> None:
        plan = check_plan(list(self.planner()), self.resources.cores_per_cpu_job)
        if job_id not in plan:
            raise PlanError(f"no job {job_id} in the plan")
        job = plan[job_id]
        needs = [plan[need] for need in job.needs]
        for need, reason in zip(needs, self.verify(needs), strict=True):
            if reason is not None:
                raise PlanError(f"job {job_id} needs {need.id}, which is not done: {reason}")
            self.done[need.id] = need.fingerprint
        self.plan = plan
        self.order = [job_id]
        self.record(job, "queued", reason="run alone on request")

    def pending(self, job: Job) -> bool:
        return (
            self.done.get(job.id) != job.fingerprint
            and (job.id, job.fingerprint) not in self.failed
        )

    def ready(self, job: Job) -> bool:
        return (
            self.pending(job)
            and job.id not in self.running
            and all(self.done.get(need) == self.plan[need].fingerprint for need in job.needs)
            and (job.family not in self.paused or self.options.job is not None)
        )

    def runnable(self, job: Job) -> bool:
        return self.ready(job) and not (job.automl and self.automl_waits())

    def automl_waits(self) -> bool:
        """automl fits wait while cpu jobs share cores, since their time budget assumes whole
        cores. with a usable gpu they also wait for an automl seat on one (device_for); without
        one they run on cpu slots like other jobs."""
        self.ensure_devices()
        return self.allocation.shared_cores

    def read_pauses(self) -> None:
        folder = self.runs_dir / "paused"
        paused = {path.name for path in folder.iterdir()} if folder.exists() else set()
        for family in self.paused - paused:
            self.family_failures[family] = 0
            self.load_broken = False
            LOG.info(f"family {family} resumed")
            self.changed = True
        self.paused = paused

    def pause(self, family: str, reason: str) -> None:
        path = self.runs_dir / "paused" / family
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(reason + "\n")
            LOG.warning(f"family {family} paused: {reason}; delete {path} to resume it")
        self.paused.add(family)

    def ensure_devices(self) -> list[Device]:
        if self.devices is None:
            gpus = self.hardware.gpus
            problem = torch_problem(gpus) if gpus else "no gpu is visible"
            if problem is None:
                self.devices = [Device(g, self.resources.gpu_workers_per_device) for g in gpus]
                LOG.info(f"gpu jobs run on {len(gpus)} devices")
            else:
                self.devices = []
                self.gpu_problem = problem
                message = (
                    "gpu jobs run on cpu slots and automl jobs on cpu, where their time budget "
                    f"buys different fits than on a gpu: {problem}"
                )
                self.notes.append(message)
                LOG.warning(message)
            allocation = self.split(tuple(d.gpu for d in self.devices))
            taken = {thread for slot in self.slots for thread in slot.threads}
            self.slots += [Slot(t) for t in allocation.cpu_slots if taken.isdisjoint(t)]
            self.allocation = replace(allocation, cpu_slots=tuple(s.threads for s in self.slots))
            self.write_allocation()
        return self.devices

    def device_for(self, job: Job) -> Device | None:
        """a device with room for job, or for an out-of-memory retry its own device once that
        device has emptied; meanwhile no other job joins it."""
        devices = self.ensure_devices()
        held = next((d for d in devices if d.gpu.index == self.held.get(job.id)), None)
        if held is not None:
            held.held_for = job.id
            return held if held.workers == 0 and held.automl == 0 else None
        open_devices = [d for d in devices if d.held_for is None]
        if job.automl:
            seats = self.resources.automl_jobs_per_device
            room = [d for d in open_devices if d.automl < seats]
            return min(room, key=lambda d: (d.automl, d.workers, d.gpu.index), default=None)
        room = [d for d in open_devices if d.workers < d.limit]
        return min(room, key=lambda d: (d.workers + d.automl, d.gpu.index), default=None)

    def release_holds(self) -> None:
        for device in self.devices or []:
            if device.held_for is None or device.held_for in self.running:
                continue
            job = self.plan.get(device.held_for)
            if job is None or not self.ready(job) or self.held.get(job.id) != device.gpu.index:
                device.held_for = None

    def place(self, job: Job) -> tuple[int, tuple[int, ...]] | None:
        """a slot and cores for a cpu job, or None. a narrow job takes one core, in a slot that
        already holds narrow jobs before a free one, so wide jobs keep finding free slots; a
        wide job takes a whole free slot. idle slots and cores win over ones holding load."""
        free = sorted(
            (k for k, slot in enumerate(self.slots) if not slot.jobs),
            key=lambda k: bool(self.slots[k].load),
        )
        if job.cores > 1:
            return (free[0], tuple(range(len(self.slots[free[0]].threads)))) if free else None
        narrow = [
            k
            for k, slot in enumerate(self.slots)
            if slot.jobs
            and slot.open_cores()
            and self.running[slot.jobs[min(slot.jobs)]].job.cores == 1
        ]
        for k in narrow + free:
            return k, (self.slots[k].open_cores()[0],)
        return None

    def reservation(self, job: Job) -> float:
        profile = self.profiles.get(job.profile_key)
        if profile is None or not profile.peak_gb:
            return job.memory_gb
        return profile.peak_gb * self.resources.memory_margin

    def memory_fits(self, job: Job) -> bool:
        used = sum(attempt.reservation_gb for attempt in self.running.values())
        return not self.running or used + self.reservation(job) <= self.allocation.memory_budget_gb

    def fill(self) -> None:
        self.release_holds()
        for job_id in self.order:
            job = self.plan[job_id]
            if not self.ready(job):
                continue
            devices = self.ensure_devices() if job.gpu or job.automl else []
            if job.automl and self.automl_waits():
                continue
            gpu_worker = job.gpu and not job.automl and bool(devices)
            placement = None if gpu_worker else self.place(job)
            if not gpu_worker and placement is None:
                continue
            device = self.device_for(job) if devices else None
            if devices and device is None:
                continue
            if self.memory_fits(job):
                self.start(job, placement, device)

    def start(
        self, job: Job, placement: tuple[int, tuple[int, ...]] | None, device: Device | None
    ) -> None:
        key = (job.id, job.fingerprint)
        self.tries[key] += 1
        number = self.tries[key]
        self.clear(job)
        shutil.rmtree(self.staging(job), ignore_errors=True)
        staging = self.staging(job) / f"{number}-{uuid.uuid4().hex[:8]}"
        for output in job.outputs:
            (staging / "out" / output).parent.mkdir(parents=True, exist_ok=True)
        slot, cores = placement if placement is not None else (None, ())
        if slot is not None:
            self.stop_load(slot, cores)
            self.slots[slot].jobs.update({core: job.id for core in cores})
            threads = tuple(self.slots[slot].threads[core] for core in cores)
        else:
            threads = self.allocation.gpu_threads
        alone = device is not None and self.held.get(job.id) == device.gpu.index
        if device is not None:
            device.automl += 1 if job.automl else 0
            device.workers += 0 if job.automl else 1
        where = ", ".join(
            [*([f"gpu {device.gpu.index}"] if device else []), f"cpu {thread_ranges(threads)}"]
        )
        process = self.spawn(
            target=job.target,
            args=job.args,
            outputs=list(job.outputs),
            staging=staging,
            threads=threads,
            count=min(job.cores, len(threads)),
            device=device,
            log=self.runs_dir / "logs" / "jobs" / f"{job.id}.log",
            header=f"{job.id} attempt {number} on {where}",
            lease=self.lease(job.id),
            record={
                "job": job.id,
                "attempt": number,
                "where": where,
                "slot": slot,
                "slot_cores": list(cores),
            },
        )
        self.running[job.id] = Attempt(
            job=job,
            number=number,
            process=process,
            staging=staging,
            started=time.monotonic(),
            slot=slot,
            slot_cores=cores,
            device=device,
            alone=alone,
            reservation_gb=self.reservation(job),
            where=where,
        )
        placed = {"slot": slot, "slot_cores": list(cores)} if slot is not None else {}
        self.record(job, "started", attempt=number, device=where, **placed)
        LOG.info(f"{name(job)} started on {where}, attempt {number}")

    def spawn(
        self,
        *,
        target: str,
        args: dict[str, Any],
        outputs: list[str],
        staging: Path,
        threads: Sequence[int],
        count: int,
        device: Device | None,
        log: Path,
        header: str,
        lease: Path,
        record: dict[str, Any],
    ) -> subprocess.Popen[bytes]:
        """start a worker, write its lease, then open its gate. a runner that dies before the
        lease is written leaves the gate closed, and the worker stops without running."""
        (staging / "out").mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        gate, opener = os.pipe()
        try:
            spec = {
                "target": target,
                "args": args,
                "out_dir": str(staging / "out"),
                "cpus": list(threads),
                "outputs": outputs,
                "result": str(staging / "result.json"),
                "gate": gate,
            }
            (staging / "spec.json").write_text(json.dumps(spec))
            with log.open("ab") as handle:
                handle.write(f"\n== {header}, {now()} ==\n".encode())
                handle.flush()
                process = subprocess.Popen(
                    [sys.executable, "-m", "windfold.runner.worker", str(staging / "spec.json")],
                    start_new_session=True,
                    pass_fds=(gate,),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=child_env(count, device),
                )
        except BaseException:
            os.close(opener)
            raise
        finally:
            os.close(gate)
        with os.fdopen(opener, "wb", buffering=0) as release:
            ticks = start_ticks(process.pid)
            stamp = now()
            identity = {"runner_pid": os.getpid(), "pgid": process.pid, "job_ticks": ticks}
            write_json(lease, {**record, **identity, "started": stamp, "heartbeat": stamp})
            release.write(b"1")
        return process

    def reap(self) -> None:
        for attempt in list(self.running.values()):
            pid = attempt.process.pid
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                continue
            if not self.settle(attempt):
                continue
            _, status, usage = os.wait4(pid, 0)
            attempt.process.returncode = os.waitstatus_to_exitcode(status)
            if attempt.restarting is None:
                self.finish(attempt, attempt.process.returncode, usage.ru_maxrss / 2**20)
            else:
                self.requeue(attempt)
        for k, slot in enumerate(self.slots):
            for process in list(dict.fromkeys(slot.load.values())):
                if process.poll() is not None:
                    for core in [c for c, held in slot.load.items() if held is process]:
                        del slot.load[core]
                    self.load_stopped(k)
        self.reap_dying(block=False)

    def settle(self, attempt: Attempt) -> bool:
        """kill what is left of an attempt's process group and wait until it is gone; true once
        it is. while the group outlives the wait, the attempt keeps its cores, staging and lease,
        so a later runner can still stop it, and later calls check again without waiting. the
        group's leader stays a zombie until reaped, so its id cannot be reused."""
        pgid = attempt.process.pid
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + (0.0 if attempt.unsettled else KILL_WAIT_SECONDS)
        while group_alive(pgid):
            if time.monotonic() > deadline:
                if not attempt.unsettled:
                    attempt.unsettled = True
                    message = (
                        f"processes of {name(attempt.job)} outlived SIGKILL, so its cores and "
                        f"lease stay held until they are gone; look at them with ps -g {pgid}"
                    )
                    LOG.error(message)
                    self.notes.append(message)
                return False
            time.sleep(POLL_SECONDS / 4)
        return True

    def load_stopped(self, slot: int) -> None:
        """a load job ended by itself: rerun every full-load fit beside it, which no longer ran
        under full load, and let the slot take a new load job. after failures_before_pause such
        stops, automl families pause."""
        self.load_stops += 1
        tail = self.tail(self.runs_dir / "logs" / "load.log", lines=3)
        cpus = thread_ranges(self.slots[slot].threads)
        LOG.warning(f"the load job on cpu {cpus} stopped by itself: {tail}")
        for attempt in [a for a in self.running.values() if a.job.full_load]:
            self.restart(attempt, f"the load job on cpu {cpus} stopped")
        if self.load_stops >= self.resources.failures_before_pause:
            self.load_broken = True
            for family in sorted({job.family for job in self.plan.values() if job.automl}):
                self.pause(family, f"{self.load_stops} load jobs stopped by themselves: {tail}")
        self.changed = True

    def restart(self, attempt: Attempt, reason: str) -> None:
        """stop a running attempt and queue it again, counted neither as a failure nor a retry;
        an attempt whose processes outlive the kill is queued once they are gone."""
        attempt.restarting = reason
        if self.settle(attempt):
            attempt.process.wait()
            self.requeue(attempt)

    def requeue(self, attempt: Attempt) -> None:
        job = attempt.job
        self.release(attempt)
        self.tries[(job.id, job.fingerprint)] -= 1
        shutil.rmtree(self.staging(job), ignore_errors=True)
        self.lease(job.id).unlink(missing_ok=True)
        self.record(job, "queued", reason=f"restarted: {attempt.restarting}")
        LOG.warning(f"{name(job)} restarted: {attempt.restarting}")

    def release(self, attempt: Attempt) -> None:
        job = attempt.job
        del self.running[job.id]
        if attempt.slot is not None:
            for core in attempt.slot_cores:
                self.slots[attempt.slot].jobs.pop(core, None)
        if attempt.device is not None:
            attempt.device.automl -= 1 if job.automl else 0
            attempt.device.workers -= 0 if job.automl else 1
            if attempt.alone:
                attempt.device.held_for = None
                self.held.pop(job.id, None)
        self.changed = True

    def reap_dying(self, block: bool) -> None:
        for process in list(self.dying):
            if block:
                process.wait()
            if process.poll() is not None:
                self.dying.remove(process)

    def tail(self, log: Path, lines: int = 20) -> str:
        try:
            text = log.read_bytes()[-20000:].decode(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def finish(self, attempt: Attempt, code: int, peak_gb: float) -> None:
        job = attempt.job
        seconds = round(time.monotonic() - attempt.started, 3)
        self.release(attempt)
        result = read_json(attempt.staging / "result.json")
        if code == 0 and result is not None and result.get("ok"):
            problem = self.recheck(job, attempt.staging, result["outputs"])
            if problem is None:
                try:
                    self.promote(job, attempt.staging, result["outputs"])
                except OSError as error:
                    self.clear(job)
                    problem = f"moving the outputs into place failed: {error}"
            if problem is None:
                self.succeed(attempt, seconds, peak_gb)
            else:
                self.fail(attempt, "promotion", problem, seconds)
        else:
            kind, error = self.diagnose(attempt, code, result)
            self.fail(attempt, kind, error, seconds)
        shutil.rmtree(self.staging(job), ignore_errors=True)
        self.lease(job.id).unlink(missing_ok=True)

    def recheck(self, job: Job, staging: Path, records: dict[str, Any]) -> str | None:
        """why the staged outputs differ from what the worker hashed or hold non-finite values,
        or None when they match; called once the job's processes are gone."""
        for output in job.outputs:
            staged = staging / "out" / output
            if not staged.is_file() or sha256(staged) != records[output]["sha256"]:
                return f"{output} changed after it was hashed"
            if staged.suffix == ".parquet" and (columns := nonfinite_columns(staged)):
                return f"{output} holds non-finite values: {columns}"
        return None

    def promote(self, job: Job, staging: Path, records: dict[str, Any]) -> None:
        folder = self.folder(job)
        for output in job.outputs:
            final = folder / output
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / "out" / output, final)
        for index, output in enumerate(job.outputs):
            record = {"job": job.id, "fingerprint": job.fingerprint, **records[output]}
            partial = staging / f"sidecar-{index}.json"
            partial.write_text(json.dumps(record))
            os.replace(partial, sidecar_path(folder / output))

    def succeed(self, attempt: Attempt, seconds: float, peak_gb: float) -> None:
        job = attempt.job
        self.done[job.id] = job.fingerprint
        self.finished.append(job.id)
        profile = self.profiles[job.profile_key]
        profile.seconds.append(seconds)
        profile.peak_gb = max(profile.peak_gb, peak_gb)
        self.record(
            job,
            "done",
            attempt=attempt.number,
            device=attempt.where,
            seconds=seconds,
            peak_memory_gb=round(peak_gb, 3),
        )
        LOG.info(f"{name(job)} done in {seconds:.1f} s, peak memory {peak_gb:.2f} gb")

    def diagnose(
        self, attempt: Attempt, code: int, result: dict[str, Any] | None
    ) -> tuple[str, str]:
        if result is not None and not result.get("ok"):
            return result["kind"], result["error"]
        tail = self.tail(self.runs_dir / "logs" / "jobs" / f"{attempt.job.id}.log")
        if code < 0:
            guess = ", possibly out of memory" if -code == signal.SIGKILL else ""
            return "killed", f"killed by {signal.Signals(-code).name}{guess}\n{tail}"
        return "error", f"exited with code {code} and no result\n{tail}"

    def fail(self, attempt: Attempt, kind: str, error: str, seconds: float) -> None:
        job = attempt.job
        current = self.plan.get(job.id)
        in_plan = current is not None and current.fingerprint == job.fingerprint
        lines = error.strip().splitlines()
        last_line = lines[-1] if lines else kind
        details = {"attempt": attempt.number, "device": attempt.where, "kind": kind}
        if kind == "gpu_out_of_memory" and attempt.device is not None:
            attempt.device.limit = max(1, attempt.device.limit - 1)
            LOG.warning(
                f"gpu {attempt.device.gpu.index} ran out of memory; "
                f"it runs at most {attempt.device.limit} gpu jobs at once from now on"
            )
        if kind == "nonfinite":
            self.pause(job.family, f"{job.id} wrote non-finite values: {last_line}")
        if in_plan and attempt.number < 2:
            if kind == "gpu_out_of_memory" and attempt.device is not None:
                self.held[job.id] = attempt.device.gpu.index
                attempt.device.held_for = job.id
            self.record(job, "retried", **details, seconds=seconds, error=error)
            LOG.warning(f"{name(job)} failed, retrying: {kind}: {last_line}")
            return
        self.failed.add((job.id, job.fingerprint))
        self.held.pop(job.id, None)
        self.record(job, "failed", **details, seconds=seconds, error=error)
        LOG.error(f"{name(job)} failed: {kind}: {last_line}")
        self.family_failures[job.family] += 1
        failures = self.family_failures[job.family]
        done = sum(
            1
            for other in self.plan.values()
            if other.family == job.family and self.done.get(other.id) == other.fingerprint
        )
        finished = done + failures
        fraction = self.resources.failure_fraction
        if failures >= self.resources.failures_before_pause and failures > fraction * finished:
            self.pause(
                job.family, f"{failures} of {finished} finished jobs failed, over {fraction:.0%}"
            )

    def balance_load(self) -> None:
        """while a full-load fit runs, keep every core busy: a load job across each slot with
        no job, and one on each idle core of a slot holding narrow jobs."""
        if self.load_broken or not any(a.job.full_load for a in self.running.values()):
            self.stop_load()
            return
        for k, slot in enumerate(self.slots):
            idle = [core for core in slot.open_cores() if core not in slot.load]
            if not slot.jobs and len(idle) == len(slot.threads):
                self.start_load(k, tuple(idle))
            else:
                for core in idle:
                    self.start_load(k, (core,))

    def start_load(self, k: int, cores: tuple[int, ...]) -> None:
        slot = self.slots[k]
        threads = tuple(slot.threads[core] for core in cores)
        staging = self.runs_dir / "load" / f"{k}-{cores[0]}-{uuid.uuid4().hex[:8]}"
        process = self.spawn(
            target=LOAD_TARGET,
            args={
                "rows": self.resources.load_rows,
                "features": self.resources.load_features,
                "seed": k * len(slot.threads) + cores[0],
            },
            outputs=[],
            staging=staging,
            threads=threads,
            count=len(threads),
            device=None,
            log=self.runs_dir / "logs" / "load.log",
            header=f"load job on cpu {thread_ranges(threads)}",
            lease=staging / "lease.json",
            record={"job": "load", "slot": k, "slot_cores": list(cores)},
        )
        slot.load.update({core: process for core in cores})
        LOG.info(f"load job started on cpu {thread_ranges(threads)}")

    def stop_load(self, k: int | None = None, cores: Sequence[int] = ()) -> None:
        """stop the load jobs on the given cores of slot k, or every load job."""
        for index, slot in enumerate(self.slots):
            if k is not None and index != k:
                continue
            held = [slot.load[core] for core in cores if core in slot.load] if cores else []
            for process in list(dict.fromkeys(held if k is not None else slot.load.values())):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                self.dying.append(process)
                freed = [core for core, owner in slot.load.items() if owner is process]
                for core in freed:
                    del slot.load[core]
                threads = [slot.threads[core] for core in freed]
                LOG.info(f"load job on cpu {thread_ranges(threads)} stopped")
        if not any(slot.load for slot in self.slots):
            shutil.rmtree(self.runs_dir / "load", ignore_errors=True)

    def heartbeat(self) -> None:
        for job_id in self.running:
            path = self.lease(job_id)
            record = read_json(path)
            if record is not None:
                write_json(path, {**record, "heartbeat": now()})
        self.write_allocation()
        health.append_health(self.runs_dir, self.resources.stall_minutes, self.notes)
        self.notes = []

    def write_allocation(self) -> None:
        profiles = {
            key: {
                "jobs": len(profile.seconds),
                "median_seconds": round(median(profile.seconds), 3),
                "peak_memory_gb": round(profile.peak_gb, 3),
                "reservation_gb": round(profile.peak_gb * self.resources.memory_margin, 3),
            }
            for key, profile in self.profiles.items()
            if profile.seconds
        }
        write_allocation(
            self.runs_dir / "allocation.json",
            self.hardware,
            self.allocation,
            profiles,
            self.gpu_problem,
        )

    def stop_all(self) -> None:
        for attempt in list(self.running.values()):
            job = attempt.job
            del self.running[job.id]
            if not self.settle(attempt):
                LOG.warning(f"{name(job)} keeps its lease, so the next runner stops what is left")
                continue
            attempt.process.wait()
            shutil.rmtree(self.staging(job), ignore_errors=True)
            self.lease(job.id).unlink(missing_ok=True)
            LOG.warning(f"{name(job)} stopped with the runner")
        self.stop_load()
        self.reap_dying(block=True)

    def summary(self) -> RunSummary:
        failed = sorted({job_id for job_id, _ in self.failed})
        waiting = [job_id for job_id in self.order if self.pending(self.plan[job_id])]
        LOG.info(
            f"runner finished: {len(self.finished)} done, {len(self.skipped)} already done, "
            f"{len(failed)} failed, {len(waiting)} not run"
        )
        return RunSummary(
            done=tuple(self.finished),
            skipped=tuple(self.skipped),
            failed=tuple(failed),
            waiting=tuple(waiting),
        )


def run(
    planner: Callable[[], Sequence[Job]],
    runs_dir: Path,
    data_dir: Path,
    resources: Resources,
    options: RunOptions | None = None,
) -> RunSummary:
    """run the jobs the planner returns, in dependency order, until nothing more can run.

    one runner per runs folder and per data folder; a second one raises RunnerBusy. the planner
    is called again every replan interval. jobs whose outputs verify are skipped. sigterm or
    sigint stops every running job and raises Stopped.
    """
    runs_dir = Path(runs_dir).resolve()
    data_dir = Path(data_dir).resolve()
    (runs_dir / "logs").mkdir(parents=True, exist_ok=True)
    locks = [lock_folder(runs_dir)]
    if data_dir != runs_dir:
        try:
            locks.append(lock_folder(data_dir))
        except RunnerBusy:
            os.close(locks[0])
            raise
    handler = logging.FileHandler(runs_dir / "logs" / "run.log")
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S UTC")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    runner: Runner | None = None
    previous: dict[int, Any] = {}

    def request_stop(signum: int, frame: object) -> None:
        if runner is not None:
            runner.stop_signal = signum

    try:
        if threading.current_thread() is threading.main_thread():
            previous = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
        runner = Runner(planner, runs_dir, data_dir, resources, options or RunOptions())
        LOG.info(f"runner started, pid {os.getpid()}")
        return runner.run()
    except Stopped as stop:
        LOG.warning(f"runner stopped by {stop}")
        raise
    finally:
        if runner is not None:
            runner.stop_all()
        for signum, before in previous.items():
            signal.signal(signum, before)
        LOG.removeHandler(handler)
        handler.close()
        for lock in locks:
            os.close(lock)
