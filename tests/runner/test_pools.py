import contextlib
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from windfold.runner import pools
from windfold.runner.health import status
from windfold.runner.jobs import (
    failed_for_good,
    read_events,
    read_json,
    sidecar_path,
    stat_fields,
    unverified,
)
from windfold.runner.pools import PlanError, RunnerBusy, check_plan


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for(condition, seconds: float = 60.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


@contextlib.contextmanager
def release_when(gate: Path, condition: Callable[[], bool]) -> Iterator[None]:
    """create gate, which the jobs held on it wait for, once condition holds."""
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set() and not condition():
            time.sleep(0.05)
        gate.touch()

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join()


def most_at_once(spans: list[tuple[datetime, datetime]]) -> int:
    """the most spans open at one time; a span that ends as another starts does not overlap it."""
    edges = sorted([(end, -1) for _, end in spans] + [(start, 1) for start, _ in spans])
    open_now = most = 0
    for _, step in edges:
        open_now += step
        most = max(most, open_now)
    return most


def interval(bench, job_id: str, attempt: int = 1) -> tuple[datetime, datetime]:
    records = [e for e in bench.events() if e["job"] == job_id and e.get("attempt") == attempt]
    start = next(e["time"] for e in records if e["status"] == "started")
    end = next(e["time"] for e in records if e["status"] in ("done", "failed", "retried"))
    return datetime.fromisoformat(start), datetime.fromisoformat(end)


def diamond(bench, b_fingerprint: str = "made-dev/b"):
    a = bench.job("prepared/a", root="data", value=1.0, sleep=0.3)
    b = bench.job("dev/b", needs=["prepared/a"], value=10.0, fingerprint=b_fingerprint)
    c = bench.job("dev/c", needs=["prepared/a"], value=100.0)
    d = bench.job(
        "dev/d",
        needs=["dev/b", "dev/c"],
        value=1000.0,
        fingerprint=f"made-dev/d+{b.fingerprint}+{c.fingerprint}",
    )
    return [a, b, c, d]


def test_jobs_run_after_their_needs_pinned_proven_and_again_only_where_changed(bench):
    plan = diamond(bench)
    summary = bench.run(plan)
    assert sorted(summary.done) == sorted(job.id for job in plan)
    traced = bench.traced()
    start = {r["job"]: r["time"] for r in traced if r["what"] == "start"}
    end = {r["job"]: r["time"] for r in traced if r["what"] == "end"}
    for job in plan:
        assert all(start[job.id] >= end[need] for need in job.needs), job.id
    assert pl.read_parquet(bench.output("dev/d"))["mean"].to_list() == [1112.0, 1113.0, 1114.0]
    assert (bench.data / "prepared/a.parquet").exists()
    for job in plan:
        assert unverified(job, bench.folder(job.root)) is None
    slots = {r["job"]: r["cpus"] for r in traced if r["what"] == "start"}
    assert all(len(cpus) == 1 for cpus in slots.values())
    started = [r for r in traced if r["what"] == "start"]
    seen = {(r["threads"], r["polars"], r["cuda"], r["cublas"]) for r in started}
    assert seen == {("1", "1", "", ":4096:8")}
    again = bench.run(plan)
    assert again.done == () and sorted(again.skipped) == sorted(job.id for job in plan)
    assert {e["status"] for e in bench.events()} == {"skipped"}
    changed = bench.run(diamond(bench, b_fingerprint="made-dev/b-changed"))
    assert sorted(changed.done) == ["dev/b", "dev/d"]
    assert sorted(changed.skipped) == ["dev/c", "prepared/a"]
    assert (
        json.loads(sidecar_path(bench.output("dev/b")).read_text())["fingerprint"]
        == "made-dev/b-changed"
    )


def test_killed_runner_resumes_to_identical_outputs_without_orphans(bench, tmp_path):
    reference = type(bench)(tmp_path / "reference")
    reference_plan = [
        reference.job("dev/a", value=1.0),
        reference.job("dev/b", needs=["dev/a"], value=2.0),
        reference.job("dev/c", needs=["dev/b"], value=3.0),
        reference.job("dev/d", value=4.0),
    ]
    reference.run(reference_plan)
    plan = [
        bench.job("dev/a", value=1.0),
        bench.job("dev/b", needs=["dev/a"], value=2.0, sleep=3.0),
        bench.job("dev/c", needs=["dev/b"], value=3.0),
        bench.job("dev/d", value=4.0),
    ]
    runner = bench.start_runner(plan)
    wait_for(lambda: any(r["job"] == "dev/b" for r in bench.traced()))
    runner.send_signal(signal.SIGKILL)
    runner.wait()
    orphan = next(r["pid"] for r in bench.traced() if r["job"] == "dev/b")
    os.kill(orphan, 0)
    bench.run(plan)
    with pytest.raises(ProcessLookupError):
        os.kill(orphan, 0)
    traced = bench.traced()
    assert not any(r["pid"] == orphan and r["what"] == "end" for r in traced)
    ends = [r["job"] for r in traced if r["what"] == "end"]
    assert sorted(ends) == ["dev/a", "dev/b", "dev/c", "dev/d"]
    assert not list(bench.trace.glob("*.concurrent"))
    assert "stopped what was left of dev/b" in bench.run_log()
    for job_id in ("dev/a", "dev/b", "dev/c", "dev/d"):
        assert digest(bench.output(job_id)) == digest(reference.output(job_id))


def test_changed_finite_value_fails_its_checksum_and_reruns(bench):
    plan = [bench.job("dev/a", value=1.0), bench.job("dev/b", value=2.0)]
    bench.run(plan)
    path = bench.output("dev/a")
    original = digest(path)
    size = path.stat().st_size
    frame = pl.read_parquet(path).with_columns(pl.Series("mean", [1.0, 2.0, 3.5]))
    frame.write_parquet(path, compression="uncompressed")
    assert path.stat().st_size == size
    summary = bench.run(plan)
    assert summary.done == ("dev/a",) and summary.skipped == ("dev/b",)
    queued = next(e for e in bench.events() if e["job"] == "dev/a" and e["status"] == "queued")
    assert queued["reason"] == "dev/a.parquet does not match its checksum"
    assert digest(path) == original


def test_second_runner_on_another_study_is_refused(bench):
    plan = [bench.job("development/slow", sleep=30.0)]
    runner = bench.start_runner(plan)
    wait_for(lambda: bool(bench.traced()))
    with pytest.raises(RunnerBusy, match=f"pid {runner.pid}"):
        bench.run([bench.job("confirmatory/other")])
    job_pid = bench.traced()[0]["pid"]
    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=30) != 0
    with pytest.raises(ProcessLookupError):
        os.kill(job_pid, 0)
    assert not bench.output("development/slow").exists()
    assert not list((bench.runs / "leases").rglob("*.lease"))
    assert not bench.output("confirmatory/other").exists()


def test_a_job_whose_id_holds_another_leaves_that_job_alone(bench):
    inner = bench.job("dev/a/b", sleep=1.0, priority=0)
    outer = bench.job("dev/a", priority=1)
    summary = bench.run([inner, outer])
    assert sorted(summary.done) == ["dev/a", "dev/a/b"]
    assert bench.statuses("dev/a/b") == ["queued", "started", "done"]


def test_outputs_that_cannot_be_moved_fail_only_their_job(bench):
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    locked = bench.runs / "locked"
    locked.mkdir(parents=True)
    locked.chmod(0o555)
    try:
        summary = bench.run([bench.job("locked/x"), bench.job("dev/fine")])
    finally:
        locked.chmod(0o755)
    assert summary.failed == ("locked/x",) and summary.done == ("dev/fine",)
    failed = bench.events()[-1] if bench.events()[-1]["job"] == "locked/x" else None
    record = failed or next(e for e in bench.events() if e["status"] == "failed")
    assert record["kind"] == "promotion" and "Permission denied" in record["error"]
    assert list(locked.iterdir()) == []


@pytest.mark.parametrize("lease", [".lease", "lease.json"])
def test_a_runner_killed_before_recording_a_worker_leaves_none_behind(bench, lease):
    from fixture_jobs import marked

    runs = str(bench.runs.resolve())
    if lease == ".lease":
        plan = [bench.job("dev/slow", sleep=30.0)]
    else:
        args = {"runs_dir": runs, "trace": str(bench.trace), "sleep": 30.0}
        plan = [bench.job("automl/fit", "automl_probe", args=args, automl=True, full_load=True)]
    runner = bench.start_runner(plan, die_before=lease)
    assert runner.wait(timeout=60) == -signal.SIGKILL
    time.sleep(2.0)
    bench.run([])
    wait_for(lambda: marked(bench.base) == {}, seconds=10.0)
    assert not [r for r in bench.traced() if r["job"] == "dev/slow"]


def test_a_staged_output_changed_after_hashing_is_never_promoted(bench):
    summary = bench.run([bench.job("dev/changed", "changes_after_hashing")])
    assert summary.failed == ("dev/changed",)
    failure = next(e for e in bench.events() if e["status"] == "failed")
    assert failure["kind"] == "promotion" and "changed after it was hashed" in failure["error"]
    assert not bench.output("dev/changed").exists()


def test_a_successor_planned_after_the_last_job_finishes_still_runs(bench):
    first = bench.job("dev/first")
    second = bench.job("dev/second", needs=["dev/first"])

    def planner():
        return [first, second] if bench.output("dev/first").exists() else [first]

    summary = bench.run(planner)
    assert summary.done == ("dev/first", "dev/second")


@pytest.mark.parametrize(
    ("usable", "problem"), [(False, "no gpu is visible"), (True, "the probe failed")]
)
def test_automl_jobs_run_on_cpu_slots_when_no_gpu_is_usable(bench, monkeypatch, usable, problem):
    bench.fake_gpu = usable
    if usable:
        monkeypatch.setattr(pools, "torch_problem", lambda gpus: problem)
    args = {"runs_dir": str(bench.runs.resolve()), "trace": str(bench.trace), "sleep": 0.5}
    automl = bench.job("automl/fit", "automl_probe", args=args, automl=True)
    summary = bench.run([automl, bench.job("dev/other")])
    assert sorted(summary.done) == ["automl/fit", "dev/other"] and summary.waiting == ()
    started = next(e for e in bench.events() if e["status"] == "started" and e["job"] == automl.id)
    assert started["device"].startswith("cpu ") and "slot" in started
    probe = next(r for r in bench.traced() if r["job"] == automl.id and r["what"] == "start")
    assert (probe["cuda"], probe["gpus"]) == ("", 0)
    budget = "where their time budget buys different fits than on a gpu"
    assert bench.run_log().count(f"automl jobs on cpu, {budget}: {problem}") == 1
    health = (bench.runs / "logs" / "health.log").read_text()
    assert f"note: automl jobs run on cpu (1 left), {budget}: {problem}" in health


def test_automl_jobs_on_cpu_still_wait_while_cpu_jobs_share_cores(bench):
    bench.fake_gpu = False
    allowed = sorted(os.sched_getaffinity(0))
    bench.cores = [[allowed[k % len(allowed)], allowed[(k + 2) % len(allowed)]] for k in range(2)]
    args = {"runs_dir": str(bench.runs.resolve()), "trace": str(bench.trace)}
    automl = bench.job("automl/fit", "automl_probe", args=args, automl=True)
    resources = {"reserved_threads": 0, "cpu_jobs_on_second_threads": True}
    summary = bench.run([automl, bench.job("dev/other")], resources=resources, cpu_jobs=None)
    assert summary.done == ("dev/other",) and summary.waiting == ("automl/fit",)
    assert bench.statuses("automl/fit") == ["queued"]
    health = (bench.runs / "logs" / "health.log").read_text()
    assert "alert: automl jobs wait while cpu jobs share cores (1 waiting)" in health


def test_automl_jobs_take_a_seat_on_a_usable_gpu_and_wait_for_one(bench):
    from fixture_jobs import FAKE_GPU

    args = {"runs_dir": str(bench.runs.resolve()), "trace": str(bench.trace), "sleep": 0.5}
    fits = [bench.job(f"automl/{n}", "automl_probe", args=args, automl=True) for n in "ab"]
    summary = bench.run(fits, resources={"automl_jobs_per_device": 1})
    assert sorted(summary.done) == ["automl/a", "automl/b"]
    started = [e for e in bench.events() if e["status"] == "started"]
    assert all(e["device"].startswith(f"gpu {FAKE_GPU.index}, cpu ") for e in started)
    probes = {(r["cuda"], r["gpus"]) for r in bench.traced() if r["what"] == "start"}
    assert probes == {(FAKE_GPU.uuid, 1)}
    assert most_at_once([interval(bench, job.id) for job in fits]) == 1
    assert "automl jobs on cpu" not in bench.run_log()


def test_shared_cores_fill_second_threads_last_while_automl_fits_wait(bench):
    allowed = sorted(os.sched_getaffinity(0))
    bench.cores = [[allowed[k % len(allowed)], allowed[(k + 2) % len(allowed)]] for k in range(2)]
    args = {"runs_dir": str(bench.runs.resolve()), "trace": str(bench.trace)}
    automl = bench.job("automl/fit", "automl_probe", args=args, automl=True)
    gate = bench.trace / "together"
    narrow = [bench.job(f"engine/{k}", release=gate) for k in range(3)]

    def together() -> bool:
        return {r["job"] for r in bench.traced() if r["what"] == "start"} >= {j.id for j in narrow}

    with release_when(gate, together):
        summary = bench.run(
            [automl, *narrow],
            resources={
                "reserved_threads": 0,
                "cpu_jobs_on_second_threads": True,
                "gpu_workers_per_device": 1,
            },
            cpu_jobs=None,
        )
    assert sorted(summary.done) == [job.id for job in narrow]
    assert summary.waiting == ("automl/fit",)
    slots = (read_json(bench.runs / "allocation.json") or {})["pools"]["cpu_slots"]
    assert slots == [[core[0]] for core in bench.cores] + [[bench.cores[0][1]]]
    started = {e["job"]: e for e in bench.events() if e["status"] == "started"}
    assert sorted(e["slot"] for e in started.values()) == [0, 1, 2]
    traced = [r for r in bench.traced() if r["job"].startswith("engine/")]
    assert max(r["time"] for r in traced if r["what"] == "start") < min(
        r["time"] for r in traced if r["what"] == "end"
    )
    third = next(job for job, e in started.items() if e["slot"] == 2)
    assert next(r["cpus"] for r in traced if r["job"] == third) == [bench.cores[0][1]]
    health = (bench.runs / "logs" / "health.log").read_text()
    assert "alert: automl jobs wait while cpu jobs share cores (1 waiting)" in health


def test_cpu_jobs_use_only_the_slots_left_beside_the_gpu_workers(bench):
    allowed = sorted(os.sched_getaffinity(0))
    bench.cores = [[allowed[k % len(allowed)], allowed[(k + 2) % len(allowed)]] for k in range(2)]
    first = bench.job("gpu/first", sleep=0.5, gpu=True, priority=0)
    narrow = [bench.job(f"engine/{k}", sleep=3.0, priority=1) for k in range(4)]
    summary = bench.run(
        [first, *narrow],
        resources={
            "reserved_threads": 0,
            "cpu_jobs_on_second_threads": True,
            "gpu_workers_per_device": 1,
        },
        cpu_jobs=None,
    )
    assert sorted(summary.done) == sorted(job.id for job in (first, *narrow))
    slots = (read_json(bench.runs / "allocation.json") or {})["pools"]["cpu_slots"]
    assert len(slots) == 3
    started = [e for e in bench.events() if e["status"] == "started" and e["job"] != first.id]
    assert max(e["slot"] for e in started) == len(slots) - 1


def test_slots_added_when_the_gpu_fails_take_only_threads_no_slot_holds(bench, monkeypatch):
    from windfold.runner import pools

    allowed = sorted(os.sched_getaffinity(0))
    bench.cores = [[allowed[k % len(allowed)], allowed[(k + 2) % len(allowed)]] for k in range(2)]
    monkeypatch.setattr(pools, "torch_problem", lambda gpus: "the probe failed")
    jobs = [bench.job("gpu/fit", sleep=0.5, gpu=True), bench.job("engine/0", sleep=0.5)]
    summary = bench.run(
        jobs,
        resources={
            "reserved_threads": 0,
            "cpu_jobs_on_second_threads": True,
            "gpu_workers_keep_whole_cores": True,
            "gpu_workers_per_device": 1,
        },
        cpu_jobs=None,
    )
    assert sorted(summary.done) == ["engine/0", "gpu/fit"]
    slots = (read_json(bench.runs / "allocation.json") or {})["pools"]["cpu_slots"]
    first, second = bench.cores
    assert slots == [[first[0]], [first[1]], [second[0]], [second[1]]]


def test_a_load_job_that_stops_reruns_budget_fits_but_not_production_fits(bench):
    from fixture_jobs import load_workers

    runs = str(bench.runs.resolve())
    args = {"runs_dir": runs, "trace": str(bench.trace), "sleep": 4.0}
    budget = bench.job(
        "calibration/budget",
        "automl_probe",
        args=args,
        automl=True,
        full_load=True,
        family="automl_budget",
    )
    production = bench.job("automl/production", "automl_probe", args=args, automl=True)
    killed: list[int] = []

    def kill_one_load_worker():
        wait_for(lambda: len(load_workers(runs)) == 3)
        killed.append(load_workers(runs)[0])
        os.kill(killed[0], signal.SIGKILL)

    killer = threading.Thread(target=kill_one_load_worker)
    killer.start()
    summary = bench.run([budget, production])
    killer.join()
    assert sorted(summary.done) == ["automl/production", "calibration/budget"]
    assert bench.statuses("calibration/budget") == [
        "queued",
        "started",
        "queued",
        "started",
        "done",
    ]
    assert bench.statuses("automl/production") == ["queued", "started", "done"]
    budget_events = [e for e in bench.events() if e["job"] == "calibration/budget"]
    attempts = [e["attempt"] for e in budget_events if "attempt" in e]
    assert set(attempts) == {1}
    assert bench.run_log().count("load job started") > 3
    assert not (bench.runs / "paused").exists()


def test_four_one_core_jobs_share_one_slot_on_four_cores_at_once(bench):
    narrow = [bench.job(f"engine/{k}", sleep=1.5) for k in range(4)]
    summary = bench.run(narrow, resources={"cores_per_cpu_job": 4}, cpu_jobs=2)
    assert sorted(summary.done) == sorted(job.id for job in narrow)
    started = {e["job"]: e for e in bench.events() if e["status"] == "started"}
    assert {e["slot"] for e in started.values()} == {0}
    assert sorted(core for e in started.values() for core in e["slot_cores"]) == [0, 1, 2, 3]
    traced = bench.traced()
    assert max(r["time"] for r in traced if r["what"] == "start") < min(
        r["time"] for r in traced if r["what"] == "end"
    )
    slot = (read_json(bench.runs / "allocation.json") or {})["pools"]["cpu_slots"][0]
    for record in [r for r in traced if r["what"] == "start"]:
        (core,) = started[record["job"]]["slot_cores"]
        assert record["cpus"] == [slot[core]] and record["threads"] == "1"


def test_a_wide_job_waits_for_a_free_slot_and_never_joins_a_narrow_one(bench):
    narrow = [bench.job(f"engine/{k}", sleep=2.0) for k in range(5)]
    wide = bench.job("gbm/wide", cores=4, priority=1)
    summary = bench.run([*narrow, wide], resources={"cores_per_cpu_job": 4}, cpu_jobs=2)
    assert len(summary.done) == 6
    started = {e["job"]: e for e in bench.events() if e["status"] == "started"}
    assert {started[job.id]["slot"] for job in narrow[:4]} == {0}
    assert sorted(c for job in narrow[:4] for c in started[job.id]["slot_cores"]) == [0, 1, 2, 3]
    assert started["engine/4"]["slot"] == 1
    assert started["gbm/wide"]["slot_cores"] == [0, 1, 2, 3]
    wide_start, wide_end = interval(bench, "gbm/wide")
    for job in narrow:
        if started[job.id]["slot"] == started["gbm/wide"]["slot"]:
            start, end = interval(bench, job.id)
            assert end <= wide_start or wide_end <= start


def test_load_jobs_fill_the_idle_cores_of_a_narrow_slot_during_a_budget_fit(bench):
    from fixture_jobs import load_workers

    runs = str(bench.runs.resolve())
    filled, finished = bench.trace / "filled", bench.trace / "finished"
    args = {"runs_dir": runs, "trace": str(bench.trace), "release": str(filled)}
    budget = bench.job(
        "calibration/budget", "automl_probe", args=args, automl=True, full_load=True, cores=4
    )
    narrow = [bench.job(f"engine/{k}", release=finished) for k in range(2)]
    loads: set[tuple[int, tuple[int, ...]]] = set()
    views: list[str] = []
    sampling = threading.Event()
    log = bench.runs / "jobs.jsonl"

    def sample():
        while not sampling.is_set():
            for path in (bench.runs / "load").glob("*/lease.json"):
                if record := read_json(path):
                    loads.add((record["slot"], tuple(record["slot_cores"])))
            views.append(status(bench.runs, 20))
            time.sleep(0.2)

    def seen_filled() -> bool:
        return len(loads) == 2 and any("narrow 2 of 4, 2 load" in view for view in views)

    def budget_done() -> bool:
        return any(e["job"] == budget.id and e["status"] == "done" for e in read_events(log))

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        with release_when(filled, seen_filled), release_when(finished, budget_done):
            summary = bench.run([budget, *narrow], resources={"cores_per_cpu_job": 4}, cpu_jobs=2)
    finally:
        sampling.set()
        sampler.join()
    assert len(summary.done) == 3
    started = {e["job"]: e for e in bench.events() if e["status"] == "started"}
    shared = started["engine/0"]["slot"]
    assert started["engine/1"]["slot"] == shared != started["calibration/budget"]["slot"]
    assert loads == {(shared, (2,)), (shared, (3,))}
    assert any(f"slot {shared}: narrow 2 of 4, 2 load" in view for view in views)
    assert load_workers(runs) == []


def test_production_automl_fits_run_without_load_jobs(bench):
    runs = str(bench.runs.resolve())
    args = {"runs_dir": runs, "trace": str(bench.trace), "sleep": 1.5}
    summary = bench.run([bench.job("automl/production", "automl_probe", args=args, automl=True)])
    assert summary.done == ("automl/production",)
    assert pl.read_parquet(bench.output("automl/production"))["mean"][0] == 0.0


def test_a_job_needs_one_core_or_a_whole_slot(bench):
    assert check_plan([bench.job("dev/one"), bench.job("dev/four", cores=4)], cores_per_cpu_job=4)
    with pytest.raises(PlanError, match="1 or 4 cores"):
        check_plan([bench.job("dev/two", cores=2)], cores_per_cpu_job=4)


def test_a_job_whose_processes_outlive_the_kill_keeps_its_cores_and_lease(bench, monkeypatch):
    survivors: set[int] = set()
    alive = pools.group_alive
    monkeypatch.setattr(pools, "group_alive", lambda pgid: pgid in survivors or alive(pgid))
    monkeypatch.setattr(pools, "KILL_WAIT_SECONDS", 0.2)
    runner = bench.runner([bench.job("dev/a"), bench.job("dev/b", sleep=60.0)])
    runner.replan(strict=True)
    runner.fill()

    def holds(job_id: str) -> tuple[bool, bool, bool]:
        cores = {job for slot in runner.slots for job in slot.jobs.values()}
        return job_id in runner.running, job_id in cores, runner.lease(job_id).is_file()

    def exited(pid: int) -> bool:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    ended = runner.running["dev/a"].process.pid
    survivors.add(ended)
    wait_for(lambda: exited(ended))
    runner.reap()
    assert holds("dev/a") == (True, True, True)
    assert bench.statuses("dev/a") == ["queued", "started"]
    survivors.clear()
    runner.reap()
    assert holds("dev/a") == (False, False, False)
    assert bench.statuses("dev/a") == ["queued", "started", "done"]

    restarted = runner.running["dev/b"].process.pid
    survivors.add(restarted)
    runner.restart(runner.running["dev/b"], "a test")
    assert holds("dev/b") == (True, True, True)
    assert bench.statuses("dev/b") == ["queued", "started"]
    survivors.clear()
    wait_for(lambda: exited(restarted))
    runner.reap()
    assert holds("dev/b") == (False, False, False)
    assert bench.statuses("dev/b") == ["queued", "started", "queued"]

    runner.fill()
    stopped = runner.running["dev/b"].process
    survivors.add(stopped.pid)
    runner.stop_all()
    assert runner.lease("dev/b").is_file()
    with pytest.raises(pools.OrphanSurvived):
        bench.runner([]).reclaim()
    survivors.clear()
    bench.runner([]).reclaim()
    assert not runner.lease("dev/b").exists()
    stopped.wait()


def test_a_group_lives_on_in_a_thread_after_its_main_thread_exits():
    code = (
        "import ctypes, threading, time\n"
        "threading.Thread(target=time.sleep, args=(60,)).start()\n"
        "ctypes.CDLL(None).pthread_exit(None)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        wait_for(lambda: (stat_fields(process.pid) or [""])[0] == "Z")
        assert pools.group_alive(process.pid)
    finally:
        os.killpg(process.pid, signal.SIGKILL)
    wait_for(lambda: not pools.group_alive(process.pid))
    process.wait()


def test_failure_mid_write_leaves_no_partial_file(bench):
    job = bench.job("dev/full", "no_space")
    summary = bench.run([job])
    assert summary.failed == ("dev/full",)
    failed = bench.events()[-1]
    assert (failed["status"], failed["kind"]) == ("failed", "no_space")
    assert "No space left on device" in failed["error"]
    leftovers = [p for p in bench.base.rglob("*full*") if p.is_file() and "logs" not in p.parts]
    assert leftovers == []


def test_retry_once_then_failure_recorded_until_the_fingerprint_changes(bench):
    flaky = bench.job("dev/flaky", "fail_first", args={"marker": str(bench.trace / "flaky")})
    broken = bench.job("dev/broken", "fail_always", fingerprint="made-1")
    fine = bench.job("dev/fine")
    summary = bench.run([flaky, broken, fine])
    assert sorted(summary.done) == ["dev/fine", "dev/flaky"]
    assert summary.failed == ("dev/broken",)
    assert bench.statuses("dev/flaky") == ["queued", "started", "retried", "started", "done"]
    assert bench.statuses("dev/broken") == ["queued", "started", "retried", "started", "failed"]
    failure = bench.events()[[e["status"] for e in bench.events()].index("failed")]
    assert "Traceback" in failure["error"] and "dev/broken always fails" in failure["error"]
    assert failed_for_good(bench.runs) == {"dev/broken": "made-1"}
    rerun = bench.run([flaky, broken, fine], job="dev/fine")
    assert rerun.done == ("dev/fine",)
    assert [e["job"] for e in bench.events()] == ["dev/fine"] * 3
    fixed = bench.job("dev/broken", fingerprint="made-2")
    bench.run([flaky, fixed, fine])
    assert failed_for_good(bench.runs) == {}


def test_family_pauses_on_failures_and_nonfinite_values(bench):
    plan = [
        bench.job("gbm/broken", "fail_always", family="gbm", priority=0),
        bench.job("gbm/later", family="gbm", priority=1),
        bench.job("engine/nan", "nonfinite", family="engine", priority=0),
        bench.job("engine/later", family="engine", priority=1),
        bench.job("graph/other", family="graph", priority=2),
    ]
    summary = bench.run(plan, resources={"failures_before_pause": 1}, cpu_jobs=1)
    assert summary.failed == ("gbm/broken",)
    assert summary.done == ("graph/other",)
    assert sorted(summary.waiting) == ["engine/later", "engine/nan", "gbm/later"]
    assert bench.statuses("engine/nan") == ["queued", "started", "retried"]
    assert not bench.output("engine/nan").exists()
    assert "1 of 1 finished jobs failed, over 2%" in (bench.runs / "paused" / "gbm").read_text()
    assert "non-finite values" in (bench.runs / "paused" / "engine").read_text()
    health = (bench.runs / "logs" / "health.log").read_text()
    assert "family gbm paused" in health and "non-finite output: engine/nan" in health
    (bench.runs / "paused" / "gbm").unlink()
    resumed = bench.run(plan[1:], cpu_jobs=1)
    assert resumed.done == ("gbm/later",)
    assert sorted(resumed.waiting) == ["engine/later", "engine/nan"]


def test_three_early_failures_pause_a_large_family(bench):
    broken = [bench.job(f"gbm/broken{k}", "fail_always", family="gbm") for k in range(3)]
    fine = [bench.job(f"gbm/fine{k}", family="gbm", priority=1) for k in range(148)]
    summary = bench.run([*broken, *fine], cpu_jobs=1)
    assert sorted(summary.failed) == [job.id for job in broken]
    assert summary.done == ()
    assert 3 < 0.02 * len(broken + fine)
    started = {e["job"] for e in bench.events() if e["status"] == "started"}
    assert started == {job.id for job in broken}
    reason = (bench.runs / "paused" / "gbm").read_text()
    assert reason.startswith("3 of 3 finished jobs failed, over 2%")


def test_a_second_runner_on_the_same_data_folder_is_refused(bench):
    plan = [bench.job("development/slow", sleep=30.0)]
    runner = bench.start_runner(plan, runs=bench.base / "other_runs")
    wait_for(lambda: bool(bench.traced()))
    with pytest.raises(RunnerBusy, match=f"works on {bench.data}"):
        bench.run([bench.job("confirmatory/other")])
    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=30) != 0
    assert not (bench.runs / "jobs.jsonl").exists()


def test_gpu_out_of_memory_retries_alone_then_device_runs_fewer(bench, real_gpu):
    oom = bench.job(
        "gpu/oom",
        "gpu_out_of_memory",
        args={"marker": str(bench.trace / "oom"), "trace": str(bench.trace)},
        gpu=True,
        priority=0,
    )
    gate = bench.trace / "retried"
    first = [bench.job(f"gpu/{n}", gpu=True, priority=1, release=gate) for n in ("b", "c")]
    later = [bench.job(f"gpu/{n}", gpu=True, priority=2) for n in ("d", "e", "f")]
    plan = [oom, *first, *later]
    log = bench.runs / "jobs.jsonl"
    with release_when(gate, lambda: any(e["status"] == "retried" for e in read_events(log))):
        summary = bench.run(plan, resources={"gpu_workers_per_device": 3})
    assert sorted(summary.done) == sorted(job.id for job in plan)
    retried = next(e for e in bench.events() if e["status"] == "retried")
    assert (retried["job"], retried["kind"]) == ("gpu/oom", "gpu_out_of_memory")
    alone = interval(bench, "gpu/oom", attempt=2)
    beside = [interval(bench, job.id) for job in first]
    after = [interval(bench, job.id) for job in later]
    assert all(end > datetime.fromisoformat(retried["time"]) for _, end in beside)
    for start, end in [*beside, *after]:
        assert end <= alone[0] or alone[1] <= start
    assert most_at_once(after) == 2
    assert {r["cuda"] for r in bench.traced() if r["what"] == "start"} == {real_gpu.uuid}
    expected = f"gpu {real_gpu.index} ran out of memory; it runs at most 2 gpu jobs at once"
    assert expected in bench.run_log()


def test_automl_job_runs_beside_load_jobs_that_yield_and_stop(bench):
    from fixture_jobs import load_workers

    runs = str(bench.runs.resolve())
    gate = bench.trace / "refilled"
    automl = bench.job(
        "automl/fit",
        "automl_probe",
        args={"runs_dir": runs, "trace": str(bench.trace), "release": str(gate)},
        automl=True,
        full_load=True,
    )
    extra = [bench.job(f"dev/{n}") for n in ("x", "y")]
    calls: list[int] = []

    def planner():
        calls.append(1)
        return [automl] if len(calls) == 1 else [automl, *extra]

    def refilled() -> bool:
        log = bench.runs / "logs" / "run.log"
        return log.is_file() and log.read_text().count("load job started") == 4 + 2

    with release_when(gate, refilled):
        summary = bench.run(planner, resources={"replan_seconds": 1.0})
    assert sorted(summary.done) == ["automl/fit", "dev/x", "dev/y"]
    assert pl.read_parquet(bench.output("automl/fit"))["mean"][0] == 4.0
    log = bench.run_log()
    assert log.count("load job started") == log.count("load job on") == 4 + 2
    fit = interval(bench, "automl/fit")
    for job_id in ("dev/x", "dev/y"):
        start, end = interval(bench, job_id)
        assert fit[0] < start and end < fit[1]
    assert load_workers(runs) == []
    gate.unlink()
    with release_when(gate, lambda: len(load_workers(runs)) == 4):
        rerun = bench.run(planner, job="automl/fit")
    assert rerun.done == ("automl/fit",)
    assert pl.read_parquet(bench.output("automl/fit"))["mean"][0] == 4.0
    assert load_workers(runs) == []


def test_load_jobs_left_by_a_killed_runner_are_stopped_before_its_jobs(bench, monkeypatch):
    from fixture_jobs import load_workers

    runs = str(bench.runs.resolve())
    automl = bench.job(
        "automl/fit",
        "automl_probe",
        args={"runs_dir": runs, "trace": str(bench.trace), "sleep": 60.0},
        automl=True,
        full_load=True,
    )
    runner = bench.start_runner([automl])
    wait_for(lambda: bool(bench.traced()) and len(load_workers(runs)) == 4)
    runner.send_signal(signal.SIGKILL)
    runner.wait()
    probe = bench.traced()[0]["pid"]
    assert len(load_workers(runs)) == 4
    alive = pools.group_alive
    monkeypatch.setattr(pools, "group_alive", lambda pgid: pgid == probe or alive(pgid))
    monkeypatch.setattr(pools, "KILL_WAIT_SECONDS", 0.2)
    with pytest.raises(pools.OrphanSurvived):
        bench.run([])
    assert load_workers(runs) == []
    assert list((bench.runs / "leases").glob("*.lease"))
    monkeypatch.setattr(pools, "group_alive", alive)
    summary = bench.run([])
    assert summary.done == ()
    with pytest.raises(ProcessLookupError):
        os.kill(probe, 0)


@pytest.mark.skipif(
    importlib.util.find_spec("autogluon") is None,
    reason="needs autogluon, from the automl dependency group",
)
def test_autogluon_fit_leaves_no_process_behind(bench):
    from fixture_jobs import marked

    seen: dict[int, tuple[str, int, int, int, str]] = {}
    sampling = threading.Event()

    def sample():
        while not sampling.is_set():
            for pid, info in marked(bench.base).items():
                seen.setdefault(pid, info)
            time.sleep(0.2)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        args = {"trace": str(bench.trace), "seconds": 120.0}
        fit = bench.job("automl/fit", "autogluon_fit", args=args, automl=True)
        summary = bench.run([fit], resources={"cores_per_cpu_job": 2}, cpu_jobs=1)
        assert summary.done == ("automl/fit",)
        assert marked(bench.base) == {}
        models = next(r["models"] for r in bench.traced() if r["what"] == "end")
        families = ("LightGBM", "CatBoost", "XGBoost", "NeuralNetTorch", "NeuralNetFastAI")
        assert all(any(m.startswith(f) for m in models) for f in families), models
        killed = bench.job("automl/killed", "autogluon_fit", args=args, automl=True)
        runner = bench.start_runner([killed], resources={"cores_per_cpu_job": 2}, cpu_jobs=1)
        staging = bench.runs / "staging"
        wait_for(lambda: any(staging.glob("automl+killed/*/out/predictor/models/*_BAG_L1")))
        runner.send_signal(signal.SIGKILL)
        runner.wait()
        assert marked(bench.base) != {}
        bench.run([])
        assert marked(bench.base) == {}
    finally:
        sampling.set()
        sampler.join()
    workers = {pid for pid, info in seen.items() if "windfold.runner.worker" in info[4]}
    runners = {runner.pid, os.getpgrp()}
    in_jobs = {pid: info for pid, info in seen.items() if info[2] not in runners}
    assert all(info[2] == info[3] and info[2] in workers for info in in_jobs.values()), in_jobs


def test_planner_adds_drops_and_renews_jobs_mid_run(bench):
    slow = bench.job("dev/slow", sleep=2.0, fingerprint="made-slow-1")
    dropped = bench.job("dev/dropped", needs=["dev/slow"])
    renewed = bench.job("dev/slow", sleep=0.5, fingerprint="made-slow-2")
    added = bench.job("dev/added")
    calls: list[int] = []

    def planner():
        calls.append(1)
        return [slow, dropped] if len(calls) == 1 else [renewed, added]

    summary = bench.run(planner, resources={"replan_seconds": 0.5})
    assert bench.statuses("dev/dropped") == ["queued", "dropped"]
    assert bench.statuses("dev/added") == ["queued", "started", "done"]
    slow_events = [e for e in bench.events() if e["job"] == "dev/slow"]
    assert [(e["status"], e["fingerprint"]) for e in slow_events] == [
        ("queued", "made-slow-1"),
        ("started", "made-slow-1"),
        ("queued", "made-slow-2"),
        ("done", "made-slow-1"),
        ("started", "made-slow-2"),
        ("done", "made-slow-2"),
    ]
    assert (
        json.loads(sidecar_path(bench.output("dev/slow")).read_text())["fingerprint"]
        == "made-slow-2"
    )
    assert sorted(summary.done) == ["dev/added", "dev/slow", "dev/slow"]
    assert not list(bench.trace.glob("*.concurrent"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"twice": True}, "planned twice"),
        ({"needs": ["dev/ghost"]}, "not in the plan"),
        ({"needs": ["dev/b"], "cycle": True}, "cycle"),
        ({"outputs": ["dev/b.parquet"]}, "both write"),
        ({"outputs": ["jobs.jsonl"]}, "a name the runner keeps"),
        ({"outputs": ["dev"]}, "as an output and as a folder"),
    ],
)
def test_bad_plans_are_refused(bench, change, message):
    a = bench.job("dev/a")
    b = bench.job("dev/b", needs=["dev/a"] if change.get("cycle") else [])
    edits = {key: value for key, value in change.items() if key in ("needs", "outputs")}
    a = a.model_copy(update=edits)
    plan = [a, b, a] if change.get("twice") else [a, b]
    with pytest.raises(PlanError, match=message):
        check_plan(plan)
