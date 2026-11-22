"""a short plain view of what the runner is doing, built only from what it leaves on disk.

progress per family and farm, time left from measured durations, devices, disk and alerts.
the runner appends the view to logs/health.log every status interval; status() gives the same
view on demand, from any process.
"""

import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from windfold.runner.jobs import read_events, read_json, runner_alive
from windfold.runner.resources import device_use

STATES = {
    "queued": "waiting",
    "retried": "waiting",
    "started": "running",
    "done": "done",
    "skipped": "done",
    "failed": "failed",
}
ALERT_KINDS = {
    "gpu_out_of_memory": "gpu out of memory",
    "out_of_memory": "out of memory",
    "killed": "killed, possibly out of memory",
    "nonfinite": "non-finite output",
    "no_space": "no space left",
}
SHOWN_FAILURES = 5


def group(record: dict[str, Any]) -> str:
    return f"{record['family']} {record['farm']}" if "farm" in record else record["family"]


def occupancy(runs_dir: Path, slots: list[list[int]]) -> str:
    """each cpu slot as free, wide (one job on every core), narrow (how many of its cores hold
    one-core jobs) or load, read from the leases of running jobs and load jobs."""
    held: dict[int, dict[int, str]] = defaultdict(dict)
    leases = [*(runs_dir / "leases").glob("*.lease"), *(runs_dir / "load").glob("*/lease.json")]
    for path in leases:
        record = read_json(path) or {}
        if isinstance(record.get("slot"), int):
            for core in record.get("slot_cores") or []:
                held[record["slot"]][core] = str(record.get("job"))
    parts = []
    for k, threads in enumerate(slots):
        cores = held.get(k, {})
        jobs = [job for job in cores.values() if job != "load"]
        load = len(cores) - len(jobs)
        if not cores:
            state = "free"
        elif not jobs:
            state = "load"
        elif len(jobs) == len(threads) and len(set(jobs)) == 1:
            state = "wide"
        else:
            state = f"narrow {len(jobs)} of {len(threads)}" + (f", {load} load" if load else "")
        parts.append(f"slot {k}: {state}")
    return "; ".join(parts)


def counts(counter: Counter[str]) -> str:
    order = ("done", "failed", "running", "interrupted", "waiting")
    return ", ".join(f"{counter[state]:,} {state}" for state in order if counter[state])


def status(runs_dir: Path, stall_minutes: int, notes: Sequence[str] = ()) -> str:
    """the current view of the runs folder as a few plain lines."""
    clock = datetime.now(UTC)
    pid = runner_alive(runs_dir)
    runner = f"runner pid {pid} running" if pid else "no runner running"
    lines = [f"{clock:%Y-%m-%d %H:%M} UTC, {runner}"]
    events = read_events(runs_dir / "jobs.jsonl")
    session = events[-1]["session"] if events else None
    durations: dict[str, list[float]] = defaultdict(list)
    latest: dict[str, dict[str, Any]] = {}
    started: dict[str, datetime] = {}
    last_finish = datetime.fromisoformat(session) if session else clock
    kinds: dict[str, dict[str, Any]] = {}
    for record in events:
        if record["status"] == "done" and "seconds" in record:
            durations[group(record)].append(record["seconds"])
            durations[record["family"]].append(record["seconds"])
        if record["session"] != session:
            continue
        latest[record["job"]] = record
        when = datetime.fromisoformat(record["time"])
        if record["status"] == "started":
            started[record["job"]] = when
        if record["status"] in ("done", "failed", "retried"):
            last_finish = max(last_finish, when)
        if record["status"] in ("retried", "failed") and record.get("kind") in ALERT_KINDS:
            kinds[record["job"]] = record
    allocation = read_json(runs_dir / "allocation.json") or {}
    pools = allocation.get("pools", {})
    gpus = [gpu["index"] for gpu in pools.get("gpus", [])]
    slots = max(len(pools.get("cpu_slots", [])), 1)
    gpu_workers = max(len(gpus) * pools.get("gpu_workers_per_device", 1), 1)
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    total: Counter[str] = Counter()
    work = {"cpu": 0.0, "gpu": 0.0}
    unmeasured: set[str] = set()
    failures: list[dict[str, Any]] = []
    for job, record in latest.items():
        state = STATES.get(record["status"])
        if state is None:
            continue
        if state == "running" and pid is None:
            state = "interrupted"
        by_group[group(record)][state] += 1
        total[state] += 1
        if state == "failed":
            failures.append(record)
        if state not in ("waiting", "running"):
            continue
        measured = durations.get(group(record)) or durations.get(record["family"])
        if not measured:
            unmeasured.add(group(record))
            continue
        estimate = median(measured)
        if state == "running" and job in started:
            estimate = max(estimate - (clock - started[job]).total_seconds(), 0.0)
        work["gpu" if record.get("gpu") and not record.get("automl") and gpus else "cpu"] += (
            estimate
        )
    hours = max(work["cpu"] / slots, work["gpu"] / gpu_workers) / 3600
    left = f"; time left about {hours:.1f} h" if total["waiting"] + total["running"] else ""
    if unmeasured:
        left += f", not counting {', '.join(sorted(unmeasured))} (no measured job yet)"
    if total:
        lines.append(f"  {counts(total)} of {sum(total.values()):,} jobs{left}")
    else:
        lines.append("  no jobs yet")
    lines += [f"  {name}: {counts(by_group[name])}" for name in sorted(by_group)]
    if pid and pools.get("cpu_slots"):
        lines.append(f"  cpu slots: {occupancy(runs_dir, pools['cpu_slots'])}")
    use = device_use() if gpus else {}
    lines += [
        f"  gpu {index}: {use[index][0]:.1f} of {use[index][1]:.1f} gb, {use[index][2]:.0f}% busy"
        for index in gpus
        if index in use
    ]
    existing = next(folder for folder in (runs_dir, *runs_dir.parents) if folder.exists())
    lines.append(f"  disk free {shutil.disk_usage(existing).free / 2**30:,.0f} gb")
    for record in failures[-SHOWN_FAILURES:]:
        error = record.get("error", "").strip().splitlines()
        kind = f" ({ALERT_KINDS[record['kind']]})" if record.get("kind") in ALERT_KINDS else ""
        last = error[-1] if error else record.get("kind")
        lines.append(f"  alert: failed {record['job']}{kind}: {last}")
    if len(failures) > SHOWN_FAILURES:
        lines.append(f"  alert: {len(failures) - SHOWN_FAILURES} more failed jobs in jobs.jsonl")
    retried = [r for job, r in kinds.items() if latest[job]["status"] != "failed"]
    lines += [
        f"  alert: {ALERT_KINDS[r['kind']]}: {r['job']} on {r.get('device')}, retried"
        for r in retried[-SHOWN_FAILURES:]
    ]
    automl = Counter(STATES.get(r["status"]) for r in latest.values() if r.get("automl"))
    waiting_automl, left_automl = automl["waiting"], automl["waiting"] + automl["running"]
    if waiting_automl and pools.get("shared_cores"):
        lines.append(
            f"  alert: automl jobs wait while cpu jobs share cores ({waiting_automl} waiting); "
            "run without cpu_jobs_on_second_threads to fit them"
        )
    paused = runs_dir / "paused"
    for path in sorted(paused.iterdir()) if paused.exists() else []:
        reason = path.read_text().strip()
        lines.append(f"  alert: family {path.name} paused: {reason}; delete {path} to resume it")
    idle_minutes = (clock - last_finish).total_seconds() / 60
    if pid and total["running"] and idle_minutes > stall_minutes:
        lines.append(
            f"  alert: stall, no job finished for {idle_minutes:.0f} minutes "
            f"while {total['running']} run"
        )
    if not pid and total["waiting"] + total["running"] + total["interrupted"]:
        lines.append("  alert: jobs are left but no runner is running")
    if left_automl and not gpus and allocation.get("gpu_problem"):
        lines.append(
            f"  note: automl jobs run on cpu ({left_automl} left), where their time budget buys "
            f"different fits than on a gpu: {allocation['gpu_problem']}"
        )
    lines += [f"  note: {note}" for note in notes]
    return "\n".join(lines)


def append_health(runs_dir: Path, stall_minutes: int, notes: Sequence[str] = ()) -> None:
    text = status(runs_dir, stall_minutes, notes)
    log = runs_dir / "logs" / "health.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n\n")
