import json
import os
from datetime import UTC, datetime, timedelta, tzinfo

from windfold.runner import health
from windfold.runner.health import status
from windfold.runner.jobs import start_ticks


class Pinned(datetime):
    """a clock stopped at noon, for the events and for the view built from them."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> "Pinned":
        return cls(2026, 11, 18, 12, 0, tzinfo=UTC)


def at(minutes_ago: float) -> str:
    return (Pinned.now() - timedelta(minutes=minutes_ago)).isoformat(timespec="milliseconds")


def test_status_of_a_runs_folder_not_made_yet(tmp_path):
    view = status(tmp_path / "runs", stall_minutes=20)
    assert "no runner running" in view and "no jobs yet" in view


def test_status_counts_time_left_alerts_and_stall(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "datetime", Pinned)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run.lock").write_text(
        json.dumps({"pid": os.getpid(), "ticks": start_ticks(os.getpid())})
    )
    (runs / "allocation.json").write_text(
        json.dumps({"pools": {"cpu_slots": [[0], [1]], "gpus": [], "gpu_workers_per_device": 2}})
    )
    old, now = at(600), at(40)
    engine = {
        "family": "engine",
        "farm": "kelmarsh",
        "gpu": False,
        "automl": False,
        "fingerprint": "made",
    }
    events = [
        {"session": old, "time": old, "job": f"e/{s}", "status": "done", "seconds": s, **engine}
        for s in (1000, 2000, 3000)
    ]
    events += [
        {"session": now, "time": now, "job": f"e/w{k}", "status": "queued", **engine}
        for k in range(3)
    ]
    events += [
        {"session": now, "time": at(10), "job": "e/r", "status": "started", **engine},
        {
            "session": now,
            "time": at(30),
            "job": "e/f",
            "status": "failed",
            "kind": "nonfinite",
            "error": "Traceback\nValueError: x",
            **engine,
        },
        {
            "session": now,
            "time": now,
            "job": "g/q",
            "status": "queued",
            "family": "gbm",
            "fingerprint": "made",
        },
        {
            "session": now,
            "time": now,
            "job": "g/d",
            "status": "dropped",
            "family": "gbm",
            "fingerprint": "made",
        },
    ]
    (runs / "jobs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (runs / "paused").mkdir()
    (runs / "paused" / "engine").write_text("e/f wrote non-finite values\n")
    lines = status(runs, stall_minutes=20).splitlines()
    assert lines[0].endswith(f"runner pid {os.getpid()} running")
    assert lines[1] == (
        "  1 failed, 1 running, 4 waiting of 6 jobs; time left about 1.0 h, "
        "not counting gbm (no measured job yet)"
    )
    assert "  engine kelmarsh: 1 failed, 1 running, 3 waiting" in lines
    assert "  gbm: 1 waiting" in lines
    assert "  alert: failed e/f (non-finite output): ValueError: x" in lines
    assert any(line.startswith("  alert: family engine paused") for line in lines)
    assert "  alert: stall, no job finished for 30 minutes while 1 run" in lines
    (runs / "run.lock").write_text(json.dumps({"pid": os.getpid(), "ticks": -1}))
    stopped = status(runs, stall_minutes=20)
    assert "no runner running" in stopped and "1 interrupted" in stopped
    assert "jobs are left but no runner is running" in stopped and "stall" not in stopped
