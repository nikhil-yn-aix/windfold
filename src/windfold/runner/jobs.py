"""the job record, the sidecars that prove its outputs, the job log and the runner's records.

a job is done when every output exists, its sidecar records the job's fingerprint and the
output's sha256 matches its sidecar. the job log, jobs.jsonl, is append-only and written only by
the runner; every line is one event of one job.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

SIDECAR = ".sidecar.json"
EVENT_KEYS = ("time", "session", "job", "status", "family", "fingerprint")
NAME = r"^[A-Za-z0-9_.=-]+(/[A-Za-z0-9_.=-]+)*$"
TARGET = r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
DESCRIBED_BY = ("farm", "model", "split", "seed")


class Job(BaseModel):
    """one unit of work the planner builds and the runner runs.

    ids are unique across studies. target is "module:function"; the worker calls
    function(**args, out_dir=staging) and the function writes each of outputs under out_dir.
    outputs are relative to the runs folder, or to the data folder when root is "data". a job
    is ready when every job in needs is done. memory_gb is the reservation until the job's
    family has been measured on its farm; lower priority runs first, then plan order. an automl
    job takes a cpu slot and, when a gpu is usable, one of its automl seats; it never runs while
    cpu jobs share cores. with full_load, as for budget fits, it also runs beside load jobs that
    keep every idle core busy, and a stop of that load reruns it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=NAME)
    target: str = Field(pattern=TARGET)
    args: dict[str, JsonValue] = {}
    needs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = Field(min_length=1)
    root: Literal["runs", "data"] = "runs"
    fingerprint: str = Field(min_length=1)
    cores: int = Field(ge=1)
    gpu: bool = False
    memory_gb: float = Field(gt=0)
    family: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    priority: int = 0
    automl: bool = False
    full_load: bool = False

    @model_validator(mode="after")
    def full_load_is_automl(self) -> "Job":
        if self.full_load and not self.automl:
            raise ValueError(f"job {self.id} asks for full_load but is not an automl job")
        return self

    @field_validator("id")
    @classmethod
    def no_dot_parts(cls, value: str) -> str:
        if any(part in (".", "..") for part in value.split("/")):
            raise ValueError(f"job id {value} has a . or .. part")
        return value

    @field_validator("outputs")
    @classmethod
    def relative_files(cls, outputs: tuple[str, ...]) -> tuple[str, ...]:
        for output in outputs:
            path = PurePosixPath(output)
            if path.is_absolute() or ".." in path.parts or output != path.as_posix():
                raise ValueError(f"output {output} must be a plain relative path")
            if output.endswith(SIDECAR):
                raise ValueError(f"output {output} ends like a sidecar")
        if len(set(outputs)) != len(outputs):
            raise ValueError("outputs repeat")
        return outputs

    @property
    def farm(self) -> str | None:
        value = self.args.get("farm")
        return value if isinstance(value, str) else None

    @property
    def profile_key(self) -> str:
        return f"{self.family}/{self.farm}" if self.farm else self.family


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def sidecar_path(output: Path) -> Path:
    return output.with_name(output.name + SIDECAR)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unverified(job: Job, folder: Path) -> str | None:
    """why job's outputs under folder do not prove it done, or None when they do."""
    for name in job.outputs:
        output = folder / name
        if not output.is_file():
            return f"{name} is missing"
        try:
            record = json.loads(sidecar_path(output).read_text())
        except (OSError, ValueError):
            return f"{name} has no readable sidecar"
        if not proper_sidecar(record):
            return f"{name} has no readable sidecar"
        if record["fingerprint"] != job.fingerprint:
            return f"{name} was made for another fingerprint"
        if output.stat().st_size != record["bytes"]:
            return f"{name} has {output.stat().st_size} bytes, its sidecar says {record['bytes']}"
        if sha256(output) != record["sha256"]:
            return f"{name} does not match its checksum"
    return None


def proper_sidecar(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("fingerprint"), str)
        and isinstance(record.get("sha256"), str)
        and type(record.get("bytes")) is int
    )


def event(job: Job, status: str, session: str, **fields: Any) -> dict[str, Any]:
    described = {key: job.args[key] for key in DESCRIBED_BY if key in job.args}
    record = {
        "time": now(),
        "session": session,
        "job": job.id,
        "status": status,
        "family": job.family,
        **described,
        "gpu": job.gpu,
        "automl": job.automl,
        "fingerprint": job.fingerprint,
        **fields,
    }
    return {key: value for key, value in record.items() if value is not None}


def append_event(log: Path, record: dict[str, Any]) -> None:
    """append one line; a last line cut short by a crash is ended first, so it stays one bad
    line instead of swallowing this one."""
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    with log.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                line = b"\n" + line
        handle.write(line)


def event_in(line: str | bytes) -> dict[str, Any] | None:
    """the event a log line holds, or None for a line cut short or of another shape."""
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in EVENT_KEYS):
        return None
    return record


def read_events(log: Path) -> list[dict[str, Any]]:
    """every whole event of the job log; a line cut short by a crash is left out."""
    if not log.exists():
        return []
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return [record for line in lines if (record := event_in(line)) is not None]


READ_SO_FAR: dict[Path, tuple[int, int, dict[str, str], dict[str, str]]] = {}


def failed_for_good(runs_dir: Path) -> dict[str, str]:
    """job id to fingerprint for every job whose latest attempt, under the fingerprint it was
    last queued with, failed after its retry. a job queued again, with any fingerprint, leaves
    this list until it fails again. each call reads only what the log gained since the last."""
    log = runs_dir / "jobs.jsonl"
    try:
        size, inode = log.stat().st_size, log.stat().st_ino
    except FileNotFoundError:
        return {}
    known, offset, planned, failed = READ_SO_FAR.get(log, (inode, 0, {}, {}))
    if known != inode or offset > size:
        offset, planned, failed = 0, {}, {}
    with log.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(size - offset)
    end = chunk.rfind(b"\n") + 1
    for line in chunk[:end].splitlines():
        record = event_in(line)
        if record is None:
            continue
        job, status, fingerprint = record["job"], record["status"], record["fingerprint"]
        if status in ("queued", "skipped", "dropped"):
            planned[job] = fingerprint
            failed.pop(job, None)
        elif planned.get(job) == fingerprint:
            if status == "failed":
                failed[job] = fingerprint
            else:
                failed.pop(job, None)
    READ_SO_FAR[log] = (inode, offset + end, planned, failed)
    return dict(failed)


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record))
    os.replace(partial, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def stat_fields(pid: int) -> list[str] | None:
    """the fields of /proc/pid/stat after the command name, or None when pid has no process."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return text[text.rindex(")") + 2 :].split()


def start_ticks(pid: int) -> int | None:
    fields = stat_fields(pid)
    return int(fields[19]) if fields else None


def runner_alive(runs_dir: Path) -> int | None:
    """the pid of the runner that holds runs_dir, or None when that runner is gone."""
    record = read_json(runs_dir / "run.lock") or {}
    pid = record.get("pid")
    if isinstance(pid, int) and start_ticks(pid) == record.get("ticks"):
        return pid
    return None
