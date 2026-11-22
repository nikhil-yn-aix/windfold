import hashlib
import json

import pytest
from pydantic import ValidationError

from windfold.runner.jobs import (
    READ_SO_FAR,
    Job,
    append_event,
    event,
    failed_for_good,
    read_events,
    sidecar_path,
    unverified,
)

GOOD = {
    "id": "development/kelmarsh/windfold/arc_120_single",
    "target": "windfold.study:fit_split",
    "args": {"farm": "kelmarsh", "seed": 1, "split": "arc_120_single", "limits": [1, 2.5, None]},
    "outputs": ["development/predictions/kelmarsh/windfold/arc_120_single.parquet"],
    "fingerprint": "abc",
    "cores": 4,
    "memory_gb": 6.0,
    "family": "engine",
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "development/../kelmarsh"),
        ("id", "/development/kelmarsh"),
        ("id", "development//kelmarsh"),
        ("target", "windfold.study.fit_split"),
        ("args", {"when": {1, 2}}),
        ("outputs", ["../escape.parquet"]),
        ("outputs", ["/absolute.parquet"]),
        ("outputs", ["a/./b.parquet"]),
        ("outputs", ["x.parquet.sidecar.json"]),
        ("outputs", ["x.parquet", "x.parquet"]),
        ("outputs", []),
        ("root", "study"),
        ("cores", 0),
        ("family", "engine/kelmarsh"),
    ],
)
def test_bad_job_records_are_refused(field, value):
    with pytest.raises(ValidationError):
        Job.model_validate({**GOOD, field: value})


def test_a_reservation_is_the_peak_its_family_reached_on_its_farm_or_its_own_guess(bench):
    farms = ("kelmarsh", "penmanshiel", "smarteole")
    jobs = {farm: Job.model_validate({**GOOD, "args": {"farm": farm}}) for farm in farms}
    log = bench.runs / "jobs.jsonl"
    bench.runs.mkdir(parents=True)
    for farm, peaks in (("kelmarsh", (2.0, 1.0)), ("penmanshiel", (3.0, 4.0))):
        for peak in peaks:
            append_event(log, event(jobs[farm], "done", "s", seconds=1.0, peak_memory_gb=peak))
    runner = bench.runner([])
    runner.seed_profiles()
    assert [jobs[farm].profile_key for farm in farms[:2]] == [
        "engine/kelmarsh",
        "engine/penmanshiel",
    ]
    assert runner.reservation(jobs["kelmarsh"]) == 2.0 * 1.25
    assert runner.reservation(jobs["penmanshiel"]) == 4.0 * 1.25
    assert runner.reservation(jobs["smarteole"]) == GOOD["memory_gb"]


def test_only_a_matching_sidecar_proves_an_output(tmp_path):
    job = Job.model_validate({**GOOD, "outputs": ["p/a.bin"]})
    output = tmp_path / "p" / "a.bin"
    assert unverified(job, tmp_path) == "p/a.bin is missing"
    output.parent.mkdir()
    output.write_bytes(b"0123456789")
    assert unverified(job, tmp_path) == "p/a.bin has no readable sidecar"
    good = {"fingerprint": "abc", "sha256": hashlib.sha256(b"0123456789").hexdigest(), "bytes": 10}
    sidecar_path(output).write_text(json.dumps({**good, "fingerprint": "older"}))
    assert unverified(job, tmp_path) == "p/a.bin was made for another fingerprint"
    sidecar_path(output).write_text(json.dumps({**good, "bytes": 11}))
    assert unverified(job, tmp_path) == "p/a.bin has 10 bytes, its sidecar says 11"
    sidecar_path(output).write_text(json.dumps(good))
    assert unverified(job, tmp_path) is None
    output.write_bytes(b"0123456780")
    assert unverified(job, tmp_path) == "p/a.bin does not match its checksum"


def test_failed_for_good_follows_the_fingerprint_last_queued(tmp_path):
    log = tmp_path / "jobs.jsonl"

    def add(*events: tuple[str, str, str]) -> None:
        with log.open("a") as handle:
            for job, status, fingerprint in events:
                record = {"job": job, "status": status, "fingerprint": fingerprint}
                record |= {"time": "t", "session": "s", "family": "f"}
                handle.write(json.dumps(record) + "\n")

    tried = ("started", "retried", "started")
    add(("a", "queued", "1"), *[("a", s, "1") for s in tried], ("a", "failed", "1"))
    add(("b", "queued", "1"), *[("b", s, "1") for s in tried], ("b", "done", "1"))
    add(("c", "queued", "1"), ("c", "started", "1"), ("c", "queued", "2"), ("c", "failed", "1"))
    assert failed_for_good(tmp_path) == {"a": "1"}
    add(("a", "queued", "2"))
    assert failed_for_good(tmp_path) == {}
    add(*[("a", s, "2") for s in tried], ("a", "failed", "2"))
    with log.open("a") as handle:
        handle.write('{"job": "d", "status": "fai')
    assert failed_for_good(tmp_path) == {"a": "2"}
    READ_SO_FAR.clear()
    assert failed_for_good(tmp_path) == {"a": "2"}


@pytest.mark.parametrize(
    "sidecar",
    [
        "[]",
        "5",
        '{"fingerprint": "abc", "sha256": 7, "bytes": 10}',
        '{"fingerprint": "abc", "sha256": "x", "bytes": "10"}',
        '{"fingerprint": "abc", "sha256": "x", "bytes": true}',
    ],
)
def test_a_sidecar_of_the_wrong_shape_proves_nothing(tmp_path, sidecar):
    job = Job.model_validate({**GOOD, "outputs": ["a.bin"]})
    (tmp_path / "a.bin").write_bytes(b"0123456789")
    sidecar_path(tmp_path / "a.bin").write_text(sidecar)
    assert unverified(job, tmp_path) == "a.bin has no readable sidecar"


def test_an_event_after_a_line_cut_short_is_kept(tmp_path):
    log = tmp_path / "jobs.jsonl"
    base = {"time": "t", "session": "s", "job": "a", "family": "f", "fingerprint": "1"}
    log.write_text(json.dumps({**base, "status": "queued"}) + "\n[]\n" + '{"time": "t", "sess')
    append_event(log, {**base, "status": "failed"})
    assert [e["status"] for e in read_events(log)] == ["queued", "failed"]
    assert failed_for_good(tmp_path) == {"a": "1"}
