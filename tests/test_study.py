import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from windfold import inflow
from windfold.cli import app
from windfold.evaluation.paths import reduction_file
from windfold.evaluation.reductions import ARCS
from windfold.locations import study_folder
from windfold.runner.jobs import Job, sidecar_path
from windfold.study import (
    Paths,
    Planner,
    development_results,
    freeze_first,
    freeze_second,
    plan,
)
from windfold.tasks import evaluation

SPLITS = [
    ("random_1", "random", "included"),
    ("random_2", "random", "included"),
    ("time_1", "time_blocked", "included"),
    ("time_2", "time_blocked", "included"),
    ("arc_000_single", "arc_single", "included"),
    ("arc_030_single", "arc_single", "included"),
    ("arc_060_single", "arc_single", "included"),
    ("arc_090_single", "arc_single", "ineligible"),
    ("history_007d_from_2018-01-01", "history", "included"),
    ("history_030d_from_2018-01-01", "history", "included"),
    ("threshold_030d_from_2018-01-01", "threshold", "included"),
]
MONITORS = [
    "direction_lookup",
    "cyclic_gam",
    "pywake",
    "pywake_inflow",
    "windfold",
    "windfold_constant_inflow",
]


def summarise(paths: Paths, study: str, farm: str, sensitivity: str | None = None) -> None:
    """a manifest summary as its manifest job would leave it, sidecar included."""
    folder = study_folder(study, sensitivity)
    summary = paths.runs / folder / "manifests" / f"{farm}_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(SPLITS, schema=["split", "regime", "status"], orient="row").write_csv(summary)
    (job,) = [j for j in plan((study,), paths) if j.id == f"{folder}/manifests/{farm}"]
    sidecar = summary.with_name(summary.name + ".sidecar.json")
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps({"fingerprint": job.fingerprint, "sha256": digest}))


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Paths:
    """a copy of the packaged configs and runs with summaries of kelmarsh. the source and the
    installed packages do not change within these tests, so the planner keeps their hashes
    between plans instead of forgetting them at every one."""
    monkeypatch.setattr("windfold.study.forget", lambda: None)
    configs = tmp_path / "configs"
    shutil.copytree(Path(str(resources.files("windfold") / "configs")), configs)
    paths = Paths(tmp_path / "data", tmp_path / "runs", configs)
    for study in ("development", "smoke"):
        summarise(paths, study, "kelmarsh")
    return paths


def release(paths: Paths, families: str) -> None:
    (paths.runs / "release.toml").write_text(f'farms = ["kelmarsh"]\nfamilies = [{families}]\n')


def engine(paths: Paths, study: str) -> dict[str, list[str]]:
    jobs = plan((study,), paths)
    prefix = f"{study}/engine/kelmarsh/"
    batches = {}
    for job in jobs:
        models = job.args["models"] if job.id.startswith(prefix) else None
        if isinstance(models, list):
            batches[job.id.removeprefix(prefix)] = [str(model) for model in models]
    return batches


def test_model_jobs_wait_for_their_release(folders: Paths):
    assert engine(folders, "development") == {}
    release(folders, '"graph"')
    assert engine(folders, "development") == {}


def test_a_released_farm_fits_every_included_split_and_monitors_only_on_thresholds(folders: Paths):
    release(folders, '"engine"')
    batches = engine(folders, "development")
    included = {name for name, _, status in SPLITS if status == "included"}
    assert set(batches) == included
    assert len(batches["random_1"]) == 23
    assert batches["threshold_030d_from_2018-01-01"] == MONITORS


def test_the_smoke_study_takes_its_declared_subset(folders: Paths):
    release(folders, "")
    assert set(engine(folders, "smoke")) == {
        "random_1",
        "time_1",
        "arc_000_single",
        "arc_030_single",
        "history_030d_from_2018-01-01",
        "threshold_030d_from_2018-01-01",
    }


def test_a_physics_setting_reruns_physics_and_fits_but_not_the_data_before_it(folders: Paths):
    release(folders, '"engine"')
    before = {j.id: j.fingerprint for j in plan(("development",), folders)}
    assert folders.configs is not None
    protocol = folders.configs / "protocol.toml"
    text = protocol.read_text()
    assert text.count("trials = 100\n") == 1
    protocol.write_text(text.replace("trials = 100\n", "trials = 101\n", 1))
    after = {j.id: j.fingerprint for j in plan(("development",), folders)}
    changed = {job for job in before if before[job] != after[job]}
    assert "physics_fit/kelmarsh" in changed and "development/engine/kelmarsh/random_1" in changed
    assert not changed & {"tables/kelmarsh", "prepared/kelmarsh", "development/manifests/kelmarsh"}


def learner_jobs(paths: Paths, study: str, family: str) -> dict[str, tuple[str, ...]]:
    return {j.id: j.needs for j in plan((study,), paths) if j.family == family}


def test_a_learner_fit_waits_for_its_released_calibration_outputs(folders: Paths):
    release(folders, '"gbm"')
    assert learner_jobs(folders, "development", "gbm") == {}
    release(folders, '"gbm", "tuning"')
    fits = learner_jobs(folders, "development", "gbm")
    assert len(fits) == 6 * 9 * 3 + 2 * 3
    needs = fits["development/identity_gbm/kelmarsh/random_1_seed0"]
    assert "calibration/tuning/kelmarsh/identity_gbm" in needs


def test_the_smoke_study_fits_learners_without_calibration_outputs(folders: Paths):
    release(folders, "")
    assert learner_jobs(folders, "development", "gbm") == {}
    gbm = learner_jobs(folders, "smoke", "gbm")
    assert gbm and not any(n.startswith("calibration/") for needs in gbm.values() for n in needs)
    assert {j.split("/")[1] for j in learner_jobs(folders, "smoke", "pretrained_graph")} == {
        "pretrained_graph"
    }


def test_every_automl_fit_starts_before_the_other_fits(folders: Paths):
    release(folders, '"automl", "automl_budget", "gbm", "tuning"')
    jobs = plan(("development",), folders)
    automl = {j.id: j.priority for j in jobs if j.family == "automl"}
    others = [j.priority for j in jobs if j.family == "gbm"]
    assert any("history" in job for job in automl) and others
    assert max(automl.values()) < min(others)


def test_a_confirmatory_decision_reads_the_same_analysis_of_the_development_data():
    jobs = [
        "development/results/kelmarsh",
        "development/sensitivities/fixed_shrinkage/results/kelmarsh",
        "development/sensitivities/binned_kernel/results/kelmarsh",
        "development/decision",
    ]
    assert development_results(jobs, None) == ["development/results/kelmarsh"]
    assert development_results(jobs, "fixed_shrinkage") == [
        "development/sensitivities/fixed_shrinkage/results/kelmarsh"
    ]


def test_confirmatory_fits_run_only_as_frozen(folders: Paths):
    summarise(folders, "confirmatory", "penmanshiel")
    (folders.runs / "release.toml").write_text('farms = ["penmanshiel"]\nfamilies = ["engine"]\n')
    fits = {j.id for j in plan(("confirmatory",), folders) if j.family == "engine"}
    assert fits == set()
    record = freeze_first("confirmatory", folders)
    sensitivity = "confirmatory/sensitivities/fixed_shrinkage/engine/penmanshiel/"
    assert len(record["jobs"]) == 20
    assert sum(job.startswith(sensitivity) for job in record["jobs"]) == 10
    record["jobs"]["confirmatory/engine/penmanshiel/random_1"] = "an older fingerprint"
    path = folders.runs / "confirmatory" / "freeze_first.json"
    path.write_text(json.dumps(record))
    fits = {j.id for j in plan(("confirmatory",), folders) if j.family == "engine"}
    assert len(fits) == 19 and "confirmatory/engine/penmanshiel/random_1" not in fits
    assert f"{sensitivity}random_1" in fits


def test_the_second_record_needs_every_fit_of_the_first_as_frozen(folders: Paths, monkeypatch):
    summarise(folders, "confirmatory", "penmanshiel")
    (folders.runs / "release.toml").write_text('farms = ["penmanshiel"]\nfamilies = ["engine"]\n')
    record = freeze_first("confirmatory", folders)
    path = folders.runs / "confirmatory" / "freeze_first.json"
    path.write_text(json.dumps(record))
    fits = [j for j in plan(("confirmatory",), folders) if j.family == "engine"]
    for job in fits:
        for output in job.outputs:
            target = folders.runs / output
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(job.id)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            proof = {"fingerprint": job.fingerprint, "sha256": digest, "bytes": len(job.id)}
            sidecar_path(target).write_text(json.dumps(proof))
    monkeypatch.setattr(evaluation, "second_record", lambda runs_dir, configs: {})
    outputs = {output for job in fits for output in job.outputs}
    assert set(freeze_second("confirmatory", folders)["predictions"]) == outputs
    record["jobs"]["confirmatory/engine/penmanshiel/random_1"] = "an older fingerprint"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="confirmatory/engine/penmanshiel/random_1"):
        freeze_second("confirmatory", folders)


def test_a_summary_from_an_older_manifest_plans_no_fits(folders: Paths):
    release(folders, '"engine"')
    assert engine(folders, "development")
    assert folders.configs is not None
    protocol = folders.configs / "protocol.toml"
    text = protocol.read_text()
    protocol.write_text(text.replace("arc_width_deg = 30.0", "arc_width_deg = 60.0", 1))
    protocol.write_text(protocol.read_text().replace("arc_count = 12", "arc_count = 6", 1))
    assert engine(folders, "development") == {}


def test_sensitivity_results_read_what_they_do_not_refit_from_the_main_study(folders: Paths):
    assert folders.configs is not None
    smoke = folders.configs / "studies" / "smoke.toml"
    text = smoke.read_text()
    text = text.replace(
        'farms = ["smarteole", "kelmarsh", "hill_of_towie", "penmanshiel"]', 'farms = ["kelmarsh"]'
    )
    sensitivities = 'sensitivities = ["fixed_shrinkage", "arc_buffer_0deg"]\n'
    smoke.write_text(text.replace("\n[limits]", f"{sensitivities}\n[limits]", 1))
    summarise(folders, "smoke", "kelmarsh")
    summarise(folders, "smoke", "kelmarsh", "arc_buffer_0deg")
    release(folders, "")
    jobs = {j.id: j for j in plan(("smoke",), folders)}
    shrinkage = jobs["smoke/sensitivities/fixed_shrinkage/reduce/kelmarsh"]
    fits = shrinkage.args["fits"]
    assert isinstance(fits, list)
    models = {str(fit[0]) for fit in fits if isinstance(fit, list)}
    assert {"no_wake", "direction_lookup", "pywake", "identity_gbm", "windfold"} <= models
    assert "smoke/engine/kelmarsh/random_1" in shrinkage.needs
    borrowing = jobs["smoke/sensitivities/arc_buffer_0deg/results/kelmarsh"]
    assert "smoke/reduce/kelmarsh" in borrowing.needs
    own = jobs["smoke/sensitivities/fixed_shrinkage/results/kelmarsh"]
    assert "smoke/reduce/kelmarsh" not in own.needs
    assert "smoke/sensitivities/arc_buffer_0deg/decision" in jobs
    tables = jobs["smoke/report/tables"]
    assert tables.needs == (
        "smoke/decision",
        "smoke/sensitivities/fixed_shrinkage/decision",
        "smoke/sensitivities/arc_buffer_0deg/decision",
    )
    assert tables.args["sensitivities"] == {"smoke": ["fixed_shrinkage", "arc_buffer_0deg"]}
    assert jobs["smoke/report/figures/validation"].cores == 1
    assert "smoke/report/figures/validation_confirmatory" not in jobs


def test_the_kernel_transfer_table_reads_the_main_and_windfold_refitting_reductions(
    folders: Paths,
):
    assert folders.configs is not None
    (folders.configs / "studies" / "development.toml").write_text(
        'name = "development"\nperiod = "development"\nfarms = ["kelmarsh"]\n'
        'models = ["no_wake", "pywake", "kernel_linear", "windfold", "windfold_no_kernel"]\n'
        'sensitivities = ["fixed_shrinkage", "fixed_physics_scale", "arc_buffer_0deg"]\n'
    )
    summarise(folders, "development", "kelmarsh")
    summarise(folders, "development", "kelmarsh", "arc_buffer_0deg")
    release(folders, '"engine", "evaluation"')
    jobs = {j.id: j for j in plan(("development",), folders)}
    assert jobs["development/kernel_transfer/kelmarsh"].needs == (
        "development/reduce/kelmarsh",
        "development/sensitivities/fixed_physics_scale/reduce/kelmarsh",
    )


def test_measured_inflow_fits_wait_for_their_farm_inflow_table(folders: Paths):
    release(folders, '"engine", "gbm", "graph", "tuning"')
    jobs = {j.id: j for j in plan(("development",), folders)}
    tables = {job for job in jobs if job.startswith("inflow_physics/")}
    assert {"inflow_physics/kelmarsh", "inflow_physics/smarteole"} <= tables
    smarteole = jobs["inflow_physics/smarteole"]
    calibrated = smarteole.args["calibrated_farms"]
    assert isinstance(calibrated, list) and calibrated
    assert {f"inflow_physics/{farm}" for farm in calibrated} <= set(smarteole.needs)
    assert jobs["inflow_physics/kelmarsh"].args["calibrated_farms"] == []
    assert {f"prepared/kelmarsh/{table}" for table in inflow.TABLES.values()} <= set(
        jobs["inflow_physics/kelmarsh"].outputs
    )
    assert "inflow_physics/kelmarsh" in jobs["development/engine/kelmarsh/random_1"].needs
    for fit, measured in (
        ("geometry_gbm_inflow/kelmarsh/random_1_seed0", True),
        ("geometry_gbm/kelmarsh/random_1_seed0", False),
        ("wind_frame_graph_inflow/kelmarsh/random_1_seed0", True),
    ):
        assert ("inflow_physics/kelmarsh" in jobs[f"development/{fit}"].needs) == measured
    tuning = jobs["calibration/tuning/kelmarsh/geometry_gbm_inflow/00"]
    assert "inflow_physics/kelmarsh" in tuning.needs


def test_matched_monitors_get_threshold_fits_beside_the_planned_ones(folders: Paths):
    release(folders, '"engine", "gbm", "tuning"')
    jobs = {j.id for j in plan(("development",), folders)}
    threshold = "threshold_030d_from_2018-01-01"
    assert f"development/identity_gbm_inflow/kelmarsh/{threshold}_seed0" in jobs
    assert f"development/identity_gbm/kelmarsh/{threshold}_seed0" in jobs
    assert f"development/geometry_gbm_inflow/kelmarsh/{threshold}_seed0" not in jobs


def small_pair(paths: Paths) -> None:
    """development and confirmatory studies of penmanshiel with two models, summarized and
    released for engine fits and evaluation."""
    assert paths.configs is not None
    for name in ("development", "confirmatory"):
        (paths.configs / "studies" / f"{name}.toml").write_text(
            f'name = "{name}"\nperiod = "{name}"\nfarms = ["penmanshiel"]\n'
            'models = ["no_wake", "windfold"]\n'
        )
        summarise(paths, name, "penmanshiel")
    families = '"engine", "evaluation"'
    (paths.runs / "release.toml").write_text(f'farms = ["penmanshiel"]\nfamilies = [{families}]\n')


def finish(paths: Paths, job: Job) -> dict[str, str]:
    """a job's outputs and sidecars as the runner leaves them, and their sha256."""
    digests = {}
    for output in job.outputs:
        target = paths.runs / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        proof = {"fingerprint": job.fingerprint, "sha256": digest, "bytes": len(output)}
        sidecar_path(target).write_text(json.dumps(proof))
        digests[output] = digest
    return digests


def test_the_first_record_holds_the_checksums_of_the_calibration_outputs_its_fits_need(
    folders: Paths,
):
    summarise(folders, "confirmatory", "penmanshiel")
    families = '"gbm", "tuning"'
    (folders.runs / "release.toml").write_text(
        f'farms = ["penmanshiel"]\nfamilies = [{families}]\n'
    )
    with pytest.raises(ValueError, match="calibration/tuning/penmanshiel/.* has not finished"):
        freeze_first("confirmatory", folders)
    jobs = {j.id: j for j in plan(("confirmatory",), folders)}
    checksums = {}
    for side in ("identity", "geometry", "physics_guided"):
        for suffix in ("", "_inflow"):
            checksums |= finish(folders, jobs[f"calibration/tuning/penmanshiel/{side}_gbm{suffix}"])
    finish(folders, jobs["calibration/tuning/penmanshiel/absolute_graph"])
    record = freeze_first("confirmatory", folders)
    assert record["calibration_outputs"] == checksums
    assert record["jobs"] and all("_gbm" in job for job in record["jobs"])
    changed = min(checksums)
    output = folders.runs / changed
    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(ValueError, match=f"{re.escape(changed)} has sha256 .*, its sidecar"):
        freeze_first("confirmatory", folders)
    output.unlink()
    with pytest.raises(ValueError, match=f"{re.escape(changed)} is missing"):
        freeze_first("confirmatory", folders)


def test_a_fit_planned_otherwise_than_frozen_takes_its_confirmatory_dependants_with_it(
    folders: Paths,
):
    small_pair(folders)
    record = freeze_first("confirmatory", folders)
    first = folders.runs / "confirmatory" / "freeze_first.json"
    first.write_text(json.dumps(record))
    (folders.runs / "confirmatory" / "freeze_second.json").write_text("{}")
    studies = ("development", "confirmatory")
    before = {j.id for j in plan(studies, folders)}
    changed = "confirmatory/engine/penmanshiel/random_1"
    gated = {
        changed,
        "confirmatory/reduce/penmanshiel",
        "confirmatory/results/penmanshiel",
        "confirmatory/decision",
        "report/tables",
    }
    assert gated <= before
    record["jobs"][changed] = "an older fingerprint"
    first.write_text(json.dumps(record))
    dropped = before - {j.id for j in plan(studies, folders)}
    assert gated <= dropped and all(job.startswith("report/") for job in dropped - gated)
    assert "confirmatory/engine/penmanshiel/arc_000_single" not in dropped


def failed(paths: Paths, job: str, fingerprint: str) -> None:
    """the runner's log of a job queued with fingerprint and failed after its retry."""
    with (paths.runs / "jobs.jsonl").open("a") as log:
        for status in ("queued", "failed"):
            event = {"time": "2026-11-18T00:00:00+00:00", "session": "s", "job": job}
            event |= {"status": status, "family": "engine", "fingerprint": fingerprint}
            log.write(json.dumps(event) + "\n")


def test_the_second_record_hashes_finished_fits_excuses_lost_ones_and_maps_development_arcs(
    folders: Paths,
):
    small_pair(folders)
    (folders.runs / "confirmatory" / "freeze_first.json").write_text(
        json.dumps(freeze_first("confirmatory", folders))
    )
    fits = {j.id: j for j in plan(("confirmatory",), folders) if j.family == "engine"}
    late = "confirmatory/engine/penmanshiel/random_1"
    outputs = {}
    for job_id, job in fits.items():
        if job_id != late:
            outputs |= finish(folders, job)
    arcs = {
        "split": ["arc_000_single", "arc_030_single"],
        "regime": ["arc_single"] * 2,
        "arc": [0, 1],
        "score": [1.0, 3.0],
        "permuted_score": [0.0, 0.0],
        "wake_strength": [0.5, 0.5],
        "angular_distance": [10.0, 10.0],
        "training_rows": [100.0, 900.0],
        "dissimilarity": [0.2, 0.2],
        "relational_mse": [2.0, 1.0],
        "physics_mse": [1.0, 2.0],
    }
    for farm in ("kelmarsh", "hill_of_towie", "smarteole"):
        path = folders.runs / reduction_file("development", farm, None, "arcs")
        path.parent.mkdir(parents=True)
        pl.DataFrame(arcs, schema=ARCS).write_parquet(path)
    with pytest.raises(
        ValueError, match=f"1 frozen fits of confirmatory have not finished: {late}"
    ):
        freeze_second("confirmatory", folders)
    failed(folders, late, "an older fingerprint")
    with pytest.raises(ValueError, match=late):
        freeze_second("confirmatory", folders)
    failed(folders, late, fits[late].fingerprint)
    record = freeze_second("confirmatory", folders)
    assert record["predictions"] == outputs
    assert record["confirmation_farms"] == ["kelmarsh", "hill_of_towie", "smarteole"]
    assert record["simple_predictor"] == {"name": "training_rows", "direction": -1.0}
    coverage = record["coverage_confirmation"]
    assert (coverage["centre"], coverage["spread"]) == (2.0, 1.0) and coverage["slope"] < 0


def test_a_freeze_record_is_replaced_only_when_asked_and_the_old_one_is_kept_whole(
    folders: Paths,
):
    small_pair(folders)
    assert folders.configs is not None
    command = ["freeze", "first", "--data", str(folders.data), "--runs", str(folders.runs)]
    command += ["--configs", str(folders.configs)]
    done = CliRunner().invoke(app, command)
    assert done.exit_code == 0, done.output
    target = folders.runs / "confirmatory" / "freeze_first.json"
    written = json.loads(target.read_text())
    assert written["jobs"] == freeze_first("confirmatory", folders)["jobs"]
    assert datetime.fromisoformat(written["written"]).utcoffset() == timedelta(0)
    assert written["commit"] is None or re.fullmatch("[0-9a-f]{40}", written["commit"])
    assert isinstance(written["uncommitted"], bool)
    before = target.read_bytes()
    refused = CliRunner().invoke(app, command)
    assert refused.exit_code == 1 and "pass --again" in refused.output
    assert target.read_bytes() == before
    again = CliRunner().invoke(app, [*command, "--again"])
    assert again.exit_code == 0, again.output
    (kept,) = (folders.runs / "confirmatory").glob("freeze_first_replaced_*.json")
    assert kept.read_bytes() == before and target.read_bytes() != before


def test_a_change_reaches_the_jobs_that_read_it_and_no_others(folders: Paths, monkeypatch):
    release(folders, '"engine", "gbm", "tuning"')
    configs = folders.configs
    assert configs is not None
    base = {j.id: j.fingerprint for j in plan(("development",), folders)}

    def changed(name: str, old: str, new: str) -> tuple[set[str], set[str]]:
        """the jobs whose fingerprints change, and the jobs that appear or vanish."""
        path = configs / name
        text = path.read_text()
        assert text.count(old) == 1
        path.write_text(text.replace(old, new))
        after = {j.id: j.fingerprint for j in plan(("development",), folders)}
        path.write_text(text)
        return {j for j in base.keys() & after.keys() if base[j] != after[j]}, base.keys() ^ after

    stable = changed("resources.toml", "reserved_threads = 2", "reserved_threads = 3")
    assert stable == (set(), set())
    refit, renamed = changed("protocol.toml", "model = [0, 1, 2]", "model = [0, 1, 5]")
    seeded = "development/identity_gbm/kelmarsh/random_1_seed"
    assert {f"{seeded}2", f"{seeded}5"} <= renamed and f"{seeded}0" in refit
    assert not refit & {"tables/kelmarsh", "prepared/kelmarsh", "physics_fit/kelmarsh"}
    features = 'features = ["turbine", "direction", "reference_speed"]'
    tuned, _ = changed(
        "models.toml",
        features,
        features.replace('"turbine", "direction"', '"direction", "turbine"'),
    )
    identity = {"identity_gbm", "identity_gbm_inflow"}
    assert {j.split("/")[1] for j in tuned if j.startswith("development/")} == identity
    assert {j.split("/")[3] for j in tuned if j.startswith("calibration/")} == identity
    monkeypatch.setattr("windfold.study.environment_hash", lambda: "another environment")
    assert all(base[j.id] != j.fingerprint for j in plan(("development",), folders))
    planner = Planner(("development",), folders)
    target, written = "windfold.tasks.data:read_tables", ("tables/kelmarsh/rows.parquet",)
    assert planner.add("read", target, {"farm": "kelmarsh"}, written, "data", (), "tables", 0)
    with pytest.raises(ValueError, match=r"does not take \['colour'\] and needs \[\]"):
        planner.add(
            "extra", target, {"farm": "kelmarsh", "colour": "red"}, written, "data", (), "t", 0
        )
    with pytest.raises(ValueError, match=r"does not take \[\] and needs \['farm'\]"):
        planner.add("missing", target, {}, written, "data", (), "tables", 0)
