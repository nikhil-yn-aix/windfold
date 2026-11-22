import json
from importlib import resources

import polars as pl
import pytest
from typer.testing import CliRunner

from windfold.analyses.development import prediction_file
from windfold.analyses.linear_clipping import linear_clipping
from windfold.cli import app


def test_analyse_writes_the_result_under_the_runs_folders_development_analyses(linear_pair):
    folders = ["--data", str(linear_pair.data), "--runs", str(linear_pair.runs)]
    done = CliRunner().invoke(app, ["analyse", "linear_clipping", "--farm", "kelmarsh", *folders])
    assert done.exit_code == 0, done.output
    written = json.loads(
        (linear_pair.runs / "development" / "analyses" / "linear_clipping.json").read_text()
    )
    assert written["analysis"] == "linear_clipping" and written["farms"] == ["kelmarsh"]
    assert written["result"] == linear_clipping(linear_pair.data, linear_pair.runs, ("kelmarsh",))


def test_analyse_refuses_unknown_names_and_heavy_analyses_without_their_refits(linear_pair):
    folders = ["--data", str(linear_pair.data), "--runs", str(linear_pair.runs)]
    unknown = CliRunner().invoke(app, ["analyse", "kernel", *folders])
    assert unknown.exit_code == 1
    assert "the analyses are arc_variants, comparison_ranks, default_physics" in unknown.output
    heavy = CliRunner().invoke(
        app, ["analyse", "strict_calibration", "--farm", "kelmarsh", *folders]
    )
    assert heavy.exit_code == 1 and "--refit" in heavy.output
    assert not (linear_pair.runs / "development" / "analyses").exists()


def test_analyse_refuses_a_farm_the_analysis_does_not_cover(linear_pair):
    folders = ["--data", str(linear_pair.data), "--runs", str(linear_pair.runs)]
    done = CliRunner().invoke(app, ["analyse", "lidar_turbulence", "--farm", "kelmarsh", *folders])
    assert done.exit_code == 1
    assert "lidar_turbulence does not analyse kelmarsh; only smarteole" in done.output
    assert not (linear_pair.runs / "development" / "analyses").exists()


def test_analyse_reads_a_confirmatory_study_only_after_its_second_freeze(confirmatory_study):
    folders = ["--data", str(confirmatory_study.data), "--runs", str(confirmatory_study.runs)]
    command = ["analyse", "practical_scale", "--study", "confirmatory", *folders]
    early = CliRunner().invoke(app, command)
    assert early.exit_code == 1 and "second freeze" in early.output
    other = CliRunner().invoke(
        app, ["analyse", "linear_clipping", "--study", "confirmatory", *folders]
    )
    assert other.exit_code == 1 and "development study only" in other.output
    target = confirmatory_study.runs / "confirmatory" / "analyses" / "practical_scale.json"
    assert not target.parent.exists()
    (confirmatory_study.runs / "confirmatory" / "freeze_second.json").write_text("{}")
    done = CliRunner().invoke(app, command)
    assert done.exit_code == 0, done.output
    written = json.loads(target.read_text())
    assert (written["study"], written["farms"]) == ("confirmatory", ["penmanshiel"])
    assert written["result"]["penmanshiel"]["strongest_learned"] == "identity_gbm_inflow"


@pytest.mark.parametrize(
    "name", ["comparison_ranks", "interval_score", "pair_skill", "sector_gains"]
)
def test_the_analyses_of_either_study_refuse_a_confirmatory_study_before_its_second_freeze(
    confirmatory_study, name
):
    folders = ["--data", str(confirmatory_study.data), "--runs", str(confirmatory_study.runs)]
    done = CliRunner().invoke(app, ["analyse", name, "--study", "confirmatory", *folders])
    assert done.exit_code == 1 and "second freeze" in done.output
    assert not (confirmatory_study.runs / "confirmatory" / "analyses").exists()


def test_analyse_follows_its_folder_options_and_a_failed_rerun_keeps_the_last_result(
    linear_pair, tmp_path
):
    packaged = resources.files("windfold") / "configs" / "resources.toml"
    text = packaged.read_text().replace('data_dir = "data"', f'data_dir = "{linear_pair.data}"')
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "resources.toml").write_text(
        text.replace('runs_dir = "runs"', f'runs_dir = "{linear_pair.runs}"')
    )
    out = tmp_path / "elsewhere" / "clipping.json"
    command = ["analyse", "linear_clipping", "--farm", "kelmarsh", "--configs", str(configs)]
    done = CliRunner().invoke(app, [*command, "--out", str(out)])
    assert done.exit_code == 0, done.output
    written = json.loads(out.read_text())
    assert (written["data"], written["runs"]) == (str(linear_pair.data), str(linear_pair.runs))
    assert written["result"] == linear_clipping(linear_pair.data, linear_pair.runs, ("kelmarsh",))
    assert not (linear_pair.runs / "development" / "analyses").exists()
    path = prediction_file(linear_pair.runs, "kelmarsh", "kernel_linear", "arc_000_single")
    pl.read_parquet(path).reverse().write_parquet(path)
    before = out.read_bytes()
    failed = CliRunner().invoke(app, [*command, "--out", str(out)])
    assert failed.exit_code == 1 and "do not predict the same rows" in failed.output
    assert out.read_bytes() == before and [p.name for p in out.parent.iterdir()] == [out.name]


GITS = {
    "missing": (None, (None, None)),
    "outside a checkout": ("echo 'fatal: not a git repository' >&2\nexit 128", (None, None)),
    "in a changed checkout": (
        '[ "$1" = rev-parse ] && echo 0123abc || echo " M cli.py"',
        ("0123abc", True),
    ),
}


@pytest.mark.parametrize(("git", "recorded"), GITS.values(), ids=list(GITS))
def test_analyse_records_the_commit_git_can_say_and_null_otherwise(
    linear_pair, tmp_path, monkeypatch, git, recorded
):
    path = tmp_path / "path"
    path.mkdir()
    if git:
        (path / "git").write_text(f"#!/bin/sh\n{git}\n")
        (path / "git").chmod(0o755)
    monkeypatch.setenv("PATH", str(path))
    folders = ["--data", str(linear_pair.data), "--runs", str(linear_pair.runs)]
    done = CliRunner().invoke(app, ["analyse", "linear_clipping", "--farm", "kelmarsh", *folders])
    assert done.exit_code == 0, done.output
    written = json.loads(
        (linear_pair.runs / "development" / "analyses" / "linear_clipping.json").read_text()
    )
    assert (written["commit"], written["uncommitted"]) == recorded
    assert written["result"] == linear_clipping(linear_pair.data, linear_pair.runs, ("kelmarsh",))
