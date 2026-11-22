"""the windfold command."""

import json
import sys
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Annotated

import typer

from windfold import config

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="learn wind-farm wake interactions from SCADA."
)


@app.callback()
def main() -> None:
    """learn wind-farm wake interactions from SCADA."""


@app.command()
def configs() -> None:
    """load and check every packaged config."""
    from windfold.report.style import load_report

    root = config.config_root()
    protocol = config.load_protocol()
    models = config.load_models()
    config.check_model_names(protocol, models)
    config.load_resources()
    load_report()
    farms = [config.load_farm(name) for name in config.names_in(root, "farms")]
    turbines = [config.load_turbine(name) for name in config.names_in(root, "turbines")]
    tuning = [config.load_menu(name) for name in config.names_in(root, "menus")]
    studies = [config.load_study(name) for name in config.names_in(root, "studies")]
    typer.echo(
        f"{len(farms)} farms, {len(turbines)} turbines, {len(models)} models, "
        f"{len(tuning)} tuning menus, {len(studies)} studies, "
        f"{len(protocol.sensitivities)} sensitivities and the report: all valid"
    )


@app.command()
def menus(
    configs: Annotated[
        Path | None, typer.Option(help="configs folder to write into; default packaged")
    ] = None,
) -> None:
    """draw the frozen tuning menus once from the protocol's search spaces; never redraws."""
    from windfold.models.menus import write_menus

    protocol = config.load_protocol(configs)
    root = configs or Path(str(resources.files("windfold") / "configs"))
    spaces = {"gbm": protocol.menus.gbm, "graph": protocol.menus.graph}
    for path in write_menus(root / "menus", spaces, protocol.menus.size, protocol.seeds.study):
        typer.echo(f"wrote {path}")


ConfigsOption = Annotated[
    Path | None, typer.Option("--configs", help="configs folder; default the packaged one")
]
DataOption = Annotated[
    Path | None, typer.Option("--data", help="data folder; default from resources")
]
RunsOption = Annotated[
    Path | None, typer.Option("--runs", help="runs folder; default from resources")
]


def folders(data: Path | None, runs: Path | None, configs: Path | None) -> tuple[Path, Path]:
    settings = config.load_resources(configs)
    return (data or Path(settings.data_dir)).resolve(), (runs or Path(settings.runs_dir)).resolve()


@app.command()
def fetch(
    farms: Annotated[list[str] | None, typer.Argument(help="farms to fetch; default all")] = None,
    data: DataOption = None,
    configs: ConfigsOption = None,
) -> None:
    """download every raw file of the farms and check its size and MD5."""
    from windfold import sources

    data_dir, _ = folders(data, None, configs)
    for name in farms or ["kelmarsh", "penmanshiel", "hill_of_towie", "smarteole"]:
        farm = config.load_farm(name, configs)
        sources.fetch(farm, data_dir / "raw" / name)
        typer.echo(f"{name}: every raw file present and verified")


@app.command()
def run(
    studies: Annotated[list[str], typer.Argument(help="studies to run, e.g. development")],
    plan: Annotated[bool, typer.Option("--plan", help="list the jobs and stop")] = False,
    job: Annotated[str | None, typer.Option(help="run this one job, even if done")] = None,
    cpu_jobs: Annotated[
        int | None, typer.Option(help="at most this many cpu slots; one-core jobs share a slot")
    ] = None,
    keep_waiting: Annotated[
        bool, typer.Option("--keep-waiting", help="keep replanning when nothing can run")
    ] = False,
    detach: Annotated[bool, typer.Option("--detach", help="run in the background")] = False,
    data: DataOption = None,
    runs: RunsOption = None,
    configs: ConfigsOption = None,
) -> None:
    """run every job of the studies that is not already done, resuming where it stopped."""
    from windfold.runner import pools
    from windfold.study import Paths
    from windfold.study import plan as plan_jobs

    data_dir, runs_dir = folders(data, runs, configs)
    paths = Paths(data_dir, runs_dir, configs.resolve() if configs else None)
    if plan:
        from windfold.runner.jobs import unverified

        cores = config.load_resources(configs).cores_per_cpu_job
        jobs = pools.check_plan(plan_jobs(tuple(studies), paths), cores)
        folder = {"data": data_dir, "runs": runs_dir}
        with ThreadPoolExecutor(8) as pool:
            done = list(pool.map(lambda j: unverified(j, folder[j.root]) is None, jobs.values()))
        families = Counter(j.family for j in jobs.values())
        finished = Counter(j.family for j, ok in zip(jobs.values(), done, strict=True) if ok)
        typer.echo(f"{len(jobs)} jobs, {sum(done)} done")
        for family, count in families.items():
            typer.echo(f"  {family}: {finished[family]} done of {count}")
        return
    if detach:
        command = [sys.executable, "-m", "windfold", "run", *studies]
        command += [f"--data={data_dir}", f"--runs={runs_dir}"]
        command += [f"--configs={paths.configs}"] if paths.configs else []
        command += [f"--cpu-jobs={cpu_jobs}"] if cpu_jobs else []
        command += [f"--job={job}"] if job else []
        command += ["--keep-waiting"] if keep_waiting else []
        pid = pools.detach(command, runs_dir / "logs" / "detached.log")
        typer.echo(f"runner started in the background, pid {pid}; see windfold status")
        return
    options = pools.RunOptions(cpu_jobs=cpu_jobs, job=job, keep_waiting=keep_waiting)
    try:
        summary = pools.run(
            lambda: plan_jobs(tuple(studies), paths),
            runs_dir,
            data_dir,
            config.load_resources(configs),
            options,
        )
    except pools.RunnerBusy as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    except pools.Stopped as error:
        typer.echo("stopped; every running job was stopped first", err=True)
        raise typer.Exit(130) from error
    typer.echo(
        f"{len(summary.done)} done, {len(summary.skipped)} already done, "
        f"{len(summary.failed)} failed, {len(summary.waiting)} waiting"
    )
    if summary.failed:
        raise typer.Exit(1)


@app.command()
def status(runs: RunsOption = None, configs: ConfigsOption = None) -> None:
    """what the runner is doing: counts, running jobs, failures, stalls."""
    from windfold.runner import health

    _, runs_dir = folders(None, runs, configs)
    typer.echo(health.status(runs_dir, config.load_resources(configs).stall_minutes))


@app.command()
def release(
    farms: Annotated[
        list[str] | None, typer.Option("--farm", help="a farm whose data passed")
    ] = None,
    families: Annotated[
        list[str] | None, typer.Option("--family", help="a model family to release")
    ] = None,
    runs: RunsOption = None,
    configs: ConfigsOption = None,
) -> None:
    """let the runner start a farm's model jobs, family by family, after their checks."""
    _, runs_dir = folders(None, runs, configs)
    path = runs_dir / "release.toml"
    current = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    released = {
        key: sorted(set(current.get(key, [])) | set(extra or []))
        for key, extra in (("farms", farms), ("families", families))
    }
    lines = [f"{key} = {json.dumps(values)}" for key, values in released.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"released farms {released['farms']}, families {released['families']}")


@app.command()
def freeze(
    record: Annotated[
        str,
        typer.Argument(
            help="first: the fits of the confirmatory study. second: the coverage map fitted on "
            "development data and the hashes of the confirmatory predictions"
        ),
    ],
    study: Annotated[str, typer.Option(help="the study to freeze")] = "confirmatory",
    again: Annotated[
        bool, typer.Option("--again", help="replace the record, keeping the old one beside it")
    ] = False,
    data: DataOption = None,
    runs: RunsOption = None,
    configs: ConfigsOption = None,
) -> None:
    """write a freeze record. only fits matching the first record run on confirmatory data, and
    their predictions are read only once the second record holds their hashes."""
    import subprocess
    from datetime import UTC, datetime

    from windfold.study import FIRST_FREEZE, SECOND_FREEZE, Paths, freeze_first, freeze_second

    if record not in ("first", "second"):
        raise typer.BadParameter("the record is first or second")
    data_dir, runs_dir = folders(data, runs, configs)
    paths = Paths(data_dir, runs_dir, configs.resolve() if configs else None)
    name = FIRST_FREEZE if record == "first" else SECOND_FREEZE
    target = runs_dir / study / name
    if target.exists() and not again:
        typer.echo(f"{target} exists; pass --again to replace it", err=True)
        raise typer.Exit(1)
    try:
        content = freeze_first(study, paths) if record == "first" else freeze_second(study, paths)
    except (ValueError, FileNotFoundError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    here = Path(__file__).parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=here,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    now = datetime.now(UTC)
    content |= {"written": now.isoformat(), "commit": commit or None, "uncommitted": bool(dirty)}
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        replaced = f"{target.stem}_replaced_{now:%Y%m%dT%H%M%SZ}.json"
        target.rename(target.with_name(replaced))
    target.write_text(json.dumps(content, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if record == "first":
        typer.echo(f"froze {len(content['jobs'])} fits of {study} in {target}")
    else:
        typer.echo(f"froze {len(content['predictions'])} prediction files of {study} in {target}")


@app.command()
def secondary(
    name: Annotated[
        str, typer.Argument(help="hybrid_gbm, monitoring_threshold or restricted_windfold")
    ],
    study: Annotated[
        str, typer.Option(help="development, or confirmatory after its second freeze")
    ],
    farm: Annotated[str, typer.Option(help="the farm to test")],
    fit: Annotated[bool, typer.Option("--fit", help="make the missing fits first")] = False,
    workers: Annotated[int, typer.Option(help="fits at a time")] = 1,
    threads: Annotated[int, typer.Option(help="threads per gbm fit")] = 4,
    out: Annotated[
        Path | None,
        typer.Option(help="the table to write; default <runs>/<study>/secondary/<model>/<farm>"),
    ] = None,
    data: DataOption = None,
    runs: RunsOption = None,
    configs: ConfigsOption = None,
) -> None:
    """run a test of the confirmation period that items 57 to 59 of the study plan specify and
    write its table under <runs>/<study>/secondary/<model>/<farm>.parquet."""
    from windfold.secondary import hybrid_gbm, monitoring_threshold, restricted_windfold
    from windfold.secondary.contrasts import output_folder

    data_dir, runs_dir = folders(data, runs, configs)
    folder = str(configs.resolve()) if configs else None
    try:
        if name == "hybrid_gbm":
            if fit:
                hybrid_gbm.fit_all(data_dir, runs_dir, study, farm, workers, threads, folder)
            table, model = (
                hybrid_gbm.score(data_dir, runs_dir, study, farm, folder),
                hybrid_gbm.MODEL,
            )
        elif name == "monitoring_threshold":
            table = monitoring_threshold.score(data_dir, runs_dir, study, farm, folder)
            model = monitoring_threshold.MODEL
        elif name == "restricted_windfold":
            if fit:
                restricted_windfold.fit_all(data_dir, runs_dir, study, farm, workers, folder)
            table = restricted_windfold.score(data_dir, runs_dir, study, farm, folder)
            model = restricted_windfold.MODEL
        else:
            raise ValueError(
                f"no secondary test {name}; the tests are hybrid_gbm, monitoring_threshold and "
                "restricted_windfold"
            )
    except (ValueError, FileNotFoundError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    target = (out or output_folder(runs_dir, study, model) / f"{farm}.parquet").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(target)
    typer.echo(str(table))
    typer.echo(f"wrote {target}")


@app.command()
def analyse(
    name: Annotated[
        str,
        typer.Argument(
            help="arc_variants, comparison_ranks, default_physics, direction_spread, "
            "energy_ratio_error, energy_ratio_intervals, inflow_smoothing, inflow_structure, "
            "interval_score, kernel_geometry, lidar_turbulence, linear_clipping, local_block, "
            "pair_skill, practical_scale, same_row_skill, secondary_accuracy, sector_gains, "
            "sensor_count, strict_alignment, "
            "strict_calibration, transfer_ceiling, turbine_levels or validation_choice"
        ),
    ],
    farms: Annotated[
        list[str] | None, typer.Option("--farm", help="a farm to analyze; default all it covers")
    ] = None,
    refit: Annotated[
        bool, typer.Option("--refit", help="make the refits a heavy analysis is missing first")
    ] = False,
    workers: Annotated[int, typer.Option(help="processes for the physics of refits")] = 1,
    work: Annotated[
        Path | None,
        typer.Option(help="folder of a heavy analysis's refits; default beside the json"),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="the json to write; default <runs>/<study>/analyses/<name>.json"),
    ] = None,
    study: Annotated[
        str, typer.Option(help="development, or confirmatory after its second freeze")
    ] = "development",
    data: DataOption = None,
    runs: RunsOption = None,
    configs: ConfigsOption = None,
) -> None:
    """write an exploratory analysis of a study as json. comparison_ranks, default_physics,
    energy_ratio_error, energy_ratio_intervals, interval_score, pair_skill, practical_scale,
    secondary_accuracy and sector_gains read either study; the rest read development data and
    results only, and the heavy ones refit models in their own folder."""
    import subprocess
    from datetime import UTC, datetime

    from windfold import analyses

    data_dir, runs_dir = folders(data, runs, configs)
    folder = runs_dir / study / "analyses"
    try:
        analysed, result = analyses.run(
            name,
            data_dir,
            runs_dir,
            tuple(farms or ()),
            configs.resolve() if configs else None,
            work.resolve() if work else folder / name,
            refit,
            workers,
            study,
        )
    except (ValueError, FileNotFoundError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    here = Path(__file__).parent

    def git(*arguments: str) -> str | None:
        try:
            found = subprocess.run(
                ["git", *arguments], cwd=here, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return found.stdout.strip() if found.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--untracked-files=no") if commit else None
    content = {
        "analysis": name,
        "study": study,
        "written": datetime.now(UTC).isoformat(),
        "commit": commit,
        "uncommitted": None if changes is None else bool(changes),
        "data": str(data_dir),
        "runs": str(runs_dir),
        "farms": analysed,
        "result": result,
    }
    target = (out or folder / f"{name}.json").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    partial.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
    partial.replace(target)
    typer.echo(f"wrote {target}")


@app.command()
def figures(
    names: Annotated[
        list[str] | None, typer.Argument(help="figures to draw; default every one")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="folder to draw into; default the runs folder")
    ] = None,
    data: DataOption = None,
    runs: RunsOption = None,
    configs: ConfigsOption = None,
) -> None:
    """draw the paper's figures that read the exploratory analyses or the secondary tests,
    which the runner does not make, into <out>/report/figures, each checked as the runner
    checks its own."""
    from windfold.report import (
        ceiling,
        confirmation,
        margin,
        monitoring,
        optimism,
        sector_gains,
        turbine_levels,
    )
    from windfold.report.check import save
    from windfold.report.paths import COMMANDED, figure_files
    from windfold.report.sources import Sources
    from windfold.report.style import load_style

    drawers = {
        "margin": margin.figure,
        "ceiling": ceiling.figure,
        "realised": ceiling.realised_figure,
        "transfer": ceiling.transfer_figure,
        "hybrid": confirmation.hybrid_figure,
        "restricted": confirmation.restricted_figure,
        "false_alarms": monitoring.false_alarms_figure,
        "detection": monitoring.detection_figure,
        "recall": monitoring.recall_figure,
        "turbine_levels": turbine_levels.figure,
        "sector_gains": sector_gains.figure,
        "optimism": optimism.figure,
    }
    if set(drawers) != set(COMMANDED):
        raise RuntimeError(f"commanded figures {sorted(drawers)} differ from {sorted(COMMANDED)}")
    unknown = sorted(set(names or ()) - set(COMMANDED))
    if unknown:
        raise typer.BadParameter(f"no figure {', '.join(unknown)}; they are {', '.join(COMMANDED)}")
    data_dir, runs_dir = folders(data, runs, configs)
    folder = configs.resolve() if configs else None
    sources = Sources(data_dir, runs_dir, folder, ("development", "confirmatory"))
    style = load_style(folder)
    target = (out or runs_dir).resolve()
    for name in names or COMMANDED:
        try:
            drawn = drawers[name](sources, style)
            files = [target / path for path in figure_files("report", name)]
            save(drawn, files, style)
        except (ValueError, FileNotFoundError) as error:
            typer.echo(f"{name}: {error}", err=True)
            raise typer.Exit(1) from error
        typer.echo(f"wrote {files[0]}")
