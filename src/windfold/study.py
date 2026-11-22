"""the job plan: every job of the chosen studies, what it needs, what it writes, its fingerprint.

the runner calls the plan again every few minutes, so jobs appear as soon as their inputs exist.
data jobs always run. a farm's model jobs wait until runs/release.toml lists the farm (its data
check passed) and the job's family (its code was reviewed and one real job of it checked).
"""

import hashlib
import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from windfold import config, inflow
from windfold.config import FarmConfig, ModelSpec, Protocol, Study, config_root, read_toml
from windfold.evaluation.paths import (
    DEVELOPMENT_STUDY,
    borrowed_regimes,
    decision_outputs,
    kernel_transfer_outputs,
    reduction_outputs,
    result_outputs,
    reuses_main_fits,
    statistics_only,
    transfer_sensitivities,
    trial_outputs,
)
from windfold.fingerprints import (
    digest,
    environment_hash,
    forget,
    function_hash,
    function_parameters,
)
from windfold.locations import (
    automl_file,
    budget_file,
    changes_kernel,
    changes_processing,
    chosen_file,
    curve_file,
    inspect_files,
    kernel_folder,
    manifest_file,
    needs_own_manifest,
    prediction_files,
    prepared_folder,
    prepared_name,
    pretraining_files,
    search_file,
    study_folder,
    tuning_file,
)
from windfold.report.paths import CONFIRMATION_ONLY, DRAWN, TABLES, figure_files, table_files
from windfold.runner.jobs import Job, failed_for_good

FARM_ORDER = ("smarteole", "kelmarsh", "hill_of_towie", "penmanshiel")
CORES = 4
PREPARED_FILES = ("rows.parquet", "ledger.csv", "provenance.json", "layout.parquet")
KERNEL_FILES = ("data.npy", "indices.npy", "indptr.npy", "shape.json")
FIT_SECTIONS = ("processing", "physics", "kernel", "engine", "smearing", "seeds")
LEARNER_SECTIONS = ("splits", "gbm", "graph", "menus", "automl", "automl_budget")
DATA, MANIFESTS, SMOKE, TUNING, BUDGET, PRETRAINING = 0, 10, 15, 20, 30, 40
EVALUATION, AUTOML, AUTOML_CONFIRMATORY = 50, 80, 90
DEVELOPMENT, CONFIRMATORY, SENSITIVITIES = 100, 200, 500
FIT_FAMILIES = ("engine", "gbm", "graph", "pretrained_graph", "automl")
type Fit = tuple[str, str, int | None]
FIRST_FREEZE = "freeze_first.json"
SECOND_FREEZE = "freeze_second.json"


@dataclass(frozen=True)
class Paths:
    data: Path
    runs: Path
    configs: Path | None = None


@dataclass(frozen=True)
class Release:
    farms: tuple[str, ...]
    families: tuple[str, ...]

    def allows(self, farm: str, family: str, study: Study | None = None) -> bool:
        """the smoke study runs every family on a farm whose data check passed: it is how a
        family earns its release for the real studies."""
        if study is not None and study.limits is not None:
            return farm in self.farms
        return farm in self.farms and family in self.families


def read_release(runs: Path) -> Release:
    path = runs / "release.toml"
    if not path.is_file():
        return Release((), ())
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return Release(tuple(data.get("farms", ())), tuple(data.get("families", ())))


def ordered(farms: tuple[str, ...] | list[str]) -> list[str]:
    known = {farm: rank for rank, farm in enumerate(FARM_ORDER)}
    return sorted(farms, key=lambda farm: (known.get(farm, len(FARM_ORDER)), farm))


def sensitivity_farms(protocol: Protocol, name: str, farms: list[str]) -> list[str]:
    named = {
        path.split(".")[1]
        for path in protocol.sensitivities[name].overrides
        if path.startswith("farms.")
    }
    return [farm for farm in farms if farm in named] if named else farms


def fit_parts(protocol: Protocol) -> dict[str, Any]:
    dumped = protocol.model_dump(mode="json")
    return {section: dumped.get(section) for section in FIT_SECTIONS}


def learner_parts(protocol: Protocol) -> dict[str, Any]:
    dumped = protocol.model_dump(mode="json")
    return fit_parts(protocol) | {section: dumped.get(section) for section in LEARNER_SECTIONS}


def learner_family(spec: ModelSpec) -> str:
    return "pretrained_graph" if spec.pretrained else spec.learner or "engine"


def automl_priority(priority: int, study: Study) -> int:
    if study.period == "confirmatory":
        return AUTOML_CONFIRMATORY + priority % 100
    return AUTOML + priority % 100


def fit_name_of(split: str, seed: int | None) -> str:
    return split if seed is None else f"{split}_seed{seed}"


def development_results(jobs: Iterable[str], sensitivity: str | None) -> list[str]:
    """the development farm results a confirmatory decision reads: those of the same analysis,
    the main one or the same sensitivity."""
    prefix = f"{study_folder(DEVELOPMENT_STUDY, sensitivity)}/results/"
    return sorted(job for job in jobs if job.startswith(prefix))


def horizon_days(split: str) -> int | None:
    if not split.startswith(("history_", "threshold_")):
        return None
    return int(split.split("_")[1].removesuffix("d"))


def limited(summary: pl.DataFrame, limits: config.Limits | None) -> pl.DataFrame:
    """the smoke study's subset: the first random and time folds, the first arcs and the
    shortest monitorable histories from the earliest starts."""
    if limits is None:
        return summary
    keep = []
    for (regime,), part in summary.group_by("regime", maintain_order=True):
        names = sorted(part["split"].to_list())
        if regime in ("random", "time_blocked"):
            keep += names[:1]
        elif str(regime).startswith("arc_"):
            keep += names[: limits.arcs] if limits.arcs else names
        else:
            by_length = sorted(names, key=lambda n: (horizon_days(n) != 30, n))
            keep += by_length[: limits.histories] if limits.histories else by_length
    return summary.filter(pl.col("split").is_in(keep))


class Planner:
    def __init__(self, studies: tuple[str, ...], paths: Paths, freezing: bool = False):
        self.studies = studies
        self.paths = paths
        self.freezing = freezing
        self.jobs: dict[str, Job] = {}
        self.expected: dict[tuple[str, str | None, str], set[Fit]] = {}
        self.planned: dict[tuple[str, str | None, str], dict[Fit, str]] = {}

    @property
    def configs(self) -> str | None:
        return str(self.paths.configs) if self.paths.configs else None

    def add(
        self,
        job_id: str,
        target: str,
        args: dict[str, Any],
        outputs: tuple[str, ...],
        root: Literal["runs", "data"],
        parts: Any,
        family: str,
        priority: int,
        needs: tuple[str, ...] = (),
        memory_gb: float = 8.0,
        gpu: bool = False,
        automl: bool = False,
        full_load: bool = False,
        cores: int = CORES,
    ) -> str:
        if job_id in self.jobs:
            return job_id
        missing = [need for need in needs if need not in self.jobs]
        if missing:
            raise ValueError(f"{job_id} needs jobs that are not planned: {missing}")
        accepted, required = function_parameters(target)
        offered = {
            "data_dir": str(self.paths.data),
            "runs_dir": str(self.paths.runs),
            "configs": self.configs,
        }
        common = {key: value for key, value in offered.items() if key in accepted}
        unknown = sorted(set(args) - set(accepted))
        missing = sorted(set(required) - set(args) - set(common) - {"out_dir"})
        if unknown or missing:
            raise ValueError(f"{job_id}: {target} does not take {unknown} and needs {missing}")
        fingerprint = digest(
            target,
            args,
            function_hash(target),
            environment_hash(),
            parts,
            [self.jobs[need].fingerprint for need in needs],
        )
        self.jobs[job_id] = Job(
            id=job_id,
            target=target,
            args={**common, **args},
            needs=needs,
            outputs=outputs,
            root=root,
            fingerprint=fingerprint,
            cores=cores,
            gpu=gpu,
            memory_gb=memory_gb,
            family=family,
            priority=priority,
            automl=automl,
            full_load=full_load,
        )
        return job_id

    def data_jobs(self, farm: FarmConfig, protocol: Protocol, rank: int, calibrated: list[str]):
        turbine = config.load_turbine(farm.turbine_type, self.paths.configs)
        name = farm.name
        tables = self.add(
            f"tables/{name}",
            "windfold.tasks.data:read_tables",
            {"farm": name},
            tuple(f"tables/{name}/{t}.parquet" for t in ("rows", "events", "layout")),
            "data",
            (farm,),
            "tables",
            DATA + rank,
            memory_gb=16.0,
        )
        prepared = self.add(
            f"prepared/{name}",
            "windfold.tasks.data:prepare_farm",
            {"farm": name, "sensitivity": None},
            tuple(f"prepared/{name}/{file}" for file in PREPARED_FILES),
            "data",
            (farm, turbine, protocol.processing),
            "prepared",
            DATA + rank,
            needs=(tables,),
            memory_gb=16.0,
        )
        own = farm.windows.calibration is not None
        fit = self.add(
            f"physics_fit/{name}",
            "windfold.tasks.data:fit_physics",
            {
                "farm": name,
                "seed": protocol.seeds.study,
                "workers": CORES,
                "calibrated_farms": calibrated,
            },
            (f"prepared/{name}/physics_fit.json",),
            "data",
            (farm.windows, turbine, protocol.physics, protocol.processing.speed_bin_edges_ms),
            "physics_fit",
            DATA + rank,
            needs=(prepared,) if own else tuple(f"physics_fit/{other}" for other in calibrated),
        )
        self.add_derived(farm, protocol, None, prepared, fit, DATA + rank, calibrated)
        self.add(
            f"inspect/{name}",
            "windfold.tasks.data:inspect_farm",
            {"farm": name},
            inspect_files(name),
            "runs",
            (farm, turbine, protocol.processing),
            "inspect",
            DATA + rank,
            needs=(prepared,),
            memory_gb=16.0,
        )

    def add_derived(
        self,
        farm: FarmConfig,
        protocol: Protocol,
        sensitivity: str | None,
        prepared: str,
        fit: str,
        priority: int,
        calibrated: list[str],
    ) -> None:
        turbine = config.load_turbine(farm.turbine_type, self.paths.configs)
        folder = prepared_name(farm.name, sensitivity)
        if sensitivity is None or changes_processing(protocol, sensitivity):
            self.add(
                f"physics_table/{folder}",
                "windfold.tasks.data:physics_table",
                {"farm": farm.name, "sensitivity": sensitivity, "workers": CORES},
                (f"prepared/{folder}/physics.parquet",),
                "data",
                (turbine, protocol.physics, protocol.processing.speed_bin_edges_ms),
                "physics_table",
                priority,
                needs=(prepared, fit),
            )
            own = farm.windows.calibration is not None
            self.add(
                f"inflow_physics/{folder}",
                "windfold.tasks.data:inflow_physics",
                {
                    "farm": farm.name,
                    "sensitivity": sensitivity,
                    "workers": CORES,
                    "calibrated_farms": [] if own else calibrated,
                },
                (
                    *(f"prepared/{folder}/{table}" for table in inflow.TABLES.values()),
                    f"prepared/{folder}/{inflow.RECORD}",
                ),
                "data",
                (
                    turbine,
                    protocol.physics,
                    protocol.inflow,
                    protocol.processing.speed_bin_edges_ms,
                    farm.windows,
                ),
                "inflow_physics",
                priority,
                needs=(prepared, fit)
                + (() if own else tuple(f"inflow_physics/{other}" for other in calibrated)),
            )
        if (
            sensitivity is None
            or changes_processing(protocol, sensitivity)
            or changes_kernel(protocol, sensitivity)
        ):
            design = kernel_folder(farm.name, sensitivity, protocol)
            self.add(
                f"kernel_design/{folder}",
                "windfold.tasks.data:kernel_designs",
                {"farm": farm.name, "sensitivity": sensitivity},
                tuple(
                    f"prepared/{design}/{sub}{file}"
                    for sub in ("", "permuted/")
                    for file in KERNEL_FILES
                ),
                "data",
                (
                    turbine,
                    protocol.kernel,
                    protocol.processing.speed_bin_edges_ms,
                    protocol.seeds.study,
                ),
                "kernel_design",
                priority,
                needs=(prepared,),
            )

    def sensitivity_data_jobs(
        self, farm: FarmConfig, protocol: Protocol, name: str, rank: int, calibrated: list[str]
    ):
        changed, farms = config.with_sensitivity(name, protocol, {farm.name: farm})
        turbine = config.load_turbine(farm.turbine_type, self.paths.configs)
        folder = prepared_name(farm.name, name)
        prepared = f"prepared/{farm.name}"
        if changes_processing(changed, name):
            prepared = self.add(
                f"prepared/{folder}",
                "windfold.tasks.data:prepare_farm",
                {"farm": farm.name, "sensitivity": name},
                tuple(f"prepared/{folder}/{file}" for file in PREPARED_FILES),
                "data",
                (farms[farm.name], turbine, changed.processing),
                "prepared",
                SENSITIVITIES + rank,
                needs=(f"tables/{farm.name}",),
                memory_gb=16.0,
            )
        self.add_derived(
            farms[farm.name],
            changed,
            name,
            prepared,
            f"physics_fit/{farm.name}",
            SENSITIVITIES + rank,
            calibrated,
        )

    def manifest_job(
        self, study: Study, farm: FarmConfig, protocol: Protocol, sensitivity: str | None, rank: int
    ) -> str:
        changed, farms = protocol, {farm.name: farm}
        if sensitivity:
            changed, farms = config.with_sensitivity(sensitivity, protocol, {farm.name: farm})
        target = manifest_file(study.name, farm.name, sensitivity, changed)
        job_id = target.removesuffix(".parquet")
        if job_id in self.jobs:
            return job_id
        f = farms[farm.name]
        source = prepared_folder(farm.name, sensitivity, changed)
        return self.add(
            job_id,
            "windfold.tasks.data:manifests",
            {
                "study": study.name,
                "farm": farm.name,
                "period": study.period,
                "sensitivity": sensitivity,
            },
            (target, target.replace(".parquet", "_summary.csv")),
            "runs",
            (
                changed.splits,
                changed.monitoring,
                changed.seeds.study,
                f.windows,
                f.histories,
                f.confirmatory_histories,
                f.time_blocks,
            ),
            "manifests",
            MANIFESTS + rank,
            needs=(f"prepared/{source}",),
        )

    def fit_jobs(
        self,
        study: Study,
        farm: FarmConfig,
        protocol: Protocol,
        models: dict[str, ModelSpec],
        sensitivity: str | None,
        manifest: str,
        priority: int,
        release: Release,
    ) -> None:
        changed = protocol
        if sensitivity:
            changed, _ = config.with_sensitivity(sensitivity, protocol, {farm.name: farm})
        summary_path = self.paths.runs / f"{manifest}_summary.csv"
        if not current(summary_path, self.jobs[manifest].fingerprint):
            return
        summary = pl.read_csv(summary_path).filter(pl.col("status") == "included")
        summary = limited(summary, study.limits)
        regime_of = dict(zip(summary["split"], summary["regime"], strict=True))
        names = list(study.models)
        regimes_in_scope = None
        if sensitivity:
            scope = changed.sensitivities[sensitivity]
            names = [m for m in names if m in scope.models]
            regimes_in_scope = set(scope.regimes)
        batches: dict[str, list[str]] = defaultdict(list)
        learners: list[tuple[str, str]] = []
        for name in names:
            spec = models[name]
            regimes = set(spec.regimes)
            if name in (*changed.monitoring.models, *changed.matched_monitoring.models):
                regimes.add("threshold")
            if regimes_in_scope is not None:
                regimes &= regimes_in_scope
            for split, regime in regime_of.items():
                if regime not in regimes:
                    continue
                days = horizon_days(split)
                if spec.history_days and regime == "history" and days not in spec.history_days:
                    continue
                if spec.learner is None:
                    batches[split].append(name)
                else:
                    learners.append((name, split))
        folder = prepared_folder(farm.name, sensitivity, changed)
        physics_table = f"physics_table/{folder}"
        inflow_table = f"inflow_physics/{folder}"
        kernel = f"kernel_design/{farm.name}"
        if changes_processing(changed, sensitivity) or changes_kernel(changed, sensitivity):
            kernel = f"kernel_design/{prepared_name(farm.name, sensitivity)}"
        folder_name = study_folder(study.name, sensitivity)
        key = (study.name, sensitivity, farm.name)
        expected = self.expected.setdefault(key, set())
        planned = self.planned.setdefault(key, {})
        expected |= {(model, split, None) for split, batch in batches.items() for model in batch}
        if release.allows(farm.name, "engine", study):
            for split, batch in batches.items():
                uses_kernel = any("shared_kernel" in models[m].blocks for m in batch)
                measured = any(models[m].inflow != "constant" for m in batch)
                needs = (
                    manifest,
                    physics_table,
                    *((kernel,) if uses_kernel else ()),
                    *((inflow_table,) if measured else ()),
                )
                outputs = tuple(
                    file
                    for model in batch
                    for file in prediction_files(
                        study.name, sensitivity, farm.name, model, split, None
                    )
                )
                job_id = self.add(
                    f"{folder_name}/engine/{farm.name}/{split}",
                    "windfold.tasks.fitting:fit_engine",
                    {
                        "study": study.name,
                        "farm": farm.name,
                        "split": split,
                        "models": batch,
                        "sensitivity": sensitivity,
                    },
                    outputs,
                    "runs",
                    ([models[m] for m in batch], fit_parts(changed), study.limits),
                    "engine",
                    priority,
                    needs=needs,
                    cores=1,
                    memory_gb=6.0,
                )
                planned |= {(model, split, None): job_id for model in batch}
        seeds = list(changed.seeds.model)
        if study.limits and study.limits.seeds:
            seeds = seeds[: study.limits.seeds]
        for name, split in learners:
            spec = models[name]
            expected |= {(name, split, seed) for seed in (seeds if spec.seeded else [None])}
        for name, split in learners:
            spec = models[name]
            family = learner_family(spec)
            if not release.allows(farm.name, family, study):
                continue
            calibration = () if study.limits else self.calibration_needs(farm.name, name, models)
            if calibration is None:
                continue
            for seed in seeds if spec.seeded else [None]:
                outputs = prediction_files(study.name, sensitivity, farm.name, name, split, seed)
                if spec.learner == "graph":
                    outputs += (curve_file(study.name, sensitivity, farm.name, name, split, seed),)
                if spec.learner == "automl":
                    outputs += (automl_file(study.name, sensitivity, farm.name, split),)
                planned[(name, split, seed)] = self.add(
                    f"{folder_name}/{name}/{farm.name}/{fit_name_of(split, seed)}",
                    f"windfold.tasks.fitting:fit_{spec.learner}",
                    {
                        "study": study.name,
                        "farm": farm.name,
                        "model": name,
                        "split": split,
                        "seed": seed,
                        "sensitivity": sensitivity,
                    },
                    outputs,
                    "runs",
                    (spec, learner_parts(changed), study.limits),
                    family,
                    automl_priority(priority, study) if spec.learner == "automl" else priority,
                    needs=(
                        manifest,
                        physics_table,
                        *((inflow_table,) if spec.inflow == "measured" else ()),
                        *calibration,
                    ),
                    gpu=spec.learner == "graph",
                    automl=spec.learner == "automl",
                    memory_gb=16.0 if spec.learner == "automl" else 8.0,
                )

    def calibration_needs(
        self, farm: str, model: str, models: dict[str, ModelSpec]
    ) -> tuple[str, ...] | None:
        """the calibration jobs a real fit needs, or None while any is not planned."""
        spec = models[model]
        needed = []
        if spec.learner in ("gbm", "graph"):
            needed.append(f"calibration/tuning/{farm}/{config.tuned_model(model, models)}")
        if spec.pretrained:
            needed.append(f"calibration/pretraining/{farm}")
        if spec.learner == "automl":
            needed.append(f"calibration/automl_budget/{farm}")
        if any(job not in self.jobs for job in needed):
            return None
        return tuple(needed)

    def calibration_jobs(
        self,
        farms: dict[str, FarmConfig],
        protocol: Protocol,
        models: dict[str, ModelSpec],
        studies: list[Study],
        release: Release,
    ) -> None:
        wanted = {m for study in studies if study.limits is None for m in study.models}
        tuned = sorted(
            {config.tuned_model(m, models) for m in wanted if models[m].learner in ("gbm", "graph")}
        )
        names = ordered(list(farms))
        calibrated = [f for f in names if farms[f].windows.calibration is not None]
        parts = learner_parts(protocol)
        rank_of = {farm: rank for rank, farm in enumerate(names)}
        for farm in calibrated + [f for f in names if f not in calibrated]:
            rank = rank_of[farm]
            own = farm in calibrated
            for model in tuned:
                if not release.allows(farm, "tuning"):
                    continue
                spec = models[model]
                menu = config.load_menu(spec.learner or "", self.paths.configs)
                if own:
                    tunes = tuple(
                        self.add(
                            f"calibration/tuning/{farm}/{model}/{index:02d}",
                            f"windfold.tasks.calibration:tune_{spec.learner}",
                            {"farm": farm, "model": model, "index": index},
                            (tuning_file(farm, model, index),),
                            "runs",
                            (spec, menu[index], parts, farms[farm].windows.calibration),
                            "tuning",
                            TUNING + rank,
                            needs=(
                                f"prepared/{farm}",
                                f"physics_table/{farm}",
                                *((f"inflow_physics/{farm}",) if spec.inflow == "measured" else ()),
                            ),
                            gpu=spec.learner == "graph",
                        )
                        for index in range(len(menu))
                    )
                    others: list[str] = []
                else:
                    others = [
                        f for f in calibrated if f"calibration/tuning/{f}/{model}" in self.jobs
                    ]
                    if len(others) != len(calibrated):
                        continue
                    tunes = tuple(f"calibration/tuning/{f}/{model}" for f in others)
                self.add(
                    f"calibration/tuning/{farm}/{model}",
                    "windfold.tasks.calibration:choose",
                    {"farm": farm, "model": model, "other_farms": others},
                    (search_file(farm, model), chosen_file(farm, model)),
                    "runs",
                    (spec, menu, parts),
                    "tuning",
                    TUNING + rank,
                    needs=tunes,
                    memory_gb=2.0,
                )
            pretrained = [m for m in wanted if models[m].pretrained]
            source = (
                f"calibration/tuning/{farm}/{config.tuned_model(pretrained[0], models)}"
                if pretrained
                else None
            )
            if source in self.jobs and release.allows(farm, "pretraining"):
                self.add(
                    f"calibration/pretraining/{farm}",
                    "windfold.tasks.calibration:pretrain",
                    {"farm": farm},
                    pretraining_files(farm),
                    "runs",
                    (parts, fit_parts(protocol), farms[farm].windows.calibration),
                    "pretraining",
                    PRETRAINING + rank,
                    needs=(f"prepared/{farm}", f"physics_fit/{farm}", source),
                    gpu=True,
                    memory_gb=16.0,
                )
            if any(models[m].learner == "automl" for m in wanted) and release.allows(
                farm, "automl_budget"
            ):
                budgets = tuple(f"calibration/automl_budget/{f}" for f in calibrated)
                if own:
                    needs: tuple[str, ...] = (f"prepared/{farm}",)
                elif all(b in self.jobs for b in budgets):
                    needs = budgets
                else:
                    continue
                self.add(
                    f"calibration/automl_budget/{farm}",
                    "windfold.tasks.calibration:budget",
                    {"farm": farm, "other_farms": [] if own else calibrated},
                    (budget_file(farm),),
                    "runs",
                    (parts, farms[farm].windows.calibration),
                    "automl_budget",
                    BUDGET + rank,
                    needs=needs,
                    automl=own,
                    full_load=own,
                    memory_gb=16.0 if own else 2.0,
                )

    def fit_list(
        self, study: str, sensitivity: str | None, farm: str, protocol: Protocol
    ) -> list[tuple[str, str, int | None, str]] | None:
        """every fit a farm's evaluation reads with the job that makes it, or None while any
        expected fit is not planned. a sensitivity that refits some models on the main data, a
        kernel or shrinkage change, reads the models it does not refit from the main study."""
        key = (study, sensitivity, farm)
        expected, planned = self.expected.get(key), self.planned.get(key, {})
        if not expected or any(fit not in planned for fit in expected):
            return None
        fits = [(m, s, k, planned[(m, s, k)]) for m, s, k in sorted(expected, key=str)]
        if reuses_main_fits(protocol, sensitivity):
            main = self.fit_list(study, None, farm, protocol)
            if main is None:
                return None
            refit = {m for m, _, _, _ in fits}
            fits += [fit for fit in main if fit[0] not in refit]
        return fits

    def evaluation_jobs(
        self,
        study: Study,
        farms: dict[str, FarmConfig],
        protocol: Protocol,
        release: Release,
    ) -> None:
        failed = {
            job_id
            for job_id, fingerprint in failed_for_good(self.paths.runs).items()
            if job_id in self.jobs and self.jobs[job_id].fingerprint == fingerprint
        }
        names = ordered(study.farms)
        record = second_freeze(self.paths.runs, study)
        if study.period == "confirmatory" and record is None:
            return
        for rank, sensitivity in enumerate((None, *study.sensitivities)):
            folder = study_folder(study.name, sensitivity)
            members = sensitivity_farms(protocol, sensitivity, names) if sensitivity else names
            reduced: dict[str, str] = {}
            for farm in members:
                changed = protocol
                if sensitivity:
                    changed, _ = config.with_sensitivity(sensitivity, protocol, {farm: farms[farm]})
                if not release.allows(farm, "evaluation", study):
                    continue
                if statistics_only(changed, sensitivity):
                    main = f"{study.name}/reduce/{farm}"
                    if main in self.jobs:
                        reduced[farm] = main
                    continue
                fits = self.fit_list(study.name, sensitivity, farm, changed)
                if fits is None:
                    continue
                manifest = manifest_file(study.name, farm, sensitivity, changed)
                table = f"physics_table/{prepared_folder(farm, sensitivity, changed)}"
                reduced[farm] = self.add(
                    f"{folder}/reduce/{farm}",
                    "windfold.tasks.evaluation:reduce_farm",
                    {
                        "study": study.name,
                        "farm": farm,
                        "sensitivity": sensitivity,
                        "fits": [[m, s, k] for m, s, k, job in fits if job not in failed],
                        "failed": [[m, s, k] for m, s, k, job in fits if job in failed],
                    },
                    reduction_outputs(study.name, farm, sensitivity),
                    "runs",
                    (changed, farms[farm], study, record),
                    "evaluation",
                    EVALUATION + rank,
                    needs=(
                        manifest.removesuffix(".parquet"),
                        table,
                        *sorted({job for _, _, _, job in fits if job not in failed}),
                    ),
                    cores=1,
                )
            if not reduced or set(reduced) != set(members):
                continue
            results = []
            for farm in members:
                changed = protocol
                if sensitivity:
                    changed, _ = config.with_sensitivity(sensitivity, protocol, {farm: farms[farm]})
                others = [other for other in members if other != farm]
                borrowed = ()
                if borrowed_regimes(changed, sensitivity):
                    borrowed = (f"{study.name}/reduce/{farm}",)
                    if borrowed[0] not in self.jobs:
                        break
                results.append(
                    self.add(
                        f"{folder}/results/{farm}",
                        "windfold.tasks.evaluation:farm_results",
                        {
                            "study": study.name,
                            "farm": farm,
                            "sensitivity": sensitivity,
                            "other_farms": others,
                        },
                        result_outputs(study.name, farm, sensitivity),
                        "runs",
                        (changed, farms[farm], study, record),
                        "evaluation",
                        EVALUATION + rank,
                        needs=(reduced[farm], *(reduced[other] for other in others), *borrowed),
                    )
                )
            if len(results) != len(members):
                continue
            development = []
            if study.period == "confirmatory":
                development = development_results(self.jobs, sensitivity)
                if not development:
                    continue
            self.add(
                f"{folder}/decision",
                "windfold.tasks.evaluation:study_decision",
                {"study": study.name, "sensitivity": sensitivity, "farms": members},
                decision_outputs(study.name, sensitivity),
                "runs",
                (protocol, study, record),
                "evaluation",
                EVALUATION + rank,
                needs=(*results, *development),
                memory_gb=2.0,
            )
        if study.period == "development" and study.limits is None:
            for rank, farm in enumerate(names):
                if farm not in release.farms:
                    continue
                self.add(
                    f"{study.name}/trial/{farm}",
                    "windfold.tasks.evaluation:trial_planning",
                    {"study": study.name, "farm": farm},
                    trial_outputs(study.name, farm),
                    "runs",
                    (
                        protocol.trial,
                        protocol.splits,
                        protocol.statistics,
                        protocol.seeds,
                        farms[farm],
                    ),
                    "trial",
                    EVALUATION + rank,
                    needs=(f"prepared/{farm}",),
                )
                transfer_needs = (
                    f"{study.name}/reduce/{farm}",
                    *(
                        f"{study_folder(study.name, name)}/reduce/{farm}"
                        for name in transfer_sensitivities(protocol, study.sensitivities)
                    ),
                )
                if all(need in self.jobs for need in transfer_needs):
                    self.add(
                        f"{study.name}/kernel_transfer/{farm}",
                        "windfold.tasks.evaluation:kernel_transfer_table",
                        {"study": study.name, "farm": farm},
                        kernel_transfer_outputs(study.name, farm),
                        "runs",
                        (protocol, farms[farm], study),
                        "evaluation",
                        EVALUATION + rank,
                        needs=transfer_needs,
                        memory_gb=6.0,
                    )

    def build(self) -> list[Job]:
        forget()
        self.jobs, self.expected, self.planned = {}, {}, {}
        folder = self.paths.configs
        protocol = config.load_protocol(folder)
        models = config.load_models(folder)
        config.check_model_names(protocol, models)
        release = read_release(self.paths.runs)
        studies = [config.load_study(name, folder) for name in self.studies]
        farm_names = ordered(list({farm for study in studies for farm in study.farms}))
        farms = {name: config.load_farm(name, folder) for name in farm_names}
        calibrated = [f for f in farm_names if farms[f].windows.calibration is not None]
        for rank, name in enumerate(farm_names):
            if farms[name].windows.calibration is not None:
                self.data_jobs(farms[name], protocol, rank, [])
        for rank, name in enumerate(farm_names):
            if farms[name].windows.calibration is None:
                self.data_jobs(farms[name], protocol, rank, calibrated)
        self.calibration_jobs(farms, protocol, models, studies, release)
        for study in studies:
            band = (
                SMOKE
                if study.limits
                else CONFIRMATORY
                if study.period == "confirmatory"
                else DEVELOPMENT
            )
            frozen = (
                study.period != "confirmatory"
                or self.freezing
                or bool(frozen_jobs(self.paths.runs, study.name))
            )
            for rank, name in enumerate(ordered(study.farms)):
                manifest = self.manifest_job(study, farms[name], protocol, None, rank)
                if not frozen:
                    continue
                self.fit_jobs(
                    study, farms[name], protocol, models, None, manifest, band + rank, release
                )
                for sensitivity in study.sensitivities:
                    if name not in sensitivity_farms(protocol, sensitivity, list(study.farms)):
                        continue
                    changed, _ = config.with_sensitivity(sensitivity, protocol, {name: farms[name]})
                    if changes_processing(changed, sensitivity) or changes_kernel(
                        changed, sensitivity
                    ):
                        self.sensitivity_data_jobs(
                            farms[name], protocol, sensitivity, rank, calibrated
                        )
                    own = (
                        self.manifest_job(study, farms[name], protocol, sensitivity, rank)
                        if needs_own_manifest(changed, sensitivity)
                        else manifest
                    )
                    self.fit_jobs(
                        study,
                        farms[name],
                        protocol,
                        models,
                        sensitivity,
                        own,
                        SENSITIVITIES + rank,
                        release,
                    )
        for study in sorted(studies, key=lambda s: s.period != "development"):
            self.evaluation_jobs(study, farms, protocol, release)
        by_name = {study.name: study for study in studies}
        for study in studies:
            self.report_jobs(study, protocol, by_name)
        for study in studies:
            if study.period == "confirmatory" and not self.freezing:
                self.keep_frozen(study.name)
        return list(self.jobs.values())

    def report_jobs(self, study: Study, protocol: Protocol, by_name: Mapping[str, Study]) -> None:
        """the tables and figures of a study once its decision is planned. a confirmatory study's
        report is the paper's: it reads the development study too. the tables also wait for the
        decision of every planned sensitivity; figures of the confirmation period are drawn for the
        paper's report alone."""
        decision = f"{study.name}/decision"
        if decision not in self.jobs:
            return
        folder, studies, needs = f"{study.name}/report", [study.name], (decision,)
        if study.period == "confirmatory":
            earlier = f"{DEVELOPMENT_STUDY}/decision"
            if earlier not in self.jobs:
                return
            folder, studies, needs = "report", [DEVELOPMENT_STUDY, study.name], (earlier, decision)
        parts = (read_toml(config_root(self.paths.configs), "report.toml"), protocol.endpoints)
        args: dict[str, Any] = {"folder": folder, "studies": studies}
        sensitivities = {
            name: [
                s
                for s in by_name[name].sensitivities
                if f"{study_folder(name, s)}/decision" in self.jobs
            ]
            for name in studies
        }
        decisions = tuple(
            f"{study_folder(name, s)}/decision"
            for name, chosen in sensitivities.items()
            for s in chosen
        )
        tables = tuple(path for name in TABLES for path in table_files(folder, name))
        common = {"memory_gb": 4.0, "cores": 1}
        self.add(
            f"{folder}/tables",
            "windfold.tasks.report:write_tables",
            args | {"sensitivities": sensitivities},
            tables,
            "runs",
            parts,
            "report",
            EVALUATION,
            needs=needs + decisions,
            **common,
        )
        for name in DRAWN:
            if name in CONFIRMATION_ONLY and study.period != "confirmatory":
                continue
            self.add(
                f"{folder}/figures/{name}",
                "windfold.tasks.report:draw_figure",
                args | {"figure": name},
                figure_files(folder, name),
                "runs",
                parts,
                "report",
                EVALUATION,
                needs=needs,
                **common,
            )

    def keep_frozen(self, study: str) -> None:
        """drop the study's fits whose fingerprint differs from the first freeze record, and
        everything that needs them, so nothing unfrozen runs on confirmatory data."""
        record = frozen_jobs(self.paths.runs, study)
        prefix = f"{study}/"
        dropped = {
            job.id
            for job in self.jobs.values()
            if job.id.startswith(prefix)
            and job.family in FIT_FAMILIES
            and record.get(job.id) != job.fingerprint
        }
        changed = True
        while changed:
            more = {
                job.id
                for job in self.jobs.values()
                if job.id not in dropped and any(need in dropped for need in job.needs)
            }
            changed = bool(more)
            dropped |= more
        for job_id in dropped:
            del self.jobs[job_id]


def current(path: Path, fingerprint: str) -> bool:
    """whether path was written by the job with this fingerprint and still holds what it wrote."""
    sidecar = path.with_name(path.name + ".sidecar.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except ValueError:
        return False
    if not isinstance(record, dict) or record.get("fingerprint") != fingerprint:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == record.get("sha256")


def second_freeze(runs: Path, study: Study) -> str | None:
    """the sha256 of the study's second freeze record, or None where it has none."""
    path = runs / study.name / SECOND_FREEZE
    if study.period != "confirmatory" or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_jobs(runs: Path, study: str) -> dict[str, str]:
    path = runs / study / FIRST_FREEZE
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["jobs"])


def freeze_first(study: str, paths: Paths) -> dict[str, Any]:
    """the first freeze record: every fit of the study with its fingerprint, which covers the
    code, configuration, data, splits and calibration outputs it rests on, and the sha256 of each
    calibration output it reads, hashed from the file. refuses while any of those outputs
    is missing or differs from its sidecar."""
    planner = Planner((study,), paths, freezing=True)
    jobs = {job.id: job for job in planner.build()}
    fits = {
        job.id: job.fingerprint
        for job in jobs.values()
        if job.id.startswith(f"{study}/") and job.family in FIT_FAMILIES
    }
    if not fits:
        raise ValueError(f"{study} has no fits to freeze; release them and build its manifests")
    calibration = sorted(
        {need for job_id in fits for need in jobs[job_id].needs if need.startswith("calibration/")}
    )
    checksums = {}
    for need in calibration:
        for output in jobs[need].outputs:
            path = paths.runs / output
            sidecar = path.with_name(f"{path.name}.sidecar.json")
            if not sidecar.is_file():
                raise ValueError(f"{need} has not finished: {output} has no sidecar")
            if not path.is_file():
                raise ValueError(f"{need} has not finished: {output} is missing")
            recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != recorded:
                raise ValueError(
                    f"{need} changed: {output} has sha256 {actual}, its sidecar {recorded}"
                )
            checksums[output] = actual
    return {"study": study, "jobs": fits, "calibration_outputs": checksums}


def freeze_second(study: str, paths: Paths) -> dict[str, Any]:
    """the second freeze record: the coverage map and simple-predictor choice fitted on the
    development arcs, and the sha256 of every output of the study's frozen fits. refuses while
    the plan's fits or fingerprints differ from the first record's, while a frozen fit has
    neither finished nor failed for good or while the development arcs are missing."""
    from windfold.runner.jobs import sha256, unverified
    from windfold.tasks.evaluation import second_record

    frozen = frozen_jobs(paths.runs, study)
    if not frozen:
        raise ValueError(f"{study} has no frozen fits; write the first record and run them")
    jobs = Planner((study,), paths).build()
    fits = [job for job in jobs if job.id.startswith(f"{study}/") and job.family in FIT_FAMILIES]
    planned = {job.id: job.fingerprint for job in fits}
    differ = sorted(
        job_id
        for job_id in frozen.keys() | planned.keys()
        if frozen.get(job_id) != planned.get(job_id)
    )
    if differ:
        raise ValueError(
            f"{len(differ)} fits of {study} are not planned as its first record froze them: "
            f"{differ[0]}"
        )
    lost = failed_for_good(paths.runs)
    predictions: dict[str, str] = {}
    waiting = []
    for job in fits:
        if unverified(job, paths.runs) is None:
            predictions |= {output: sha256(paths.runs / output) for output in job.outputs}
        elif lost.get(job.id) != job.fingerprint:
            waiting.append(job.id)
    if waiting:
        raise ValueError(f"{len(waiting)} frozen fits of {study} have not finished: {waiting[0]}")
    configs = str(paths.configs) if paths.configs else None
    return {"study": study, **second_record(str(paths.runs), configs), "predictions": predictions}


def plan(studies: tuple[str, ...], paths: Paths) -> list[Job]:
    return Planner(studies, paths).build()
