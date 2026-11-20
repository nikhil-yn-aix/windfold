"""typed configuration: schemas, loading and model resolution.

configs ship inside the package; a folder with the same layout replaces them for custom runs.
"""

import json
import tomllib
from collections.abc import Iterator
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

BLOCKS = (
    "turbine_offset",
    "turbine_speed_cells",
    "turbine_direction_table",
    "turbine_direction_harmonics",
    "turbine_direction_spline",
    "physics",
    "shared_kernel",
    "local",
)
Regime = Literal[
    "random", "time_blocked", "arc_single", "arc_opposite", "arc_control", "history", "threshold"
]
MAIN_REGIMES: tuple[Regime, ...] = (
    "random",
    "time_blocked",
    "arc_single",
    "arc_opposite",
    "arc_control",
    "history",
)


class ConfigError(ValueError):
    pass


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Window(Frozen):
    start: AwareDatetime
    end: AwareDatetime
    reason: str | None = None

    @model_validator(mode="after")
    def ordered(self) -> "Window":
        if self.end <= self.start:
            raise ValueError(f"window ends at {self.end}, not after its start {self.start}")
        return self

    def overlaps(self, other: "Window") -> bool:
        return self.start < other.end and other.start < self.end


class Dataset(Frozen):
    doi: str
    licence: str
    citation: str
    version: str | None = None
    concept_doi: str | None = None
    acknowledgement: str | None = None


class RawFile(Frozen):
    name: str
    bytes: int | None = None
    md5: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    url: str | None = None


class Windows(Frozen):
    development: Window
    calibration: Window | None = None
    confirmatory: Window | None = None

    @model_validator(mode="after")
    def disjoint(self) -> "Windows":
        windows = [
            (name, window)
            for name in ("calibration", "development", "confirmatory")
            if (window := getattr(self, name)) is not None
        ]
        for i, (a, wa) in enumerate(windows):
            for b, wb in windows[i + 1 :]:
                if wa.overlaps(wb):
                    raise ValueError(f"{a} and {b} windows overlap")
        return self


class Histories(Frozen):
    starts: tuple[AwareDatetime, ...]
    test: Window
    horizons_days: tuple[int, ...] | None = None


class ProviderCorrection(Frozen):
    file: str
    start: AwareDatetime


class Northing(Frozen):
    farm_rotation: Literal["wake_dips", "none"]
    provider_corrections: tuple[ProviderCorrection, ...] = ()


class TimeBlocks(Frozen):
    folds: int
    buffer_days: int


class Source(Frozen):
    archives: str
    turbines: str
    null_values: tuple[str, ...] = ()
    speed_missing_codes: tuple[float, ...] = ()
    event_files: tuple[str, ...] = ()
    hub_height_m: float | None = None


class MessageKind(Frozen):
    status: str
    message: str
    kind: Literal["stop", "curtailment", "derate", "icing", "manual_yaw", "unknown"]


class FarmRules(Frozen):
    fit_power_curve: bool
    causal_offsets: bool = True
    interval_minutes: int = 10
    availability: str | None = None
    downtime_loss: str | None = None
    setpoints: tuple[str, ...] = ()
    setpoint_end: str | None = None
    status_kinds: dict[str, Literal["stop", "curtailment", "derate", "icing", "unknown"]] = {}
    message_kinds: tuple[MessageKind, ...] = ()
    state_codes: tuple[str, ...] = ()
    running_seconds: str | None = None
    min_running_seconds: float | None = None
    stop_seconds: tuple[str, ...] = ()
    linear_mean_angles: tuple[tuple[str, str, str], ...] = ()
    alarm_kinds: dict[str, Literal["stop", "icing"]] = {}
    derated: str | None = None
    low_wind_states: tuple[str, ...] = ()


class FarmConfig(Frozen):
    name: str
    reader: Literal["greenbyte", "hill_of_towie", "smarteole"]
    turbine_type: str
    dataset: Dataset
    raw_files: tuple[RawFile, ...]
    windows: Windows
    histories: Histories
    northing: Northing
    source: Source
    channels: dict[str, str]
    rules: FarmRules
    exclusions: tuple[Window, ...] = ()
    confirmatory_histories: Histories | None = None
    time_blocks: TimeBlocks | None = None


class Curve(Frozen):
    speed_ms: tuple[float, ...]
    value: tuple[float, ...]
    source: str

    @model_validator(mode="after")
    def matched(self) -> "Curve":
        if len(self.speed_ms) != len(self.value) or len(self.speed_ms) < 2:
            raise ValueError("a curve needs matching speed and value lists of length 2 or more")
        if any(b <= a for a, b in zip(self.speed_ms, self.speed_ms[1:], strict=False)):
            raise ValueError("curve speeds must increase")
        return self


class TurbineConfig(Frozen):
    name: str
    rated_power_kw: float
    rotor_diameter_m: float
    source: str
    power_curve: Curve
    thrust_curve: Curve


class ModelSpec(Frozen):
    """one model of models.toml, its base resolved.

    inflow "measured" gives the model each timestamp's measured free-stream turbulence: as the
    ambient turbulence of its wake physics, and as an input of a gbm or graph learner. "average",
    "climatology" and "diurnal" give its wake physics the expectation of that measurement over
    other timestamps instead (windfold.inflow), as controls. scales
    "fixed" or "evidence" sets how its penalized engine priors get their variances, overriding
    the protocol's engine.scales. the text of a spec, which fingerprints its fits, leaves out
    inflow and scales while they hold their defaults, so a model that does not set them keeps
    the fingerprints it had before they existed.
    """

    name: str
    display_name: str
    representation: Literal[
        "reference", "identity", "peer", "wind_relative", "physics", "hybrid", "windfold"
    ]
    blocks: tuple[str, ...] = ()
    physics_scale: Literal["learned", "fixed"] | None = None
    physics_parameters: Literal["calibrated", "default"] = "calibrated"
    kernel_layout: Literal["real", "permuted"] = "real"
    learner: Literal["gbm", "graph", "automl"] | None = None
    features: tuple[str, ...] = ()
    edges: Literal["absolute", "wind_frame"] | None = None
    pretrained: bool = False
    seeded: bool = False
    regimes: tuple[Regime, ...] = MAIN_REGIMES
    history_days: tuple[int, ...] = ()
    inflow: Literal["constant", "measured", "average", "climatology", "diurnal"] = "constant"
    scales: Literal["evidence", "fixed"] | None = None

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        for name, value in super().__repr_args__():
            if name in ("inflow", "scales") and value == type(self).model_fields[name].default:
                continue
            yield name, value

    @model_validator(mode="after")
    def coherent(self) -> "ModelSpec":
        if "threshold" in self.regimes:
            raise ValueError("threshold fits come from monitoring.models, not a model's regimes")
        unknown = [b for b in self.blocks if b not in BLOCKS]
        if unknown:
            raise ValueError(f"unknown blocks {unknown}; known: {list(BLOCKS)}")
        if bool(self.blocks) == bool(self.learner):
            raise ValueError("a model has either blocks or a learner, not both or neither")
        if "physics" in self.blocks and self.physics_scale is None:
            raise ValueError("a model with the physics block sets physics_scale")
        if self.kernel_layout == "permuted" and "shared_kernel" not in self.blocks:
            raise ValueError("a permuted layout needs the shared_kernel block")
        if self.learner == "graph" and self.edges is None:
            raise ValueError("a graph model sets edges")
        if self.pretrained and self.learner != "graph":
            raise ValueError("only graph models can be pretrained")
        if self.inflow == "measured" and not (
            "physics" in self.blocks or (self.learner in ("gbm", "graph") and not self.pretrained)
        ):
            raise ValueError(
                "measured inflow needs the physics block or a gbm or graph learner that is not "
                "pretrained"
            )
        if self.inflow not in ("constant", "measured") and "physics" not in self.blocks:
            raise ValueError(f"{self.inflow} inflow sets the physics block's inflow")
        if self.scales is not None and not self.blocks:
            raise ValueError("scales set the priors of engine blocks; a learner has none")
        return self


class Processing(Frozen):
    speed_bin_edges_ms: tuple[float, ...]
    direction_bucket_deg: float
    steady_change_deg: float
    steady_resultant: float
    stuck_repeats: int
    stopped_power_fraction: float
    control_min_valid_minutes: int
    control_washout_minutes: int
    validity_distance_d: float
    setpoint_binding_fraction: float
    setpoint_tolerance_fraction: float
    offset_window_days: int
    direction_calibration: bool
    max_speed_ms: float
    max_power_fraction: float
    frozen_min_speed_ms: float
    curve_speed_tolerance: float
    curve_power_tolerance: float
    curve_bin_ms: float
    curve_min_rows: int
    consensus_min_others: int
    speed_factor_min_timestamps: int
    min_temperature_c: float
    max_temperature_c: float
    offset_short_window_hours: float
    offset_jump_deg: float
    offset_min_rows: int
    dip_distance_d: tuple[float, float]
    dip_speed_ms: tuple[float, float]
    dip_half_width_deg: float
    dip_min_rows: int
    dip_min_depth: float
    min_valid_turbines: int
    linear_mean_north_deg: float
    linear_mean_distance_deg: float

    @property
    def speed_bin_centres_ms(self) -> tuple[float, ...]:
        edges = self.speed_bin_edges_ms
        return tuple((low + high) / 2 for low, high in zip(edges[:-1], edges[1:], strict=True))


class Splits(Frozen):
    random_folds: int
    time_folds: int
    time_buffer_days: int
    arc_count: int
    arc_width_deg: float
    arc_buffer_deg: float
    arc_min_rows: int
    arc_min_days: int
    control_tolerance: float
    history_horizons_days: tuple[int, ...]
    history_gap_days: int
    history_min_timestamps: int
    tuning_validation_fraction: float
    tuning_embargo_days: int


class Statistics(Frozen):
    block_days: int
    replicates: int
    interval: float
    alpha: float
    min_finite: float = Field(gt=0, le=1)
    coverage_levels: tuple[float, ...]
    turbulence_edges: tuple[float, ...]


class GbmFeatures(Frozen):
    upstream_neighbours: int
    rotor_points: int


class GraphTraining(Frozen):
    steps_per_round: int
    pretrain_rounds: int


class Menus(Frozen):
    size: int
    gbm: dict[str, Any]
    graph: dict[str, Any]


class ValidationMisleads(Frozen):
    physics: str
    baselines: tuple[str, ...]


class WindFrameTransfers(Frozen):
    pairs: tuple[tuple[str, str], ...]


class CoveragePredictsFailure(Frozen):
    relational: str
    physics: str
    covariates: tuple[
        Literal["wake_strength", "angular_distance", "training_rows", "dissimilarity"], ...
    ]
    min_class: int
    confirmation_farms: tuple[str, ...]


class EarlyHistoryAdvantage(Frozen):
    model: str
    horizon_days: int
    accuracy_comparators: tuple[str, ...]
    monitor_comparators: tuple[str, ...]
    drop: float


class Endpoints(Frozen):
    validation_misleads: ValidationMisleads
    wind_frame_transfers: WindFrameTransfers
    coverage_predicts_failure: CoveragePredictsFailure
    early_history_advantage: EarlyHistoryAdvantage

    def models(self) -> dict[str, tuple[str, ...]]:
        v, w = self.validation_misleads, self.wind_frame_transfers
        c, e = self.coverage_predicts_failure, self.early_history_advantage
        return {
            "validation_misleads": (v.physics, *v.baselines),
            "wind_frame_transfers": tuple(m for pair in w.pairs for m in pair),
            "coverage_predicts_failure": (c.relational, c.physics),
            "early_history_advantage": (
                e.model,
                *e.accuracy_comparators,
                *e.monitor_comparators,
            ),
        }


class Monitoring(Frozen):
    min_history_days: int
    fit_fraction: float
    calibrate_from_fraction: float
    min_rows_per_day: int
    min_calibration_days: int
    min_monitorable_share: float
    false_alarms_per_turbine_month: float
    drops: tuple[float, ...]
    copies: int
    episode_days: int
    censor_days: int
    models: tuple[str, ...]
    lookup_min_rows: int


class MatchedMonitoring(Frozen):
    """monitors fitted beside the planned ones and compared with windfold's on the same
    injected copies, outside the decision rule: each sees what windfold sees, so the comparison
    matches information."""

    models: tuple[str, ...]


class Decision(Frozen):
    min_development_farms: int
    min_confirmatory_datasets: int
    exceedance_margin: float
    noninferiority_margin: float


class Seeds(Frozen):
    study: int
    model: tuple[int, ...]


class Smearing(Frozen):
    min_rows_per_bin: int


class AutoMLBudget(Frozen):
    candidate_minutes: tuple[float, ...]
    min_relative_gain: float


class AutoML(Frozen):
    preset: str
    hyperparameters: str
    models: tuple[str, ...]
    gpu_models: tuple[str, ...]
    bag_folds: int
    stack_levels: int


class Physics(Frozen):
    trials: int
    power_floor_fraction: float
    truncate_sd: float
    kernel_family: Literal["jensen", "niayifar", "turbopark"]
    search: dict[str, dict[str, tuple[float, float]]]
    defaults: dict[str, dict[str, float]]

    @model_validator(mode="after")
    def families_agree(self) -> "Physics":
        families = {"jensen", "niayifar", "turbopark"}
        if set(self.search) != families or set(self.defaults) != families:
            raise ValueError(f"search and defaults must cover exactly {sorted(families)}")
        for family, ranges in self.search.items():
            if set(ranges) != set(self.defaults[family]):
                raise ValueError(f"{family}: search and defaults name different parameters")
            for name, (low, high) in ranges.items():
                if not low <= self.defaults[family][name] <= high:
                    raise ValueError(f"{family}.{name}: default outside its search range")
        return self


class Engine(Frozen):
    starts: tuple[float, ...]
    noise_floor: float
    variance_floor: float
    variance_ceiling: float
    offset_variance: float
    physics_variance: float
    harmonics_order: int
    spline_knots: int
    spline_ridge: float
    lookup_window_deg: int
    scales: Literal["evidence", "fixed"]
    fixed_prior_variance: float


class Inflow(Frozen):
    """the grid of turbulence ratios, two to the power low, low + step, ..., high, on which
    measured-inflow physics is evaluated, ratio one on it; and the direction sectors and parts of
    the day within which the inflow controls average (windfold.inflow)."""

    log2_ratio_low: float
    log2_ratio_high: float
    log2_ratio_step: float
    climatology_sector_deg: float
    diurnal_hours: int

    @model_validator(mode="after")
    def holds_one(self) -> "Inflow":
        if not self.log2_ratio_low < 0 < self.log2_ratio_high or self.log2_ratio_step <= 0:
            raise ValueError("the ratio grid runs from below one to above one in positive steps")
        steps = -self.log2_ratio_low / self.log2_ratio_step
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("the ratio grid must hold ratio one exactly")
        span = (self.log2_ratio_high - self.log2_ratio_low) / self.log2_ratio_step
        if abs(span - round(span)) > 1e-9:
            raise ValueError("the ratio grid must end on its high bound")
        if self.climatology_sector_deg <= 0 or 360 % self.climatology_sector_deg:
            raise ValueError("climatology sectors must divide the circle")
        if self.diurnal_hours <= 0 or 24 % self.diurnal_hours:
            raise ValueError("diurnal parts must divide the day")
        return self


class Kernel(Frozen):
    along_max_d: float
    across_max_d: float
    along_first_knot_d: float
    along_knots: int
    across_first_knot_d: float
    across_knots: int
    degree: Literal[0, 3]
    ridge: float


class Coverage(Frozen):
    noise_variance: float
    kernel_variance: float
    dissimilarity_sample: int


class Trial(Frozen):
    toggle_minutes: int
    durations_days: tuple[int, ...]
    replicates: int


class Sensitivity(Frozen):
    overrides: dict[str, Any]
    changes_processing: bool = False
    models: tuple[str, ...] = ()
    regimes: tuple[Regime, ...] = ()

    @model_validator(mode="after")
    def scoped(self) -> "Sensitivity":
        if bool(self.models) != bool(self.regimes):
            raise ValueError(
                "a sensitivity names both the models and the regimes it refits, or neither"
            )
        return self


class Protocol(Frozen):
    processing: Processing
    splits: Splits
    statistics: Statistics
    monitoring: Monitoring
    matched_monitoring: MatchedMonitoring
    decision: Decision
    seeds: Seeds
    smearing: Smearing
    automl_budget: AutoMLBudget
    automl: AutoML
    physics: Physics
    inflow: Inflow
    engine: Engine
    kernel: Kernel
    coverage: Coverage
    trial: Trial
    endpoints: Endpoints
    gbm: GbmFeatures
    graph: GraphTraining
    menus: Menus
    hypotheses: tuple[str, ...]
    sensitivities: dict[str, Sensitivity] = {}


class Limits(Frozen):
    days: int | None = None
    arcs: int | None = None
    histories: int | None = None
    seeds: int | None = None
    replicates: int | None = None
    copies: int | None = None
    automl_minutes: float | None = None


class Study(Frozen):
    name: str
    period: Literal["development", "confirmatory"]
    farms: tuple[str, ...]
    models: tuple[str, ...]
    sensitivities: tuple[str, ...] = ()
    limits: Limits | None = None


class Resources(Frozen):
    data_dir: str
    runs_dir: str
    reserved_threads: int
    cores_per_cpu_job: int
    gpu_workers_per_device: int
    automl_jobs_per_device: int
    status_interval_seconds: int
    stall_minutes: int
    replan_seconds: float = Field(gt=0)
    failure_fraction: float = Field(gt=0, lt=1)
    failures_before_pause: int = Field(ge=1)
    memory_margin: float = Field(ge=1)
    reserved_memory_gb: float = Field(ge=0)
    load_rows: int = Field(gt=0)
    load_features: int = Field(gt=0)
    cpu_jobs_on_second_threads: bool = False
    gpu_workers_keep_whole_cores: bool = False


def config_root(folder: Path | None = None) -> Traversable:
    return Path(folder) if folder is not None else resources.files("windfold") / "configs"


def read_toml(root: Traversable, *parts: str) -> dict[str, Any]:
    target = root.joinpath(*parts)
    if not target.is_file():
        raise ConfigError(f"no config file {'/'.join(parts)} in {root}")
    try:
        return tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{'/'.join(parts)}: {error}") from error


def build[T: BaseModel](schema: type[T], data: dict[str, Any], where: str) -> T:
    try:
        return schema.model_validate(data)
    except ValidationError as error:
        lines = []
        for problem in error.errors():
            key = ".".join(str(p) for p in problem["loc"]) or "(top level)"
            lines.append(f"  {key}: {problem['msg']}")
        raise ConfigError(f"{where} is not valid:\n" + "\n".join(lines)) from None


def names_in(root: Traversable, folder: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            p.name.removesuffix(".toml")
            for p in root.joinpath(folder).iterdir()
            if p.name.endswith(".toml")
        )
    )


def load_farm(name: str, folder: Path | None = None) -> FarmConfig:
    root = config_root(folder)
    if name not in names_in(root, "farms"):
        raise ConfigError(f"unknown farm {name}; known: {list(names_in(root, 'farms'))}")
    farm = build(FarmConfig, read_toml(root, "farms", f"{name}.toml"), f"farms/{name}.toml")
    if farm.name != name:
        raise ConfigError(f"farms/{name}.toml names itself {farm.name}")
    return farm


def load_turbine(name: str, folder: Path | None = None) -> TurbineConfig:
    root = config_root(folder)
    if name not in names_in(root, "turbines"):
        raise ConfigError(f"unknown turbine type {name}; known: {list(names_in(root, 'turbines'))}")
    return build(
        TurbineConfig, read_toml(root, "turbines", f"{name}.toml"), f"turbines/{name}.toml"
    )


def load_protocol(folder: Path | None = None) -> Protocol:
    return build(Protocol, read_toml(config_root(folder), "protocol.toml"), "protocol.toml")


def load_resources(folder: Path | None = None) -> Resources:
    return build(Resources, read_toml(config_root(folder), "resources.toml"), "resources.toml")


def load_study(name: str, folder: Path | None = None) -> Study:
    root = config_root(folder)
    if name not in names_in(root, "studies"):
        raise ConfigError(f"unknown study {name}; known: {list(names_in(root, 'studies'))}")
    study = build(Study, read_toml(root, "studies", f"{name}.toml"), f"studies/{name}.toml")
    if study.name != name:
        raise ConfigError(f"studies/{name}.toml names itself {study.name}")
    return study


def resolve_models(raw: dict[str, dict[str, Any]]) -> dict[str, ModelSpec]:
    resolved: dict[str, dict[str, Any]] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> dict[str, Any]:
        if name in chain:
            raise ConfigError(f"models.toml: inheritance cycle {' -> '.join((*chain, name))}")
        if name not in raw:
            raise ConfigError(f"models.toml: unknown model {name} (base of {chain[-1]})")
        if name in resolved:
            return resolved[name]
        entry = dict(raw[name])
        base = entry.pop("base", None)
        drop = tuple(entry.pop("drop", ()))
        merged: dict[str, Any] = {}
        if base is not None:
            merged = {k: v for k, v in resolve(base, (*chain, name)).items() if k != "display_name"}
            missing = [b for b in drop if b not in merged.get("blocks", ())]
            if missing:
                raise ConfigError(
                    f"models.toml: {name} drops {missing}, which {base} does not have"
                )
            merged["blocks"] = tuple(b for b in merged.get("blocks", ()) if b not in drop)
            if "physics" in drop:
                merged.pop("physics_scale", None)
                merged.pop("physics_parameters", None)
                merged.pop("inflow", None)
        elif drop:
            raise ConfigError(f"models.toml: {name} drops blocks but has no base")
        merged.update(entry)
        merged["name"] = name
        resolved[name] = merged
        return merged

    return {name: build(ModelSpec, resolve(name, ()), f"models.toml [{name}]") for name in raw}


def load_models(folder: Path | None = None) -> dict[str, ModelSpec]:
    return resolve_models(read_toml(config_root(folder), "models.toml"))


def load_menu(learner: str, folder: Path | None = None) -> tuple[dict[str, Any], ...]:
    """the frozen menu of one learner, its configurations in index order."""
    menu = read_toml(config_root(folder), "menus", f"{learner}.toml")
    configurations = menu.get("configuration", [])
    if [c.get("index") for c in configurations] != list(range(menu.get("size", -1))):
        raise ConfigError(f"menus/{learner}.toml lists its configurations out of order")
    return tuple({k: v for k, v in c.items() if k != "index"} for c in configurations)


def folder_of(configs: str | None) -> Path | None:
    return Path(configs) if configs else None


def for_farm(
    farm: str, sensitivity: str | None, folder: Path | None = None
) -> tuple[Protocol, FarmConfig, TurbineConfig]:
    """the protocol, farm and turbine configs one farm runs with, a sensitivity applied."""
    protocol = load_protocol(folder)
    farms = {farm: load_farm(farm, folder)}
    if sensitivity:
        protocol, farms = with_sensitivity(sensitivity, protocol, farms)
    return protocol, farms[farm], load_turbine(farms[farm].turbine_type, folder)


def tuned_model(model: str, specs: dict[str, ModelSpec]) -> str:
    """the model whose search gives model its configuration: itself, or for a pretrained graph
    the graph with the same edges and inflow that is not pretrained."""
    spec = specs[model]
    if not spec.pretrained:
        return model
    (source,) = [
        name
        for name, other in specs.items()
        if other.learner == spec.learner
        and other.edges == spec.edges
        and other.inflow == spec.inflow
        and not other.pretrained
    ]
    return source


def check_model_names(protocol: Protocol, models: dict[str, ModelSpec]) -> None:
    named = (
        {
            "monitoring.models": protocol.monitoring.models,
            "matched_monitoring.models": protocol.matched_monitoring.models,
        }
        | {
            f"sensitivities.{name}.models": entry.models
            for name, entry in protocol.sensitivities.items()
        }
        | {f"endpoints.{name}": names for name, names in protocol.endpoints.models().items()}
    )
    for where, names in named.items():
        unknown = sorted(set(names) - set(models))
        if unknown:
            raise ConfigError(f"protocol.toml {where}: unknown models {unknown}")


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any], where: str) -> dict[str, Any]:
    result = json.loads(json.dumps(data, default=str))
    for path, value in overrides.items():
        keys = path.split(".")
        node = result
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"{where}: override {path} names no existing setting")
            node = node[key]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise ConfigError(f"{where}: override {path} names no existing setting")
        node[keys[-1]] = value
    return result


def with_sensitivity(
    name: str, protocol: Protocol, farms: dict[str, FarmConfig]
) -> tuple[Protocol, dict[str, FarmConfig]]:
    if name not in protocol.sensitivities:
        raise ConfigError(f"unknown sensitivity {name}; known: {sorted(protocol.sensitivities)}")
    overrides = protocol.sensitivities[name].overrides
    on_protocol = {k: v for k, v in overrides.items() if not k.startswith("farms.")}
    on_farms: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        if key.startswith("farms."):
            _, farm, rest = key.split(".", 2)
            if farm not in farms:
                raise ConfigError(f"sensitivity {name} overrides unknown farm {farm}")
            on_farms.setdefault(farm, {})[rest] = value
    where = f"sensitivity {name}"
    changed = protocol.model_dump(mode="json")
    changed = apply_overrides(changed, on_protocol, where)
    new_protocol = build(Protocol, changed, where)
    new_farms = dict(farms)
    for farm, farm_overrides in on_farms.items():
        data = apply_overrides(farms[farm].model_dump(mode="json"), farm_overrides, where)
        new_farms[farm] = build(FarmConfig, data, where)
    return new_protocol, new_farms
