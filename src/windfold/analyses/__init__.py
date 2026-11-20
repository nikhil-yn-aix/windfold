"""exploratory analyses, behind statements of the paper but outside the study plan. each
is a function of explicit folders that returns plain data, and `windfold analyse <name>` writes
one as json under the runs folder's <study>/analyses. those of ANY_STUDY read either study, and
confirmatory results only after the second freeze; the others read development data and results
only. the heavy ones refit models in a folder of their own and make missing refits only when
asked to.
"""

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from windfold.analyses.development import FARMS, LARGE_FARMS, STUDY
from windfold.config import load_study

ANY_STUDY = (
    "comparison_ranks",
    "interval_score",
    "pair_skill",
    "practical_scale",
    "sector_gains",
)
COVERED = {
    "direction_spread": ("kelmarsh", "penmanshiel"),
    "lidar_turbulence": ("smarteole",),
    "strict_calibration": LARGE_FARMS,
}


def covered(name: str, study: str = STUDY, configs: Path | None = None) -> tuple[str, ...]:
    """the farms an analysis can analyze: the study's for an analysis of ANY_STUDY, otherwise
    those of COVERED or every farm."""
    if name in ANY_STUDY:
        return tuple(load_study(study, configs).farms)
    return COVERED.get(name, FARMS)


def run(
    name: str,
    data: Path,
    runs: Path,
    farms: Sequence[str] = (),
    configs: Path | None = None,
    work: Path | None = None,
    refit: bool = False,
    workers: int = 1,
    study: str = STUDY,
) -> tuple[list[str], dict[str, Any]]:
    """the farms analyzed and the result of one analysis of the study, over farms or every farm
    it covers; a farm it does not cover is refused. a heavy analysis keeps its refits in work, by
    default runs/<study>/analyses/<name>."""
    folder = work or runs / study / "analyses" / name
    calls: dict[str, Callable[[ModuleType], dict[str, Any]]] = {
        "comparison_ranks": lambda m: m.comparison_ranks(runs, study, chosen, configs),
        "direction_spread": lambda m: m.direction_spread(data, chosen, configs),
        "inflow_smoothing": lambda m: m.inflow_smoothing(
            data, runs, folder, chosen, refit, workers, configs
        ),
        "inflow_structure": lambda m: m.inflow_structure(data, chosen, configs),
        "interval_score": lambda m: m.interval_score(data, runs, study, chosen, configs),
        "kernel_geometry": lambda m: m.kernel_geometry(data, runs, chosen, configs),
        "lidar_turbulence": lambda m: m.lidar_turbulence(data),
        "linear_clipping": lambda m: m.linear_clipping(data, runs, chosen),
        "local_block": lambda m: m.local_block(runs, chosen),
        "pair_skill": lambda m: m.pair_skill(runs, study, chosen, configs),
        "practical_scale": lambda m: m.practical_scale(runs, study, chosen, configs),
        "sector_gains": lambda m: m.sector_gains(runs, study, chosen, configs),
        "sensor_count": lambda m: m.sensor_count(data, runs, chosen),
        "strict_calibration": lambda m: m.strict_calibration(
            data, runs, folder, chosen, refit, workers, configs
        ),
        "transfer_ceiling": lambda m: m.transfer_ceiling(data, runs, chosen, configs),
        "turbine_levels": lambda m: m.turbine_levels(data, runs, chosen, configs),
        "validation_choice": lambda m: m.validation_choice(runs, chosen, configs),
    }
    if name not in calls:
        raise ValueError(f"no analysis {name}; the analyses are {', '.join(calls)}")
    if study != STUDY and name not in ANY_STUDY:
        raise ValueError(f"{name} analyses the {STUDY} study only")
    every = covered(name, study, configs)
    if unknown := [farm for farm in farms if farm not in every]:
        raise ValueError(f"{name} does not analyse {', '.join(unknown)}; only {', '.join(every)}")
    chosen = tuple(farms) or every
    return list(chosen), calls[name](importlib.import_module(f"windfold.analyses.{name}"))
