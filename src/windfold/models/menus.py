"""the frozen tuning menus and the calibration-window search.

a menu is a fixed number of configurations drawn once, with a fixed seed, from a search space in
the packaged config, then written once and never redrawn. the search fits each configuration on
the first part of a farm's calibration window and scores it on the last part, after an embargo.
"""

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.models.interface import FitContext, Fitted, Model

KINDS = ("uniform", "log", "integer")


def sample(spec: Any, rng: np.random.Generator) -> Any:
    """one draw: a list is a choice of its values; {uniform, log or integer: [low, high]} is a
    range, uniform on the value, uniform on its logarithm or uniform on the integers in it."""
    if isinstance(spec, list | tuple):
        return spec[int(rng.integers(len(spec)))]
    if not isinstance(spec, Mapping) or len(spec) != 1 or next(iter(spec)) not in KINDS:
        raise ValueError(f"a search range is a list of choices or one of {KINDS}, got {spec!r}")
    kind, (low, high) = next(iter(spec.items()))
    if kind == "integer":
        return int(rng.integers(int(low), int(high) + 1))
    if kind == "log":
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    return float(f"{value:.6g}")


def draw(space: Mapping[str, Any], size: int, seed: int) -> list[dict[str, Any]]:
    """size configurations, each drawing every parameter in the space's order from one seeded
    generator, so the same space, size and seed give the same menu."""
    rng = np.random.default_rng(seed)
    return [{name: sample(spec, rng) for name, spec in space.items()} for _ in range(size)]


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"cannot write {value!r} to a menu")


def write_menu(path: Path, menu: Sequence[Mapping[str, Any]], seed: int) -> None:
    """write the menu once; an existing file is never replaced."""
    if path.exists():
        raise FileExistsError(f"{path} exists; menus are frozen and written once")
    lines = [f"seed = {seed}", f"size = {len(menu)}"]
    for index, configuration in enumerate(menu):
        lines += ["", "[[configuration]]", f"index = {index}"]
        lines += [f"{name} = {toml_value(value)}" for name, value in configuration.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.link(part, path)
    part.unlink()


def write_menus(
    folder: Path, spaces: Mapping[str, Mapping[str, Any]], size: int, seed: int
) -> list[Path]:
    """draw and write each learner's menu as folder/<learner>.toml, refusing any that exists."""
    clashes = [folder / f"{name}.toml" for name in spaces if (folder / f"{name}.toml").exists()]
    if clashes:
        raise FileExistsError(f"menus are frozen and written once; found {clashes}")
    paths = []
    for name, space in spaces.items():
        path = folder / f"{name}.toml"
        write_menu(path, draw(space, size, seed), seed)
        paths.append(path)
    return paths


@dataclass(frozen=True)
class CalibrationSplit:
    fit: pl.DataFrame
    validation: pl.DataFrame
    record: dict[str, Any]


def calibration_split(
    rows: pl.DataFrame,
    start: datetime,
    end: datetime,
    validation_fraction: float,
    embargo_days: float,
) -> CalibrationSplit:
    """with c the calendar-time point (1 - validation_fraction) of [start, end): fit on
    [start, c - embargo), drop the embargo [c - embargo, c), validate on [c, end)."""
    stamp = pl.col("timestamp")
    outside = rows.filter((stamp < start) | (stamp >= end)).height
    if outside:
        raise ValueError(f"{outside} rows lie outside the calibration window [{start}, {end})")
    cut = start + (end - start) * (1.0 - validation_fraction)
    embargo = cut - timedelta(days=embargo_days)
    fit = rows.filter(stamp < embargo)
    held = rows.filter((stamp >= embargo) & (stamp < cut))
    validation = rows.filter(stamp >= cut)
    record: dict[str, Any] = {
        "window_start": start.isoformat(),
        "embargo_start": embargo.isoformat(),
        "validation_start": cut.isoformat(),
        "window_end": end.isoformat(),
    }
    for name, part in (("fit", fit), ("embargo", held), ("validation", validation)):
        record[f"{name}_rows"] = part.height
        record[f"{name}_timestamps"] = part["timestamp"].n_unique()
        record[f"{name}_first"] = None if part.is_empty() else str(part["timestamp"].min())
        record[f"{name}_last"] = None if part.is_empty() else str(part["timestamp"].max())
    return CalibrationSplit(fit, validation, record)


def score(
    model: Model, split: CalibrationSplit, context: FitContext
) -> tuple[dict[str, Any], Fitted]:
    """fit on the split's fit rows and score the centered contrast on its validation rows: mse
    over all rows and over waked rows."""
    start = time.perf_counter()
    fitted = model.fit(split.fit, context)
    fit_seconds = time.perf_counter() - start
    prediction = fitted.predict(split.validation)
    error = (prediction["mean"] - split.validation["y"]).to_numpy()
    waked = split.validation["waked"].to_numpy()
    row = {
        "mse": float(np.mean(error**2)),
        "waked_mse": float(np.mean(error[waked] ** 2)) if waked.any() else None,
        "fit_seconds": fit_seconds,
        "fit_rows": split.fit.height,
        "validation_rows": split.validation.height,
    }
    return row, fitted


def choose(mse: Sequence[float]) -> int:
    """the configuration with the lowest validation mse, ties to the earliest in the menu."""
    return min(range(len(mse)), key=lambda index: (mse[index], index))


def ranks(mse: Sequence[float]) -> np.ndarray:
    """rank 1 for the lowest mse, ties by menu order."""
    order = sorted(range(len(mse)), key=lambda index: (mse[index], index))
    out = np.empty(len(mse), dtype=np.int64)
    out[order] = np.arange(1, len(mse) + 1)
    return out


def best_mean_rank(searches: Sequence[Sequence[float]]) -> int:
    """the configuration with the best mean rank over farms' searches; ties in mean rank go to
    the earliest in the menu."""
    sizes = {len(mse) for mse in searches}
    if len(sizes) != 1:
        raise ValueError(f"every search covers the same menu, got sizes {sorted(sizes)}")
    total = np.sum([ranks(mse) for mse in searches], axis=0)
    return min(range(sizes.pop()), key=lambda index: (total[index], index))
