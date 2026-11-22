"""a tiny development study for the heavy analyses and the secondary tests: a four-turbine farm
under kelmarsh's name and configs, with a whole prepared folder of rows in a calibration and a
development period, its physics and inflow tables and kernel designs, a manifest of two single
arcs, a tuned geometry gbm, the runner's sidecars of those, the study's fits of the compared
models and the restricted windfold. the modules that build it load only when a test asks for it."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl
import pytest

FARM = "kelmarsh"
SIDECAR = ".sidecar.json"
SPLITS = {"arc_090_single": 90.0, "arc_270_single": 270.0}
EAST = np.array([0.0, 460.0, 920.0, 230.0])
NORTH = np.array([0.0, 10.0, -10.0, 420.0])
DIAMETER = 92.5
PARAMETERS = {
    "jensen": {"k": 0.06, "spread_deg": 1.0},
    "niayifar": {"ti": 0.09, "spread_deg": 1.0},
    "turbopark": {"a": 0.04, "ti": 0.09, "spread_deg": 1.0},
}
STUDY_MODELS = (
    "no_wake",
    "windfold",
    "pywake_inflow",
    "windfold_diurnal_inflow",
    "windfold_no_local",
)


@dataclass(frozen=True)
class TinyStudy:
    configs: Path
    data: Path
    runs: Path
    splits: tuple[str, ...]


def tiny_configs(folder: Path) -> Path:
    """the packaged configs with two calibration trials, a coarser ratio grid and the farm's
    development window cut to the three weeks of the tiny study."""
    shutil.copytree(Path(str(resources.files("windfold") / "configs")), folder)
    protocol = folder / "protocol.toml"
    text = protocol.read_text().replace("trials = 100", "trials = 2")
    protocol.write_text(text.replace("log2_ratio_step = 0.25", "log2_ratio_step = 0.5"))
    farm = folder / "farms" / f"{FARM}.toml"
    window = "end = 2025-01-01T00:00:00Z"
    farm.write_text(farm.read_text().replace(window, "end = 2018-01-22T00:00:00Z", 1))
    return folder


def stamps(start: datetime, count: int, hours: int) -> list[datetime]:
    return [start + timedelta(hours=hours * k) for k in range(count)]


def prepared_rows(seed: int = 4) -> pl.DataFrame:
    from windfold import geometry
    from windfold.models import engine
    from windfold.schemas import PREPARED_ROWS, conform

    rng = np.random.default_rng(seed)
    times = stamps(datetime(2017, 12, 1, tzinfo=UTC), 64, 3)
    times += stamps(datetime(2018, 1, 1, tzinfo=UTC), 252, 2)
    periods = ["calibration"] * 64 + ["development"] * 252
    count, n = len(times), len(EAST)
    along_row = rng.choice([90.0, 100.0, 110.0, 270.0, 280.0, 290.0], count)
    direction = np.where(rng.random(count) < 0.5, along_row, 10.0 * rng.integers(0, 36, count))
    full = (1 << n) - 1
    masks = np.where(rng.random(count) < 0.1, full & ~(1 << rng.integers(0, n, count)), full)
    turbulence = np.exp(rng.normal(np.log(0.1), 0.3, count))
    turbulence[rng.random(count) < 0.05] = np.nan
    t, i = np.nonzero((masks[:, None] >> np.arange(n)) & 1)
    along, across = geometry.relations(EAST, NORTH, direction[t], DIAMETER)
    receiver = np.arange(t.size)
    along, across = along[receiver, i], across[receiver, i]
    operating = ((masks[t][:, None] >> np.arange(n)) & 1).astype(bool)
    source = geometry.upstream(along, across, 30.0, 3.0) & operating
    deficit = 0.4 * np.exp(-along / 10.0) * np.exp(-0.5 * (across / 0.8) ** 2)
    scale = np.where(np.isfinite(turbulence[t]), turbulence[t] / 0.1, 1.0) ** -0.5
    raw = np.array([0.02, -0.01, 0.0, 0.03])[i] - (deficit * source).sum(1) * scale
    y = engine.centre(raw + rng.normal(0, 0.03, t.size), t.astype(np.int64))
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series([times[k] for k in t]).dt.cast_time_unit("us"),
            "turbine": [f"T{k}" for k in i],
            "turbine_index": pl.Series(i, dtype=pl.UInt8),
            "period": [periods[k] for k in t],
            "power_kw": 900.0 * np.exp(y),
            "speed": np.full(t.size, 7.5),
            "speed_std": np.full(t.size, 0.75),
            "direction_raw": direction[t],
            "direction": direction[t],
            "nacelle_position": direction[t],
            "farm_direction": direction[t],
            "reference_speed": np.full(t.size, 7.5),
            "speed_bin": pl.Series(rng.integers(0, 3, count)[t], dtype=pl.UInt8),
            "direction_bucket": pl.Series((direction[t] // 5).astype(int), dtype=pl.UInt8),
            "steady": np.ones(t.size, bool),
            "operating_mask": pl.Series(masks[t], dtype=pl.UInt32),
            "waked": source.any(1),
            "valid_turbines": pl.Series(np.bincount(t)[t], dtype=pl.UInt8),
            "y": y,
            "temperature": np.full(t.size, 5.0),
            "turbulence_intensity": turbulence[t],
        }
    )
    return conform(frame, PREPARED_ROWS)


def manifest(rows: pl.DataFrame) -> pl.DataFrame:
    """each split tests on the development timestamps in its arc and trains on those outside it
    and its five-degree buffers."""
    from windfold.schemas import MANIFEST, conform

    times = rows.filter(pl.col("period") == "development").unique("timestamp").sort("timestamp")
    parts = []
    for split, start in SPLITS.items():
        offset = (pl.col("farm_direction") - start) % 360.0
        role = pl.when(offset < 30.0).then(pl.lit("test"))
        role = role.when((offset < 35.0) | (offset >= 355.0)).then(pl.lit("buffer"))
        chosen = times.select("timestamp", role=role.otherwise(pl.lit("train")))
        parts.append(chosen.with_columns(split=pl.lit(split), regime=pl.lit("arc_single")))
    return conform(pl.concat(parts), MANIFEST)


def summary(rows: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    tests = entries.filter(pl.col("role") == "test").join(rows, on="timestamp")
    waked = tests.group_by("split").agg(test_waked_rows=pl.col("waked").sum())
    return (
        pl.DataFrame({"split": list(SPLITS)})
        .join(waked, on="split", how="left")
        .with_columns(regime=pl.lit("arc_single"), status=pl.lit("included"))
    )


def prove(folder: Path) -> None:
    """a sidecar beside every file under folder, as the runner leaves one beside each output."""
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and SIDECAR not in p.name):
        content = path.read_bytes()
        record = {"fingerprint": "tiny", "sha256": hashlib.sha256(content).hexdigest()}
        sidecar = path.with_name(path.name + SIDECAR)
        sidecar.write_text(json.dumps(record | {"bytes": len(content)}))


@pytest.fixture(scope="session")
def tiny_study(tmp_path_factory) -> TinyStudy:
    from windfold.config import load_menu
    from windfold.locations import chosen_file
    from windfold.schemas import LAYOUT, conform
    from windfold.secondary import restricted_windfold
    from windfold.tasks.data import inflow_physics, kernel_designs, physics_table
    from windfold.tasks.fitting import fit_engine

    root = tmp_path_factory.mktemp("tiny_study")
    configs = tiny_configs(root / "configs")
    data, runs = root / "data", root / "runs"
    prepared = data / "prepared" / FARM
    prepared.mkdir(parents=True)
    rows = prepared_rows()
    rows.write_parquet(prepared / "rows.parquet")
    layout = pl.DataFrame(
        {
            "turbine": [f"T{k}" for k in range(len(EAST))],
            "latitude": np.zeros(len(EAST)),
            "longitude": np.zeros(len(EAST)),
            "x_m": EAST,
            "y_m": NORTH,
            "hub_height_m": np.full(len(EAST), 78.5),
        }
    )
    conform(layout, LAYOUT).write_parquet(prepared / "layout.parquet")
    fit = {"seed": 5, "parameters": PARAMETERS, "teacher": "niayifar"}
    (prepared / "physics_fit.json").write_text(json.dumps(fit))
    (prepared / "ledger.csv").write_text(f"step,rule,rows_kept\n1,read,{rows.height}\n")
    (prepared / "provenance.json").write_text(json.dumps({"farm": FARM}))
    physics_table(FARM, None, str(data), str(configs), 1, data)
    inflow_physics(FARM, None, str(data), str(configs), 1, [], data)
    kernel_designs(FARM, None, str(data), str(configs), data)
    entries = manifest(rows)
    folder = runs / "development" / "manifests"
    folder.mkdir(parents=True)
    entries.write_parquet(folder / f"{FARM}.parquet")
    summary(rows, entries).write_csv(folder / f"{FARM}_summary.csv")
    chosen = runs / chosen_file(FARM, "geometry_gbm")
    chosen.parent.mkdir(parents=True)
    configuration = load_menu("gbm", configs)[0] | {"num_boost_round": 20}
    chosen.write_text(json.dumps({"configuration": configuration}))
    for proven in (data / "prepared", folder, chosen.parent):
        prove(proven)
    for split in SPLITS:
        fit_engine(
            "development",
            FARM,
            split,
            list(STUDY_MODELS),
            None,
            str(data),
            str(runs),
            str(configs),
            runs,
        )
        restricted_windfold.fit(data, runs, "development", FARM, split, str(configs))
    return TinyStudy(configs, data, runs, tuple(SPLITS))
