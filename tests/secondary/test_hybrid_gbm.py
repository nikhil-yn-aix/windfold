import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from windfold.config import load_farm, load_protocol, load_turbine
from windfold.locations import chosen_file, manifest_file
from windfold.schemas import MANIFEST, PREPARED_ROWS, conform
from windfold.secondary import hybrid_gbm, restricted_windfold
from windfold.secondary.contrasts import confirmatory_ready, model_errors
from windfold.secondary.hybrid_gbm import MODEL, fit


def test_confirmatory_data_wait_for_the_second_freeze_record(tmp_path):
    configs = shutil.copytree(Path(str(resources.files("windfold") / "configs")), tmp_path / "c")
    (configs / "studies" / "second_confirmatory.toml").write_text(
        'name = "second_confirmatory"\nperiod = "confirmatory"\n'
        'farms = ["penmanshiel"]\nmodels = ["no_wake"]\n'
    )
    folder, data, farm, arc = str(configs), tmp_path / "data", "penmanshiel", "arc_000_single"
    confirmatory_ready(tmp_path, "development", folder)
    entries = [
        lambda study: hybrid_gbm.fit(data, tmp_path, study, farm, arc, 0, 1, folder),
        lambda study: hybrid_gbm.fit_all(data, tmp_path, study, farm, 1, 1, folder),
        lambda study: hybrid_gbm.score(data, tmp_path, study, farm, folder),
        lambda study: restricted_windfold.fit(data, tmp_path, study, farm, arc, folder),
        lambda study: restricted_windfold.fit_all(data, tmp_path, study, farm, 1, folder),
        lambda study: restricted_windfold.score(data, tmp_path, study, farm, folder),
    ]
    for study in ("confirmatory", "second_confirmatory"):
        for entry in entries:
            with pytest.raises(FileNotFoundError, match="second freeze"):
                entry(study)
    record = tmp_path / "second_confirmatory" / "freeze_second.json"
    record.parent.mkdir()
    record.write_text("{not json")
    with pytest.raises(ValueError):
        confirmatory_ready(tmp_path, "second_confirmatory", folder)
    record.write_text(json.dumps({"predictions": {}}))
    confirmatory_ready(tmp_path, "second_confirmatory", folder)


def test_a_fit_is_kept_only_while_its_provenance_matches(tiny_study, tmp_path):
    runs = shutil.copytree(tiny_study.runs, tmp_path / "runs")
    configs = shutil.copytree(tiny_study.configs, tmp_path / "configs")
    split = "arc_090_single"
    args = (tiny_study.data, runs, "development", "kelmarsh", split, 0, 1, str(configs))
    table = fit(*args)
    written = table.stat().st_mtime_ns
    assert fit(*args) == table and table.stat().st_mtime_ns == written
    protocol = configs / "protocol.toml"
    text = protocol.read_text()
    protocol.write_text(text.replace("upstream_neighbours = 3", "upstream_neighbours = 4"))
    with pytest.raises(ValueError, match="made by other code, settings or inputs; delete"):
        fit(*args)
    protocol.write_text(text)
    note = table.with_suffix(".json")
    record = json.loads(note.read_text())
    record.pop("provenance", None)
    note.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=f"{note.name} has no provenance; delete .* to fit it"):
        fit(*args)
    scored = model_errors(
        tiny_study.data,
        runs,
        "development",
        "kelmarsh",
        MODEL,
        [0],
        [split],
        load_protocol(configs),
    )
    assert scored["rows"].sum() > 0


def proven(path: Path) -> None:
    """a runner sidecar beside path."""
    content = path.read_bytes()
    proof = {"fingerprint": "f", "sha256": hashlib.sha256(content).hexdigest()}
    path.with_name(path.name + ".sidecar.json").write_text(
        json.dumps(proof | {"bytes": len(content)})
    )


OFFSETS = np.array([0.03, -0.01, 0.02, -0.04])


def write_row_farm(data: Path, runs: Path, configuration: dict) -> list[Path]:
    """four kelmarsh turbines five diameters apart from west to east, whose y holds their own
    offsets and noise. the single arc tests on winds from 0 to 30 degrees, where no turbine has
    an upstream neighbor, and trains on winds from 40 to 340 degrees; the geometry gbm's tuned
    configuration is given. returns the proven files a fit rests on."""
    diameter = load_turbine(load_farm("kelmarsh").turbine_type).rotor_diameter_m
    rng = np.random.default_rng(3)
    start = datetime(2018, 3, 1, tzinfo=UTC)
    directions = np.r_[rng.uniform(40, 340, 300), rng.uniform(0, 30, 60)]
    records, roles = [], []
    for k, direction in enumerate(directions):
        stamp = start + timedelta(minutes=10 * k)
        y = OFFSETS + rng.normal(0, 0.002, 4)
        y -= y.mean()
        roles.append((stamp, "test" if direction < 30 else "train"))
        for i in range(4):
            records.append(
                {
                    "timestamp": stamp,
                    "turbine": f"T{i}",
                    "turbine_index": i,
                    "period": "development",
                    "power_kw": 1000.0 * np.exp(y[i]),
                    "speed": 8.0,
                    "speed_std": 0.8,
                    "direction_raw": direction,
                    "direction": direction,
                    "nacelle_position": direction,
                    "farm_direction": direction,
                    "reference_speed": 8.0,
                    "speed_bin": 1,
                    "direction_bucket": int(direction // 5),
                    "steady": True,
                    "operating_mask": 0b1111,
                    "waked": False,
                    "valid_turbines": 4,
                    "y": float(y[i]),
                    "temperature": 10.0,
                    "turbulence_intensity": 0.08,
                }
            )
    prepared = data / "prepared" / "kelmarsh"
    prepared.mkdir(parents=True)
    rows = pl.DataFrame(records).cast(dict(PREPARED_ROWS.items()))
    conform(rows, PREPARED_ROWS).write_parquet(prepared / "rows.parquet")
    pl.DataFrame(
        {
            "turbine": [f"T{i}" for i in range(4)],
            "latitude": [0.0] * 4,
            "longitude": [0.0] * 4,
            "x_m": [5 * diameter * i for i in range(4)],
            "y_m": [0.0] * 4,
            "hub_height_m": [80.0] * 4,
        }
    ).write_parquet(prepared / "layout.parquet")
    manifest = runs / manifest_file("development", "kelmarsh", None, load_protocol())
    manifest.parent.mkdir(parents=True)
    pl.DataFrame(
        [("arc_000_single", "arc_single", stamp, role) for stamp, role in roles],
        schema=MANIFEST,
        orient="row",
    ).write_parquet(manifest)
    chosen = runs / chosen_file("kelmarsh", "geometry_gbm")
    chosen.parent.mkdir(parents=True)
    chosen.write_text(json.dumps({"configuration": configuration}))
    inputs = [manifest, prepared / "rows.parquet", prepared / "layout.parquet"]
    for path in (*inputs, chosen):
        proven(path)
    return inputs


def test_the_hybrid_adds_turbine_identity_to_the_tuned_geometry_gbm_and_keeps_its_seed(
    tmp_path, monkeypatch
):
    data, runs = tmp_path / "data", tmp_path / "runs"
    configuration = {
        "num_boost_round": 60,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_data_in_leaf": 10,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l2": 0.0,
        "min_data_per_group": 10,
        "cat_smooth": 10.0,
    }
    inputs = write_row_farm(data, runs, configuration)
    monkeypatch.setattr(hybrid_gbm, "fit_inputs", lambda *_: inputs)
    table = hybrid_gbm.fit(data, runs, "development", "kelmarsh", "arc_000_single", 2, 1)
    record = json.loads(table.with_suffix(".json").read_text())
    assert (record["model"], record["seed"]) == ("hybrid_gbm", 2)
    assert record["configuration"] == configuration
    by_turbine = pl.read_parquet(table).group_by("turbine").agg(pl.col("mean").mean())
    np.testing.assert_allclose(by_turbine.sort("turbine")["mean"].to_numpy(), OFFSETS, atol=0.005)
