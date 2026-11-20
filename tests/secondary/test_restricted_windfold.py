import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from windfold.config import load_models, load_protocol
from windfold.locations import manifest_file
from windfold.schemas import MANIFEST
from windfold.secondary import restricted_windfold
from windfold.secondary.restricted_windfold import KERNEL_PRIOR_VARIANCE, fit, restricted


def test_only_the_declared_kernel_scale_changes():
    protocol = load_protocol()
    changed = restricted(protocol)
    assert changed.engine.fixed_prior_variance == KERNEL_PRIOR_VARIANCE == 0.04 / 64
    assert protocol.engine.fixed_prior_variance == 0.04
    before = protocol.model_dump()
    after = changed.model_dump()
    before["engine"].pop("fixed_prior_variance")
    after["engine"].pop("fixed_prior_variance")
    assert before == after


def test_a_fit_from_before_provenance_is_refused_not_reused(tiny_study, tmp_path):
    runs = shutil.copytree(tiny_study.runs, tmp_path / "runs")
    args = (tiny_study.data, runs, "development", "kelmarsh", "arc_270_single")
    table = fit(*args, str(tiny_study.configs))
    written = table.stat().st_mtime_ns
    assert fit(*args, str(tiny_study.configs)) == table and table.stat().st_mtime_ns == written
    note = table.with_suffix(".json")
    record = json.loads(note.read_text())
    record.pop("provenance", None)
    note.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=f"{note.name} has no provenance; delete .* to fit it"):
        fit(*args, str(tiny_study.configs))


def test_a_restricted_fit_is_windfold_without_its_local_block_under_the_tighter_prior(
    tmp_path, monkeypatch
):
    """the engine fit itself is replaced: this checks the model and protocol it is given."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    stamps = [datetime(2018, 3, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(4)]
    rows = data / "prepared" / "kelmarsh" / "rows.parquet"
    rows.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [s for s in stamps for _ in range(2)],
            "turbine": ["T0", "T1"] * 4,
            "turbine_index": pl.Series([0, 1] * 4, dtype=pl.UInt8),
            "y": [0.1, -0.1] * 4,
            "waked": [False, True] * 4,
        }
    ).write_parquet(rows)
    manifest = runs / manifest_file("development", "kelmarsh", None, load_protocol())
    manifest.parent.mkdir(parents=True)
    roles = ["train", "train", "test", "test"]
    pl.DataFrame(
        [("arc_000_single", "arc_single", s, r) for s, r in zip(stamps, roles, strict=True)],
        schema=MANIFEST,
        orient="row",
    ).write_parquet(manifest)
    for path in (manifest, rows):
        content = path.read_bytes()
        proof = {"fingerprint": "f", "sha256": hashlib.sha256(content).hexdigest()}
        path.with_name(path.name + ".sidecar.json").write_text(
            json.dumps(proof | {"bytes": len(content)})
        )
    monkeypatch.setattr(restricted_windfold, "fit_inputs", lambda *_: [manifest, rows])
    given = {}

    def fit_one(model, train, test, context, protocol):
        given.update(model=model, protocol=protocol, train=train.height, test=test.height)
        predictions = test.select(
            "timestamp",
            "turbine",
            "y",
            "waked",
            mean=pl.lit(0.0),
            variance=pl.lit(None, pl.Float64),
            support=pl.lit(None, pl.Int64),
        )
        return predictions, {"model": model.name}, None

    monkeypatch.setattr(restricted_windfold, "fit_one", fit_one)
    table = restricted_windfold.fit(data, runs, "development", "kelmarsh", "arc_000_single")
    model, protocol = given["model"], given["protocol"]
    blocks = tuple(b for b in load_models()["windfold"].blocks if b != "local")
    assert model.spec.name == "windfold_no_local" and model.spec.blocks == blocks
    assert model.settings.fixed_prior_variance == KERNEL_PRIOR_VARIANCE
    assert protocol.engine.fixed_prior_variance == KERNEL_PRIOR_VARIANCE
    assert (given["train"], given["test"]) == (4, 4)
    record = json.loads(table.with_suffix(".json").read_text())
    assert record["base"] == "windfold_no_local"
    assert record["kernel_prior_variance"] == KERNEL_PRIOR_VARIANCE
