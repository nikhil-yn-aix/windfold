import hashlib
import json
import shutil
from pathlib import Path

import pytest

from windfold.config import load_protocol
from windfold.provenance import fit_inputs, provenance
from windfold.secondary import hybrid_gbm


def written(path: Path, content: bytes) -> None:
    """content at path, with the sidecar the runner leaves beside an output."""
    path.write_bytes(content)
    proof = {"fingerprint": "f", "sha256": hashlib.sha256(content).hexdigest()}
    path.with_name(path.name + ".sidecar.json").write_text(
        json.dumps(proof | {"bytes": len(content)})
    )


def test_a_provenance_follows_its_settings_and_what_each_input_holds(tmp_path):
    inputs = [tmp_path / "rows.parquet", tmp_path / "manifest.parquet"]
    for path in inputs:
        written(path, path.name.encode())
    made = provenance(hybrid_gbm.fit, inputs, "arc_000_single", 0)
    assert provenance(hybrid_gbm.fit, inputs, "arc_000_single", 0) == made
    assert provenance(hybrid_gbm.fit, inputs, "arc_000_single", 1) != made
    written(inputs[0], b"rows prepared again")
    assert provenance(hybrid_gbm.fit, inputs, "arc_000_single", 0) != made
    (tmp_path / "manifest.parquet.sidecar.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.parquet has no runner sidecar"):
        provenance(hybrid_gbm.fit, inputs, "arc_000_single", 0)


@pytest.mark.parametrize("hidden", ["rows.parquet", "kernel_design/permuted/indices.npy"])
def test_a_prepared_file_without_its_sidecar_is_refused_not_left_out(tiny_study, tmp_path, hidden):
    data = shutil.copytree(tiny_study.data, tmp_path / "data")
    protocol = load_protocol(tiny_study.configs)

    def made() -> str:
        inputs = fit_inputs(data, tiny_study.runs, "development", "kelmarsh", protocol)
        return provenance(hybrid_gbm.fit, inputs)

    made()
    (data / "prepared" / "kelmarsh" / f"{hidden}.sidecar.json").unlink()
    with pytest.raises(FileNotFoundError, match=f"{Path(hidden).name} has no runner sidecar"):
        made()
