import functools
import hashlib
import http.server
import threading
from pathlib import Path

import pooch
import pytest

from windfold import sources
from windfold.config import FarmConfig, RawFile, load_farm
from windfold.sources import SourceError

ZIP_BYTES = b"scada archive bytes"
YAML_BYTES = b"turbine: 1.5\n"


def raw_file(name: str, content: bytes, url: str | None = None) -> RawFile:
    return RawFile(name=name, bytes=len(content), md5=hashlib.md5(content).hexdigest(), url=url)


def farm_with(*files: RawFile) -> FarmConfig:
    farm = load_farm("kelmarsh")
    dataset = farm.dataset.model_copy(update={"doi": "10.5281/zenodo.1"})
    return farm.model_copy(update={"raw_files": files, "dataset": dataset})


@pytest.fixture
def no_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a download was attempted")

    monkeypatch.setattr(pooch, "retrieve", refuse)


@pytest.fixture
def served(tmp_path: Path):
    root = tmp_path / "served"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_a_good_file_is_not_downloaded_again(tmp_path: Path, no_download: None):
    (tmp_path / "a.zip").write_bytes(ZIP_BYTES)
    sources.fetch(farm_with(raw_file("a.zip", ZIP_BYTES)), tmp_path)
    assert (tmp_path / "a.zip").read_bytes() == ZIP_BYTES


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"scada archive bytez", r"a\.zip has md5 [0-9a-f]{32}, expected"),
        (b"short", r"a\.zip has 5 bytes, expected 19"),
    ],
    ids=["same size, other content", "truncated"],
)
def test_a_wrong_file_is_caught_and_left_alone(
    tmp_path: Path, no_download: None, content: bytes, reason: str
):
    (tmp_path / "a.zip").write_bytes(content)
    with pytest.raises(SourceError, match=reason) as caught:
        sources.fetch(farm_with(raw_file("a.zip", ZIP_BYTES)), tmp_path)
    assert "windfold fetch kelmarsh" in str(caught.value)
    assert (tmp_path / "a.zip").read_bytes() == content


def test_missing_files_resolve_by_doi_or_pinned_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pinned = "https://raw.githubusercontent.com/org/repo/0123abc/northing.yaml"
    published = {"doi:10.5281/zenodo.1/a.zip": ZIP_BYTES, pinned: YAML_BYTES}
    calls = []

    def retrieve(url, known_hash, fname, path, progressbar):
        calls.append((url, known_hash, fname, Path(path)))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / fname).write_bytes(published[url])
        return str(Path(path) / fname)

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    files = (raw_file("a.zip", ZIP_BYTES), raw_file("companion/n.yaml", YAML_BYTES, url=pinned))
    sources.fetch(farm_with(*files), tmp_path)
    assert calls == [
        ("doi:10.5281/zenodo.1/a.zip", f"md5:{files[0].md5}", "a.zip", tmp_path),
        (pinned, f"md5:{files[1].md5}", "n.yaml", tmp_path / "companion"),
    ]
    assert (tmp_path / "companion" / "n.yaml").read_bytes() == YAML_BYTES


def test_a_pinned_url_downloads_through_pooch(tmp_path: Path, served):
    root, base = served
    (root / "n.yaml").write_bytes(YAML_BYTES)
    raw_dir = tmp_path / "raw"
    sources.fetch(farm_with(raw_file("companion/n.yaml", YAML_BYTES, f"{base}/n.yaml")), raw_dir)
    assert (raw_dir / "companion" / "n.yaml").read_bytes() == YAML_BYTES


def test_a_download_that_differs_from_its_md5_leaves_nothing(tmp_path: Path, served):
    root, base = served
    (root / "n.yaml").write_bytes(b"turbine: 9.9\n")
    raw_dir = tmp_path / "raw"
    farm = farm_with(raw_file("n.yaml", YAML_BYTES, f"{base}/n.yaml"))
    with pytest.raises(SourceError, match=r"n\.yaml"):
        sources.fetch(farm, raw_dir)
    assert not (raw_dir / "n.yaml").exists()


def test_verify_names_missing_files_and_the_fix(tmp_path: Path):
    farm = farm_with(raw_file("a.zip", ZIP_BYTES), raw_file("b.zip", ZIP_BYTES))
    with pytest.raises(SourceError, match="kelmarsh has no raw files yet. run windfold fetch"):
        sources.verify(farm, tmp_path)
    (tmp_path / "a.zip").write_bytes(ZIP_BYTES)
    with pytest.raises(SourceError, match=r"b\.zip is missing") as caught:
        sources.verify(farm, tmp_path)
    assert "a.zip" not in str(caught.value)
