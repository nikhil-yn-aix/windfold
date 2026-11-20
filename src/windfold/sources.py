"""raw files: each checked against its md5, and downloaded only when missing."""

import hashlib
import logging
from pathlib import Path

import pooch

from windfold.config import FarmConfig, RawFile

log = logging.getLogger(__name__)


class SourceError(RuntimeError):
    pass


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def url_of(raw: RawFile, doi: str) -> str:
    """where a raw file is published: its pinned url, otherwise its name in the doi record."""
    return raw.url or f"doi:{doi}/{Path(raw.name).name}"


def problem(raw: RawFile, raw_dir: Path) -> str | None:
    """why the file on disk is not the raw file the config lists, or None when it is."""
    path = raw_dir / raw.name
    if not path.is_file():
        return f"{raw.name} is missing"
    if raw.md5 is None:
        return f"{raw.name} has no md5 in the farm config"
    size = path.stat().st_size
    if raw.bytes is not None and size != raw.bytes:
        return f"{raw.name} has {size} bytes, expected {raw.bytes}"
    digest = md5sum(path)
    if digest != raw.md5:
        return f"{raw.name} has md5 {digest}, expected {raw.md5}"
    return None


def verify(farm: FarmConfig, raw_dir: Path) -> None:
    """raise SourceError naming every raw file of farm that is missing or differs from its entry."""
    if not any((raw_dir / raw.name).exists() for raw in farm.raw_files):
        raise SourceError(f"{farm.name} has no raw files yet. run windfold fetch {farm.name}")
    problems = [found for raw in farm.raw_files if (found := problem(raw, raw_dir))]
    if problems:
        raise SourceError(
            f"{farm.name} raw files in {raw_dir} do not match the farm config:\n  "
            + "\n  ".join(problems)
            + f"\nrun windfold fetch {farm.name} to download missing files;"
            " delete a file that differs before fetching it again"
        )


def fetch(farm: FarmConfig, raw_dir: Path) -> None:
    """download every raw file of farm that is missing, then verify them all.

    a file already on disk is never downloaded again, even when it differs.
    """
    for raw in farm.raw_files:
        path = raw_dir / raw.name
        if path.exists():
            continue
        if raw.md5 is None:
            raise SourceError(f"{raw.name} has no md5 in the farm config, so it cannot be checked")
        url = url_of(raw, farm.dataset.doi)
        log.info("downloading %s from %s", raw.name, url)
        try:
            pooch.retrieve(
                url,
                known_hash=f"md5:{raw.md5}",
                fname=path.name,
                path=path.parent,
                progressbar=False,
            )
        except ValueError as error:
            raise SourceError(f"{raw.name} from {url} does not match its md5: {error}") from error
    verify(farm, raw_dir)
