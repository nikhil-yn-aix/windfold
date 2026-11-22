"""the refit folders of the heavy analyses: a data folder whose prepared farm links the study's
prepared files but for the tables the analysis rebuilds, each written whole or not at all."""

import os
from collections.abc import Callable, Collection
from pathlib import Path


def link_prepared(source: Path, target: Path, own: Collection[str]) -> None:
    """target holds a link to every entry of source but the files named in own and their
    sidecars."""
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        link = target / entry.name
        if entry.name.removesuffix(".sidecar.json") not in own and not link.is_symlink():
            link.symlink_to(entry.resolve())


def write_atomic(path: Path, write: Callable[[Path], object]) -> None:
    """write(a partial path), then move the file into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    write(partial)
    os.replace(partial, path)
