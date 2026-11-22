from importlib import metadata
from pathlib import Path

import pytest

from windfold.fingerprints import environment_hash, forget, function_hash

JOBS = '''
from windfold import reached
from windfold.config import load

LIMIT = 3


def helper():
    return LIMIT


def first(out_dir):
    """does the first thing."""
    return reached.value() + helper() + load()


def second(out_dir):
    return 2
'''


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def package(tmp_path: Path) -> Path:
    root = tmp_path / "windfold"
    write(root, "__init__.py", "")
    write(root, "tasks/__init__.py", "")
    write(root, "tasks/jobs.py", JOBS)
    reached = "from windfold import deeper\n\n\ndef value():\n    return deeper.x\n"
    write(root, "reached.py", reached)
    write(root, "deeper.py", "x = 1\n")
    write(root, "unrelated.py", "y = 1\n")
    write(root, "config.py", "def load():\n    return 1\n")
    return root


def edited(root: Path, name: str, old: str, new: str) -> str:
    path = root / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    forget()
    return function_hash("windfold.tasks.jobs:first", root)


def test_a_job_hash_changes_with_the_code_it_reaches_and_nothing_else(tmp_path: Path):
    root = package(tmp_path)
    forget()
    base = function_hash("windfold.tasks.jobs:first", root)
    assert edited(root, "tasks/jobs.py", "does the first thing", "does it") == base
    assert edited(root, "tasks/jobs.py", "return 2", "return 3") == base
    assert edited(root, "unrelated.py", "y = 1", "y = 2") == base
    assert edited(root, "config.py", "return 1", "return 5") == base
    assert edited(root, "deeper.py", "x = 1", "x = 2") != base
    forget()
    reached = function_hash("windfold.tasks.jobs:first", root)
    assert edited(root, "tasks/jobs.py", "LIMIT = 3", "LIMIT = 4") != reached


def test_formatting_and_comments_do_not_count(tmp_path: Path):
    root = package(tmp_path)
    forget()
    base = function_hash("windfold.tasks.jobs:first", root)
    changed = edited(root, "reached.py", "    return deeper.x\n", "    return (deeper.x)  # same\n")
    assert changed == base


def test_unpacked_constants_aliases_and_named_modules_count(tmp_path: Path):
    root = package(tmp_path)
    write(root, "helpers.py", "def first():\n    return 1\n\n\ndef second():\n    return 2\n")
    write(root, "late.py", "z = 1\n")
    write(
        root,
        "tasks/more.py",
        "from windfold.helpers import first as helper\n\n"
        "LOW, HIGH = (1, 2)\n"
        "TARGET = 'windfold.late'\n\n\n"
        "def job(out_dir):\n    return helper() + LOW + len(TARGET)\n",
    )
    forget()
    base = function_hash("windfold.tasks.more:job", root)

    def changed(name: str, old: str, new: str) -> str:
        path = root / name
        path.write_text(path.read_text().replace(old, new))
        forget()
        return function_hash("windfold.tasks.more:job", root)

    assert changed("tasks/more.py", "LOW, HIGH = (1, 2)", "LOW, HIGH = (3, 2)") != base
    forget()
    before = function_hash("windfold.tasks.more:job", root)
    assert changed("tasks/more.py", "import first as helper", "import second as helper") != before
    forget()
    before = function_hash("windfold.tasks.more:job", root)
    assert changed("late.py", "z = 1", "z = 2") != before


class Installed:
    def __init__(self, version: str, requires: list[str]):
        self.version = version
        self.requires = requires


REQUIRES = {
    "windfold": ["scipy>=1.16", "sphinx>=7; extra == 'docs'"],
    "scipy": ["numpy>=1.25"],
    "py-wake": ["xarray"],
    "xarray": ["pandas>=2"],
    "pytest": ["pluggy"],
}


@pytest.fixture
def environment(monkeypatch):
    """the environment hash of fake installed distributions, by version; the real hash is
    forgotten afterward."""

    def hashed(versions: dict[str, str]) -> str:
        def distribution(name: str) -> Installed:
            if name not in versions:
                raise metadata.PackageNotFoundError(name)
            return Installed(versions[name], REQUIRES.get(name, []))

        monkeypatch.setattr(metadata, "distribution", distribution)
        forget()
        return environment_hash()

    yield hashed
    monkeypatch.undo()
    forget()


def test_the_environment_hash_follows_runtime_requirements_and_not_tools_or_extras(environment):
    versions = {
        "windfold": "0.1.0",
        "scipy": "1.16.0",
        "numpy": "2.3.0",
        "py-wake": "2.6.20",
        "xarray": "2025.1",
        "pandas": "2.2.0",
        "pytest": "8.4.0",
        "pluggy": "1.5.0",
        "sphinx": "7.0.0",
    }
    base = environment(versions)
    for name in ("numpy", "pandas", "py-wake"):
        assert environment(versions | {name: "9.9"}) != base, name
    for name in ("windfold", "pytest", "pluggy", "sphinx"):
        assert environment(versions | {name: "9.9"}) == base, name
