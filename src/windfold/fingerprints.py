"""fingerprints: what a job's result depends on, so a change reruns exactly the jobs it touches.

a job's code is its target function, the helpers and constants of its own module that the
function uses and every windfold module those reach through imports. configuration code and the
runner are left out: the configuration values a job uses enter its fingerprint directly.
"""

import ast
import hashlib
import json
import re
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import windfold

SOURCE = Path(windfold.__file__).parent
LEFT_OUT = ("windfold.config", "windfold.runner", "windfold.cli", "windfold.study")
PACKAGES = (
    "numpy",
    "scipy",
    "polars",
    "pyarrow",
    "py-wake",
    "lightgbm",
    "torch",
    "scikit-learn",
    "optuna",
    "pydantic",
    "autogluon.tabular",
)


def digest(*parts: Any) -> str:
    payload = [p.model_dump(mode="json") if isinstance(p, BaseModel) else p for p in parts]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def without_docstrings[T: ast.AST](tree: T) -> T:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            first = body[0] if body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_text(node: ast.AST) -> str:
    return ast.dump(without_docstrings(node), include_attributes=False)


@cache
def parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def module_path(name: str, source: Path = SOURCE) -> Path | None:
    if name != "windfold" and not name.startswith("windfold."):
        return None
    base = source.joinpath(*name.split(".")[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def imported_name(module: str, name: str, source: Path) -> str:
    full = f"{module}.{name}"
    return full if module_path(full, source) else module


def module_imports(node: ast.AST, source: Path) -> set[str]:
    """windfold modules a node imports, or names in a string, as a dynamic import would."""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            names.update(alias.name for alias in sub.names)
        elif isinstance(sub, ast.ImportFrom) and sub.module and not sub.level:
            names.update(imported_name(sub.module, alias.name, source) for alias in sub.names)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            names.add(sub.value.split(":")[0])
    return {name for name in names if module_path(name, source)}


@cache
def module_closure(name: str, source: Path = SOURCE) -> frozenset[Path]:
    seen: set[Path] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current.startswith(LEFT_OUT):
            continue
        path = module_path(current, source)
        if path is None or path in seen:
            continue
        seen.add(path)
        pending.extend(module_imports(parsed(path), source))
    return frozenset(seen)


def top_level(tree: ast.Module) -> dict[str, ast.stmt]:
    named: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            named[node.name] = node
        elif isinstance(node, ast.TypeAlias):
            named[node.name.id] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        named[sub.id] = node
    return named


def aliases(tree: ast.Module, source: Path) -> dict[str, tuple[str, str]]:
    """each local name a windfold import binds: the module it reaches and the import itself."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                full = imported_name(node.module, alias.name, source)
                found[alias.asname or alias.name] = (full, ast.dump(alias))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found[alias.asname or alias.name.split(".")[0]] = (alias.name, ast.dump(alias))
    return {local: bound for local, bound in found.items() if module_path(bound[0], source)}


@cache
def function_parameters(
    target: str, source: Path = SOURCE
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """a "module:function" target's parameter names and those of them without a default."""
    module, function = target.split(":")
    path = module_path(module, source)
    if path is None:
        raise ValueError(f"no windfold module {module}")
    node = top_level(parsed(path)).get(function)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError(f"{module} has no function {function}")
    arguments = [*node.args.posonlyargs, *node.args.args]
    defaults = len(node.args.defaults)
    required = [a.arg for a in arguments[: len(arguments) - defaults]]
    keyword = [a.arg for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)]
    required += [
        a.arg for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True) if d is None
    ]
    return tuple(a.arg for a in arguments) + tuple(keyword), tuple(required)


@cache
def function_hash(target: str, source: Path = SOURCE) -> str:
    """the hash of a "module:function" target's own code and everything it reaches."""
    module, function = target.split(":")
    path = module_path(module, source)
    if path is None:
        raise ValueError(f"no windfold module {module}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    named, local = top_level(tree), aliases(tree, source)
    if function not in named:
        raise ValueError(f"{module} has no function {function}")
    used: set[str] = set()
    modules: set[str] = set()
    bindings: set[str] = set()
    pending = [function]
    while pending:
        name = pending.pop()
        if name in used:
            continue
        used.add(name)
        node = named[name]
        modules |= module_imports(node, source)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id != name:
                if sub.id in named:
                    pending.append(sub.id)
                elif sub.id in local:
                    modules.add(local[sub.id][0])
                    bindings.add(f"{sub.id}={local[sub.id][1]}")
    h = hashlib.sha256()
    for name in sorted(used):
        h.update(code_text(named[name]).encode())
    for binding in sorted(bindings):
        h.update(binding.encode())
    files = sorted(set().union(*(module_closure(m, source) for m in modules)))
    for file in files:
        h.update(str(file.relative_to(source)).encode())
        h.update(code_text(ast.parse(file.read_text(encoding="utf-8"))).encode())
    return h.hexdigest()


def requirement_name(requirement: str) -> str | None:
    """the distribution a requirement line names, or None when only an extra needs it."""
    if "extra ==" in requirement.replace("extra==", "extra =="):
        return None
    found = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    return found.group(0) if found else None


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@cache
def environment_hash() -> str:
    """versions of every distribution the runtime reaches from windfold and the learners'
    packages, following their requirements; development tools are not reached."""
    versions: dict[str, str | None] = {}
    pending = [canonical(name) for name in ("windfold", *PACKAGES)]
    while pending:
        name = pending.pop()
        if name in versions:
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            continue
        versions[name] = None if name == "windfold" else distribution.version
        for requirement in distribution.requires or []:
            needed = requirement_name(requirement)
            if needed:
                pending.append(canonical(needed))
    return digest(versions)


def forget() -> None:
    """drop every cached hash, so the next plan sees code or packages changed since the last."""
    for cached in (parsed, module_closure, function_hash, function_parameters, environment_hash):
        cached.cache_clear()
