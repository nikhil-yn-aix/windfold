import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "tests", "docs", ".github", "README.md", "pyproject.toml")
CODE = ("src", "tests", ".github", "pyproject.toml")
SUFFIXES = {".py", ".toml", ".md", ".yml", ".yaml", ".txt"}
INTERNAL = re.compile(
    r"\b(phase|stage|wave|round)[ _-]?\d+\b|_v\d+\b|\bv\d+_|\b(todo|fixme|hack|xxx)\b",
    re.IGNORECASE,
)
HARDWARE = re.compile(r"\b((gb|mi|[a-z])[1-9]00|(dgx|rtx|gtx)\w*|sm_\d{2,3})\b", re.IGNORECASE)
PROVIDER_NAMES = ("WindCube_v1",)
RETIRED = re.compile(
    r"seal|registered|registers|registration|repair|recipe|clarification|bank", re.IGNORECASE
)
ON_DISK = {"registered", "registered_comparator", "repair"}


def files_under(names):
    for name in names:
        path = ROOT / name
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (
                p for p in path.rglob("*") if p.suffix in SUFFIXES and p.name != "test_naming.py"
            )


def test_no_internal_words_anywhere_and_no_hardware_names_in_code():
    hits = []
    for path in files_under(SCANNED):
        relative = path.relative_to(ROOT)
        patterns = (INTERNAL, HARDWARE) if relative.parts[0] in CODE else (INTERNAL,)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            quoted = line
            for name in PROVIDER_NAMES:
                quoted = quoted.replace(name, "")
            for pattern in patterns:
                if match := pattern.search(quoted):
                    hits.append(f"{relative}:{number}: {match.group(0)}")
    assert not hits, "internal words, or hardware names in code:\n" + "\n".join(hits)


def docstrings(tree: ast.Module) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def retired_words(path: Path) -> list[tuple[int, str]]:
    """the retired words of a python file's names, comments, docstrings and strings, but for
    strings that name outputs already on disk; of any other file, every line that has one."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return [(n, line) for n, line in enumerate(text.splitlines(), 1) if RETIRED.search(line)]
    tree = ast.parse(text)
    documented = docstrings(tree)
    found = [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type in (tokenize.NAME, tokenize.COMMENT) and RETIRED.search(token.string)
    ]
    found += [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and RETIRED.search(node.value)
        and (id(node) in documented or node.value not in ON_DISK)
    ]
    return found


def test_retired_words_stay_out_of_code():
    hits = []
    for path in files_under(("src", "tests", ".github")):
        relative = path.relative_to(ROOT)
        if RETIRED.search(path.name):
            hits.append(f"{relative}: in the file name")
        hits += [f"{relative}:{number}: {found[:60]!r}" for number, found in retired_words(path)]
    assert not hits, "retired words in code:\n" + "\n".join(hits)
