"""Static scans of src/ (spec/gate.md): residue, test doubles, error handling, print,
subprocess use, and the ``# kta:fixture`` markers that must be declared.

Every scan takes the repository root so tests can point it at a synthetic tree.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from kta_gate.result import CheckResult, from_findings

ALLOWED_MARK = re.compile(r"#\s*kta:allowed\s+\S")
FIXTURE_MARK = "# kta:fixture"
RESIDUE = re.compile(r"NotImplementedError|TODO|FIXME|XXX|MagicMock|pytest\.skip|(?i:\bmock)")
DOUBLE_NAME = re.compile(r"(?i)fake|stub|dummy|mock")
PRINT_ALLOWED = {"src/kta/cli/output.py"}
SUBPROCESS_ALLOWED_PREFIXES = ("src/kta/execution/", "src/kta/core/envprobe.py")
SUBPROCESS_MODULES = {"subprocess", "pty", "pexpect"}
SUBPROCESS_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"),
    ("os", "execvp"),
    ("os", "spawnv"),
    ("asyncio", "create_subprocess_exec"),
    ("asyncio", "create_subprocess_shell"),
}


def py_files(root: Path) -> Iterator[tuple[str, Path]]:
    src = root / "src"
    for path in sorted(src.rglob("*.py")):
        yield path.relative_to(root).as_posix(), path


def _allowed(lines: list[str], lineno: int) -> bool:
    return 0 < lineno <= len(lines) and bool(ALLOWED_MARK.search(lines[lineno - 1]))


def _parse(path: Path) -> tuple[ast.Module | None, list[str], str | None]:
    text = path.read_text(encoding="utf-8")
    try:
        return ast.parse(text, filename=str(path)), text.splitlines(), None
    except SyntaxError as exc:
        return None, text.splitlines(), f"syntax error: {exc}"


def scan_residue(root: Path) -> CheckResult:
    findings: list[str] = []
    for rel, path in py_files(root):
        lines = path.read_text(encoding="utf-8").splitlines()
        for no, line in enumerate(lines, 1):
            if RESIDUE.search(line) and not _allowed(lines, no):
                findings.append(f"{rel}:{no}: {line.strip()[:100]}")
    return from_findings("residue", findings)


def _is_broad(handler: ast.ExceptHandler) -> bool:
    names: list[ast.expr] = []
    if isinstance(handler.type, ast.Tuple):
        names = list(handler.type.elts)
    elif handler.type is not None:
        names = [handler.type]
    return any(isinstance(n, ast.Name) and n.id in {"Exception", "BaseException"} for n in names)


def _swallows(body: list[ast.stmt]) -> bool:
    """True when a handler body only passes or returns a constant."""
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Return):
        return stmt.value is None or isinstance(stmt.value, ast.Constant)
    return False


def _only_pass(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(func.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def scan_error_handling(root: Path) -> CheckResult:
    findings: list[str] = []
    for rel, path in py_files(root):
        tree, lines, err = _parse(path)
        if tree is None:
            findings.append(f"{rel}: {err}")
            continue
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            if _allowed(lines, line):
                continue
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    findings.append(f"{rel}:{line}: bare except")
                elif _is_broad(node) and _swallows(node.body):
                    findings.append(f"{rel}:{line}: broad except that only passes/returns")
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _only_pass(node):
                findings.append(f"{rel}:{line}: function {node.name} body is only pass")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute | ast.Name)
                and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
                == "suppress"
                and any(
                    isinstance(a, ast.Name) and a.id in {"Exception", "BaseException"}
                    for a in node.args
                )
            ):
                findings.append(f"{rel}:{line}: suppress(Exception)")
    return from_findings("error handling", findings)


def scan_print(root: Path) -> CheckResult:
    findings: list[str] = []
    for rel, path in py_files(root):
        if rel in PRINT_ALLOWED:
            continue
        tree, lines, err = _parse(path)
        if tree is None:
            findings.append(f"{rel}: {err}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _allowed(lines, node.lineno):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                findings.append(f"{rel}:{node.lineno}: print()")
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "write"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr in {"stdout", "stderr"}
            ):
                findings.append(f"{rel}:{node.lineno}: direct {func.value.attr}.write()")
    return from_findings("print", findings)


def _imported_modules(tree: ast.Module) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module


def scan_doubles(root: Path) -> CheckResult:
    """Test doubles must live in tests/fakes, and src must not import tests."""
    findings: list[str] = []
    for rel, path in py_files(root):
        if DOUBLE_NAME.search(path.stem):
            findings.append(f"{rel}: file name looks like a test double")
        tree, lines, err = _parse(path)
        if tree is None:
            findings.append(f"{rel}: {err}")
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                and DOUBLE_NAME.search(node.name)
                and not _allowed(lines, node.lineno)
            ):
                findings.append(f"{rel}:{node.lineno}: definition {node.name}")
        for line, module in _imported_modules(tree):
            top = module.split(".")[0]
            if top in {"tests", "fakes", "conftest"}:
                findings.append(f"{rel}:{line}: imports {module}")
    return from_findings("test doubles", findings)


def scan_subprocess(root: Path) -> CheckResult:
    """Only the execution module (and the read-only env probe) may start processes."""
    findings: list[str] = []
    for rel, path in py_files(root):
        if rel.startswith(SUBPROCESS_ALLOWED_PREFIXES):
            continue
        tree, lines, err = _parse(path)
        if tree is None:
            findings.append(f"{rel}: {err}")
            continue
        for line, module in _imported_modules(tree):
            if module.split(".")[0] in SUBPROCESS_MODULES and not _allowed(lines, line):
                findings.append(f"{rel}:{line}: imports {module}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and (node.func.value.id, node.func.attr) in SUBPROCESS_CALLS
                and not _allowed(lines, node.lineno)
            ):
                findings.append(f"{rel}:{node.lineno}: {node.func.value.id}.{node.func.attr}()")
    return from_findings("subprocess outside Executor", findings)


def fixture_markers(root: Path) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for path in sorted((root / "src").rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".toml", ".md"}:
            continue
        rel = path.relative_to(root).as_posix()
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FIXTURE_MARK in line:
                hits.append((rel, no))
    return hits
