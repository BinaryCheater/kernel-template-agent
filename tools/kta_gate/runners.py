"""Checks that run external tools: ruff, pyright, import-linter, pytest, doctor, git."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from kta.core.secrets import find_secrets
from kta_gate.result import CheckResult, from_findings

TAIL_LINES = 30
MAX_SCAN_BYTES = 2_000_000


@dataclass(frozen=True)
class Proc:
    code: int
    out: str


def run(argv: list[str], root: Path, env: dict[str, str] | None = None) -> Proc:
    merged = {**os.environ, **(env or {})}
    proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, env=merged, check=False)
    return Proc(proc.returncode, (proc.stdout + proc.stderr).strip())


def _tool(name: str) -> str:
    """Tools come from the project venv that runs the gate."""
    return str(Path(sys.executable).parent / name)


def _tail(text: str) -> list[str]:
    return text.splitlines()[-TAIL_LINES:]


def tool_check(
    name: str, argv: list[str], root: Path, env: dict[str, str] | None = None
) -> CheckResult:
    proc = run(argv, root, env)
    if proc.code == 0:
        return CheckResult(name, "PASS")
    return CheckResult(name, "FAIL", f"exit {proc.code}", _tail(proc.out))


def check_ruff(root: Path) -> list[CheckResult]:
    return [
        tool_check("ruff format", [_tool("ruff"), "format", "--check", "."], root),
        tool_check("ruff check", [_tool("ruff"), "check", "."], root),
    ]


def check_pyright(root: Path) -> CheckResult:
    # Use the Node runtime from the nodejs-wheel in the venv, and keep caches in the project.
    env = {
        "PYRIGHT_PYTHON_GLOBAL_NODE": "0",
        "PYRIGHT_PYTHON_CACHE_DIR": str(root / ".venv" / ".pyright-cache"),
    }
    return tool_check("pyright", [_tool("pyright")], root, env)


def check_imports(root: Path, config: Path | None = None) -> CheckResult:
    argv = [_tool("lint-imports"), "--no-cache"]
    if config is not None:
        argv += ["--config", str(config)]
    env = {"PYTHONPATH": str(root / "src")}
    return tool_check("import layers", argv, root, env)


@dataclass(frozen=True)
class PytestRun:
    result: CheckResult
    skipped: list[str]
    passed: set[str]


def run_pytest(root: Path, name: str, args: list[str], allow_empty: bool = False) -> PytestRun:
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        proc = run([_tool("pytest"), "-q", f"--junitxml={junit}", *args], root)
        skipped: list[str] = []
        passed: set[str] = set()
        if junit.exists():
            for case in ET.parse(junit).getroot().iter("testcase"):
                test_id = f"{case.get('classname')}::{case.get('name')}"
                if case.find("skipped") is not None:
                    skipped.append(test_id)
                elif case.find("failure") is None and case.find("error") is None:
                    passed.add(test_id)
    if proc.code == 0 or (allow_empty and proc.code == 5):
        detail = "no tests collected" if proc.code == 5 else f"{len(passed)} passed"
        return PytestRun(CheckResult(name, "PASS", detail), skipped, passed)
    return PytestRun(
        CheckResult(name, "FAIL", f"exit {proc.code}", _tail(proc.out)), skipped, passed
    )


def fake_classes(root: Path) -> list[str]:
    names: list[str] = []
    for path in sorted((root / "tests" / "fakes").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names += [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    return names


def check_contract(root: Path) -> list[CheckResult]:
    fakes = fake_classes(root)
    contract_text = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted((root / "tests" / "contract").rglob("*.py"))
    )
    uncovered = [
        f"tests/fakes class {n} has no contract test" for n in fakes if n not in contract_text
    ]
    run_result = run_pytest(root, "contract tests", ["-m", "contract"], allow_empty=not fakes)
    return [
        run_result.result,
        from_findings("fake coverage", uncovered, f"{len(fakes)} fake(s) covered"),
    ]


def check_doctor(root: Path, wp: str, data_root: Path) -> CheckResult:
    env = {"KTA_ENV": "dev", "KTA_WP": wp, "KTA_DATA": str(data_root)}
    proc = run([_tool("kta"), "doctor"], root, env)
    if proc.code == 0:
        return CheckResult("registry (kta doctor, dev)", "PASS")
    return CheckResult("registry (kta doctor, dev)", "FAIL", f"exit {proc.code}", _tail(proc.out))


def repo_files(root: Path) -> list[Path]:
    proc = run(["git", "ls-files", "-co", "--exclude-standard", "-z"], root)
    if proc.code != 0:
        return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    return [root / rel for rel in proc.out.split("\0") if rel]


def check_secrets(root: Path, files: list[Path] | None = None) -> CheckResult:
    findings: list[str] = []
    for path in files if files is not None else repo_files(root):
        if not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for no, line in enumerate(text.splitlines(), 1):
            # Report only the pattern name: the gate output must not repeat the secret.
            findings += [f"{path.relative_to(root)}:{no}: {name}" for name, _ in find_secrets(line)]
    return from_findings("secrets", findings)
