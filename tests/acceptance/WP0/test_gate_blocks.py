"""WP0 acceptance: the gate stops each listed violation (spec/workplan.md WP0).

Each test builds a minimal synthetic tree that contains exactly one violation and runs
the real gate check on it; a control run on the clean tree must pass.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from kta.core.config import config_from_mapping, repo_root
from kta.core.context import Context
from kta.core.doctor import run_doctor
from kta.core.errors import PermissionDenied
from kta.core.registry import Registry, resolve
from kta_gate import runners, scans
from kta_gate.specific import MODULES

pytestmark = pytest.mark.acceptance


def _tree(tmp_path: Path) -> Path:
    for module in MODULES:
        pkg = tmp_path / "src" / "kta" / module
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
    (tmp_path / "src" / "kta" / "__init__.py").write_text("")
    return tmp_path


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).write_text(text)


def _importlinter_config(dest: Path) -> Path:
    """Copy only the import-linter tables of the real pyproject.toml."""
    keep: list[str] = []
    inside = False
    for line in (repo_root() / "pyproject.toml").read_text().splitlines():
        if line.startswith("["):
            inside = line.lstrip("[").startswith("tool.importlinter")
        if inside:
            keep.append(line)
    config = dest / "pyproject.toml"
    config.write_text("\n".join(keep) + "\n")
    return config


def test_gate_blocks_layer_violation(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    config = _importlinter_config(root)
    assert runners.check_imports(root, config).status == "PASS"
    _write(root, "src/kta/core/bad.py", "import kta.store\n")
    assert runners.check_imports(root, config).status == "FAIL"


def test_gate_blocks_checks_importing_knowledge(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    config = _importlinter_config(root)
    _write(root, "src/kta/checks/bad.py", "import kta.knowledge\n")
    assert runners.check_imports(root, config).status == "FAIL"


def test_gate_blocks_fake_class_in_src(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert scans.scan_doubles(root).status == "PASS"
    _write(root, "src/kta/store/memory.py", "class FakeStore:\n    x = 1\n")
    result = scans.scan_doubles(root)
    assert result.status == "FAIL" and "FakeStore" in result.findings[0]


def test_gate_blocks_bare_except(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(
        root, "src/kta/core/a.py", "def f():\n    try:\n        g()\n    except:\n        raise\n"
    )
    assert scans.scan_error_handling(root).status == "FAIL"


def test_gate_blocks_swallowed_exception(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(
        root,
        "src/kta/core/a.py",
        "def f():\n    try:\n        return g()\n    except Exception:\n        return 0\n",
    )
    assert scans.scan_error_handling(root).status == "FAIL"


def test_gate_blocks_print_in_src(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _write(root, "src/kta/store/a.py", "def f():\n    print('x')\n")
    assert scans.scan_print(root).status == "FAIL"
    shutil.rmtree(root / "src" / "kta" / "store")
    (root / "src" / "kta" / "cli" / "output.py").write_text("def emit():\n    print('ok')\n")
    assert scans.scan_print(root).status == "PASS"


def test_gate_blocks_fake_api_key(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    leaked = root / "notes.md"
    # Built at runtime so this file itself never contains a key-shaped string.
    leaked.write_text("key: " + "sk-" + "ant-" + "A1b2" * 10 + "\n")
    result = runners.check_secrets(root, files=[leaked])
    assert result.status == "FAIL"
    assert "A1b2" not in "\n".join(result.findings)


def test_gate_blocks_fixture_impl_in_dev(ctx: Context) -> None:
    reg = Registry()
    reg.register("clock", "manual", lambda c: c.clock, fixture=True)
    cfg = config_from_mapping({"impl": {"clock": "manual"}})
    dev = dataclasses.replace(ctx, env="dev", registry=reg, config=cfg)
    with pytest.raises(PermissionDenied):
        resolve("clock", dev)
    report = run_doctor(dev)
    assert not report.ok and "fixture" in report.problems[0]


def test_clean_repository_passes_static_scans() -> None:
    root = repo_root()
    for check in (
        scans.scan_residue,
        scans.scan_doubles,
        scans.scan_error_handling,
        scans.scan_print,
        scans.scan_subprocess,
    ):
        assert check(root).status == "PASS", check.__name__
