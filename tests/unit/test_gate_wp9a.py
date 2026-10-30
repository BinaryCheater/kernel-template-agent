"""The WP9a gate's web build sees only what a clean checkout has (review of WP9a, D320)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kta_gate.specific import web_build

# The build fails unless the API client a page imports is present.
BUILD = "node -e \"require('fs').accessSync('src/data/source.ts')\""


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _repo(tmp: Path, ignore: str) -> Path:
    web = tmp / "web"
    (web / "src" / "data").mkdir(parents=True)
    (web / "node_modules").mkdir()
    (web / "package.json").write_text(json.dumps({"scripts": {"build": BUILD}}))
    (web / "src" / "data" / "source.ts").write_text("export const api = {};\n")
    (tmp / ".gitignore").write_text(ignore)
    _git(tmp, "init", "-q")
    _git(tmp, "add", "-A")
    _git(tmp, "commit", "-q", "-m", "init")
    return tmp


def test_web_build_fails_when_a_source_dir_is_git_ignored(tmp_path: Path) -> None:
    root = _repo(tmp_path, "node_modules/\ndata/\n")
    result = web_build(root)
    assert result.status == "FAIL", result.render()


def test_web_build_passes_when_sources_are_committed(tmp_path: Path) -> None:
    root = _repo(tmp_path, "node_modules/\ndata/\n!web/src/data/\n")
    result = web_build(root)
    assert result.status == "PASS", result.render()


def test_web_build_fails_on_uncommitted_web_files(tmp_path: Path) -> None:
    root = _repo(tmp_path, "node_modules/\n")
    (root / "web" / "src" / "extra.ts").write_text("export {};\n")
    result = web_build(root)
    assert result.status == "FAIL" and "commit it first" in result.render()
