"""Compare the running environment with the committed lock files (spec/conventions.md §7)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from kta.core.envprobe import run_probe


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def python_mismatches(root: Path) -> list[str]:
    pinned_file = root / ".python-version"
    if not pinned_file.exists():
        return [".python-version is missing"]
    pinned = pinned_file.read_text(encoding="utf-8").strip()
    running = ".".join(str(part) for part in sys.version_info[:3])
    if running != pinned:
        return [f"python {running} != .python-version {pinned}"]
    return []


def uv_lock_mismatches(root: Path) -> list[str]:
    """Installed distributions whose version differs from uv.lock, and missing direct deps."""
    lock_file = root / "uv.lock"
    if not lock_file.exists():
        return ["uv.lock is missing"]
    lock: dict[str, Any] = tomllib.loads(lock_file.read_text(encoding="utf-8"))
    packages: list[dict[str, Any]] = lock.get("package", [])
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project_name = _normalize(str(project["project"]["name"]))
    locked: dict[str, str] = {}
    direct: set[str] = set()
    for pkg in packages:
        name = _normalize(str(pkg["name"]))
        if name == project_name:
            direct.update(_normalize(str(dep["name"])) for dep in pkg.get("dependencies", []))
            continue
        if "version" in pkg:
            locked[name] = str(pkg["version"])
    mismatches: list[str] = []
    for name, version in sorted(locked.items()):
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name in direct:
                mismatches.append(f"{name} {version} is locked but not installed")
            continue
        if installed != version:
            mismatches.append(f"{name} installed {installed} != locked {version}")
    return mismatches


def node_mismatches(root: Path) -> list[str]:
    """Only meaningful once the web/ project exists (WP9a)."""
    web = root / "web"
    if not web.exists():
        return []
    problems: list[str] = []
    nvmrc = web / ".nvmrc"
    if not nvmrc.exists():
        problems.append("web/.nvmrc is missing")
    else:
        pinned = nvmrc.read_text(encoding="utf-8").strip().lstrip("v")
        running = (run_probe(["node", "--version"]) or "absent").lstrip("v")
        if running != pinned:
            problems.append(f"node {running} != web/.nvmrc {pinned}")
    if not (web / "package-lock.json").exists():
        problems.append("web/package-lock.json is missing")
    return problems


def lock_mismatches(root: Path) -> list[str]:
    return python_mismatches(root) + uv_lock_mismatches(root) + node_mismatches(root)
