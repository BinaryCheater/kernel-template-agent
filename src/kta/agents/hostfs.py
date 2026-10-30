"""File-system work of the launcher on the agent host: install, manifest, watch, pack.

No processes are started here (those are in ``kta.agents.launch``); everything is
plain file handling so it can be unit-tested directly.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from kta.agents.launchspec import (
    BUNDLE_BATCH,
    BUNDLE_WORKSPACE,
    REPLACED_DIRS,
    WORKSPACE_ENTRIES,
)
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.secrets import find_secrets

MANIFEST_SCHEMA = "kta-context-manifest/1"
CREDENTIAL = "credential: provided"  # what the manifest says instead of a credential's hash
UPWARD_NAMES = ("AGENTS.md", "AGENTS.override.md")
# Files a CLI may read from outside its home; listed in the manifest when present.
SYSTEM_CONFIG = (
    "/etc/codex/config.toml",
    "/etc/codex/managed_config.toml",
    "/etc/codex/requirements.toml",
)
SECRET_SCAN_MAX_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(name: str) -> PurePosixPath:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ConfigError(f"bundle path {name!r} must be relative")
    return rel


def upward_agents_md(workspace: Path) -> list[str]:
    """AGENTS.md files on the path from / to the workspace (exclusive): must be none."""
    found: list[str] = []
    for parent in workspace.resolve().parents:
        found += [str(parent / n) for n in UPWARD_NAMES if (parent / n).exists()]
    return found


def _files(root: Path, skip_top: Iterable[str] = ()) -> dict[str, Path]:
    skip = set(skip_top)
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel.split("/")[0] in skip or path.is_dir():
            continue
        out[rel] = path
    return out


def install_workspace(job_dir: Path, workspace: Path, names: list[str]) -> list[str]:
    """Copy bundle files into the workspace; returns the names written.

    ``NOTES.md`` is only seeded (the agent owns it); directories in REPLACED_DIRS are
    replaced wholesale so the agent cannot keep edits to skills between steps.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "out").mkdir(exist_ok=True)
    (workspace / "inputs").mkdir(exist_ok=True)
    for d in REPLACED_DIRS:
        if (workspace / d).exists():
            shutil.rmtree(workspace / d)
    written: list[str] = []
    for name in names:
        rel = safe_rel(name)
        if rel.parts[0] not in WORKSPACE_ENTRIES or rel.parts[0] in ("out", ".git"):
            raise ConfigError(f"bundle file {name} is outside the allowed workspace entries")
        dest = workspace / rel
        if rel.as_posix() == "NOTES.md" and dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(job_dir / (BUNDLE_WORKSPACE + name), dest)
        written.append(name)
    return written


def install_home(template: Path, home: Path) -> bool:
    """Create the CLI home from the batch's agent-home-template only on first use."""
    if home.exists():
        return False
    shutil.copytree(template, home, symlinks=False)
    return True


def ensure_batch(job_dir: Path, root: Path, names: list[str], lock_name: str) -> list[str]:
    """The agent host's copy of a batch root: created from the job inputs on first use,
    afterwards compared with them; returns the problems (empty = identical to the lock)."""
    lock = json.loads((job_dir / (BUNDLE_BATCH + lock_name)).read_text(encoding="utf-8"))
    expected: dict[str, str] = dict(lock["files"])
    if set(names) != {*expected, lock_name}:
        return ["job inputs do not match the batch lock"]
    if not (root / lock_name).exists():
        tmp = root.with_name(f".{root.name}.installing")
        if tmp.exists():
            shutil.rmtree(tmp)  # an interrupted install of the same batch
        for name in names:
            dest = tmp / safe_rel(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(job_dir / (BUNDLE_BATCH + name), dest)
        root.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(root)
    problems: list[str] = []
    if (root / lock_name).read_bytes() != (job_dir / (BUNDLE_BATCH + lock_name)).read_bytes():
        problems.append("agent host batch lock differs from the control plane's")
    for rel, sha in sorted(expected.items()):
        path = root / rel
        if path.is_symlink() or not path.is_file():
            problems.append(f"missing {rel}")
        elif sha256_file(path) != sha:
            problems.append(f"changed: {rel}")
    return problems


def link_credentials(home: Path, auth_dir: Path, files: list[str]) -> list[str]:
    """Symlink credential files into the home; raises PermissionDenied when absent."""
    missing = [f for f in files if not (auth_dir / f).is_file()]
    if missing:
        raise PermissionDenied(f"credentials not provided in {auth_dir}: {missing}")
    for f in files:
        link = home / f
        if link.is_symlink() or link.exists():
            if link.is_symlink() and Path(os.readlink(link)) == auth_dir / f:
                continue
            link.unlink()
        link.symlink_to(auth_dir / f)
    return list(files)


def restore_credentials(home: Path, auth_dir: Path, files: list[str]) -> list[str]:
    """A CLI that refreshed a token by rename leaves a real file: move it back, relink."""
    moved: list[str] = []
    for f in files:
        link = home / f
        if link.exists() and not link.is_symlink():
            tmp = auth_dir / f".{f}.kta-new"
            shutil.copyfile(link, tmp)
            os.chmod(tmp, 0o600)
            os.replace(tmp, auth_dir / f)
            link.unlink()
            link.symlink_to(auth_dir / f)
            moved.append(f)
    return moved


def manifest(
    *,
    spec: dict[str, Any],
    workspace: Path,
    home: Path,
    home_created: bool,
    credentials: list[str],
    cli: dict[str, Any],
    env: dict[str, str],
    secret_env: list[str],
    host_home: Path,
) -> dict[str, Any]:
    """Every context source of the run with its hash (spec/modules/agents.md "上下文清单")."""
    ws = {rel: sha256_file(p) for rel, p in _files(workspace, skip_top=(".git",)).items()}
    home_files: dict[str, str] = {}
    for rel, p in _files(home).items():
        home_files[rel] = CREDENTIAL if rel in credentials else sha256_file(p)
    for c in credentials:
        home_files[c] = CREDENTIAL
    user_codex = host_home / ".codex"
    return {
        "schema": MANIFEST_SCHEMA,
        "run": spec["run"],
        "work_unit": spec["work_unit"],
        "provider": spec["provider"],
        "mode": spec["mode"],
        "session": spec.get("session"),
        "argv": spec["argv"],
        "run_as": spec.get("run_as"),
        "cli": cli,
        "workspace": {"path": str(workspace), "files": ws},
        "home": {"path": str(home), "created_this_run": home_created, "files": home_files},
        "env": {k: ("<provided>" if k in secret_env else v) for k, v in sorted(env.items())},
        "upward_agents_md": upward_agents_md(workspace),
        "user_level": {
            str(user_codex): "present, not used (CODEX_HOME points to home)"
            if user_codex.exists()
            else "absent"
        },
        "system_config": {
            p: (sha256_file(Path(p)) if Path(p).is_file() else "absent") for p in SYSTEM_CONFIG
        },
    }


def share_with_group(root: Path) -> None:
    """Group read/write on a tree (setgid directories keep the shared group)."""
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode & 0o7777
        path.chmod(mode | (0o2070 if path.is_dir() else 0o060))


def workspace_extra(workspace: Path) -> list[str]:
    return sorted(p.name for p in workspace.iterdir() if p.name not in WORKSPACE_ENTRIES)


def snapshot(paths: list[Path]) -> dict[str, list[Any]]:
    """Depth-1 listing (mtime_ns, size) of each watched path, to spot writes after a run."""
    out: dict[str, list[Any]] = {}
    for root in paths:
        if not root.exists():
            out[str(root)] = ["absent"]
            continue
        st = root.stat()
        out[str(root)] = [st.st_mtime_ns, st.st_size]
        if root.is_dir():
            for child in root.iterdir():
                try:
                    cst = child.lstat()
                except FileNotFoundError:
                    continue
                out[str(child)] = [cst.st_mtime_ns, cst.st_size]
    return out


def snapshot_diff(before: dict[str, list[Any]], after: dict[str, list[Any]]) -> list[str]:
    changed = [p for p in after if before.get(p) != after[p]]
    removed = [p for p in before if p not in after]
    return sorted(changed + removed)


def pack(
    root: Path, exclude_top: Iterable[str] = (), skip_symlinks: bool = True
) -> tuple[bytes, list[str]]:
    """tar.gz of a directory (deterministic order), without excluded top-level names.

    Symlinks are never followed or stored, so linked credentials cannot enter an archive.
    Returns the archive and the members whose text matched a secret pattern (reported,
    conventions §8).
    """
    skip = set(exclude_top)
    buf = io.BytesIO()
    flagged: list[str] = []
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if rel.split("/")[0] in skip or (skip_symlinks and path.is_symlink()):
                continue
            tar.add(path, arcname=rel, recursive=False)
            if path.is_file() and path.stat().st_size <= SECRET_SCAN_MAX_BYTES:
                if find_secrets(path.read_bytes().decode("utf-8", "replace")):
                    flagged.append(rel)
    # gzip without a timestamp, so packing the same tree twice gives the same bytes.
    return gzip.compress(buf.getvalue(), mtime=0), flagged


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=1), encoding="utf-8")
