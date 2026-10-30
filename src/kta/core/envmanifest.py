"""Environment manifest collection (spec/conventions.md §7, spec/schemas.md EnvManifest).

This produces the body of an EnvManifest record. Storing it as a record is WP1's job;
probing kernel backend versions (``backends``) is WP2's.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from kta.core.envprobe import git_state, run_probe
from kta.core.locks import file_sha256, lock_mismatches

Role = Literal["control", "worker"]


@dataclass(frozen=True)
class EnvManifestData:
    role: Role
    host: str
    os: str
    arch: str
    python: str
    uv: str
    uv_lock_sha256: str
    node: str | None
    npm: str | None
    npm_lock_sha256: str | None
    cuda: str | None
    driver: str | None
    nvcc: str | None
    ptxas: str | None
    backends: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    image_digest: str | None = None
    commit: str = "unknown"
    dirty: bool = True
    lock_mismatches: list[str] = field(default_factory=lambda: list[str]())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _last_version(text: str | None) -> str | None:
    """Pick the last dotted version number from tool output."""
    if text is None:
        return None
    found = re.findall(r"\d+(?:\.\d+)+", text)
    return found[-1] if found else None


def _cuda_version(cuda_home: Path) -> str | None:
    meta = cuda_home / "version.json"
    if not meta.exists():
        return None
    data: dict[str, Any] = json.loads(meta.read_text(encoding="utf-8"))
    cuda: dict[str, Any] = data.get("cuda", {})
    version = cuda.get("version")
    return str(version) if version is not None else None


def _os_name() -> str:
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]}"
    release = Path("/etc/os-release")
    if release.exists():
        for line in release.read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    return f"{platform.system()} {platform.release()}"


def collect_manifest(role: Role, root: Path, cuda_home: Path) -> EnvManifestData:
    uv_exe = os.environ.get("UV", "uv")
    driver = run_probe(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    commit, dirty = git_state(root)
    return EnvManifestData(
        role=role,
        host=platform.node(),
        os=_os_name(),
        arch=platform.machine(),
        python=platform.python_version(),
        uv=_last_version(run_probe([uv_exe, "--version"])) or "absent",
        uv_lock_sha256=file_sha256(root / "uv.lock") or "absent",
        node=_last_version(run_probe(["node", "--version"])),
        npm=_last_version(run_probe(["npm", "--version"])),
        npm_lock_sha256=file_sha256(root / "web" / "package-lock.json"),
        cuda=_cuda_version(cuda_home),
        driver=driver.splitlines()[0].strip() if driver else None,
        nvcc=_last_version(run_probe([str(cuda_home / "bin" / "nvcc"), "--version"])),
        ptxas=_last_version(run_probe([str(cuda_home / "bin" / "ptxas"), "--version"])),
        image_digest=os.environ.get("KTA_IMAGE_DIGEST"),
        commit=commit,
        dirty=dirty,
        lock_mismatches=lock_mismatches(root),
    )
