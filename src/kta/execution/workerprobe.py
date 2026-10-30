"""Worker-side capability probe: EnvManifest body, capability labels and devices.

Labels are derived from what the worker can actually run (decision D182):

- ``gpu=sm_<cc>``, ``gpu_name``, ``devices`` from ``nvidia-smi`` (only when the worker is
  configured to offer GPU pools, so a GPU without a toolchain is not advertised);
- ``cuda`` (major.minor of the toolkit), ``backend.cuda`` (nvcc version) when nvcc runs;
- ``backend.tilelang`` from the worker image's ``org.kta.tilelang`` label when Docker can
  inspect the configured image;
- ``pool.<name>=<concurrency>`` for each configured execution pool;
- ``arch``, ``cpus``, ``commit``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kta.core.config import repo_root
from kta.core.envmanifest import collect_manifest, parse_version

PROBE_TIMEOUT_S = 60.0
TILELANG_LABEL = "org.kta.tilelang"


@dataclass(frozen=True)
class WorkerSpec:
    """What the control plane tells a worker about itself (from config [workers.<name>])."""

    name: str
    pools: dict[str, int] = field(default_factory=lambda: {"cpu": 1})
    image: str | None = None
    cuda_home: str = "/usr/local/cuda"

    @property
    def offers_gpu(self) -> bool:
        return any(p.startswith("gpu") for p in self.pools)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerSpec:
        return cls(**data)


def _run(argv: list[str]) -> str | None:
    exe = shutil.which(argv[0]) if "/" not in argv[0] else argv[0]
    if exe is None or not Path(exe).exists():
        return None
    try:
        proc = subprocess.run(
            [exe, *argv[1:]], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def gpus() -> list[dict[str, str]]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader"])
    found: list[dict[str, str]] = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[2].replace(".", "").isdigit():
            found.append(
                {"index": parts[0], "name": parts[1], "sm": "sm_" + parts[2].replace(".", "")}
            )
    return found


def image_info(docker: str, image: str) -> tuple[str | None, str | None]:
    """(image ID digest, TileLang version label) of a local image, or (None, None)."""
    out = _run([docker, "image", "inspect", image, "--format", "{{json .}}"])
    if not out:
        return None, None
    data: dict[str, Any] = json.loads(out)
    labels: dict[str, str] = (data.get("Config") or {}).get("Labels") or {}
    return str(data.get("Id")), labels.get(TILELANG_LABEL)


def probe(spec: WorkerSpec) -> dict[str, Any]:
    cuda_home = Path(spec.cuda_home)
    manifest = collect_manifest("worker", repo_root(), cuda_home)
    labels: dict[str, str] = {
        "arch": platform.machine(),
        "cpus": str(os.cpu_count() or 1),
        "commit": manifest.commit,
    }
    labels.update({f"pool.{name}": str(n) for name, n in spec.pools.items()})
    backends: dict[str, str] = {}
    devices: list[str] = []
    if manifest.nvcc:
        labels["backend.cuda"] = manifest.nvcc
        backends["cuda"] = manifest.nvcc
        if manifest.cuda:
            labels["cuda"] = ".".join(manifest.cuda.split(".")[:2])
    if spec.offers_gpu:
        found = gpus()
        devices = [g["index"] for g in found]
        if found:
            labels["gpu"] = found[0]["sm"]
            labels["gpu_name"] = found[0]["name"]
    image_digest = None
    docker = shutil.which("docker")
    if docker and spec.image:
        image_digest, tilelang = image_info(docker, spec.image)
        if tilelang and image_digest:
            labels["backend.tilelang"] = tilelang
            backends["tilelang"] = tilelang
        docker_version = parse_version(_run([docker, "--version"]))
        if docker_version:
            labels["docker"] = docker_version
    manifest = dataclasses.replace(manifest, backends=backends, image_digest=image_digest)
    return {
        "manifest": manifest.to_dict(),
        "labels": labels,
        "devices": devices,
        "commit": manifest.commit,
        "dirty": manifest.dirty,
        "host": manifest.host,
    }
