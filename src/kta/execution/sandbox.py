"""Container execution for external code (TileLang codegen, third-party builds).

KTA's own jobs run in the worker's host environment; only jobs that name a
``container`` image go through Docker (WP2's allowed simplification, decision D187).
The job directory is mounted at ``/job``; the host CUDA toolkit is mounted read-only
at ``/opt/kta-cuda`` so code built inside a container uses the same nvcc and ptxas as
host builds. The container has no network and runs as the worker's user.
"""

from __future__ import annotations

import os
from pathlib import Path

CONTAINER_JOB_DIR = "/job"
CONTAINER_CUDA_HOME = "/opt/kta-cuda"


def container_name(job_id: str) -> str:
    return f"kta-job-{job_id.lower()}"


def docker_argv(
    docker: str,
    image: str,
    job_id: str,
    job_dir: Path,
    argv: list[str],
    env: dict[str, str],
    cuda_home: Path | None,
    device: str | None,
    mem_limit_mb: int | None,
) -> list[str]:
    cmd = [
        docker,
        "run",
        "--rm",
        "--name",
        container_name(job_id),
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{job_dir}:{CONTAINER_JOB_DIR}",
        "-w",
        CONTAINER_JOB_DIR,
        "-e",
        f"HOME={CONTAINER_JOB_DIR}/.home",
    ]
    if cuda_home is not None:
        cmd += ["-v", f"{cuda_home}:{CONTAINER_CUDA_HOME}:ro"]
    if device is not None:
        cmd += ["--runtime", "nvidia", "--gpus", f"device={device}"]
    if mem_limit_mb is not None:
        cmd += ["--memory", f"{mem_limit_mb}m"]
    for key, value in sorted(env.items()):
        cmd += ["-e", f"{key}={value}"]
    return [*cmd, image, *argv]


def kill_argv(docker: str, job_id: str) -> list[str]:
    return [docker, "kill", container_name(job_id)]
