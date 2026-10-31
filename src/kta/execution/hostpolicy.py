"""Per-job host-lock mode and the unified-memory cap (WP23, D603, D607).

Pure functions the Executor calls while it places and runs a job; kept apart so the
policy can be read (and tested) without the Executor's threads.

Host-lock mode of a job on a worker whose ``[workers.<w>] host_lock`` is set:

| job | mode |
| --- | --- |
| pool ``agent`` (the agent CLI itself) | none: its heavy work goes through ``kta`` → Executor |
| inside a measurement window the same Executor holds (``quiet_lease``) | ``held`` |
| gpu-exclusive that took the device lease itself | ``exclusive`` |
| cpu, gpu-shared | ``shared`` |

Memory: on a ``unified_memory`` worker, concurrently running cpu-pool jobs may together
reserve at most ``cpu_jobs_gb`` (controls ``execution.unified_memory_cap``); a job without
``mem_limit_mb`` counts ``job_default_gb``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from kta.core.jobs import Job
from kta.execution.base import WorkerConfig

HostMode = Literal["none", "held", "shared", "exclusive"]


def host_mode(job: Job, own_window: bool, self_leased: bool) -> HostMode:
    if job.pool == "agent":
        return "none"
    if own_window:
        return "held"
    if job.pool == "gpu-exclusive" or self_leased:
        return "exclusive"
    return "shared"


def lock_request(
    config: WorkerConfig,
    mode: HostMode,
    holder: str,
    shared_wait_s: float,
    exclusive_wait_s: float,
    max_s: float,
) -> dict[str, Any] | None:
    """The ``host_lock`` of a worker ``run`` request, or None for a worker without one."""
    if not config.host_lock or mode == "none":
        return None
    wait = exclusive_wait_s if mode == "exclusive" else shared_wait_s
    return {"dir": config.host_lock, "mode": mode, "wait_s": wait, "max_s": max_s, "holder": holder}


def job_mem_gb(job: Job, cap: Mapping[str, float]) -> float:
    if job.mem_limit_mb is not None:
        return job.mem_limit_mb / 1024
    return float(cap.get("job_default_gb", 0.0))


def mem_blocks(
    config: WorkerConfig, job: Job, running_gb: float, cap: Mapping[str, float]
) -> str | None:
    """Why a cpu job may not start now on this worker, or None."""
    if job.pool != "cpu" or not config.unified_memory:
        return None
    need = job_mem_gb(job, cap)
    limit = float(cap.get("cpu_jobs_gb", 0.0))
    if running_gb > 0 and running_gb + need > limit:
        return (
            f"unified-memory cap on {config.name}: {running_gb:.1f} GB running "
            f"+ {need:.1f} GB > {limit:.1f} GB"
        )
    return None
