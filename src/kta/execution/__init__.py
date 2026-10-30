"""kta.execution (see README.md).

``resolve("executor", ctx)`` returns a JobExecutor over the configured, registered
workers whose commit equals the control plane's.
"""

from __future__ import annotations

from kta.core.context import Context
from kta.core.registry import Registry, resolve
from kta.execution.base import Executor, Job, JobResult, WorkerConfig
from kta.execution.executor import JobExecutor
from kta.execution.registry import register_worker
from kta.store import code_state


def open_executor(ctx: Context) -> JobExecutor:
    return JobExecutor(ctx, resolve("store", ctx), resolve("ops", ctx), code_state().commit)


def register_implementations(registry: Registry) -> None:
    registry.register("executor", "workers", open_executor)


__all__ = [
    "Executor",
    "Job",
    "JobExecutor",
    "JobResult",
    "WorkerConfig",
    "open_executor",
    "register_implementations",
    "register_worker",
]
