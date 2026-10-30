"""kta.agents (see README.md)."""

from __future__ import annotations

from typing import Any

from kta.core.context import Context
from kta.core.registry import Registry


def _batch_envs(ctx: Context) -> Any:
    from kta.agents.batch_port import BatchEnvs

    return BatchEnvs(ctx)


def register_implementations(registry: Registry) -> None:
    # The experiment process resolves batch environments without importing agents (D403).
    registry.register("batch_envs", "agents.batches", _batch_envs)
