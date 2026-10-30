"""kta.targets (see README.md).

``resolve("target", ctx)`` returns the configured TargetPlugin (GB10 in phase 1).
"""

from __future__ import annotations

from kta.core.context import Context
from kta.core.registry import Registry, resolve
from kta.targets.base import ProbeReport, TargetPlugin
from kta.targets.gb10 import Gb10Target


def open_gb10(ctx: Context) -> Gb10Target:
    return Gb10Target(resolve("store", ctx), ctx.clock)


def register_implementations(registry: Registry) -> None:
    registry.register("target", "gb10", open_gb10)


__all__ = ["Gb10Target", "ProbeReport", "TargetPlugin", "open_gb10", "register_implementations"]
