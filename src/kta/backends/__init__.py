"""kta.backends (see README.md).

``resolve("backend.cuda", ctx)`` and ``resolve("backend.tilelang", ctx)`` return the
two backends for the configured target.
"""

from __future__ import annotations

from kta.backends.base import Backend, BuiltModule, GeneratedKernel, LaunchSpec, Param
from kta.backends.cuda import CudaBackend
from kta.backends.tilelang import TileLangBackend
from kta.core.context import Context
from kta.core.registry import Registry, resolve


def open_cuda(ctx: Context) -> CudaBackend:
    return CudaBackend(resolve("store", ctx), ctx.clock, resolve("target", ctx))


def open_tilelang(ctx: Context) -> TileLangBackend:
    return TileLangBackend(resolve("store", ctx), ctx.clock, resolve("target", ctx))


def register_implementations(registry: Registry) -> None:
    registry.register("backend.cuda", "nvcc", open_cuda)
    registry.register("backend.tilelang", "tilelang", open_tilelang)


__all__ = [
    "Backend",
    "BuiltModule",
    "CudaBackend",
    "GeneratedKernel",
    "LaunchSpec",
    "Param",
    "TileLangBackend",
    "open_cuda",
    "open_tilelang",
    "register_implementations",
]
