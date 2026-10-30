"""Backend protocol, GeneratedKernel, BuiltModule (spec/modules/target-exec.md)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from kta.core.jobs import JobRunner
from kta.core.schemas.evaluation import Build
from kta.core.schemas.hardware import Target

# Parameter types the harness can pass: a pointer to a buffer or a scalar.
SCALAR_TYPES = ("i32", "u32", "i64", "u64", "f32", "f64")
POINTER_TYPES = ("f16*", "bf16*", "f32*", "f64*", "i8*", "u8*", "i32*", "u32*", "i64*")


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # one of POINTER_TYPES or SCALAR_TYPES

    @property
    def is_pointer(self) -> bool:
        return self.type.endswith("*")


@dataclass(frozen=True)
class LaunchSpec:
    grid: list[int]
    block: list[int]
    smem_bytes: int = 0
    params: list[Param] = field(default_factory=lambda: list[Param]())

    @property
    def threads_per_block(self) -> int:
        n = 1
        for b in self.block:
            n *= b
        return n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaunchSpec:
        params = [Param(**p) for p in data.get("params", [])]
        return cls(list(data["grid"]), list(data["block"]), int(data.get("smem_bytes", 0)), params)


@dataclass(frozen=True)
class GeneratedKernel:
    backend: str
    files: dict[str, str]  # file name -> source text
    entry: str  # symbol name (TileLang: the prim_func's name in kernel.py)
    launch: LaunchSpec
    markers: list[str] = field(default_factory=lambda: list[str]())  # kta:step markers


@dataclass(frozen=True)
class BuiltModule:
    sha256: str  # blob of the cubin
    kind: Literal["so", "cubin"]
    feedback: dict[str, Any]  # regs, spills, smem, ptxas summary
    sass_blob: str | None
    entry: str  # symbol to launch (TileLang appends "_kernel")
    build_id: str  # the Build record (decision D189)


class Backend(Protocol):
    name: str

    def supports(self, target: Target) -> bool: ...

    def build(
        self, gk: GeneratedKernel, target: Target, ex: JobRunner
    ) -> tuple[Build, BuiltModule]: ...
