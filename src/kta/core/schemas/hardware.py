"""Hardware and environment: Target, Capability, CapabilityDecl, HwFeature, HwFact,
Worker, EnvManifest, HwModel (spec/schemas.md §4.6)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from kta.core.schemas.common import Body, Unit
from kta.core.schemas.hwmodel import EfficiencySpec, InteractionSpec, Provenance, TableSpec
from kta.core.schemas.knowledge import ParamSpec


class Target(Body):
    name: str
    gpu: str
    sm: str
    driver: str
    cuda: str
    ptxas: str
    image: str
    backends: dict[str, str] = Field(default_factory=lambda: dict[str, str]())


class Capability(Body):
    """An abstract hardware capability a Template can require (D-T8, D442), e.g. a
    managed scratchpad or asynchronous bulk copy; Targets declare it with parameters."""

    name: str
    description: str
    unit: Unit | None = None
    params: list[ParamSpec] = Field(default_factory=lambda: list[ParamSpec]())
    # ISA features (HwFeature names) that provide it; a probe of one infers support.
    realised_by: list[str] = Field(default_factory=lambda: list[str]())


CapabilitySupport = Literal["yes", "no", "unknown"]


class CapabilityDecl(Body):
    """One Target's declaration of one Capability; the newest per pair wins (D442)."""

    target: str  # Target name
    capability: str  # Capability name
    status: CapabilitySupport
    params: dict[str, int | float | str | bool] = Field(
        default_factory=lambda: dict[str, int | float | str | bool]()
    )
    basis: Literal["documented", "probed", "inferred"]
    note: str = ""


class HwFeature(Body):
    """An ISA instruction family and its probe result per Target (realisation level)."""

    name: str
    unit: Unit
    instructions: list[str] = Field(default_factory=lambda: list[str]())
    description: str
    support: dict[str, Literal["yes", "no", "unknown"]] = Field(
        default_factory=lambda: dict[str, Literal["yes", "no", "unknown"]]()
    )


class HwFact(Body):
    feature: str | None = None
    unit: Unit | None = None
    quantity: str
    value: float | str
    ci_low: float | None = None
    ci_high: float | None = None
    method: Literal["measured", "documented"]


class Worker(Body):
    name: str
    host: str
    transport: Literal["local", "ssh", "pull"]
    labels: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    devices: list[str] = Field(default_factory=lambda: list[str]())
    commit: str


class EnvManifest(Body):
    role: Literal["control", "worker"]
    host: str
    os: str
    arch: str
    python: str
    uv: str
    uv_lock_sha256: str
    node: str | None = None
    npm: str | None = None
    npm_lock_sha256: str | None = None
    cuda: str | None = None
    driver: str | None = None
    nvcc: str | None = None
    ptxas: str | None = None
    backends: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    image_digest: str | None = None
    commit: str
    dirty: bool
    lock_mismatches: list[str] = Field(default_factory=lambda: list[str]())


class HwModel(Body):
    """The chip instance of the hardware model (schema 2, D463).

    ``peaks`` are measured chip-level peaks per resource (amount per second); they
    override the architecture class's documented peak expressions. ``limits`` are the
    ``hw.*`` values. A schema-1 body (no ``arch``) still reads: it prices only legacy
    ``unit:<u>`` transactions, with ``peaks`` as the resources.
    """

    target: str
    version: int
    peaks: dict[Unit, float] = Field(default_factory=lambda: dict[Unit, float]())
    # The hw.* names available in expressions, e.g. smem_per_cta, regs_per_thread.
    limits: dict[str, int | float] = Field(default_factory=lambda: dict[str, int | float]())
    arch: str | None = None  # "<HwArch name>@<version>"
    # "peaks.<resource>" / "limits.<name>" -> where the value comes from; required for
    # every value when ``arch`` is set.
    provenance: dict[str, Provenance] = Field(default_factory=lambda: dict[str, Provenance]())
    tables: list[TableSpec] = Field(default_factory=lambda: list[TableSpec]())
    efficiency: list[EfficiencySpec] = Field(default_factory=lambda: list[EfficiencySpec]())
    interactions: list[InteractionSpec] = Field(default_factory=lambda: list[InteractionSpec]())

    @model_validator(mode="after")
    def _provenance(self) -> HwModel:
        if self.arch is None:
            return self
        keys = [f"peaks.{k}" for k in self.peaks] + [f"limits.{k}" for k in self.limits]
        missing = [k for k in keys if k not in self.provenance]
        if missing:
            raise ValueError(f"HwModel {self.target}@{self.version}: no provenance for {missing}")
        return self
