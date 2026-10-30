"""TargetPlugin protocol (spec/modules/target-exec.md).

Targets run their probes through the Executor; they only need its ``run`` method and
blob access, so they depend on the structural ``JobRunner`` (kta.core.jobs), not on
kta.execution (a higher layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from kta.core.jobs import JobRunner
from kta.core.schemas.hardware import HwFeature, Target


class TargetPlugin(Protocol):
    name: str

    def probe(self, ex: JobRunner) -> Target: ...

    def units(self) -> list[str]: ...

    def limits(self) -> dict[str, int | float]: ...

    def features(self) -> list[HwFeature]: ...

    def compile_flags(self, backend: str) -> list[str]: ...


@dataclass(frozen=True)
class ProbeReport:
    """Record IDs written by one probe run."""

    target_id: str
    worker_id: str
    env_id: str
    facts: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # quantity -> id
    features: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # name -> id
    support: dict[str, str] = field(
        default_factory=lambda: dict[str, str]()
    )  # name -> yes/no/unknown
