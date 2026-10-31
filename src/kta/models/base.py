"""Interfaces of kta.models (conventions §3.1): the L3 SASS-level simulator (D506)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from kta.core.jobs import Job


@dataclass(frozen=True)
class L3Result:
    total_us: float
    sm_us: float  # the SM term
    bus_us: float  # the memory-bus term
    model: str  # "l3:<bundle commit>"


class L3Simulator(Protocol):
    def job(self, cubin_sha256: str, launch: Mapping[str, int]) -> Job: ...

    def simulate(self, cubin_sha256: str, launch: Mapping[str, int]) -> L3Result: ...
