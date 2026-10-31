"""Result types of solving (module-internal dataclasses, conventions §3.1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from kta.core.schemas.evaluation import Candidate as CandidateBody
from kta.core.schemas.evaluation import Prediction

Method = Literal["enumerate", "mip", "projection"]  # mip: the comparison arm (D-T1)


def assignment_key(assignment: dict[str, Any], bucket: str | None = None) -> str:
    text = json.dumps([bucket, assignment], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Candidate:
    """An assignment with its prediction; ``key`` names it in ``rerank``'s builds."""

    assignment: dict[str, Any]
    bucket: str | None
    prediction: Prediction

    @property
    def key(self) -> str:
        return assignment_key(self.assignment, self.bucket)

    def body(self) -> CandidateBody:
        return CandidateBody(assignment=self.assignment, bucket=self.bucket)


@dataclass(frozen=True)
class SolveResult:
    candidates: list[Candidate]  # best predicted first
    method: Method
    space_size: int  # product of the family variables' domain sizes
    feasible: int  # feasible points visited (all of them when exhaustive)
    pruned: int  # points excluded by constraints (feasible + pruned = space when exhaustive)
    pruned_by: dict[str, int]  # constraint label "<item key>:<id>" -> points it excluded
    exhaustive: bool
    model_optimal: bool  # the first candidate is optimal for this model (see ``scope``)
    scope: dict[str, Any]  # family, formulation, model version, hw model, bucket, shape
    mip_status: str | None = None
    mip_gap: float | None = None
    notes: list[str] = field(default_factory=lambda: list[str]())

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "space_size": self.space_size,
            "feasible": self.feasible,
            "pruned": self.pruned,
            "pruned_by": self.pruned_by,
            "exhaustive": self.exhaustive,
            "model_optimal": self.model_optimal,
            "scope": self.scope,
            "mip_status": self.mip_status,
            "mip_gap": self.mip_gap,
            "notes": self.notes,
            "candidates": len(self.candidates),
        }
