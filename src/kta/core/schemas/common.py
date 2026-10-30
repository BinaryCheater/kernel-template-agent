"""Shared value types (spec/schemas.md §2) and the base class of every body model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

EvidenceStatus = Literal[
    "proposed",
    "reproduced",
    "calibrated",
    "holdout_validated",
    "rejected",
    "deprecated",
    "stale",
    "suspect",
]
Assurance = Literal["declared", "tested", "bounded", "proved"]
# Refinement levels of the paper: 0 graph, 1 operator transforms, 2 tensor execution
# mapping, 3 instruction scheduling, 4 system architecture, 5 microarchitecture.
Level = Literal[0, 1, 2, 3, 4, 5]
ChangeKind = Literal["vertical", "expansion", "horizontal", "restriction"]
Binding = Literal["compile", "runtime"]
VarRole = Literal["software", "hardware", "boundary"]
VarKind = Literal["structure", "numeric"]
# Hardware unit names come from the Target's unit table (open set), e.g. "dram", "smem".
Unit = str
Origin = Literal["real", "fixture", "simulated", "historical"]
ActorType = Literal["tool", "checker", "agent", "human", "system", "import"]


class Body(BaseModel):
    """Base of all record bodies and their nested specs.

    Unknown fields are refused so that typos in hand-written YAML surface as errors
    instead of silently disappearing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
