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
# Refinement levels 1-7 of the research frontend (D-T7, D441): 1 workload, 2 graph,
# 3 operators, 4 tensor map, 5 instructions, 6 architecture, 7 microarchitecture. Records
# written before D441 used the paper's 0-5; ``kta.core.schemas.levels`` maps them on read.
Level = Literal[1, 2, 3, 4, 5, 6, 7]
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
