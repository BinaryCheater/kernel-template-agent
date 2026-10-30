"""Solving and evaluation: Problem, Contract, Case, Candidate, Prediction, Build, Check,
Measurement, Profile, Bound, Solution (spec/schemas.md §4.8)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from kta.core.schemas.common import Body, Unit


class TensorSpec(Body):
    name: str
    shape: list[str] = Field(default_factory=lambda: list[str]())
    dtype: str
    layout: str


class Problem(Body):
    name: str
    reference_impl: str
    inputs: list[TensorSpec] = Field(default_factory=lambda: list[TensorSpec]())
    outputs: list[TensorSpec] = Field(default_factory=lambda: list[TensorSpec]())
    params: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())


class Contract(Body):
    # dtype -> {"rtol": ..., "atol": ..., "rel_l2": ...}
    tolerances: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: dict[str, dict[str, Any]]()
    )
    input_dists: list[str] = Field(default_factory=lambda: list[str]())
    edge_shapes: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())
    timing_boundary: str


class Case(Body):
    problem: str
    # Each item: {"shape": {...}, "weight": w}.
    shape_dist: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())
    objective: Literal["weighted_latency", "max_normalized_latency"]
    target: str


class Candidate(Body):
    assignment: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    bucket: str | None = None


class Prediction(Body):
    per_unit: dict[Unit, float] = Field(default_factory=lambda: dict[Unit, float]())
    serial: float
    total: float
    model: str


class Build(Body):
    ok: bool
    regs: int | None = None
    spills: int | None = None
    smem: int | None = None
    log_summary: str


class Check(Body):
    passed: bool
    # Each item: check name, input distribution, shape, max error, ...; never hidden shapes.
    failures: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())


class Measurement(Body):
    median_us: float
    ci_low_us: float
    ci_high_us: float
    n: int
    clocks: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    temps: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    protocol: str


class Profile(Body):
    metrics: dict[str, float] = Field(default_factory=lambda: dict[str, float]())


class Bound(Body):
    per_unit_us: dict[Unit, float] = Field(default_factory=lambda: dict[Unit, float]())
    bound_us: float
    bottleneck: Unit


class Solution(Body):
    bucket: str | None = None
    assignment: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    pct_of_bound: float
