"""Knowledge objects: Pattern, Template, Chain, Family, Reference, Reconstruction,
Structure, Regime, Negative, Gap (spec/schemas.md §4.1-§4.5, §4.7)."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import Field, model_validator

from kta.core.schemas.common import Binding, Body, ChangeKind, Level, Unit, VarKind, VarRole
from kta.core.schemas.effects import (
    LEGACY_KIND_PREFIX,
    Effects,
    upcast_template_body,
)

# ---- 4.1 Pattern ---------------------------------------------------------------


class ParamSpec(Body):
    name: str
    type: Literal["bool", "int", "float", "str", "enum"]
    values: list[str] | None = None
    meaning: str


class Pattern(Body):
    name: str
    level: Level
    category: Literal["reduce", "contract", "map", "store", "transfer", "sync", "access", "other"]
    params: list[ParamSpec] = Field(default_factory=lambda: list[ParamSpec]())
    description: str
    examples: list[str] = Field(min_length=2)


# ---- 4.2 Template --------------------------------------------------------------


class ObjectSpec(Body):
    type: str
    description: str


class RangeSpec(Body):
    lo: int
    hi: int
    step: int = 1
    pow2: bool = False


class VarSpec(Body):
    name: str
    kind: VarKind
    binding: Binding
    role: VarRole
    domain: list[int | str | bool] | RangeSpec
    unit: str | None = None
    meaning: str
    when: str | None = None


class ExprSpec(Body):
    id: str
    expr: str
    meaning: str
    cls: Literal["semantic", "resource", "implementation", "performance_assumption"]
    same_as: str | None = None


class ProgramSpec(Body):
    description: str
    fragments: dict[str, str] = Field(default_factory=lambda: dict[str, str]())


class Obligation(Body):
    name: str
    statement: str
    method: Literal["test", "exhaustive", "smt", "proof", "unspecified"]


class CostTerm(Body):
    """Schema-1 cost term; kept as the read-only legacy view ``Template.cost`` (D461)."""

    unit: Unit
    work: str
    overlap: Literal["overlappable", "serial"] = "overlappable"


class ScopeSpec(Body):
    predicate: str
    text: str


class Relation(Body):
    type: Literal["requires", "excludes", "enables", "refines", "alternative_of"]
    target: str
    note: str | None = None


class SignatureSpec(Body):
    algorithm: str
    reuse: str | None = None
    storage: str | None = None
    boundary: str | None = None
    resources: list[str] = Field(default_factory=lambda: list[str]())


class SassExpect(Body):
    when: str
    must_contain: list[str] = Field(default_factory=lambda: list[str]())
    must_not_contain: list[str] = Field(default_factory=lambda: list[str]())


class Template(Body):
    name: str
    version: int
    level_from: Level
    level_to: Level
    change: ChangeKind
    refines: str
    produces: list[str] = Field(default_factory=lambda: list[str]())
    object: ObjectSpec
    variables: list[VarSpec] = Field(default_factory=lambda: list[VarSpec]())
    constraints: list[ExprSpec] = Field(default_factory=lambda: list[ExprSpec]())
    preconditions: list[ExprSpec] = Field(default_factory=lambda: list[ExprSpec]())
    program: ProgramSpec
    backward_map: str
    backward_map_fn: str | None = None
    obligations: list[Obligation] = Field(default_factory=lambda: list[Obligation]())
    effects: Effects = Field(default_factory=Effects)
    scope: ScopeSpec
    requires_hw: list[str] = Field(default_factory=lambda: list[str]())
    relations: list[Relation] = Field(default_factory=lambda: list[Relation]())
    signature: SignatureSpec
    numeric_risk: str | None = None
    expected_sass: list[SassExpect] = Field(default_factory=lambda: list[SassExpect]())
    summary: str
    example_code: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _upcast(cls, data: object) -> object:
        # Schema-1 bodies carry ``cost``; read them as effects (D461).
        if not isinstance(data, dict):
            return data
        body = cast(dict[str, Any], data)
        return upcast_template_body(body) if "cost" in body else body

    @property
    def cost(self) -> list[CostTerm]:
        """Legacy view: the transactions of ``unit:<u>`` classes as schema-1 cost terms,
        in order (empty for a Template written with effects)."""
        serial = {
            o.between[0]
            for o in self.effects.overlap
            if len(o.between) == 1 and o.when is not None and o.when.strip() == "False"
        }
        return [
            CostTerm(
                unit=t.kind[len(LEGACY_KIND_PREFIX) :],
                work=t.count,
                overlap="serial" if t.id in serial else "overlappable",
            )
            for t in self.effects.transactions
            if t.kind.startswith(LEGACY_KIND_PREFIX)
        ]


# ---- 4.3 Chain / 4.4 Family ----------------------------------------------------


class ChainItem(Body):
    key: str
    template: str
    applies_to: str
    bindings: dict[str, str | int | bool] = Field(
        default_factory=lambda: dict[str, str | int | bool]()
    )
    after: list[str] = Field(default_factory=lambda: list[str]())
    code_markers: list[str] = Field(default_factory=lambda: list[str]())


class Chain(Body):
    workload: str
    items: list[ChainItem] = Field(default_factory=lambda: list[ChainItem]())


class DispatchSpec(Body):
    runtime_vars: list[str] = Field(default_factory=lambda: list[str]())
    buckets: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())
    rule: str


class Family(Body):
    name: str
    version: int
    backend: Literal["tilelang", "cuda"]
    variables: list[VarSpec] = Field(default_factory=lambda: list[VarSpec]())
    glue_constraints: list[ExprSpec] = Field(default_factory=lambda: list[ExprSpec]())
    dispatch: DispatchSpec | None = None
    edge_cases: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())
    numeric_risk: list[str] = Field(default_factory=lambda: list[str]())
    entry: str


# ---- 4.5 Reference / Reconstruction --------------------------------------------


class Reference(Body):
    name: str
    problem: str
    build: str
    runnable_on: list[str] = Field(default_factory=lambda: list[str]())


class UnmatchedTx(Body):
    level: Level
    pattern_guess: str
    description: str
    where: str


class Reconstruction(Body):
    matched: int
    total: int
    unmatched_transactions: list[UnmatchedTx] = Field(default_factory=lambda: list[UnmatchedTx]())
    reproduction_error: float | None = None
    notes: str


# ---- 4.7 Other knowledge -------------------------------------------------------


class Structure(Body):
    name: str
    effect: Literal["objective", "variables", "enumeration"]
    description: str
    evidence_needed: str


class Regime(Body):
    structure: str
    context: str
    holds: bool
    how: Literal["proved", "observed"]


class Negative(Body):
    subject: str
    condition: str
    effect: str


class Gap(Body):
    missing: Literal[
        "pattern",
        "template",
        "relation",
        "hw_fact",
        "family",
        "model_term",
        "check",
        "other",
    ]
    description: str
    level: Level | None = None
    pattern_guess: str | None = None
    attribution: Literal[
        "space",
        "model",
        "intent_sass",
        "verification",
        "process",
        "granularity",
        "other",
    ]
