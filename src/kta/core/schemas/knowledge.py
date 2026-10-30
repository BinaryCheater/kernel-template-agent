"""Knowledge objects: Pattern, Template, Realisation, Chain, Family, Reference,
Reconstruction, Structure, Finding, Coupling, Gap, and the legacy Regime and Negative
(spec/schemas.md §4.1-§4.5, §4.7)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from kta.core.schemas.common import Binding, Body, ChangeKind, Level, Unit, VarKind, VarRole

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
    """The abstract program of a Template; code lives in its Realisations (D-T8, D443)."""

    description: str


class Obligation(Body):
    name: str
    statement: str
    method: Literal["test", "exhaustive", "smt", "proof", "unspecified"]


class CostTerm(Body):
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
    cost: list[CostTerm] = Field(default_factory=lambda: list[CostTerm]())
    scope: ScopeSpec
    # Abstract capabilities (Capability names) the mechanism needs (D-T8, D442).
    requires: list[str] = Field(default_factory=lambda: list[str]())
    # Targets on which a family using this Template compiled and ran (D-T27, D445).
    probed_on: list[str] = Field(default_factory=lambda: list[str]())
    relations: list[Relation] = Field(default_factory=lambda: list[Relation]())
    signature: SignatureSpec
    numeric_risk: str | None = None
    summary: str
    example_code: str | None = None


class IsaRange(Body):
    """An inclusive ISA range, e.g. family "sm", min "80", max None (no upper bound)."""

    family: str
    min: str | None = None
    max: str | None = None


class Realisation(Body):
    """How a Template is written on one backend over an ISA range (D-T8, D443).

    ``backend`` is an open name (tilelang, cuda, triton, ...); ``any`` marks a
    backend-independent realisation such as SASS expectations alone.
    """

    template: str  # "<name>@<version>"
    backend: str
    isa: IsaRange
    # ISA-level features (HwFeature names, e.g. cp_async) this realisation uses.
    requires: list[str] = Field(default_factory=lambda: list[str]())
    isa_constraints: list[ExprSpec] = Field(default_factory=lambda: list[ExprSpec]())
    fragments: dict[str, str] = Field(default_factory=lambda: dict[str, str]())  # name -> blob
    code_markers: list[str] = Field(default_factory=lambda: list[str]())
    expected_sass: list[SassExpect] = Field(default_factory=lambda: list[SassExpect]())
    probed_on: list[str] = Field(default_factory=lambda: list[str]())
    summary: str


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
    """Legacy (read as a Finding of kind ``regime``, D444)."""

    structure: str
    context: str
    holds: bool
    how: Literal["proved", "observed"]


class Negative(Body):
    """Legacy (read as a Finding of kind ``negative``, D444)."""

    subject: str
    condition: str
    effect: str


class Scope(Body):
    """Where an empirical claim holds; empty fields mean "not restricted"."""

    targets: list[str] = Field(default_factory=lambda: list[str]())  # Target names
    workload: str = ""
    condition: str = ""
    # Optional machine-checkable form of ``condition`` over p.*, hw.*, case.*.
    predicate: str | None = None


class Finding(Body):
    """An empirical claim with scope (D-T9, D444); evidence is refs.evidence, status is
    derived from StatusChanges on the one ladder of empirical knowledge (D446).

    ``kind`` is an open name: regime, negative, positive, observation, other, ...
    """

    claim: str
    kind: str
    subject: str | None = None  # name of the Template, Family, Structure or Capability
    scope: Scope
    holds: bool = True
    basis: Literal["observed", "measured", "proved", "claimed"]
    # Record IDs or model-term names ("HwModel:<id>#<term>") that explain the claim.
    explained_by: list[str] = Field(default_factory=lambda: list[str]())


class CouplingMember(Body):
    kind: Literal["variable", "template", "resource", "other"]
    name: str


class Coupling(Body):
    """Members whose joint effect deviates from the product of their single effects
    beyond the noise band (02 §8.4, D447); evidence is refs.evidence."""

    members: list[CouplingMember] = Field(min_length=2)
    scope: Scope
    effect: str
    joint_ratio: float | None = None  # joint effect / product of single effects
    noise_band: float | None = None  # relative half-width the deviation exceeded
    basis: Literal["observed", "measured", "proved", "claimed"]


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
