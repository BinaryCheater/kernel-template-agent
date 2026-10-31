"""Declarative hardware model: architecture class, chip-instance parts, model versions,
regression sets and the interference matrix (spec/schemas.md §4.6.1, docs/design/02 §4,
decisions D462-D464, D469, D470).

The architecture class (``HwArch``) holds structure shared by every chip of the class:
resources and their scopes, walls with their cliffs, operation classes with reservation
tables, table terms, efficiency and interaction structure, the composition rule. The chip
instance (``HwModel`` in ``hardware.py``) holds the numbers: ``limits`` (the ``hw.*``
values), measured ``peaks``, instance tables, fitted efficiency and interactions, and a
provenance entry for every value.

Expressions use the language of ``kta.core.expr`` with the roots ``hw`` (instance
values), ``tx`` (per-transaction amounts), ``f`` (features such as occupancy) and ``tbl``
(table terms, looked up by L1 before evaluation). Every value names its source.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from kta.core.expr import Expr, ExprError
from kta.core.schemas.common import Body

ValueStatus = Literal["measured", "documented", "default"]
HW_ROOTS = ("hw", "tx", "f", "tbl")
# Which roots each kind of expression may read (D462).
ROOTS_SCOPE = ("hw",)
ROOTS_RESERVATION = ("hw", "tx", "tbl")
ROOTS_FEATURE = ("hw", "f", "tbl")


def check_hw_expr(src: str, roots: tuple[str, ...], where: str) -> Expr:
    try:
        return Expr.parse(src, set(), roots)
    except ExprError as exc:
        raise ValueError(f"{where}: {exc}") from exc


class Provenance(Body):
    """Where one value comes from (D463). ``fact`` is the HwFact record when there is one."""

    status: ValueStatus
    source: str
    fact: str | None = None


class ResourceSpec(Body):
    name: str  # dram, l2, smem, tensor, fma, issue, launch, ... (open)
    kind: str  # throughput | capacity | latency (open)
    scope: str  # a key of HwArch.scopes
    amount: str  # what one unit of use counts: bytes, flops, wavefronts, instructions, ctas
    # amount per second per scope instance; absent -> infinite unless HwModel.peaks has it
    peak: str | None = None
    latency_cycles: str | None = None
    max_inflight: str | None = None
    ncu: list[str] = Field(default_factory=lambda: list[str]())
    source: str


class WallSpec(Body):
    """A capacity or legality limit on the storage effects at one level (D471)."""

    name: str
    kind: str  # capacity | legality (open)
    level: str  # storage level the wall limits
    scope: str  # executor the storage is counted per (cta, sm, ...)
    limit: str  # expression over hw.*
    reserved: str | None = None  # added per executor instance, e.g. hw.smem_reserved_per_cta
    cliff: str  # no_launch | spill | lower_occupancy | other (open)
    per: str | None = None  # for lower_occupancy: the scope whose residency it limits (sm)
    source: str


class ReservationRow(Body):
    resource: str
    amount: str  # per transaction, over tx.* and hw.*
    latency_cycles: str | None = None


class OperationClass(Body):
    name: str  # = TxEffect.kind
    patterns: list[str] = Field(default_factory=lambda: list[str]())
    reservation: list[ReservationRow] = Field(min_length=1)
    latency_cycles: str | None = None  # issue to completion of one transaction
    serial: bool = False  # never overlaps with anything (grid launch, grid-wide sync)
    sass: list[str] = Field(default_factory=lambda: list[str]())  # SASS classes covered
    source: str


class TableSpec(Body):
    """A table term: ``tbl.<name>`` is the value of the row matching the keys (D462)."""

    name: str
    keys: list[str] = Field(min_length=1)  # attribute refs: "tx.stride", "f.occupancy", "hw.x"
    rows: list[list[float]] = Field(min_length=1)  # key values..., value
    lookup: Literal["exact", "floor"] = "exact"  # floor: largest key row <= the keys (1 key)
    source: str

    @model_validator(mode="after")
    def _shape(self) -> TableSpec:
        width = len(self.keys) + 1
        if any(len(r) != width for r in self.rows):
            raise ValueError(f"table {self.name}: every row needs {width} numbers")
        if self.lookup == "floor" and len(self.keys) != 1:
            raise ValueError(f"table {self.name}: floor lookup takes one key")
        for k in self.keys:
            root = k.partition(".")[0]
            if root not in ("hw", "tx", "f"):
                raise ValueError(f"table {self.name}: key {k!r} is not hw.*, tx.* or f.*")
        return self


class EfficiencySpec(Body):
    """Discount on a resource's peak as a function of few features (02 §4)."""

    resource: str
    expr: str  # over f.*, tbl.*, hw.*; in (0, 1]
    features: list[str] = Field(default_factory=lambda: list[str]())
    status: ValueStatus
    source: str
    fit_data: list[str] = Field(default_factory=lambda: list[str]())  # record IDs
    holdout_error: float | None = None


class InteractionSpec(Body):
    """Interference between two resources: adds gamma * min(busy_a, busy_b) when it applies."""

    name: str
    resources: list[str] = Field(min_length=2, max_length=2)
    gamma: float
    applies_when: str = "True"  # over f.*, hw.*
    evidence: list[str] = Field(default_factory=lambda: list[str]())  # record IDs
    status: ValueStatus
    source: str


class CompositionSpec(Body):
    rule: Literal["steady_state_max"] = "steady_state_max"  # the L1 rule (D464)
    clock_hz: str = "hw.sm_clock_hz"  # converts latency cycles to seconds
    note: str = ""


def _expr_checks(items: list[tuple[str, str | None, tuple[str, ...]]]) -> None:
    for where, src, roots in items:
        if src is not None:
            check_hw_expr(src, roots, where)


class HwArch(Body):
    """Architecture class: structure shared by the chips of one class (D463)."""

    name: str
    version: int
    description: str
    scopes: dict[str, str]  # scope -> instance count expression over hw.*
    params: list[str] = Field(default_factory=lambda: list[str]())  # hw.* an instance gives
    resources: list[ResourceSpec] = Field(min_length=1)
    walls: list[WallSpec] = Field(default_factory=lambda: list[WallSpec]())
    operation_classes: list[OperationClass] = Field(default_factory=lambda: list[OperationClass]())
    tables: list[TableSpec] = Field(default_factory=lambda: list[TableSpec]())
    efficiency: list[EfficiencySpec] = Field(default_factory=lambda: list[EfficiencySpec]())
    interactions: list[InteractionSpec] = Field(default_factory=lambda: list[InteractionSpec]())
    composition: CompositionSpec = Field(default_factory=CompositionSpec)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @model_validator(mode="after")
    def _consistent(self) -> HwArch:
        names = [r.name for r in self.resources]
        resources = set(names)
        problems = [f"resource {n} declared twice" for n in sorted(resources) if names.count(n) > 1]
        ops = [o.name for o in self.operation_classes]
        problems += [f"operation class {n} declared twice" for n in set(ops) if ops.count(n) > 1]
        problems += [
            f"resource {r.name}: scope {r.scope} not in scopes"
            for r in self.resources
            if r.scope not in self.scopes
        ]
        problems += [
            f"operation class {o.name}: resource {row.resource} not declared"
            for o in self.operation_classes
            for row in o.reservation
            if row.resource not in resources
        ]
        problems += [
            f"{kind} {name}: resource {r} not declared"
            for kind, name, rs in [
                *(("efficiency", e.resource, [e.resource]) for e in self.efficiency),
                *(("interaction", i.name, i.resources) for i in self.interactions),
            ]
            for r in rs
            if r not in resources
        ]
        if problems:
            raise ValueError("; ".join(problems))
        _expr_checks(self._expressions())
        return self

    def _expressions(self) -> list[tuple[str, str | None, tuple[str, ...]]]:
        out: list[tuple[str, str | None, tuple[str, ...]]] = [
            (f"scopes.{k}", v, ROOTS_SCOPE) for k, v in self.scopes.items()
        ]
        for r in self.resources:
            out += [
                (f"resources.{r.name}.{f}", getattr(r, f), ROOTS_SCOPE)
                for f in ("peak", "latency_cycles", "max_inflight")
            ]
        for w in self.walls:
            out += [(f"walls.{w.name}.limit", w.limit, ROOTS_SCOPE)]
            out += [(f"walls.{w.name}.reserved", w.reserved, ROOTS_SCOPE)]
        for o in self.operation_classes:
            out += [(f"operation_classes.{o.name}.latency_cycles", o.latency_cycles, ROOTS_SCOPE)]
            for i, row in enumerate(o.reservation):
                where = f"operation_classes.{o.name}.reservation[{i}]"
                out += [(where + ".amount", row.amount, ROOTS_RESERVATION)]
                out += [(where + ".latency_cycles", row.latency_cycles, ROOTS_SCOPE)]
        out += [(f"efficiency.{e.resource}", e.expr, ROOTS_FEATURE) for e in self.efficiency]
        out += [
            (f"interactions.{i.name}", i.applies_when, ROOTS_FEATURE) for i in self.interactions
        ]
        out += [("composition.clock_hz", self.composition.clock_hz, ROOTS_SCOPE)]
        return out


# ---- model versions and the regression set (D470) -------------------------------------


class RegressionCase(Body):
    """One measured kernel family with several configurations (ranking needs several)."""

    kernel: str  # family or reference name
    shape: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    configurations: list[dict[str, Any]]
    measurements: list[str] = Field(default_factory=lambda: list[str]())  # Measurement IDs
    holdout: bool = False


class RegressionSet(Body):
    name: str
    version: int
    target: str
    min_configurations: int = Field(default=2, ge=2)
    cases: list[RegressionCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _several(self) -> RegressionSet:
        for c in self.cases:
            if len(c.configurations) < self.min_configurations:
                raise ValueError(
                    f"regression case {c.kernel}: {len(c.configurations)} configurations, "
                    f"at least {self.min_configurations} are required"
                )
            if c.measurements and len(c.measurements) != len(c.configurations):
                raise ValueError(f"regression case {c.kernel}: one measurement per configuration")
        return self


class RegressionResult(Body):
    kernel: str
    predicted_us: list[float]
    measured_us: list[float]
    rank_correlation: float | None = None
    error_by_resource: dict[str, float] = Field(default_factory=lambda: dict[str, float]())


class ModelVersion(Body):
    """An immutable hardware-model version: parameter table + simulator commit +
    regression results (02 §4 "Maintenance"). refs: hw_model, arch, regression_set,
    previous."""

    target: str
    label: str
    hw_model: str  # "<target>@<version>" of the HwModel (chip instance)
    arch: str | None = None  # "<name>@<version>" of the HwArch
    simulators: dict[str, str] = Field(default_factory=lambda: dict[str, str]())  # level -> commit
    change: Literal["initial", "adapter_fix", "machine_model_change"]
    reason: str
    results: list[RegressionResult] = Field(default_factory=lambda: list[RegressionResult]())
    summary: dict[str, float] = Field(default_factory=lambda: dict[str, float]())
    admitted: bool | None = None  # hold-out decision; None until regression ran


# ---- interference matrix (D469; an artefact, validated by this model) -------------------


class InterferenceCell(Body):
    a: str  # operation class (or SASS instruction class)
    b: str
    status: Literal["measured", "declared_independent", "unmeasured"]
    alone_a_s: float | None = None
    alone_b_s: float | None = None
    together_s: float | None = None
    # together / max(alone_a, alone_b): 1 = full overlap, 2 = fully serialised (equal work)
    slowdown: float | None = None
    measurements: list[str] = Field(default_factory=lambda: list[str]())
    note: str | None = None


class InterferenceMatrix(Body):
    target: str
    arch: str | None = None
    protocol: str | None = None
    classes: list[str] = Field(min_length=1)
    cells: list[InterferenceCell] = Field(default_factory=lambda: list[InterferenceCell]())

    def coverage(self) -> dict[str, int]:
        """Pairs measured, declared independent and still unknown (D-T14 coverage)."""
        n = len(self.classes)
        pairs = n * (n + 1) // 2
        counts = {"pairs": pairs, "measured": 0, "declared_independent": 0}
        for c in self.cells:
            if c.status != "unmeasured":
                counts[c.status] += 1
        counts["unmeasured"] = pairs - counts["measured"] - counts["declared_independent"]
        return counts
