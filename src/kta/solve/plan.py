"""FormulationPlan: the deterministic default, diff and score (WP19; 02 §5.3, D-T18, D-T19).

The default plan is built from what the family already declares and what the projection
compiler decides (D541):

- F1 chain: the family's chain (``chain_source`` names where it came from: the family, a
  retrieved chain or the chain enumerator's best-bound chain);
- F2 bindings: the chain items' bindings; F3 all variables open, the compiler's outer
  variables ``outer``; F4 the declared domains (``menu: None``);
- F5 every effect the compiler tabulates over two or more variables;
- F6 the family's glue and dispatch (none and one bucket for a family without them);
- F7 no new generator.

A plan is applied to a formulation as ``fixed``/``menus``/``outer`` for
``projection.compile_and_solve``; ``diff_plans`` lists its non-default decisions and
``score_plan`` counts them, checks their citations and compares space and bound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kta.core.errors import SchemaError
from kta.core.schemas import Actor, Record
from kta.core.schemas.plan import (
    DispatchPlan,
    FormulationPlan,
    GlueSpec,
    PlanBinding,
    PlanStep,
    TableTerm,
    VarStatus,
)
from kta.formulation import Formulation
from kta.knowledge.types import FamilyPackage
from kta.models import MachineModel
from kta.solve import projection
from kta.solve.mip import NotExportable
from kta.store.base import Store

SECTIONS = ("chain", "bindings", "variables", "table_terms", "glue", "dispatch", "generator")


def default_plan(
    fp: FamilyPackage,
    f: Formulation,
    mm: MachineModel,
    shape: Mapping[str, Any],
    chain_source: str = "family",
) -> FormulationPlan:
    """The deterministic default (D-T19) for ``f`` at a representative ``shape``."""
    menus = projection.menus_of(f, shape, mm.hw)
    try:
        outer, _ = projection.outer_vars(f, mm, shape, menus)
    except NotExportable:
        outer = set()
    terms: dict[tuple[str, tuple[str, ...]], TableTerm] = {}
    fixed = {x: menus[x][0] for x in outer}
    for a in projection.atoms(f, mm, shape, menus, fixed):
        if a.rule != "constraint" and len(a.support) >= 2:
            terms.setdefault(
                (a.source, a.support), TableTerm(effect=a.source, over=list(a.support))
            )
    family = fp.family
    d = family.dispatch
    items = fp.chain.items
    return FormulationPlan(
        family=f.family,
        author="default",
        chain_source=chain_source,
        chain=[PlanStep(key=i.key, template=i.template, applies_to=i.applies_to) for i in items],
        bindings=[
            PlanBinding(item=i.key, var=k, to=v) for i in items for k, v in i.bindings.items()
        ],
        variables=[
            VarStatus(name=x, status="outer" if x in outer else "open") for x in f.variables
        ],
        table_terms=sorted(terms.values(), key=lambda t: (t.effect, t.over)),
        glue=[GlueSpec(id=g.id, expr=g.expr, why=g.meaning) for g in family.glue_constraints],
        dispatch=DispatchPlan(runtime_vars=list(d.runtime_vars), buckets=list(d.buckets))
        if d
        else DispatchPlan(),
        generator=None,
    )


def plan_inputs(plan: FormulationPlan, f: Formulation) -> dict[str, Any]:
    """``fixed``, ``menus`` and ``outer`` of ``plan`` for ``compile_and_solve`` on ``f``."""
    fixed: dict[str, Any] = {}
    menus: dict[str, list[Any]] = {}
    outer: list[str] = []
    for v in plan.variables:
        if v.name not in f.variables:
            raise SchemaError(f"plan variable {v.name} is not a variable of {f.family}")
        if v.status == "fixed":
            if v.value is None:
                raise SchemaError(f"plan fixes {v.name} without a value")
            fixed[v.name] = v.value
        elif v.menu is not None:
            menus[v.name] = list(v.menu)
        if v.status == "outer":
            outer.append(v.name)
    return {"fixed": fixed, "menus": menus, "outer": outer}


@dataclass(frozen=True)
class PlanChange:
    section: str
    key: str
    default: Any
    plan: Any
    cited: bool


def _cited_var(v: VarStatus) -> bool:
    return bool(v.structure) and bool(v.evidence)


def _keyed(plan: FormulationPlan, section: str) -> dict[str, Any]:
    match section:
        case "chain":
            return {s.key: s.template for s in plan.chain}
        case "bindings":
            return {f"{b.item}.{b.var}": b.to for b in plan.bindings}
        case "variables":
            return {v.name: (v.status, v.value, v.menu) for v in plan.variables}
        case "table_terms":
            return {f"{t.effect}@{','.join(t.over)}": t.by for t in plan.table_terms}
        case "glue":
            return {g.id: g.expr for g in plan.glue}
        case "dispatch":
            return {"dispatch": plan.dispatch.model_dump(mode="json")}
        case _:
            return {"generator": plan.generator}


def diff_plans(default: FormulationPlan, plan: FormulationPlan) -> list[PlanChange]:
    """Every decision of ``plan`` that differs from ``default``, with whether it is cited."""
    out: list[PlanChange] = []
    for section in SECTIONS:
        a, b = _keyed(default, section), _keyed(plan, section)
        for key in sorted(a.keys() | b.keys()):
            if a.get(key) == b.get(key):
                continue
            if section == "variables":
                v = plan.var(key)
                cited = v is not None and _cited_var(v)
            else:
                c = plan.citations.get(section)
                cited = c is not None and bool(c.structure) and bool(c.evidence)
            out.append(PlanChange(section, key, a.get(key), b.get(key), cited))
    return out


@dataclass(frozen=True)
class PlanScore:
    changes: int
    uncited: list[str]  # "<section>:<key>" of non-default decisions without citation
    open_dims: int
    space: int  # points of the plan's menus (open and outer variables)
    default_space: int
    bound_us: float | None = None  # projection optimum under the plan (same chain only)
    default_bound_us: float | None = None
    notes: list[str] = field(default_factory=lambda: list[str]())

    @property
    def complete(self) -> bool:
        """The stop rule of 02 §5.4: every non-default decision cites evidence."""
        return not self.uncited

    def to_dict(self) -> dict[str, Any]:
        ratio = (
            self.bound_us / self.default_bound_us
            if self.bound_us and self.default_bound_us
            else None
        )
        return {
            "changes": self.changes, "uncited": self.uncited, "complete": self.complete,
            "open_dims": self.open_dims, "space": self.space, "default_space": self.default_space,
            "bound_us": self.bound_us, "default_bound_us": self.default_bound_us,
            "bound_ratio": ratio, "notes": self.notes,
        }  # fmt: skip


def _space(
    plan: FormulationPlan, f: Formulation, mm: MachineModel, shape: Mapping[str, Any]
) -> tuple[int, int]:
    inputs = plan_inputs(plan, f)
    menus = projection.menus_of(f, shape, mm.hw, inputs["fixed"], inputs["menus"])
    return math.prod(len(v) for v in menus.values()), sum(len(v) > 1 for v in menus.values())


def score_plan(
    plan: FormulationPlan,
    default: FormulationPlan,
    f: Formulation,
    mm: MachineModel,
    shape: Mapping[str, Any],
    require_citation: bool = True,
    solve: bool = True,
) -> PlanScore:
    """Score ``plan`` against ``default`` (02 §5.3); ``require_citation`` is the knob
    ``guard.evidence_citation_in_plans``."""
    changes = diff_plans(default, plan)
    uncited = [f"{c.section}:{c.key}" for c in changes if not c.cited] if require_citation else []
    space, dims = _space(plan, f, mm, shape)
    default_space, _ = _space(default, f, mm, shape)
    notes: list[str] = []
    bounds: list[float | None] = [None, None]
    same_chain = _keyed(plan, "chain") == {k: b.template for k, b in f.blocks.items()}
    if solve and same_chain:
        for i, p in enumerate((plan, default)):
            try:
                out = projection.compile_and_solve(f, shape, mm, shortlist=1, **plan_inputs(p, f))
            except NotExportable as exc:
                notes.append(f"{p.author}: not compiled ({exc})")
                continue
            bounds[i] = out.objective_us[0] if out.objective_us else None
    elif solve:
        notes.append("the plan changes the chain: its bound needs the new chain's formulation")
    return PlanScore(len(changes), uncited, dims, space, default_space, bounds[0], bounds[1], notes)


def record_plan(
    store: Store,
    plan: FormulationPlan,
    actor: Actor,
    at: datetime,
    case: str | None = None,
    default: str | None = None,
) -> str:
    """Write ``plan`` as an immutable FormulationPlan record; refs case and default."""
    refs: dict[str, Any] = {}
    if case:
        refs["case"] = case
    if default:
        refs["default"] = default
    return store.put(Record.new("FormulationPlan", plan.model_dump(mode="json"), actor, at, refs))
