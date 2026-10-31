"""Assemble a family's formulation from its chain's blocks plus the glue (22 §3.3).

The formulation is compositional: each chain item contributes one block (blocks.py);
swapping a step rebuilds only that block (``previous`` reuses unchanged blocks by
fingerprint).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.errors import SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.schemas.evaluation import Case
from kta.core.schemas.knowledge import Chain, Template, VarSpec
from kta.formulation.blocks import (
    Block,
    BoundCost,
    BoundExpr,
    BoundOverlap,
    BoundStorage,
    BoundTx,
    build_block,
    fingerprint,
)
from kta.knowledge.types import FamilyPackage

GLUE = "glue"


@dataclass(frozen=True)
class Formulation:
    family: str  # "<name>@<version>"
    variables: dict[str, VarSpec]  # family variables
    constraints: list[BoundExpr]  # every item's constraints and preconditions, then glue
    cost_terms: list[BoundCost]  # legacy unit:<u> transactions (schema-1 cost, D461)
    linear: bool  # every constraint is in the linear subset
    cost_linear: bool  # every legacy cost term is linear in the family variables
    blocks: dict[str, Block]
    reused: tuple[str, ...]  # item keys whose block came from ``previous``
    runtime_vars: tuple[str, ...]  # dispatch variables (per shape bucket)
    hw_refs: frozenset[str]  # hw.* names the formulation reads
    case_refs: frozenset[str]  # case.* names (shape values) it reads
    missing_hw: tuple[str, ...]  # hw.* names absent from the hw contract given to assemble
    fingerprint: str
    transactions: list[BoundTx] = field(default_factory=lambda: list[BoundTx]())
    storage: list[BoundStorage] = field(default_factory=lambda: list[BoundStorage]())
    overlaps: list[BoundOverlap] = field(default_factory=lambda: list[BoundOverlap]())

    def constraint(self, label: str) -> BoundExpr:
        return next(c for c in self.constraints if c.label == label)


def _glue(family_vars: frozenset[str], fp: FamilyPackage) -> list[BoundExpr]:
    out: list[BoundExpr] = []
    for g in fp.family.glue_constraints:
        try:
            expr = Expr.parse(g.expr, set(family_vars))
        except ExprError as exc:
            raise SchemaError(f"glue {g.id}: {exc}") from exc
        out.append(BoundExpr(GLUE, g.id, GLUE, g.cls, g.meaning, expr))
    return out


def _refs(exprs: list[Expr], root: str) -> frozenset[str]:
    prefix = root + "."
    return frozenset(r[len(prefix) :] for e in exprs for r in e.attr_refs if r.startswith(prefix))


def assemble(
    family: FamilyPackage,
    chain: Chain,
    templates: dict[str, Template],
    case: Case,
    hw: Mapping[str, Any],
    previous: Formulation | None = None,
) -> Formulation:
    """Blocks of ``chain`` (Template ref -> Template in ``templates``) plus the family glue.

    ``hw`` is the hardware contract (``hw.*`` values); names the formulation reads but
    ``hw`` lacks are listed in ``missing_hw`` (solving then raises NotSupported). ``case``
    is not used to build the formulation; shapes enter at solve time.
    """
    del case  # shapes enter at solve time; kept for the spec signature
    fam = family.family
    family_vars = frozenset(v.name for v in fam.variables)
    blocks: dict[str, Block] = {}
    reused: list[str] = []
    errors: list[str] = []
    for item in chain.items:
        t = templates.get(item.template)
        if t is None:
            errors.append(f"chain item {item.key}: Template {item.template} not given")
            continue
        old = previous.blocks.get(item.key) if previous else None
        if old is not None and old.fingerprint == fingerprint(item, t, family_vars):
            blocks[item.key] = old
            reused.append(item.key)
            continue
        try:
            blocks[item.key] = build_block(item, t, family_vars)
        except SchemaError as exc:
            errors.append(str(exc))
    if errors:
        raise SchemaError("cannot assemble formulation: " + " | ".join(errors))
    constraints = [c for b in blocks.values() for c in b.constraints] + _glue(family_vars, family)
    costs = [c for b in blocks.values() for c in b.cost_terms]
    exprs = [c.expr for c in constraints] + [e for b in blocks.values() for e in b.exprs()]
    hw_refs = _refs(exprs, "hw")
    digest = hashlib.sha256(
        json.dumps(
            [
                fam.name,
                fam.version,
                [v.model_dump(mode="json") for v in fam.variables],
                [c.expr.src for c in constraints],
                [(c.unit, c.overlap, c.work.src) for c in costs],
                [_effects_key(b) for b in blocks.values()],
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return Formulation(
        family=f"{fam.name}@{fam.version}",
        variables={v.name: v for v in fam.variables},
        constraints=constraints,
        cost_terms=costs,
        linear=all(c.expr.is_linear for c in constraints),
        cost_linear=all(c.work.is_linear for c in costs),
        blocks=blocks,
        reused=tuple(reused),
        runtime_vars=tuple(fam.dispatch.runtime_vars) if fam.dispatch else (),
        hw_refs=hw_refs,
        case_refs=_refs(exprs, "case"),
        missing_hw=tuple(sorted(hw_refs - set(hw))),
        fingerprint=digest[:16],
        transactions=[t for b in blocks.values() for t in b.transactions],
        storage=[s for b in blocks.values() for s in b.storage],
        overlaps=[o for b in blocks.values() for o in b.overlaps],
    )


def _effects_key(b: Block) -> list[object]:
    return [
        [(t.id, t.pattern, t.kind, t.count.src, sorted((k, v.src) for k, v in t.amounts.items()))
         for t in b.transactions],
        [(s.id, s.level, s.scope, s.bytes.src) for s in b.storage],
        [(o.between, o.when.src if o.when else None) for o in b.overlaps],
    ]  # fmt: skip


def violated(
    f: Formulation, assignment: Mapping[str, Any], shape: Mapping[str, Any], hw: Mapping[str, Any]
) -> list[BoundExpr]:
    """Constraints the point violates: "which step's constraint excludes this point"."""
    return [c for c in f.constraints if not c.expr.eval(assignment, hw=hw, case=shape)]
