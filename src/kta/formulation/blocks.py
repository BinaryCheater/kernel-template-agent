"""One chain item's block: its constraints and cost terms rewritten over family variables.

Binding semantics (decision D246, settles WP4's D205):

- A Template variable bound to a family variable name becomes that variable.
- Bound to a number or boolean, or to a value of its list domain, it becomes a constant
  and leaves the search.
- Bound to any other string, the string is an expression over family variables,
  ``case.*`` (shape values) and ``hw.*``.
- ``p.<name>`` bindings give the refined transaction's parameters, as constants or as
  such expressions. A ``p.*`` used by a constraint or cost term must be bound.
- ``p.instances`` (reserved) is how many instances of the refined transaction the whole
  kernel runs; every cost term of the item is multiplied by it (default 1, decision D247).

Constraints of a block are the Template's C plus its preconditions without ``same_as``
(those repeat a constraint verbatim).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import dataclass

from kta.core.errors import SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.schemas.knowledge import ChainItem, RangeSpec, Template, VarSpec

PARAM_PREFIX = "p."
INSTANCES = "p.instances"


@dataclass(frozen=True)
class BoundExpr:
    source: str  # chain item key, or "glue"
    id: str  # the constraint / precondition ID in its Template or family
    kind: str  # "constraint", "precondition" or "glue"
    cls: str
    meaning: str
    expr: Expr  # over family variables, case.* and hw.*

    @property
    def label(self) -> str:
        return f"{self.source}:{self.id}"


@dataclass(frozen=True)
class BoundCost:
    source: str
    index: int  # position in the Template's cost list
    unit: str
    overlap: str  # "overlappable" or "serial"
    work: Expr


@dataclass(frozen=True)
class Block:
    key: str
    template: str
    fingerprint: str
    constraints: tuple[BoundExpr, ...]
    cost_terms: tuple[BoundCost, ...]
    fixed: dict[str, int | str | bool]  # Template variables bound to constants
    uses: frozenset[str]  # family variables the block refers to


def _referenced(item: ChainItem) -> set[str]:
    """Names that string bindings may refer to (the value itself, or its identifiers)."""
    names: set[str] = set()
    for value in item.bindings.values():
        if not isinstance(value, str):
            continue
        names.add(value)
        try:
            names |= {
                n.id for n in ast.walk(ast.parse(value, mode="eval")) if isinstance(n, ast.Name)
            }
        except SyntaxError:
            continue  # not an expression; the value itself is already recorded
    return names


def fingerprint(item: ChainItem, t: Template, family_vars: frozenset[str]) -> str:
    """Identity of a block: it depends on the item, the Template and which of the names
    its bindings mention are family variables, not on the rest of the family."""
    text = json.dumps(
        [
            item.model_dump(mode="json"),
            t.model_dump(mode="json"),
            sorted(family_vars & _referenced(item)),
        ],
        sort_keys=True,
    )
    return hashlib.sha256(text.encode()).hexdigest()


class _Rewriter(ast.NodeTransformer):
    def __init__(self, subst: dict[str, ast.expr], params: dict[str, ast.expr], where: str):
        self.subst = subst
        self.params = params
        self.where = where
        self.missing: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.subst:
            return copy.deepcopy(self.subst[node.id])
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if isinstance(node.value, ast.Name) and node.value.id == "p":
            if node.attr not in self.params:
                self.missing.add(f"p.{node.attr}")
                return node
            return copy.deepcopy(self.params[node.attr])
        return node


def _value_node(
    value: str | int | bool, var: VarSpec | None, family_vars: frozenset[str], where: str
) -> ast.expr:
    if not isinstance(value, str):
        return ast.Constant(value)
    if value in family_vars:
        return ast.Name(value, ast.Load())
    if var is not None and not isinstance(var.domain, RangeSpec) and value in var.domain:
        return ast.Constant(value)
    try:
        Expr.parse(value, set(family_vars))
    except ExprError as exc:
        raise SchemaError(f"{where}: binding {value!r} is not a valid expression: {exc}") from exc
    return ast.parse(value, mode="eval").body


def _to_expr(node: ast.expr, src: str, family_vars: frozenset[str], where: str) -> Expr:
    out = ast.unparse(ast.fix_missing_locations(node))
    try:
        expr = Expr.parse(out, set(family_vars))
    except ExprError as exc:
        raise SchemaError(f"{where}: {src!r} becomes invalid after binding: {exc}") from exc
    if any(r.startswith(PARAM_PREFIX) for r in expr.attr_refs):
        raise SchemaError(f"{where}: {src!r} still refers to p.* after binding ({out})")
    return expr


def _rewrite(src: str, rw: _Rewriter) -> ast.expr:
    return rw.visit(ast.parse(src.strip(), mode="eval").body)


def build_block(item: ChainItem, t: Template, family_vars: frozenset[str]) -> Block:
    """Rewrite ``t``'s constraints and cost terms through ``item``'s bindings.

    Raises SchemaError listing unbound Template variables and ``p.*`` parameters.
    """
    where = f"chain item {item.key} ({item.template})"
    variables = {v.name: v for v in t.variables}
    unbound = sorted(n for n in variables if n not in item.bindings)
    subst: dict[str, ast.expr] = {}
    fixed: dict[str, int | str | bool] = {}
    for name, var in variables.items():
        if name not in item.bindings:
            continue
        value = item.bindings[name]
        node = _value_node(value, var, family_vars, f"{where} {name}")
        subst[name] = node
        if isinstance(node, ast.Constant):
            fixed[name] = value
    params = {
        k[len(PARAM_PREFIX) :]: _value_node(v, None, family_vars, f"{where} {k}")
        for k, v in item.bindings.items()
        if k.startswith(PARAM_PREFIX)
    }
    rw = _Rewriter(subst, params, where)
    pre = [c for c in t.preconditions if c.same_as is None]
    specs = [("constraint", c) for c in t.constraints] + [("precondition", c) for c in pre]
    trees = [(kind, c, _rewrite(c.expr, rw)) for kind, c in specs]
    scale = params.pop(INSTANCES[len(PARAM_PREFIX) :], None)
    cost_trees: list[tuple[int, str, str, str, ast.expr]] = []
    for i, term in enumerate(t.cost):
        node = _rewrite(term.work, rw)
        if scale is not None:
            node = ast.BinOp(node, ast.Mult(), copy.deepcopy(scale))
        cost_trees.append((i, term.unit, term.overlap, term.work, node))
    problems = [f"variable {n} is not bound" for n in unbound]
    problems += [f"{p} is used but not bound" for p in sorted(rw.missing)]
    if problems:
        raise SchemaError(f"{where}: " + "; ".join(problems))
    constraints = tuple(
        BoundExpr(item.key, c.id, kind, c.cls, c.meaning, _to_expr(n, c.expr, family_vars, where))
        for kind, c, n in trees
    )
    costs = [
        BoundCost(item.key, i, unit, overlap, _to_expr(n, src, family_vars, where))
        for i, unit, overlap, src, n in cost_trees
    ]
    uses = frozenset[str]().union(
        *(c.expr.free_vars for c in constraints), *(c.work.free_vars for c in costs)
    )
    return Block(
        key=item.key,
        template=item.template,
        fingerprint=fingerprint(item, t, family_vars),
        constraints=constraints,
        cost_terms=tuple(costs),
        fixed=fixed,
        uses=frozenset(uses),
    )
