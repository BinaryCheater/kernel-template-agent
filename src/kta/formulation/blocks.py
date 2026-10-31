"""One chain item's block: its constraints and effects rewritten over family variables.

Binding semantics (decision D246, settles WP4's D205):

- A Template variable bound to a family variable name becomes that variable.
- Bound to a number or boolean, or to a value of its list domain, it becomes a constant
  and leaves the search.
- Bound to any other string, the string is an expression over family variables,
  ``case.*`` (shape values) and ``hw.*``.
- ``p.<name>`` bindings give the refined transaction's parameters, as constants or as
  such expressions. A ``p.*`` used by a constraint or cost term must be bound.
- ``p.instances`` (reserved) is how many instances of the refined transaction the whole
  kernel runs; every transaction count of the item is multiplied by it (default 1,
  decisions D247, D460). Storage is per executor instance and is not multiplied.
- Effects (D460): transaction counts and amounts, storage bytes and overlap conditions
  are rewritten like constraints. A storage ``level`` naming a Template variable or a
  ``p.*`` parameter takes its constant string binding. Overlap selectors naming one of
  the Template's transactions become ``<item key>.<id>``.
- ``cost_terms`` are the legacy ``unit:<u>`` transactions (schema-1 cost, D461), kept so
  that a formulation over upcast Templates prices exactly as before.

Constraints of a block are the Template's C plus its preconditions without ``same_as``
(those repeat a constraint verbatim).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from kta.core.errors import SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.schemas.effects import LEGACY_KIND_PREFIX, SELECT_KIND, SELECT_PATTERN
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
class BoundTx:
    source: str
    id: str
    pattern: str
    kind: str
    count: Expr  # already multiplied by p.instances
    amounts: dict[str, Expr]
    when: Expr | None
    # When the kind is chosen by a family variable (D460): its name and the value -> class
    # map; ``kind`` is then empty and ``kind_at`` gives the class at a point.
    kind_var: str | None = None
    kind_map: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        return f"{self.source}.{self.id}"

    def kind_at(self, assignment: Mapping[str, Any]) -> str:
        if self.kind_var is None:
            return self.kind
        value = str(assignment[self.kind_var])
        return dict(self.kind_map).get(value, value)


@dataclass(frozen=True)
class BoundStorage:
    source: str
    id: str
    level: str
    scope: str
    bytes: Expr
    when: Expr | None
    level_var: str | None = None  # family variable choosing the level (``level`` is empty)

    def level_at(self, assignment: Mapping[str, Any]) -> str:
        return self.level if self.level_var is None else str(assignment[self.level_var])


@dataclass(frozen=True)
class BoundOverlap:
    source: str
    between: tuple[str, ...]  # pattern:/kind: selectors or "<item key>.<id>"
    when: Expr | None
    inflight: Expr | None
    trips: Expr | None


@dataclass(frozen=True)
class Block:
    key: str
    template: str
    fingerprint: str
    constraints: tuple[BoundExpr, ...]
    cost_terms: tuple[BoundCost, ...]
    fixed: dict[str, int | str | bool]  # Template variables bound to constants
    uses: frozenset[str]  # family variables the block refers to
    transactions: tuple[BoundTx, ...] = ()
    storage: tuple[BoundStorage, ...] = ()
    overlaps: tuple[BoundOverlap, ...] = ()

    def exprs(self) -> list[Expr]:
        out = [c.expr for c in self.constraints]
        for t in self.transactions:
            out += [t.count, *t.amounts.values(), *([t.when] if t.when else [])]
        out += [x for s in self.storage for x in (s.bytes, s.when) if x is not None]
        out += [x for o in self.overlaps for x in (o.when, o.inflight, o.trips) if x is not None]
        return out


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
    eff = _Effects(t, item, rw, scale, family_vars, where)
    problems = [f"variable {n} is not bound" for n in unbound]
    problems += [f"{p} is used but not bound" for p in sorted(rw.missing)]
    if problems:
        raise SchemaError(f"{where}: " + "; ".join(problems))
    constraints = tuple(
        BoundExpr(item.key, c.id, kind, c.cls, c.meaning, _to_expr(n, c.expr, family_vars, where))
        for kind, c, n in trees
    )
    txs = eff.transactions()
    costs = _legacy_costs(t, txs)
    block = Block(
        key=item.key,
        template=item.template,
        fingerprint=fingerprint(item, t, family_vars),
        constraints=constraints,
        cost_terms=tuple(costs),
        fixed=fixed,
        uses=frozenset(),
        transactions=txs,
        storage=eff.storage(),
        overlaps=eff.overlaps(),
    )
    uses = frozenset[str]().union(*(e.free_vars for e in block.exprs()))
    return replace(block, uses=uses)


class _Effects:
    """Rewrites a Template's effects through a chain item's bindings."""

    def __init__(
        self,
        t: Template,
        item: ChainItem,
        rw: _Rewriter,
        scale: ast.expr | None,
        family_vars: frozenset[str],
        where: str,
    ) -> None:
        self.t, self.item, self.rw, self.scale = t, item, rw, scale
        self.family_vars, self.where = family_vars, where
        # Rewrite eagerly so that missing p.* bindings are reported with the constraints'.
        e = t.effects
        self._tx = [
            (tx, _rewrite(tx.count, rw), {k: _rewrite(v, rw) for k, v in tx.amounts.items()},
             _rewrite(tx.when, rw) if tx.when else None)
            for tx in e.transactions
        ]  # fmt: skip
        self._st = [
            (s, _rewrite(s.bytes, rw), _rewrite(s.when, rw) if s.when else None) for s in e.storage
        ]
        self._ov = [
            (o, *(_rewrite(x, rw) if x else None for x in (o.when, o.inflight, o.trips)))
            for o in e.overlap
        ]

    def _expr(self, node: ast.expr | None, src: str | None) -> Expr | None:
        if node is None or src is None:
            return None
        return _to_expr(node, src, self.family_vars, self.where)

    def transactions(self) -> tuple[BoundTx, ...]:
        out: list[BoundTx] = []
        for tx, count, amounts, when in self._tx:
            if self.scale is not None:
                count = ast.BinOp(count, ast.Mult(), copy.deepcopy(self.scale))
            bound = _to_expr(count, tx.count, self.family_vars, self.where)
            out.append(
                BoundTx(
                    source=self.item.key,
                    id=tx.id,
                    pattern=tx.pattern,
                    kind=self._kind(tx.kind, tx.kind_map),
                    kind_var=self._var(tx.kind),
                    kind_map=tuple(sorted(tx.kind_map.items())),
                    count=bound,
                    amounts={
                        k: _to_expr(n, tx.amounts[k], self.family_vars, self.where)
                        for k, n in amounts.items()
                    },
                    when=self._expr(when, tx.when),
                )
            )
        return tuple(out)

    def storage(self) -> tuple[BoundStorage, ...]:
        return tuple(
            BoundStorage(
                source=self.item.key,
                id=s.id,
                level=self._level(s.level),
                level_var=self._var(s.level),
                scope=s.scope,
                bytes=_to_expr(n, s.bytes, self.family_vars, self.where),
                when=self._expr(when, s.when),
            )
            for s, n, when in self._st
        )

    def _var(self, name: str) -> str | None:
        """The family variable a kind or level name is bound to, if any."""
        value = self.item.bindings.get(name) if self._names(name) else None
        return value if isinstance(value, str) and value in self.family_vars else None

    def _kind(self, kind: str, kind_map: dict[str, str]) -> str:
        if not self._names(kind):
            return kind
        if self._var(kind):
            return ""
        value = self._constant(kind, "transaction kind")
        return kind_map.get(value, value)

    def _names(self, name: str) -> bool:
        return name in {v.name for v in self.t.variables} or name.startswith(PARAM_PREFIX)

    def _constant(self, name: str, what: str) -> str:
        value = self.item.bindings.get(name)
        if not isinstance(value, str) or value in self.family_vars:
            raise SchemaError(f"{self.where}: {what} {name} needs a constant name binding")
        return value

    def _level(self, level: str) -> str:
        if not self._names(level):
            return level
        return "" if self._var(level) else self._constant(level, "storage level")

    def overlaps(self) -> tuple[BoundOverlap, ...]:
        local = {tx.id for tx in self.t.effects.transactions}

        def qualify(sel: str) -> str:
            if sel.startswith((SELECT_PATTERN, SELECT_KIND)) or sel not in local:
                return sel
            return f"{self.item.key}.{sel}"

        return tuple(
            BoundOverlap(
                source=self.item.key,
                between=tuple(qualify(s) for s in o.between),
                when=self._expr(when, o.when),
                inflight=self._expr(inflight, o.inflight),
                trips=self._expr(trips, o.trips),
            )
            for o, when, inflight, trips in self._ov
        )


def _legacy_costs(t: Template, txs: tuple[BoundTx, ...]) -> list[BoundCost]:
    """The ``unit:<u>`` transactions as cost terms, indexed like ``Template.cost``."""
    legacy = [x for x in txs if x.kind.startswith(LEGACY_KIND_PREFIX)]
    return [
        BoundCost(x.source, i, spec.unit, spec.overlap, x.count)
        for i, (x, spec) in enumerate(zip(legacy, t.cost, strict=True))
    ]
