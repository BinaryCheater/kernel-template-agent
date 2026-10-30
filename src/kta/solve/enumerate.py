"""Finite-domain enumeration with constraint pruning (decision D248).

Depth-first over the family variables: list-domain variables first, then range domains,
each group by increasing size. A constraint is checked as soon as its last variable is
assigned. For a range variable, a single comparison that is affine in it (the other
variables fixed) narrows its range to an interval before the values are tried; every
value is still checked against every constraint, so narrowing never admits a point.
Excluded points are counted per constraint, so feasible + pruned = space when the search
is exhaustive. ``max_nodes`` bounds the search; ``rng`` shuffles value order (sampling).
"""

from __future__ import annotations

import ast
import math
import random
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.expr import Expr
from kta.core.schemas.knowledge import RangeSpec, VarSpec
from kta.formulation import BoundExpr, Formulation

_AFFINE_CMP = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq)


def domain_size(v: VarSpec) -> int:
    d = v.domain
    if not isinstance(d, RangeSpec):
        return len(d)
    if d.hi < d.lo:
        return 0
    if d.pow2:
        return sum(1 for _ in range_values(d, d.lo, d.hi))
    return (d.hi - d.lo) // d.step + 1


def range_values(d: RangeSpec, lo: int, hi: int) -> Iterator[int]:
    """Values of ``d`` within [lo, hi], ascending."""
    lo, hi = max(lo, d.lo), min(hi, d.hi)
    if d.pow2:
        x = 1
        while x <= hi:
            if x >= lo and (x - d.lo) % d.step == 0:
                yield x
            x *= 2
        return
    first = d.lo + -(-(lo - d.lo) // d.step) * d.step
    yield from range(first, hi + 1, d.step)


def space_size(f: Formulation) -> int:
    return math.prod(domain_size(v) for v in f.variables.values())


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _affine(node: ast.expr, x: str) -> bool:
    if x not in _names(node):
        return True
    match node:
        case ast.Name():
            return True
        case ast.BinOp(left=left, op=ast.Add() | ast.Sub(), right=right):
            return _affine(left, x) and _affine(right, x)
        case ast.BinOp(left=left, op=ast.Mult(), right=right):
            return (x not in _names(left) and _affine(right, x)) or (
                x not in _names(right) and _affine(left, x)
            )
        case ast.BinOp(left=left, op=ast.Div(), right=right):
            return x not in _names(right) and _affine(left, x)
        case ast.UnaryOp(op=ast.USub() | ast.UAdd(), operand=operand):
            return _affine(operand, x)
        case _:
            return False


@dataclass(frozen=True)
class _Narrower:
    """``lhs op rhs`` with both sides affine in one range variable."""

    check: BoundExpr
    op: type[ast.cmpop]
    lhs: Expr
    rhs: Expr


def _narrower(c: BoundExpr, x: str, declared: set[str]) -> _Narrower | None:
    tree = ast.parse(c.expr.src.strip(), mode="eval").body
    if not isinstance(tree, ast.Compare) or len(tree.ops) != 1:
        return None
    op = type(tree.ops[0])
    left, right = tree.left, tree.comparators[0]
    if op not in _AFFINE_CMP or not (_affine(left, x) and _affine(right, x)):
        return None
    return _Narrower(
        c, op, Expr.parse(ast.unparse(left), declared), Expr.parse(ast.unparse(right), declared)
    )


@dataclass
class EnumStats:
    feasible: int = 0
    pruned: int = 0
    nodes: int = 0
    exhaustive: bool = True
    pruned_by: dict[str, int] = field(default_factory=lambda: dict[str, int]())

    def prune(self, label: str, count: int) -> None:
        self.pruned += count
        self.pruned_by[label] = self.pruned_by.get(label, 0) + count


class _Stop(Exception):
    """The node budget is spent."""


@dataclass
class _Search:
    f: Formulation
    shape: Mapping[str, Any]
    hw: Mapping[str, Any]
    visit: Callable[[dict[str, Any]], None]
    max_nodes: int | None
    rng: random.Random | None
    order: list[str] = field(default_factory=lambda: list[str]())
    checks: list[list[BoundExpr]] = field(default_factory=lambda: list[list[BoundExpr]]())
    narrowers: list[list[_Narrower]] = field(default_factory=lambda: list[list[_Narrower]]())
    rest: list[int] = field(default_factory=lambda: list[int]())  # points below depth i
    stats: EnumStats = field(default_factory=EnumStats)

    def plan(self) -> list[BoundExpr]:
        """Variable order and per-depth checks; returns the constant constraints."""
        vs = self.f.variables
        lists = sorted(
            (n for n, v in vs.items() if not isinstance(v.domain, RangeSpec)),
            key=lambda n: (domain_size(vs[n]), n),
        )
        ranges = sorted(
            (n for n, v in vs.items() if isinstance(v.domain, RangeSpec)),
            key=lambda n: (domain_size(vs[n]), n),
        )
        self.order = lists + ranges
        pos = {n: i for i, n in enumerate(self.order)}
        self.checks = [[] for _ in self.order]
        self.narrowers = [[] for _ in self.order]
        constant: list[BoundExpr] = []
        declared = set(vs)
        for c in self.f.constraints:
            if not c.expr.free_vars:
                constant.append(c)
                continue
            depth = max(pos[v] for v in c.expr.free_vars)
            self.checks[depth].append(c)
            x = self.order[depth]
            if isinstance(vs[x].domain, RangeSpec) and (n := _narrower(c, x, declared)):
                self.narrowers[depth].append(n)
        sizes = [domain_size(vs[n]) for n in self.order]
        self.rest = [math.prod(sizes[i + 1 :]) for i in range(len(sizes))]
        return constant

    def _ok(self, c: BoundExpr, env: Mapping[str, Any]) -> bool:
        return bool(c.expr.eval(env, hw=self.hw, case=self.shape))

    def _interval(self, n: _Narrower, x: str, env: dict[str, Any], lo: int) -> tuple[float, float]:
        def g(v: int) -> float:
            e = {**env, x: v}
            return float(n.lhs.eval(e, hw=self.hw, case=self.shape)) - float(
                n.rhs.eval(e, hw=self.hw, case=self.shape)
            )

        g0 = g(lo)
        a = g(lo + 1) - g0
        if a == 0:
            holds = self._ok(n.check, {**env, x: lo})
            return (-math.inf, math.inf) if holds else (math.inf, -math.inf)
        root = lo - g0 / a
        slack = 1e-9 * max(1.0, abs(root))
        if n.op is ast.Eq:
            return root - slack, root + slack
        below = (n.op in (ast.Lt, ast.LtE)) == (a > 0)  # g <= 0 holds below the root
        return (-math.inf, root + slack) if below else (root - slack, math.inf)

    def _values(self, depth: int, env: dict[str, Any]) -> list[Any]:
        x = self.order[depth]
        d = self.f.variables[x].domain
        if not isinstance(d, RangeSpec):
            values: list[Any] = list(d)
        else:
            lo, hi = float(d.lo), float(d.hi)
            cutters: list[str] = []
            for n in self.narrowers[depth]:
                a, b = self._interval(n, x, env, d.lo)
                if a > lo or b < hi:
                    cutters.append(n.check.label)
                lo, hi = max(lo, a), min(hi, b)
            kept = list(range_values(d, math.ceil(lo), math.floor(hi))) if lo <= hi else []
            excluded = domain_size(self.f.variables[x]) - len(kept)
            if excluded:
                # Attributed to the first constraint that narrowed the range.
                self.stats.prune(cutters[0] if cutters else "interval", excluded * self.rest[depth])
            values = kept
        if self.rng is not None:
            self.rng.shuffle(values)
        return values

    def descend(self, depth: int, env: dict[str, Any]) -> None:
        if depth == len(self.order):
            self.stats.feasible += 1
            self.visit(dict(env))
            return
        x = self.order[depth]
        for value in self._values(depth, env):
            self.stats.nodes += 1
            if self.max_nodes is not None and self.stats.nodes > self.max_nodes:
                raise _Stop
            env[x] = value
            failed = next((c for c in self.checks[depth] if not self._ok(c, env)), None)
            if failed is not None:
                self.stats.prune(failed.label, self.rest[depth])
            else:
                self.descend(depth + 1, env)
            del env[x]


def enumerate_space(
    f: Formulation,
    shape: Mapping[str, Any],
    hw: Mapping[str, Any],
    visit: Callable[[dict[str, Any]], None],
    max_nodes: int | None = None,
    rng: random.Random | None = None,
) -> EnumStats:
    """Call ``visit`` on every feasible assignment (until ``max_nodes`` nodes)."""
    s = _Search(f, shape, hw, visit, max_nodes, rng)
    constant = s.plan()
    total = space_size(f)
    for c in constant:
        if not s._ok(c, {}):
            s.stats.prune(c.label, total)
            return s.stats
    if not s.order:
        s.stats.feasible = 1
        visit({})
        return s.stats
    try:
        s.descend(0, {})
    except _Stop:
        s.stats.exhaustive = False
    return s.stats
