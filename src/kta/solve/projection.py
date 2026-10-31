"""The steady-state projection compiler (WP17, folded into WP19; docs/design/02 §5.1, §5.2).

From a formulation's bound effects, a FormulationPlan's menus and statuses, and the
hardware model, build the MIP of L1's projection at peak efficiency::

    minimise  T + serial
    T >= busy_r                      for every resource r          (ResMII, epigraph max)
    T >= u_a + u_b,  u_g >= t_g,r    for every overlap that does not hold (pair bound)
    T >= trips * (lat_a + lat_b) / inflight   for every overlap that holds (RecMII)
    storage bytes (+ reserved) <= wall limit                        (walls)
    every constraint of the formulation

Every quantity is an *atom*: one transaction's (storage item's, overlap's, constraint's)
contribution as a function of the few variables it reads. The rules of 02 §5.2, applied
here so that the agent writes none of it:

- one-hot menus: every open variable is ``sum_v z_{x,v} = 1``; an atom over one variable
  is linear in its z;
- menu products and table terms: an atom over k >= 2 variables is tabulated over their
  joint menu, one indicator per combination linked to the menus (``sum_{c: c_i = v} y_c =
  z_{i,v}``); k = 2 is a menu product, k >= 3 a table term; non-algebraic effects
  (``tbl.*``, ``min``, ``ceil``) are evaluated inside the table;
- integer ceil: an atom too large to tabulate whose expression is linear up to ``ceil``
  becomes a linear row, each ``ceil(L)`` an integer column ``n >= L``;
- outer enumeration: what fits neither (an overlap whose truth depends on a variable, a
  kind selector over a variable operation class, an atom over too many values) moves its
  variables outside; the MIP is solved once per outer point.

Efficiencies and interaction terms are relaxed (eff = 1, gamma dropped): the objective
is L1's projection at peak, a lower bound of L1. ``projection_us`` recomputes it at a point
(translation correctness is the compiler's, 02 §5.2). The MIP <-> L2 consistency check
is ``kta.solve.consistency`` (WP16's simulator on the other side).
"""

from __future__ import annotations

import ast
import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import highspy

from kta.core.errors import NotSupported
from kta.core.expr import Expr, ExprError
from kta.core.schemas.effects import SELECT_KIND, SELECT_PATTERN
from kta.formulation import Formulation, evaluate_effects
from kta.formulation.blocks import BoundExpr, BoundOverlap, BoundStorage, BoundTx
from kta.formulation.pricing import row_term
from kta.models import MachineModel
from kta.models.l1 import US_PER_S, compose, price_parts
from kta.solve.enumerate import domain_size, enumerate_space, range_values
from kta.solve.mip import FEASIBLE, GAP_ZERO, INF, MENU_MAX, Linear, NotExportable, _add_row, _Lin

TABLE_MAX = 4096  # default of knob formulation.table_max_entries
OUTER_MAX = 256  # default of knob formulation.outer_max_points
CEIL_PREFIX = "__ceil"
Combo = tuple[Any, ...]
Value = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True)
class Atom:
    """One quantity of the projection over the open variables in ``support``."""

    rule: str  # resmii | serial | pair | recmii | wall | constraint
    row: str  # what it aggregates into: resource, bound, wall or constraint label
    source: str  # transaction key, storage key, overlap name or constraint label
    support: tuple[str, ...]
    value: Value
    linear: Expr | None = None  # the same quantity as an expression (the linear path)
    scale: float = 1.0  # us per unit of ``linear``


@dataclass
class ProjectionOutcome:
    """Assignments in objective order across outer points, with the compiler's counts."""

    assignments: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    objective_us: list[float] = field(default_factory=lambda: list[float]())
    status: str = "not_run"
    first_optimal: bool = False
    solves: int = 0
    runtime_s: float = 0.0
    outer: tuple[str, ...] = ()
    outer_points: int = 0
    menus: dict[str, int] = field(default_factory=lambda: dict[str, int]())  # var -> size
    rules: dict[str, int] = field(default_factory=lambda: dict[str, int]())  # rule -> uses
    notes: list[str] = field(default_factory=lambda: list[str]())


class _NeedOuter(Exception):
    def __init__(self, names: Iterable[str], why: str) -> None:
        super().__init__(why)
        self.names = set(names)
        self.why = why


# ---- menus (F4) -----------------------------------------------------------------------


def _declared(f: Formulation, x: str) -> list[Any]:
    d = f.variables[x].domain
    return list(d) if isinstance(d, list) else list(range_values(d, d.lo, d.hi))


def _narrowed(
    f: Formulation,
    x: str,
    menus: Mapping[str, list[Any]],
    shape: Mapping[str, Any],
    hw: Mapping[str, Any],
) -> list[Any]:
    """Values of a large range variable reachable under the constraints tying it to small
    menus (e.g. KT = ceil_div(K, BK)); the declared domain when none does."""
    cons = [
        c
        for c in f.constraints
        if x in c.expr.free_vars and all(v in menus for v in c.expr.free_vars - {x})
    ]
    if not cons:
        return _declared(f, x)
    sub_vars = {x: f.variables[x]}
    for v in sorted({v for c in cons for v in c.expr.free_vars} - {x}):
        sub_vars[v] = f.variables[v].model_copy(update={"domain": list(menus[v])})
    seen: set[Any] = set()
    enumerate_space(replace(f, variables=sub_vars, constraints=cons), shape, hw,
                    lambda a: seen.add(a[x]))  # fmt: skip
    return sorted(seen)


def menus_of(
    f: Formulation,
    shape: Mapping[str, Any],
    hw: Mapping[str, Any],
    fixed: Mapping[str, Any] | None = None,
    menus: Mapping[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    """Menus: the plan's, else the declared domain; a fixed variable has one value; a range
    above MENU_MAX is narrowed by the constraints that pin it."""
    out: dict[str, list[Any]] = {}
    big: list[str] = []
    for x in f.variables:
        if fixed and x in fixed:
            out[x] = [fixed[x]]
        elif menus and x in menus:
            out[x] = list(menus[x])
        elif domain_size(f.variables[x]) > MENU_MAX:
            big.append(x)
        else:
            out[x] = _declared(f, x)
    for x in big:
        out[x] = _narrowed(f, x, out, shape, hw)
        if len(out[x]) > MENU_MAX:
            raise NotExportable(f"variable {x} keeps {len(out[x])} values after narrowing")
    return out


# ---- atoms ----------------------------------------------------------------------------


def _vars(*exprs: Expr | None) -> set[str]:
    return {v for e in exprs if e is not None for v in e.free_vars}


def _members(sel: str, txs: list[BoundTx], fixed: Mapping[str, Any]) -> list[BoundTx]:
    if sel.startswith(SELECT_PATTERN):
        return [t for t in txs if t.pattern == sel[len(SELECT_PATTERN) :]]
    if sel.startswith(SELECT_KIND):
        loose = {t.kind_var for t in txs if t.kind_var and t.kind_var not in fixed}
        if loose:
            raise _NeedOuter(loose, f"selector {sel} over a variable operation class")
        return [t for t in txs if t.kind_at(fixed) == sel[len(SELECT_KIND) :]]
    return [t for t in txs if t.key == sel]


@dataclass
class _Ctx:
    f: Formulation
    mm: MachineModel
    shape: Mapping[str, Any]
    fixed: Mapping[str, Any]
    menus: Mapping[str, list[Any]]
    cache: dict[tuple[str, Combo], dict[str, float]] = field(
        default_factory=lambda: dict[tuple[str, Combo], dict[str, float]]()
    )

    def env(self, a: Mapping[str, Any]) -> dict[str, Any]:
        return {**self.fixed, **a}

    def ev(self, e: Expr, a: Mapping[str, Any]) -> Any:
        return e.eval(self.env(a), hw=self.mm.hw, case=self.shape)

    def on(self, e: Expr | None, a: Mapping[str, Any]) -> bool:
        return True if e is None else bool(self.ev(e, a))

    def open(self, names: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(n for n in names if n not in self.fixed))

    def times(self, t: BoundTx, a: Mapping[str, Any]) -> dict[str, float]:
        """Time per resource (us, at peak) of ``t`` at ``a``; {} when it does not occur."""
        key = (t.key, tuple(sorted(a.items())))
        if key in self.cache:
            return self.cache[key]
        out: dict[str, float] = {}
        op = self.mm.op_class(t.kind_at(self.env(a)))
        if op is not None and self.on(t.when, a):  # unpriced classes: no time (D465)
            count = float(self.ev(t.count, a))
            amounts = {k: float(self.ev(v, a)) for k, v in t.amounts.items()}
            for row in op.rows:
                s = self.mm.time_s(row.resource, count * self.mm.evaluate(row.amount, amounts))
                if s:
                    out[row.resource] = out.get(row.resource, 0.0) + s * US_PER_S
        self.cache[key] = out
        return out

    def resources(self, t: BoundTx) -> set[str]:
        kinds = {t.kind_at({**self.fixed, **({t.kind_var: v} if t.kind_var else {})})
                 for v in (self.menus[t.kind_var] if t.kind_var else [None])}  # fmt: skip
        return {r.resource for k in kinds if (op := self.mm.op_class(k)) for r in op.rows}


def _linear_rows(c: _Ctx, t: BoundTx) -> dict[str, tuple[Expr, float]]:
    """Resource -> (work expression, us per unit) of an always-overlappable transaction."""
    op = c.mm.op_class(t.kind)
    if t.kind_var is not None or op is None or op.serial:
        return {}
    out: dict[str, tuple[Expr, float]] = {}
    for row in op.rows:
        seconds = c.mm.time_s(row.resource, 1.0)
        try:
            term = row_term(t, row.resource, row.amount, False, c.f)
        except NotSupported:
            continue  # a table term: tabulated or moved outward
        if seconds:
            out[row.resource] = (term.work, seconds * US_PER_S)
    return out


def _recurrence(c: _Ctx, o: BoundOverlap, groups: list[list[BoundTx]]) -> Value:
    def value(a: Mapping[str, Any]) -> float:
        lats: list[float] = []
        for g in groups[:2]:
            live = [c.mm.op_class(t.kind_at(c.env(a))) for t in g if c.times(t, a)]
            if not live or any(op is None or op.latency_s is None for op in live):
                return 0.0  # unmodelled latency: L1 reports it and drops the bound
            lats.append(min(op.latency_s for op in live if op and op.latency_s is not None))
        assert o.inflight is not None and o.trips is not None
        return float(c.ev(o.trips, a)) * sum(lats) / float(c.ev(o.inflight, a)) * US_PER_S

    return value


def _wall_bytes(c: _Ctx, s: BoundStorage, level: str, scope: str) -> Value:
    def value(a: Mapping[str, Any]) -> float:
        if s.scope != scope or s.level_at(c.env(a)) != level or not c.on(s.when, a):
            return 0.0
        return float(c.ev(s.bytes, a))

    return value


def _holds(c: _Ctx, k: BoundExpr) -> Value:
    return lambda a: 1.0 if c.ev(k.expr, a) else 0.0


def atoms(
    f: Formulation,
    mm: MachineModel,
    shape: Mapping[str, Any],
    menus: Mapping[str, list[Any]],
    fixed: Mapping[str, Any],
) -> list[Atom]:
    """The projection's atoms with the variables of ``fixed`` substituted."""
    c = _Ctx(f, mm, shape, fixed, menus)
    txs = list(f.transactions)
    serial_by: dict[str, list[Expr | None]] = {}
    for o in f.overlaps:
        if len(o.between) == 1:
            for t in _members(o.between[0], txs, fixed):
                serial_by.setdefault(t.key, []).append(o.when)

    def serial(t: BoundTx, a: Mapping[str, Any]) -> bool:
        op = mm.op_class(t.kind_at(c.env(a)))
        return bool(op and op.serial) or any(not c.on(w, a) for w in serial_by.get(t.key, []))

    def busy(t: BoundTx, r: str) -> Value:
        return lambda a: 0.0 if serial(t, a) else c.times(t, a).get(r, 0.0)

    def serial_time(t: BoundTx) -> Value:
        return lambda a: max(c.times(t, a).values(), default=0.0) if serial(t, a) else 0.0

    out: list[Atom] = []
    support: dict[str, tuple[str, ...]] = {}
    for t in txs:
        names = _vars(t.count, t.when, *t.amounts.values(), *serial_by.get(t.key, []))
        support[t.key] = s = c.open(names | ({t.kind_var} if t.kind_var else set()))
        lin = {} if t.key in serial_by else _linear_rows(c, t)
        for r in sorted(c.resources(t)):
            expr, scale = lin.get(r, (None, 1.0))
            out.append(Atom("resmii", r, t.key, s, busy(t, r), expr, scale))
        out.append(Atom("serial", "serial", t.key, s, serial_time(t)))
    for o in f.overlaps:
        if len(o.between) < 2:
            continue
        name = f"{o.source}:{'|'.join(o.between)}"
        if o.when is not None and (loose := c.open(o.when.free_vars)):
            raise _NeedOuter(loose, f"whether overlap {name} holds depends on {list(loose)}")
        groups = [_members(sel, txs, fixed) for sel in o.between]
        if not c.on(o.when, {}):
            for side, g in zip("ab", groups[:2], strict=False):
                res = sorted(set().union(*(c.resources(t) for t in g)))
                out += [
                    Atom("pair", f"{name}#{side}#{r}", t.key, support[t.key], busy(t, r))
                    for t in g
                    for r in res
                ]
        elif o.inflight is not None and o.trips is not None:
            names = _vars(o.inflight, o.trips) | {
                v for g in groups for t in g for v in support[t.key]
            }
            out.append(
                Atom("recmii", f"recurrence {name}", name, c.open(names), _recurrence(c, o, groups))
            )
    for w in mm.walls:
        for s in f.storage:
            names = _vars(s.bytes, s.when) | ({s.level_var} if s.level_var else set())
            value = _wall_bytes(c, s, w.spec.level, w.spec.scope)
            out.append(Atom("wall", w.spec.name, f"{s.source}.{s.id}", c.open(names), value))
    for k in f.constraints:
        out.append(Atom("constraint", k.label, k.label, c.open(k.expr.free_vars), _holds(c, k)))
    return out


# ---- the MIP --------------------------------------------------------------------------


@dataclass
class _Mip:
    h: highspy.Highs
    menus: dict[str, list[Any]]
    cols: dict[str, list[int]] = field(default_factory=lambda: dict[str, list[int]]())
    joint: dict[tuple[str, ...], dict[Combo, int]] = field(
        default_factory=lambda: dict[tuple[str, ...], dict[Combo, int]]()
    )
    rules: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    infeasible: str | None = None

    def count(self, rule: str) -> None:
        self.rules[rule] = self.rules.get(rule, 0) + 1

    def col(self, lo: float = 0.0, hi: float = INF, integer: bool = False) -> int:
        i = self.h.getNumCol()
        self.h.addVar(lo, hi)
        if integer:
            self.h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
        return i

    def indicators(self, support: tuple[str, ...]) -> dict[Combo, int]:
        """One column per combination of ``support``'s menus (k = 1: the menu itself)."""
        if len(support) == 1:
            x = support[0]
            return {(v,): col for v, col in zip(self.menus[x], self.cols[x], strict=True)}
        if support not in self.joint:
            combos = list(itertools.product(*(self.menus[x] for x in support)))
            ys = {cb: self.col(0.0, 1.0) for cb in combos}  # integral through the links
            for i, x in enumerate(support):
                for v, z in zip(self.menus[x], self.cols[x], strict=True):
                    idx = [ys[cb] for cb in combos if cb[i] == v]
                    _add_row(self.h, 0.0, 0.0, [*idx, z], [1.0] * len(idx) + [-1.0])
            self.joint[support] = ys
            self.count("menu_product" if len(support) == 2 else "table_term")
        return self.joint[support]

    def forbid(self, support: tuple[str, ...], combo: Combo) -> None:
        cols = [self.cols[x][self.menus[x].index(v)] for x, v in zip(support, combo, strict=True)]
        _add_row(self.h, -INF, len(cols) - 1.0, cols, [1.0] * len(cols))
        self.count("undefined_excluded")

    def expand(self, coeffs: Mapping[str, float]) -> tuple[list[int], list[float]]:
        idx: list[int] = []
        val: list[float] = []
        for name, cf in coeffs.items():
            for col, value in zip(self.cols[name], self.menus[name], strict=True):
                idx.append(col)
                val.append(cf * float(value))
        return idx, val


Form = tuple[list[int], list[float], float]  # columns, coefficients, constant


def _size(menus: Mapping[str, list[Any]], support: Iterable[str]) -> int:
    return math.prod(len(menus[x]) for x in support)


def _tabulate(m: _Mip, a: Atom) -> Form:
    if not a.support:
        return [], [], a.value({})
    idx: list[int] = []
    val: list[float] = []
    for cb, col in m.indicators(a.support).items():
        try:
            v = a.value(dict(zip(a.support, cb, strict=True)))
        except ExprError:
            m.forbid(a.support, cb)  # undefined there (e.g. no CTA fits): not a point
            continue
        if v:
            idx.append(col)
            val.append(v)
    return idx, val, 0.0


class _Subst(ast.NodeTransformer):
    def __init__(self, fixed: Mapping[str, Any]) -> None:
        self.fixed = fixed

    def visit_Name(self, node: ast.Name) -> ast.expr:
        return ast.Constant(self.fixed[node.id]) if node.id in self.fixed else node


class _CeilLin(_Lin):
    """``_Lin`` with each ``ceil(L)`` / ``ceil_div(L, k)`` over variables as an integer
    column n >= L (valid where n only bounds the objective from below)."""

    def __init__(self, m: _Mip, shape: Mapping[str, Any], hw: Mapping[str, Any]) -> None:
        super().__init__(set(m.cols), shape, hw)
        self.m = m

    def tree(self, src: str, fixed: Mapping[str, Any]) -> ast.expr:
        node = _Subst(fixed).visit(ast.parse(src.strip(), mode="eval").body)
        return ast.fix_missing_locations(node)

    def form(self, node: ast.expr) -> Linear:
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        match node:
            case ast.Call(func=ast.Name(id="ceil" | "ceil_div" as fn), args=args) if (
                names & self.variables
            ):
                coeffs, const = super().form(args[0])
                k = self.const(args[1]) if fn == "ceil_div" and len(args) == 2 else 1.0
                name = f"{CEIL_PREFIX}{len(self.m.cols)}"
                self.m.cols[name] = [self.m.col(integer=True)]
                self.m.menus[name] = [1]
                idx, val = self.m.expand({x: v / k for x, v in coeffs.items()})
                row = [self.m.cols[name][0], *idx]
                _add_row(self.m.h, const / k, INF, row, [1.0, *(-v for v in val)])
                self.m.count("integer_ceil")
                return {name: 1.0}, 0.0
            case _:
                return super().form(node)


def _build(
    f: Formulation,
    mm: MachineModel,
    shape: Mapping[str, Any],
    menus: dict[str, list[Any]],
    fixed: Mapping[str, Any],
    table_max: int,
) -> tuple[_Mip, int]:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)
    h.setOptionValue("random_seed", 0)
    m = _Mip(h, {x: v for x, v in menus.items() if x not in fixed})
    for x, values in list(m.menus.items()):
        m.cols[x] = [m.col(0.0, 1.0, integer=True) for _ in values]
        _add_row(h, 1.0, 1.0, m.cols[x], [1.0] * len(values))
        m.count("one_hot")
    t_col = m.col()
    lin = _CeilLin(m, shape, mm.hw)
    sums: dict[tuple[str, str], Form] = {}
    serial_cost: dict[int, float] = {t_col: 1.0}
    offset = 0.0  # constant serial time
    for a in atoms(f, mm, shape, menus, fixed):
        big = _size(m.menus, a.support) > table_max
        if a.rule == "constraint":
            _constraint(m, f.constraint(a.row), a, lin, fixed, big)
            continue
        if big and a.linear is None:
            raise _NeedOuter(a.support, f"{a.source} spans {_size(m.menus, a.support)} values")
        if big:
            assert a.linear is not None
            try:
                coeffs, const = lin.form(lin.tree(a.linear.src, fixed))
            except NotExportable as exc:
                raise _NeedOuter(a.support, f"{a.source} spans too many values: {exc}") from None
            idx, val = m.expand({k: v * a.scale for k, v in coeffs.items()})
            form: Form = (idx, val, const * a.scale)
        else:
            form = _tabulate(m, a)
        if a.rule == "serial":
            for i, v in zip(form[0], form[1], strict=True):
                serial_cost[i] = serial_cost.get(i, 0.0) + v
            offset += form[2]
            if form[0] or form[2]:
                m.count("serial_term")
            continue
        idx, val, const = sums.get((a.rule, a.row), ([], [], 0.0))
        sums[(a.rule, a.row)] = (idx + form[0], val + form[1], const + form[2])
    for i, cost in serial_cost.items():
        h.changeColCost(i, cost)
    h.changeObjectiveOffset(offset)
    pairs: dict[str, dict[str, int]] = {}
    for (rule, row), (idx, val, const) in sums.items():
        if rule in ("resmii", "recmii"):
            _add_row(h, const, INF, [t_col, *idx], [1.0, *(-v for v in val)])  # T >= sum
            m.count(rule)
        elif rule == "pair":
            name, side, _ = row.rsplit("#", 2)
            u = pairs.setdefault(name, {}).setdefault(side, m.col())
            _add_row(h, const, INF, [u, *idx], [1.0, *(-v for v in val)])  # u_g >= t_g,r
        elif rule == "wall":
            w = next(w for w in mm.walls if w.spec.name == row)
            if not idx and not const:
                continue
            limit = w.limit - (w.reserved if w.spec.cliff == "lower_occupancy" else 0.0)
            if not idx:
                m.infeasible = m.infeasible or (row if const > limit else None)
                continue
            _add_row(h, -INF, limit - const, idx, val)
            m.count("wall")
    for sides in pairs.values():
        us = list(sides.values())
        _add_row(h, 0.0, INF, [t_col, *us], [1.0, *([-1.0] * len(us))])  # T >= u_a + u_b
        m.count("pair")
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    return m, t_col


def _constraint(
    m: _Mip, k: BoundExpr, a: Atom, lin: _CeilLin, fixed: Mapping[str, Any], big: bool
) -> None:
    if not a.support:
        if not a.value({}):
            m.infeasible = a.row
        return
    try:
        rows = lin.rows(lin.tree(k.expr.src, fixed))
    except NotExportable:
        if big:
            raise _NeedOuter(
                a.support, f"{a.row} spans {_size(m.menus, a.support)} values"
            ) from None
        rows = None
    if rows is not None:
        for (coeffs, const), op in rows:
            idx, val = m.expand(coeffs)
            lo, hi = -INF, INF
            if op in (ast.LtE, ast.Lt, ast.Eq):
                hi = -const - (1e-9 if op is ast.Lt else 0.0)
            if op in (ast.GtE, ast.Gt, ast.Eq):
                lo = -const + (1e-9 if op is ast.Gt else 0.0)
            _add_row(m.h, lo, hi, idx, val)
        m.count("constraint_linear")
        return
    for cb in itertools.product(*(m.menus[x] for x in a.support)):
        try:
            ok = a.value(dict(zip(a.support, cb, strict=True)))
        except ExprError:
            ok = False
        if not ok:
            m.forbid(a.support, cb)
    m.count("constraint_table")


def _shortlist(
    m: _Mip, n: int, time_limit_s: float
) -> tuple[bool, str, list[tuple[float, dict[str, Any]]], float]:
    """Up to ``n`` distinct solutions in objective order (no-good cuts)."""
    h = m.h
    out: list[tuple[float, dict[str, Any]]] = []
    status_text = "not_run"
    optimal_first = False
    start = h.getRunTime()
    picks = [x for x in m.cols if not x.startswith(CEIL_PREFIX)]
    while len(out) < n and (left := time_limit_s - (h.getRunTime() - start)) > 0:
        h.setOptionValue("time_limit", left)
        h.run()
        status = h.getModelStatus()
        info = h.getInfo()
        optimal = status == highspy.HighsModelStatus.kOptimal
        if status_text == "not_run":
            status_text = h.modelStatusToString(status)
            optimal_first = optimal and float(info.mip_gap) <= GAP_ZERO
        if not optimal and info.primal_solution_status != FEASIBLE:
            break
        x = list(h.getSolution().col_value)
        chosen: dict[str, Any] = {}
        cols: list[int] = []
        for name in picks:
            i = max(range(len(m.cols[name])), key=lambda j: x[m.cols[name][j]])
            chosen[name] = m.menus[name][i]
            cols.append(m.cols[name][i])
        out.append((float(info.objective_function_value), chosen))
        if not cols or not optimal:
            break
        _add_row(h, -INF, len(cols) - 1.0, cols, [1.0] * len(cols))  # no-good cut
    return optimal_first, status_text, out, h.getRunTime() - start


def outer_vars(
    f: Formulation,
    mm: MachineModel,
    shape: Mapping[str, Any],
    menus: dict[str, list[Any]],
    outer: set[str] | None = None,
    table_max: int = TABLE_MAX,
) -> tuple[set[str], list[str]]:
    """Grow ``outer`` until the first outer point compiles; the reasons, in order."""
    why: list[str] = []
    outer = set(outer or ())
    while True:
        try:
            _build(f, mm, shape, menus, {x: menus[x][0] for x in outer}, table_max)
            return outer, why
        except _NeedOuter as exc:
            grow = exc.names - outer
            if not grow:
                raise NotExportable(exc.why) from None
            if "spans" in exc.why:
                grow = {max(grow, key=lambda x: (len(menus[x]), x))}
            why.append(f"outer {sorted(grow)}: {exc.why}")
            outer = outer | grow


def compile_and_solve(
    f: Formulation,
    shape: Mapping[str, Any],
    mm: MachineModel,
    *,
    fixed: Mapping[str, Any] | None = None,
    menus: Mapping[str, list[Any]] | None = None,
    outer: Iterable[str] = (),
    shortlist: int = 16,
    time_limit_s: float = 60.0,
    table_max: int = TABLE_MAX,
    outer_max: int = OUTER_MAX,
) -> ProjectionOutcome:
    """Compile the projection with a plan's fixed values, menus and outer variables, and
    solve it; raises NotExportable when outer enumeration would exceed ``outer_max``."""
    all_menus = menus_of(f, shape, mm.hw, fixed, menus)
    out_vars, why = outer_vars(f, mm, shape, all_menus, set(outer), table_max)
    names = tuple(sorted(out_vars))
    points = math.prod(len(all_menus[x]) for x in names)
    if points > outer_max:
        raise NotExportable(f"outer enumeration over {list(names)}: {points} > {outer_max}")
    res = ProjectionOutcome(outer=names, outer_points=points, notes=why)
    res.menus = {x: len(v) for x, v in all_menus.items()}
    found: list[tuple[float, dict[str, Any]]] = []
    optimal = True
    for values in itertools.product(*(all_menus[x] for x in names)):
        point = dict(zip(names, values, strict=True))
        try:
            m, _ = _build(f, mm, shape, all_menus, point, table_max)
        except _NeedOuter as exc:
            raise NotExportable(f"at outer point {point}: {exc.why}") from None
        for k, v in m.rules.items():
            res.rules[k] = max(res.rules.get(k, 0), v)
        if m.infeasible is not None:
            res.notes.append(f"outer point {point}: {m.infeasible} excludes it")
            continue
        left = max(time_limit_s - res.runtime_s, 1e-3)
        first, status, sols, runtime = _shortlist(m, shortlist, left)
        res.solves += 1
        res.runtime_s += runtime
        optimal = optimal and (first or not sols)
        if not first and sols:
            res.status = status
        found += [(obj, {**point, **a}) for obj, a in sols]
    found.sort(key=lambda p: p[0])
    found = found[:shortlist]
    res.assignments = [a for _, a in found]
    res.objective_us = [o for o, _ in found]
    res.first_optimal = optimal and bool(found)
    if res.status == "not_run":
        res.status = "Optimal" if found else "Infeasible"
    return res


def projection_us(
    f: Formulation, assignment: Mapping[str, Any], shape: Mapping[str, Any], mm: MachineModel
) -> float:
    """L1's projection at one point at peak (eff = 1, no interaction terms): what the
    compiled MIP's objective equals there (translation correctness)."""
    parts = price_parts(evaluate_effects(f, assignment, shape, mm.hw), mm)[0]
    return compose(parts).total_us
