"""Export of the linear subset to HiGHS (WP5 allowed simplification, decision D249).

Every family variable is a finite menu encoded one-hot (sum of binaries = 1, value =
sum of value * binary); range domains are expanded when they have at most MENU_MAX
values. Constraints are linear comparisons joined by ``and``; cost terms are linear.
The objective is the model's pre-compile total without interaction terms:
T >= overlappable time of each unit, plus the serial time. Work per unit comes from
``kta.formulation.work_terms``: legacy cost terms and each priced transaction's
reservation-table use (D472); L1's pair and recurrence bounds are relaxed (the MIP's
value is a lower bound of L1; the shortlist is re-priced by L1). A shortlist is obtained by
re-solving with a no-good cut on each solution found.

Since WP19/WP17 this export of the agent-written (hand) formulation is the comparison arm
of D-T1: the compiled projection (``kta.solve.projection``) is the system's own path;
``solve(..., projection=...)`` selects it, and this path stays callable for comparison.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import highspy
import numpy as np

from kta.core.errors import NotSupported
from kta.core.expr import Expr
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.knowledge import RangeSpec, VarSpec
from kta.formulation import BoundCost, Formulation, work_terms
from kta.models import MachineModel, UnitModel, machine, unit_time_us
from kta.solve.enumerate import domain_size, range_values

MENU_MAX = 4096
COMPARISON_ARM = "agent-written formulation (comparison arm, D-T1)"
INF = highspy.kHighsInf
# HiGHS reports a relative gap; with mip_rel_gap = mip_abs_gap = 0 an "Optimal" run can
# still show rounding noise near a zero objective, so this is what counts as gap 0.
GAP_ZERO = 1e-6
FEASIBLE = 2  # HiGHS solution status kSolutionStatusFeasible
Linear = tuple[dict[str, float], float]  # coefficients over family variables, constant


class NotExportable(NotSupported):
    """The formulation is outside the exportable subset; enumeration is used instead."""


@dataclass
class MipOutcome:
    assignments: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    status: str = "not_run"
    gap: float | None = None
    first_optimal: bool = False
    solves: int = 0
    runtime_s: float = 0.0


def _menu(v: VarSpec) -> list[Any]:
    d = v.domain
    if not isinstance(d, RangeSpec):
        return list(d)
    if domain_size(v) > MENU_MAX:
        raise NotExportable(f"variable {v.name} has more than {MENU_MAX} values")
    return list(range_values(d, d.lo, d.hi))


def _numeric(value: Any, var: str) -> float:
    if isinstance(value, str):
        raise NotExportable(f"variable {var} is used arithmetically but has string values")
    return float(value)


class _Lin:
    def __init__(self, variables: set[str], shape: Mapping[str, Any], hw: Mapping[str, Any]):
        self.variables = variables
        self.shape = shape
        self.hw = hw

    def const(self, node: ast.expr) -> float:
        return float(Expr.parse(ast.unparse(node), set()).eval({}, hw=self.hw, case=self.shape))

    def form(self, node: ast.expr) -> Linear:
        if not any(isinstance(n, ast.Name) and n.id in self.variables for n in ast.walk(node)):
            return {}, self.const(node)
        match node:
            case ast.Name(id=name):
                return {name: 1.0}, 0.0
            case ast.BinOp(left=left, op=ast.Add() | ast.Sub() as op, right=right):
                (a, ca), (b, cb) = self.form(left), self.form(right)
                sign = 1.0 if isinstance(op, ast.Add) else -1.0
                coeffs = dict(a)
                for k, v in b.items():
                    coeffs[k] = coeffs.get(k, 0.0) + sign * v
                return coeffs, ca + sign * cb
            case ast.BinOp(left=left, op=ast.Mult(), right=right):
                (a, ca), (b, cb) = self.form(left), self.form(right)
                if a and b:
                    raise NotExportable(f"product of variables in {ast.unparse(node)}")
                coeffs, const, scale = (b, cb, ca) if not a else (a, ca, cb)
                return {k: v * scale for k, v in coeffs.items()}, const * scale
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                a, ca = self.form(operand)
                return {k: -v for k, v in a.items()}, -ca
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return self.form(operand)
            case _:
                raise NotExportable(f"{ast.unparse(node)} is not linear")

    def rows(self, node: ast.expr) -> list[tuple[Linear, type[ast.cmpop]]]:
        """``lhs - rhs`` forms with their comparison, for a conjunction of comparisons."""
        match node:
            case ast.BoolOp(op=ast.And(), values=values):
                return [r for v in values for r in self.rows(v)]
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                out: list[tuple[Linear, type[ast.cmpop]]] = []
                terms = [left, *comparators]
                for op, a, b in zip(ops, terms, terms[1:], strict=False):
                    (ca, ka), (cb, kb) = self.form(a), self.form(b)
                    coeffs = dict(ca)
                    for k, v in cb.items():
                        coeffs[k] = coeffs.get(k, 0.0) - v
                    out.append(((coeffs, ka - kb), type(op)))
                return out
            case _:
                raise NotExportable(f"{ast.unparse(node)} is not a linear comparison")


@dataclass
class _Model:
    h: highspy.Highs
    cols: dict[str, list[int]]  # variable -> binary column per menu value
    menus: dict[str, list[Any]]
    infeasible: str | None = None


def _add_row(h: highspy.Highs, lo: float, hi: float, idx: list[int], val: list[float]) -> None:
    h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=float))


def _expand(m: _Model, coeffs: Mapping[str, float]) -> tuple[list[int], list[float]]:
    idx: list[int] = []
    val: list[float] = []
    for name, c in coeffs.items():
        for col, value in zip(m.cols[name], m.menus[name], strict=True):
            idx.append(col)
            val.append(c * _numeric(value, name))
    return idx, val


def _constraint_rows(m: _Model, f: Formulation, lin: _Lin) -> None:
    for c in f.constraints:
        tree = ast.parse(c.expr.src.strip(), mode="eval").body
        if not c.expr.free_vars:
            if not c.expr.eval({}, hw=lin.hw, case=lin.shape):
                m.infeasible = c.label
            continue
        for (coeffs, const), op in lin.rows(tree):
            idx, val = _expand(m, coeffs)
            integral = all(float(v).is_integer() for v in [*val, const])
            delta = 1.0 if integral else 1e-9
            lo, hi = -INF, INF
            if op in (ast.LtE, ast.Lt, ast.Eq):
                hi = -const - (delta if op is ast.Lt else 0.0)
            if op in (ast.GtE, ast.Gt, ast.Eq):
                lo = -const + (delta if op is ast.Gt else 0.0)
            _add_row(m.h, lo, hi, idx, val)


def _terms(f: Formulation, hw: MachineModel) -> list[BoundCost]:
    try:
        terms = work_terms(f, hw)
    except NotSupported as exc:
        raise NotExportable(str(exc)) from exc
    unpriced = sorted({t.unit for t in terms if hw.rate(t.unit) is None})
    if unpriced:
        # Infinite capacity (D465) would drop these rows; enumeration reports them instead.
        raise NotExportable(f"resources without a peak: {unpriced}")
    return terms


def _objective(m: _Model, f: Formulation, lin: _Lin, model: UnitModel, hw: MachineModel) -> None:
    t_col = m.h.getNumCol()
    m.h.addVar(0.0, INF)
    m.h.changeColCost(t_col, 1.0)
    over: dict[str, tuple[dict[str, float], float]] = {}
    serial_cost: dict[int, float] = {}
    for term in _terms(f, hw):
        coeffs, const = lin.form(ast.parse(term.work.src.strip(), mode="eval").body)
        scale = unit_time_us(term.unit, 1.0, hw)
        if term.overlap == "serial":
            idx, val = _expand(m, {k: v * scale * model.serial_coef for k, v in coeffs.items()})
            for i, v in zip(idx, val, strict=True):
                serial_cost[i] = serial_cost.get(i, 0.0) + v
            continue
        acc, k0 = over.get(term.unit, ({}, 0.0))
        for k, v in coeffs.items():
            acc[k] = acc.get(k, 0.0) + v
        over[term.unit] = (acc, k0 + const)
    for col, cost in serial_cost.items():
        m.h.changeColCost(col, cost)
    for unit, (coeffs, const) in over.items():
        scale = unit_time_us(unit, 1.0, hw) / model.eff.get(unit, 1.0)
        idx, val = _expand(m, {k: v * scale for k, v in coeffs.items()})
        # T - sum(scaled work) >= scaled constant
        _add_row(m.h, const * scale, INF, [t_col, *idx], [1.0, *(-v for v in val)])


def build(
    f: Formulation, shape: Mapping[str, Any], hw: HwModel | MachineModel, model: UnitModel
) -> _Model:
    mm = machine(hw)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)
    h.setOptionValue("random_seed", 0)
    m = _Model(h, {}, {})
    for name, v in f.variables.items():
        m.menus[name] = _menu(v)
        start = h.getNumCol()
        m.cols[name] = list(range(start, start + len(m.menus[name])))
        for col in m.cols[name]:
            h.addVar(0.0, 1.0)
            h.changeColIntegrality(col, highspy.HighsVarType.kInteger)
        _add_row(h, 1.0, 1.0, m.cols[name], [1.0] * len(m.cols[name]))
    lin = _Lin(set(f.variables), shape, mm.hw)
    _constraint_rows(m, f, lin)
    _objective(m, f, lin, model, mm)
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    return m


def solve_mip(
    f: Formulation,
    shape: Mapping[str, Any],
    hw: HwModel | MachineModel,
    model: UnitModel,
    shortlist: int,
    time_limit_s: float,
) -> MipOutcome:
    """Up to ``shortlist`` distinct solutions in objective order (no-good cuts)."""
    m = build(f, shape, hw, model)
    out = MipOutcome()
    if m.infeasible is not None:
        out.status = f"infeasible ({m.infeasible} is false)"
        return out
    h = m.h
    while len(out.assignments) < shortlist:
        remaining = time_limit_s - out.runtime_s
        if remaining <= 0:
            break
        h.setOptionValue("time_limit", remaining)
        h.run()
        out.solves += 1
        out.runtime_s = h.getRunTime()  # HiGHS' run clock accumulates over runs
        status = h.getModelStatus()
        info = h.getInfo()
        if out.solves == 1:
            out.status = h.modelStatusToString(status)
            out.gap = float(info.mip_gap)
            out.first_optimal = status == highspy.HighsModelStatus.kOptimal and out.gap <= GAP_ZERO
        optimal = status == highspy.HighsModelStatus.kOptimal
        if not optimal and info.primal_solution_status != FEASIBLE:
            break
        x = list(h.getSolution().col_value)
        chosen: dict[str, Any] = {}
        picked: list[int] = []
        for name, cols in m.cols.items():
            k = max(range(len(cols)), key=lambda i: x[cols[i]])
            chosen[name] = m.menus[name][k]
            picked.append(cols[k])
        out.assignments.append(chosen)
        _add_row(h, -INF, len(picked) - 1.0, picked, [1.0] * len(picked))
        if not optimal:
            break  # time limit with an incumbent: keep it, stop searching
    return out
