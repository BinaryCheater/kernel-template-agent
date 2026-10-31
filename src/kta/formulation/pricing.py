"""A formulation's effects at one point, priced by the hardware model (decision D464).

- ``evaluate_effects``: the bound effects at an assignment and shape, as numbers (the
  input of L1, and of WP16's L2 simulator).
- ``price_point``: L1 at that point.
- ``per_unit_work``: work per resource, (overlappable, serial) (decision D240). Legacy
  ``unit:<u>`` terms need no hardware model; other transactions need a MachineModel for
  their reservation tables.
- ``work_terms``: every transaction's use of every resource as an expression over the
  family variables (count times the reservation amount), the linear objective rows the
  MIP exports. Pair and recurrence bounds are not in these rows: the MIP relaxes them
  (WP17 compiles the full projection).
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from kta.core.errors import NotSupported, SchemaError
from kta.core.expr import Expr, ExprError
from kta.core.schemas.effects import LEGACY_KIND_PREFIX
from kta.formulation.assemble import Formulation
from kta.formulation.blocks import BoundCost, BoundTx
from kta.models.l1 import EvalEffects, EvalOverlap, EvalStorage, EvalTx, L1Result, price
from kta.models.machine import MachineModel


def _num(e: Expr, a: Mapping[str, Any], hw: Mapping[str, Any], shape: Mapping[str, Any]) -> float:
    return float(e.eval(a, hw=hw, case=shape))


def _on(
    when: Expr | None, a: Mapping[str, Any], hw: Mapping[str, Any], shape: Mapping[str, Any]
) -> bool:
    return True if when is None else bool(when.eval(a, hw=hw, case=shape))


def evaluate_effects(
    f: Formulation, assignment: Mapping[str, Any], shape: Mapping[str, Any], hw: Mapping[str, Any]
) -> EvalEffects:
    """The bound effects of ``f`` at one point (``hw`` = the hw.* values)."""
    a = assignment
    txs = tuple(
        EvalTx(
            t.source,
            t.id,
            t.pattern,
            t.kind_at(a),
            _num(t.count, a, hw, shape) if _on(t.when, a, hw, shape) else 0.0,
            {k: _num(v, a, hw, shape) for k, v in t.amounts.items()},
        )
        for t in f.transactions
    )
    storage = tuple(
        EvalStorage(s.source, s.id, s.level_at(a), s.scope, _num(s.bytes, a, hw, shape))
        for s in f.storage
        if _on(s.when, a, hw, shape)
    )
    overlaps = tuple(
        EvalOverlap(
            o.source,
            o.between,
            _on(o.when, a, hw, shape),
            _num(o.inflight, a, hw, shape) if o.inflight else None,
            _num(o.trips, a, hw, shape) if o.trips else None,
        )
        for o in f.overlaps
    )
    return EvalEffects(txs, storage, overlaps)


def price_point(
    f: Formulation,
    assignment: Mapping[str, Any],
    shape: Mapping[str, Any],
    mm: MachineModel,
    features: Mapping[str, Any] | None = None,
) -> L1Result:
    return price(evaluate_effects(f, assignment, shape, mm.hw), mm, features)


def _serial_keys(f: Formulation, mm: MachineModel, ev: EvalEffects) -> set[str]:
    keys = {
        t.key
        for t in ev.txs
        if (op := mm.op_class(t.kind)) is not None and op.serial and not op.legacy
    }
    for o in ev.overlaps:
        if len(o.between) == 1 and not o.holds:
            keys |= {t.key for t in ev.txs if t.key == o.between[0]}
    return keys


def per_unit_work(
    f: Formulation,
    assignment: Mapping[str, Any],
    shape: Mapping[str, Any],
    hw: Mapping[str, Any] | MachineModel,
) -> tuple[dict[str, float], dict[str, float]]:
    """Work per resource, summed over transactions: (overlappable, serial) (D240)."""
    env = hw.hw if isinstance(hw, MachineModel) else hw
    over: dict[str, float] = {}
    serial: dict[str, float] = {}
    for term in f.cost_terms:
        value = float(term.work.eval(assignment, hw=env, case=shape))
        target = serial if term.overlap == "serial" else over
        target[term.unit] = target.get(term.unit, 0.0) + value
    priced = [t for t in f.transactions if t.kind_var or not t.kind.startswith(LEGACY_KIND_PREFIX)]
    if not priced:
        return over, serial
    if not isinstance(hw, MachineModel):
        raise NotSupported("pricing effects needs the hardware model (a MachineModel)")
    ev = evaluate_effects(f, assignment, shape, env)
    serial_keys = _serial_keys(f, hw, ev)
    for t in ev.txs:
        op = hw.op_class(t.kind)
        if op is None or op.legacy or t.count == 0:
            continue  # unpriced classes are reported by L1 (D465)
        target = serial if t.key in serial_keys else over
        for row in op.rows:
            use = t.count * hw.evaluate(row.amount, tx=t.amounts)
            target[row.resource] = target.get(row.resource, 0.0) + use
    return over, serial


def cost_breakdown(
    f: Formulation,
    assignment: Mapping[str, Any],
    shape: Mapping[str, Any],
    hw: Mapping[str, Any] | MachineModel,
) -> list[dict[str, Any]]:
    """Each legacy cost term's work, and with a MachineModel each transaction's time per
    resource, with its source: "which step's cost is off"."""
    env = hw.hw if isinstance(hw, MachineModel) else hw
    rows: list[dict[str, Any]] = [
        {
            "source": t.source,
            "index": t.index,
            "unit": t.unit,
            "overlap": t.overlap,
            "work": float(t.work.eval(assignment, hw=env, case=shape)),
        }
        for t in f.cost_terms
    ]
    if isinstance(hw, MachineModel):
        res = price_point(f, assignment, shape, hw)
        rows += [
            {"source": key.split(".", 1)[0], "tx": key, "kind": kind, "time_us": times}
            for key, kind, times in res.per_tx
            if not kind.startswith(LEGACY_KIND_PREFIX)
        ]
    return rows


class _TxSubst(ast.NodeTransformer):
    def __init__(self, amounts: Mapping[str, Expr], where: str) -> None:
        self.amounts = amounts
        self.where = where

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if isinstance(node.value, ast.Name) and node.value.id == "tx":
            if node.attr not in self.amounts:
                raise SchemaError(f"{self.where}: the transaction declares no amount {node.attr}")
            return ast.parse(self.amounts[node.attr].src.strip(), mode="eval").body
        if isinstance(node.value, ast.Name) and node.value.id == "tbl":
            raise NotSupported(f"{self.where}: table term tbl.{node.attr} is not exportable")
        return node


def _row_term(t: BoundTx, resource: str, amount: Expr, serial: bool, f: Formulation) -> BoundCost:
    where = f"{t.key} ({t.kind}) on {resource}"
    tree = _TxSubst(t.amounts, where).visit(ast.parse(amount.src.strip(), mode="eval").body)
    count = ast.parse(t.count.src.strip(), mode="eval").body
    node: ast.expr = ast.BinOp(count, ast.Mult(), tree)
    if t.when is not None:
        node = ast.IfExp(ast.parse(t.when.src.strip(), mode="eval").body, node, ast.Constant(0))
    src = ast.unparse(ast.fix_missing_locations(node))
    try:
        work = Expr.parse(src, set(f.variables))
    except ExprError as exc:
        raise SchemaError(f"{where}: {exc}") from exc
    return BoundCost(t.source, -1, resource, "serial" if serial else "overlappable", work)


def work_terms(f: Formulation, mm: MachineModel) -> list[BoundCost]:
    """Legacy cost terms plus each priced transaction's per-resource use as an expression.

    Raises NotSupported for a table term (not exportable as an expression).
    """
    out = list(f.cost_terms)
    serial_ids = {
        o.between[0]
        for o in f.overlaps
        if len(o.between) == 1 and o.when is not None and o.when.src.strip() == "False"
    }
    for t in f.transactions:
        if t.kind_var is not None:
            raise NotSupported(f"{t.key}: its operation class depends on {t.kind_var}")
        op = mm.op_class(t.kind)
        if op is None or op.legacy:
            continue
        serial = op.serial or t.key in serial_ids
        out += [_row_term(t, row.resource, row.amount, serial, f) for row in op.rows]
    return out
