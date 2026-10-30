"""Theoretical lower bound (docs/design/50 §6).

The bound is the largest per-unit time "minimum work / peak" over the units. Minimum work
is what the problem semantically needs (each input read once, each output written once,
semantic FLOPs); it does not depend on any family.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from kta.core.errors import NotSupported, SchemaError
from kta.core.expr import Expr
from kta.core.schemas.evaluation import Bound, Case, Problem
from kta.core.schemas.hardware import HwModel

US_PER_S = 1e6
# Problem.params key holding {unit: expression over case.* (shape values)} (decision D243).
MIN_WORK_KEY = "min_work"


def unit_time_us(unit: str, work: float, hw: HwModel) -> float:
    """Time of ``work`` on ``unit`` at the unit's peak, in microseconds."""
    peak = hw.peaks.get(unit)
    if peak is None or peak <= 0:
        raise NotSupported(f"HwModel {hw.target} v{hw.version} has no peak for unit {unit!r}")
    return work / peak * US_PER_S


def min_work(problem: Problem, shape: Mapping[str, Any]) -> dict[str, float]:
    """Semantic minimum work per unit from the Problem's ``params.min_work`` expressions."""
    raw = problem.params.get(MIN_WORK_KEY)
    if not isinstance(raw, dict) or not raw:
        raise NotSupported(
            f"Problem {problem.name} has no params.{MIN_WORK_KEY} (unit -> expression over case.*)"
        )
    exprs = cast(dict[str, Any], raw)
    return {
        str(unit): float(Expr.parse(str(src), set()).eval({}, case=shape))
        for unit, src in exprs.items()
    }


def bound(case: Case, shape: dict[str, Any], hw: HwModel, work: dict[str, float]) -> Bound:
    """Per-unit time of the minimum work at measured peaks; the bound is the largest."""
    per_unit = {u: unit_time_us(u, w, hw) for u, w in sorted(work.items()) if w > 0}
    if not per_unit:
        raise SchemaError(f"no positive work for case {case.problem} at shape {shape}")
    bottleneck = max(per_unit, key=lambda u: (per_unit[u], u))
    return Bound(per_unit_us=per_unit, bound_us=per_unit[bottleneck], bottleneck=bottleneck)
