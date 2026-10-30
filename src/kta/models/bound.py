"""Theoretical lower bound (docs/design/50 §6), on the hardware model (D464).

The bound is the largest per-resource time "minimum work / peak" over the resources: the
per-resource speed of light (``kta.models.l1.sol_us``). Minimum work is what the problem
semantically needs (each input read once, each output written once, semantic FLOPs); it
does not depend on any family.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from kta.core.errors import NotSupported, SchemaError
from kta.core.expr import Expr
from kta.core.schemas.evaluation import Bound, Case, Problem
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.hwmodel import HwArch
from kta.models.l1 import US_PER_S, sol_us
from kta.models.machine import MachineModel

# Problem.params key holding {unit: expression over case.* (shape values)} (decision D243).
MIN_WORK_KEY = "min_work"


def machine(hw: HwModel | MachineModel, arch: HwArch | None = None) -> MachineModel:
    """``hw`` resolved on ``arch`` (an already resolved model is returned as it is)."""
    return hw if isinstance(hw, MachineModel) else MachineModel.build(hw, arch)


def unit_time_us(
    unit: str, work: float, hw: HwModel | MachineModel, arch: HwArch | None = None
) -> float:
    """Time of ``work`` on resource ``unit`` at its peak, in microseconds."""
    mm = machine(hw, arch)
    seconds = mm.time_s(unit, work)
    if seconds is None:
        raise NotSupported(f"HwModel {mm.target}@{mm.hw_version} has no peak for unit {unit!r}")
    return seconds * US_PER_S


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


def bound(
    case: Case,
    shape: dict[str, Any],
    hw: HwModel | MachineModel,
    work: dict[str, float],
    arch: HwArch | None = None,
) -> Bound:
    """Per-resource speed of light of the minimum work; the bound is the largest.

    A resource of the minimum work without a peak raises NotSupported: a bound that
    silently left it out would claim more than is known.
    """
    mm = machine(hw, arch)
    missing = sorted(u for u, w in work.items() if w > 0 and mm.rate(u) is None)
    if missing:
        raise NotSupported(f"HwModel {mm.target}@{mm.hw_version} has no peak for units {missing}")
    per_unit = sol_us(work, mm)
    if not per_unit:
        raise SchemaError(f"no positive work for case {case.problem} at shape {shape}")
    bottleneck = max(per_unit, key=lambda u: (per_unit[u], u))
    return Bound(per_unit_us=per_unit, bound_us=per_unit[bottleneck], bottleneck=bottleneck)
