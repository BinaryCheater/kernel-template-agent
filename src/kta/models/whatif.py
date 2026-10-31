"""What-if runs on L2 (docs/design/02 §4 "Sustained saturation and walls"; D504).

Move a wall limit or a resource peak, re-simulate, and report the shadow price: the idle
time of the base bottleneck that the move removes. On a fixed chip it separates scheduling
problems from hardware limits; in co-design it is what relaxing the wall buys. The moved
model exists only in memory; the hardware-model records are never changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.models.l1 import EvalEffects
from kta.models.l2 import L2Result, moved, simulate
from kta.models.machine import MachineModel
from kta.store.base import Store


@dataclass(frozen=True)
class WhatIf:
    peaks: Mapping[str, float]  # resource -> chip-level rate per second
    walls: Mapping[str, float]  # wall -> limit
    base: L2Result
    moved: L2Result
    shadow_price_us: float | None  # bottleneck idle removed; None when the base cannot launch
    delta_total_us: float | None

    def summary(self) -> dict[str, Any]:
        return {
            "peaks": dict(self.peaks),
            "walls": dict(self.walls),
            "bottleneck": self.base.bottleneck,
            "shadow_price_us": self.shadow_price_us,
            "delta_total_us": self.delta_total_us,
            "base_total_us": self.base.total_us if self.base.feasible else None,
            "moved_total_us": self.moved.total_us if self.moved.feasible else None,
            "base_model": self.base.model,
            "moved_model": self.moved.model,
            "idle_base": self.base.idle_us.get(self.base.bottleneck, {}),
            "idle_moved": self.moved.idle_us.get(self.base.bottleneck, {}),
        }


def _idle(r: L2Result, resource: str) -> float:
    return sum(r.idle_us.get(resource, {}).values())


def what_if(
    effects: EvalEffects,
    mm: MachineModel,
    peaks: Mapping[str, float] | None = None,
    walls: Mapping[str, float] | None = None,
    features: Mapping[str, Any] | None = None,
) -> WhatIf:
    """Re-simulate ``effects`` with ``peaks`` and/or ``walls`` moved (unknown names: KeyError)."""
    base = simulate(effects, mm, features)
    after = simulate(effects, moved(mm, peaks, walls), features)
    if not base.feasible or not after.feasible:
        price, delta = None, None
    else:
        b = base.bottleneck
        price = _idle(base, b) - _idle(after, b)
        delta = after.total_us - base.total_us
    return WhatIf(dict(peaks or {}), dict(walls or {}), base, after, price, delta)


def record_l2(
    store: Store, result: L2Result, actor: Actor, now: datetime, refs: Mapping[str, str]
) -> str:
    """Store an L2 result as a Report (title ``l2_simulation``)."""
    body = Report(
        title="l2_simulation",
        subject=result.model,
        summary=(
            f"L2 {result.total_us:.1f} us (L1 {result.l1_total_us:.1f} us), "
            f"bottleneck {result.bottleneck}, {len(result.waves)} waves"
        ),
        data=result.summary(),
    )
    return store.put(Record.new("Report", body, actor, now, refs=dict(refs)))


def record_what_if(
    store: Store, w: WhatIf, actor: Actor, now: datetime, refs: Mapping[str, str]
) -> str:
    """Store a what-if run as a Report (title ``what_if``)."""
    moves = ", ".join(
        [
            *(f"peak {k}={v:g}" for k, v in w.peaks.items()),
            *(f"wall {k}={v:g}" for k, v in w.walls.items()),
        ]
    )
    price = "n/a" if w.shadow_price_us is None else f"{w.shadow_price_us:.2f} us"
    body = Report(
        title="what_if",
        subject=w.base.model,
        summary=f"{moves}: shadow price {price} on {w.base.bottleneck}",
        data=w.summary(),
    )
    return store.put(Record.new("Report", body, actor, now, refs=dict(refs)))
