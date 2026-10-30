"""L1: analytic steady-state pricing of a chain's effects (docs/design/02 §4, §5.1;
spec/modules/formulation-solve-models.md "L1"; decisions D464, D465, D471).

Input: evaluated effects (transactions with numeric counts and amounts, storage, overlap
statements) and a ``MachineModel``. Every transaction reserves, per its operation
class's reservation table, ``count * amount`` on each resource; resource time is use over
the chip-level rate. Composition (rule ``steady_state_max``)::

    busy_r      = sum over overlappable transactions of use_r / (rate_r * eff_r)
    pair bound  = t(G_a) + t(G_b)                for an overlap statement that does not hold
                  t(G) = max_r sum_{t in G} use_r / (rate_r * eff_r)
    recurrence  = trips * (lat(G_a) + lat(G_b)) / inflight      for one that holds
                  lat(G) = the smallest operation-class latency in G
    additive    = sum over serial transactions of max_r use_r / (rate_r * eff_r)
    total       = max(max_r busy_r, bounds) + additive + sum gamma_ab * min(busy_a, busy_b)

Every term is a lower bound of the steady state; the total is the steady-state
projection that WP17 exports to a MIP and WP16's L2 simulator refines. Walls check the
storage effects; a ``no_launch`` cliff makes the point infeasible, ``lower_occupancy``
gives the residency feature. Unpriced operation classes and unmodelled resources cost no
time (infinite capacity) and are counted in the result (D465).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.schemas.effects import SELECT_KIND, SELECT_PATTERN
from kta.core.schemas.evaluation import Prediction
from kta.core.schemas.hwmodel import ROOTS_FEATURE
from kta.models.machine import MachineModel, OpClass, parse_hw

US_PER_S = 1e6


@dataclass(frozen=True)
class EvalTx:
    source: str  # chain item key
    id: str
    pattern: str
    kind: str
    count: float
    amounts: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())

    @property
    def key(self) -> str:
        return f"{self.source}.{self.id}"


@dataclass(frozen=True)
class EvalStorage:
    source: str
    id: str
    level: str
    scope: str
    bytes: float


@dataclass(frozen=True)
class EvalOverlap:
    source: str
    between: tuple[str, ...]  # selectors, local ids already qualified as "<source>.<id>"
    holds: bool
    inflight: float | None = None
    trips: float | None = None


@dataclass(frozen=True)
class EvalEffects:
    txs: tuple[EvalTx, ...] = ()
    storage: tuple[EvalStorage, ...] = ()
    overlaps: tuple[EvalOverlap, ...] = ()


@dataclass(frozen=True)
class WallCheck:
    name: str
    cliff: str
    usage: float
    limit: float
    ok: bool
    residency: int | None = None  # executors per ``per`` scope this wall allows


@dataclass(frozen=True)
class Parts:
    """Peak-time vectors (us, eff = 1) that ``compose`` turns into a total."""

    busy_peak_us: Mapping[str, float]
    serial_peak_us: tuple[Mapping[str, float], ...] = ()
    pairs_peak_us: tuple[tuple[str, Mapping[str, float], Mapping[str, float]], ...] = ()
    recurrence_us: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())


@dataclass(frozen=True)
class Composed:
    total_us: float
    bottleneck: str
    busy_us: dict[str, float]
    bounds_us: dict[str, float]
    additive_us: float
    interaction_us: float


@dataclass(frozen=True)
class L1Result:
    parts: Parts
    composed: Composed
    walls: tuple[WallCheck, ...]
    unpriced: dict[str, float]  # operation class -> transactions without a reservation table
    unmodelled: dict[str, float]  # resource -> amount used with no peak (infinite capacity)
    per_tx: tuple[tuple[str, str, dict[str, float]], ...]  # (tx key, kind, resource -> us)
    model: str
    interactions: tuple[tuple[str, str, float], ...] = ()  # the hardware model's, applied

    @property
    def feasible(self) -> bool:
        return all(w.ok or w.cliff != "no_launch" for w in self.walls)

    @property
    def complete(self) -> bool:
        return not self.unpriced and not self.unmodelled

    @property
    def total_us(self) -> float:
        return self.composed.total_us

    @property
    def residency(self) -> int | None:
        values = [w.residency for w in self.walls if w.residency is not None]
        return min(values) if values else None

    def prediction(self) -> Prediction:
        p = self.parts
        return Prediction(
            per_unit=dict(p.busy_peak_us),
            serial=sum(max(v.values(), default=0.0) for v in p.serial_peak_us),
            total=self.composed.total_us,
            model=self.model,
            bounds=dict(self.composed.bounds_us),
            unpriced={**self.unpriced, **{f"resource:{r}": a for r, a in self.unmodelled.items()}},
        )


# ---- composition --------------------------------------------------------------------


def _scaled(vec: Mapping[str, float], eff: Mapping[str, float]) -> dict[str, float]:
    return {r: t / eff.get(r, 1.0) for r, t in vec.items()}


def compose(
    parts: Parts,
    eff: Mapping[str, float] | None = None,
    interactions: Iterable[tuple[str, str, float]] = (),
    serial_coef: float = 1.0,
    serial_const_us: float = 0.0,
) -> Composed:
    """Total of ``parts`` under per-resource efficiencies (the one composition rule)."""
    e = eff or {}
    busy = _scaled(parts.busy_peak_us, e)
    bounds: dict[str, float] = {}
    for name, a, b in parts.pairs_peak_us:
        bounds[name] = max(_scaled(a, e).values(), default=0.0) + max(
            _scaled(b, e).values(), default=0.0
        )
    bounds.update(parts.recurrence_us)
    candidates = {**busy, **bounds}
    bottleneck = max(candidates, key=lambda k: (candidates[k], k)) if candidates else ""
    steady = candidates.get(bottleneck, 0.0)
    additive = sum(max(_scaled(v, e).values(), default=0.0) for v in parts.serial_peak_us)
    additive = serial_coef * additive + serial_const_us
    inter = sum(g * min(busy.get(a, 0.0), busy.get(b, 0.0)) for a, b, g in interactions)
    return Composed(steady + additive + inter, bottleneck, busy, bounds, additive, inter)


# ---- pricing ------------------------------------------------------------------------


def _tx_times(
    t: EvalTx, op: OpClass, mm: MachineModel, unmodelled: dict[str, float]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in op.rows:
        use = t.count * mm.evaluate(row.amount, tx=t.amounts)
        if use == 0:
            continue
        seconds = mm.time_s(row.resource, use)
        if seconds is None:
            unmodelled[row.resource] = unmodelled.get(row.resource, 0.0) + use
            continue
        out[row.resource] = out.get(row.resource, 0.0) + seconds * US_PER_S
    return out


def _selected(sel: str, txs: Iterable[EvalTx]) -> list[EvalTx]:
    if sel.startswith(SELECT_PATTERN):
        return [t for t in txs if t.pattern == sel[len(SELECT_PATTERN) :]]
    if sel.startswith(SELECT_KIND):
        return [t for t in txs if t.kind == sel[len(SELECT_KIND) :]]
    return [t for t in txs if t.key == sel]


def _sum(vectors: Iterable[Mapping[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for v in vectors:
        for r, x in v.items():
            out[r] = out.get(r, 0.0) + x
    return out


def check_walls(storage: Iterable[EvalStorage], mm: MachineModel) -> tuple[WallCheck, ...]:
    out: list[WallCheck] = []
    items = list(storage)
    for w in mm.walls:
        usage = sum(s.bytes for s in items if s.level == w.spec.level and s.scope == w.spec.scope)
        if usage == 0:
            continue
        if w.spec.cliff == "lower_occupancy":
            per_executor = usage + w.reserved
            resident = math.floor(w.limit / per_executor) if per_executor > 0 else None
            out.append(
                WallCheck(
                    w.spec.name, w.spec.cliff, per_executor, w.limit, bool(resident), resident
                )
            )
            continue
        out.append(WallCheck(w.spec.name, w.spec.cliff, usage, w.limit, usage <= w.limit))
    return tuple(out)


def price(
    effects: EvalEffects, mm: MachineModel, features: Mapping[str, Any] | None = None
) -> L1Result:
    """Price ``effects`` on ``mm`` with the model's own efficiency and interactions."""
    walls = check_walls(effects.storage, mm)
    feats: dict[str, Any] = dict(features or {})
    resident = [w.residency for w in walls if w.residency is not None]
    if resident and "residency" not in feats:
        feats["residency"] = min(resident)
    parts, unpriced, unmodelled, per_tx = price_parts(effects, mm)
    eff = {r: mm.efficiency_of(r, feats) for r in parts.busy_peak_us}
    for vec in parts.serial_peak_us:
        eff.update({r: mm.efficiency_of(r, feats) for r in vec if r not in eff})
    inter = [
        (i.resources[0], i.resources[1], i.gamma)
        for i in mm.interactions
        if _applies(mm, i.applies_when, feats)
    ]
    return L1Result(
        parts=parts,
        composed=compose(parts, eff, inter),
        walls=walls,
        unpriced=unpriced,
        unmodelled=unmodelled,
        per_tx=per_tx,
        model=f"l1:{mm.version}",
        interactions=tuple(inter),
    )


def _applies(mm: MachineModel, src: str, feats: Mapping[str, Any]) -> bool:
    return bool(mm.evaluate(parse_hw(src, ROOTS_FEATURE, "applies_when"), f=feats))


def price_parts(
    effects: EvalEffects, mm: MachineModel
) -> tuple[
    Parts, dict[str, float], dict[str, float], tuple[tuple[str, str, dict[str, float]], ...]
]:
    """Peak-time parts of ``effects``, plus unpriced classes and unmodelled resources."""
    unpriced: dict[str, float] = {}
    unmodelled: dict[str, float] = {}
    times: dict[str, dict[str, float]] = {}
    serial_keys: set[str] = set()
    latency: dict[str, float | None] = {}
    for t in effects.txs:
        if t.count == 0:
            continue
        op = mm.op_class(t.kind)
        if op is None:
            unpriced[t.kind] = unpriced.get(t.kind, 0.0) + t.count
            continue
        times[t.key] = _tx_times(t, op, mm, unmodelled)
        latency[t.key] = op.latency_s
        if op.serial:
            serial_keys.add(t.key)
    live = [t for t in effects.txs if t.key in times]
    pairs: list[tuple[str, Mapping[str, float], Mapping[str, float]]] = []
    recurrence: dict[str, float] = {}
    for o in effects.overlaps:
        groups = [[t.key for t in _selected(s, live)] for s in o.between]
        if len(groups) == 1:
            if not o.holds:
                serial_keys.update(groups[0])
            continue
        name = f"{o.source}:{'|'.join(o.between)}"
        if not o.holds:
            a, b = (_sum(times[k] for k in g if k not in serial_keys) for g in groups)
            pairs.append((f"serial {name}", a, b))
        elif o.inflight and o.trips:
            lats = [[latency[k] for k in g] for g in groups]
            if all(lat is not None for g in lats for lat in g) and all(lats):
                # min: the bound must stay optimistic (MIP <= simulator, 02 §5.2)
                la, lb = (min(x for x in g if x is not None) for g in lats)
                recurrence[f"recurrence {name}"] = o.trips * (la + lb) / o.inflight * US_PER_S
            else:
                unmodelled[f"latency:{name}"] = 1.0
    busy = _sum(v for k, v in times.items() if k not in serial_keys)
    serial = tuple(times[k] for k in sorted(serial_keys))
    per_tx = tuple((t.key, t.kind, times[t.key]) for t in live)
    parts = Parts(busy, serial, tuple(pairs), recurrence)
    return parts, unpriced, unmodelled, per_tx


def sol_us(min_work: Mapping[str, float], mm: MachineModel) -> dict[str, float]:
    """Speed of light per resource: the semantic minimum work at peak (02 §4, D-T11)."""
    out: dict[str, float] = {}
    for r, w in sorted(min_work.items()):
        if w <= 0:
            continue
        seconds = mm.time_s(r, w)
        if seconds is not None:
            out[r] = seconds * US_PER_S
    return out
