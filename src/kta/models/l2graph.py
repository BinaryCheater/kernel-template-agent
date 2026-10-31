"""L2 graph and one wave: the per-CTA transaction graph of a chain's effects and its
list schedule on FIFO resources (decisions D500–D503). ``kta.models.l2`` composes waves
into a kernel; the module docstring there describes the phases, edges and idle categories.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.schemas.effects import SELECT_KIND, SELECT_PATTERN
from kta.models.l1 import US_PER_S, EvalEffects, EvalTx, L1Result
from kta.models.machine import MachineModel

PHASES = ("load", "sync", "compute", "epilogue")
_PHASE_OF_PATTERN = {"transfer": "load", "sync": "sync", "store": "epilogue", "reduce": "epilogue"}
_LATENCY_CAT = {"load": "data", "sync": "sync", "compute": "other_unit", "epilogue": "other_unit"}
EPS = 1e-12


@dataclass(frozen=True)
class Interval:
    resource: str
    start_us: float  # wave-local
    end_us: float
    phase: str
    utilisation: float  # share of the chip-level capacity the wave can use (tail < 1)


# ---- the graph ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Op:
    phase: str
    base_us: Mapping[str, float]  # per CTA and trip (epilogue: per CTA), chip rate, eff applied
    latency_s: float


@dataclass(frozen=True)
class Graph:
    ops: Mapping[str, _Op]  # phase -> op
    prefix_us: float
    prefix_busy_us: Mapping[str, float]
    ctas: int
    trips: int
    inflight: int
    notes: tuple[str, ...] = ()


def _selected(sel: str, txs: Iterable[EvalTx]) -> set[str]:
    if sel.startswith(SELECT_PATTERN):
        return {t.key for t in txs if t.pattern == sel[len(SELECT_PATTERN) :]}
    if sel.startswith(SELECT_KIND):
        return {t.key for t in txs if t.kind == sel[len(SELECT_KIND) :]}
    return {t.key for t in txs if t.key == sel}


def _loop(effects: EvalEffects, live: list[EvalTx], notes: list[str]) -> tuple[int, int, dict]:
    """(trips, inflight, tx key -> phase) from the loop overlap statement (D500)."""
    phase: dict[str, str] = {}
    for o in effects.overlaps:
        if len(o.between) != 2 or not o.trips:
            continue
        a, b = (_selected(s, live) for s in o.between)
        phase |= {k: "load" for k in a} | {k: "compute" for k in b - a}
        if o.holds and o.inflight:
            inflight = max(1, round(o.inflight))
        elif o.holds:
            inflight = 2
            notes.append(f"loop {o.source}: overlap holds without inflight; 2 assumed (D500)")
        else:
            inflight = 1  # groups do not overlap: the next load waits for this compute
        return max(1, round(o.trips)), inflight, phase
    notes.append("no loop overlap statement with trips: one trip per CTA")
    return 1, 1, phase


def build_graph(effects: EvalEffects, l1: L1Result, mm: MachineModel, eff: Mapping) -> Graph:
    notes: list[str] = []
    per_tx = {k: times for k, _, times in l1.per_tx}
    live = [t for t in effects.txs if t.key in per_tx]
    serial = {
        k
        for o in effects.overlaps
        if len(o.between) == 1 and not o.holds
        for k in _selected(o.between[0], live)
    }
    launch = [t for t in live if t.pattern == "launch" and not _serial(mm, t)]
    ctas = round(sum(t.count for t in launch))
    if ctas < 1:
        ctas = 1
        notes.append("no CTA launch transaction: the kernel is simulated as one CTA")
    trips, inflight, phase = _loop(effects, live, notes)
    ops: dict[str, dict[str, float]] = {}
    lat: dict[str, float] = {}
    prefix: dict[str, float] = {}
    for t in live:
        times = {r: us / eff.get(r, 1.0) for r, us in per_tx[t.key].items()}
        if t.key in serial or _serial(mm, t):
            prefix = {r: prefix.get(r, 0.0) + us for r, us in times.items()}
            continue
        if t in launch:
            continue
        p = phase.get(t.key) or _PHASE_OF_PATTERN.get(t.pattern, "compute")
        per = ctas * (1 if p == "epilogue" else trips)
        acc = ops.setdefault(p, {})
        for r, us in times.items():
            acc[r] = acc.get(r, 0.0) + us / per
        op = mm.op_class(t.kind)
        lat[p] = max(lat.get(p, 0.0), (op.latency_s if op else None) or 0.0)
    built = {p: _Op(p, ops[p], lat.get(p, 0.0)) for p in PHASES if p in ops}
    # The prefix is serial: its length is its slowest resource (= L1's additive term).
    return Graph(
        built, max(prefix.values(), default=0.0), prefix, ctas, trips, inflight, tuple(notes)
    )


def _serial(mm: MachineModel, t: EvalTx) -> bool:
    op = mm.op_class(t.kind)
    return bool(op and op.serial and not op.legacy)


# ---- one wave ---------------------------------------------------------------------------


@dataclass
class _Entry:
    phase: str
    ready: float
    first_start: float
    res_end: float
    done: float
    binding: _Entry | None


@dataclass
class Wave:
    duration: float
    busy: dict[str, float]
    idle: dict[str, dict[str, float]]
    intervals: list[Interval] = field(default_factory=lambda: list[Interval]())


def _explain(entry: _Entry, a: float, b: float, add: Any) -> None:
    """Attribute the gap [a, b] that ends when ``entry`` becomes ready (D503)."""
    hi, cur = b, entry
    while hi > a + EPS:
        d = cur.binding
        if d is None:
            add("prologue_epilogue_tail", hi - a)
            return
        for lo, cat in (
            (d.res_end, _LATENCY_CAT[d.phase]),
            (d.first_start, "sync" if d.phase == "sync" else "other_unit"),
            (d.ready, "other_unit"),  # the producer waited for its own resources
        ):
            lo = max(a, lo)
            if hi > lo:
                add(cat, hi - lo)
                hi = lo
        cur = d


class _Booker:
    def __init__(
        self,
        dur: Mapping[str, Mapping[str, float]],
        share: Mapping[str, float],
        lat: Mapping[str, float],
    ) -> None:
        self.dur, self.share, self.lat = dur, share, lat
        self.free: dict[str, float] = {}
        self.wave = Wave(0.0, {}, {})

    def add(self, r: str) -> Any:
        def _add(cat: str, us: float) -> None:
            slot = self.wave.idle.setdefault(r, {})
            slot[cat] = slot.get(cat, 0.0) + us

        return _add

    def book(self, phase: str, deps: list[_Entry]) -> _Entry:
        binding = max(deps, key=lambda e: e.done) if deps else None
        ready = binding.done if binding else 0.0
        entry = _Entry(phase, ready, math.inf, ready, ready, binding)
        for r, d in self.dur[phase].items():
            free = self.free.get(r, 0.0)
            start = max(ready, free)
            if start > free + EPS:
                _explain(entry, free, start, self.add(r))
            end = start + d
            self.free[r] = end
            share = self.share[r]
            self.wave.busy[r] = self.wave.busy.get(r, 0.0) + d * share
            if share < 1.0:
                self.add(r)("prologue_epilogue_tail", d * (1.0 - share))
            self.wave.intervals.append(Interval(r, start, end, phase, share))
            entry.first_start = min(entry.first_start, start)
            entry.res_end = max(entry.res_end, end)
        entry.first_start = min(entry.first_start, entry.res_end)
        entry.done = entry.res_end + self.lat[phase]
        return entry


def simulate_wave(g: Graph, resident: int, share: Mapping[str, float]) -> Wave:
    dur = {
        p: {r: resident * us / share[r] for r, us in op.base_us.items()} for p, op in g.ops.items()
    }
    bk = _Booker(dur, share, {p: op.latency_s * US_PER_S for p, op in g.ops.items()})
    has = set(g.ops)
    loads: list[_Entry] = []
    comps: list[_Entry] = []
    last: list[_Entry] = []
    if "load" in has:
        loads = [bk.book("load", []) for _ in range(min(g.inflight, g.trips))]
    for i in range(g.trips):
        dep = [loads[i]] if loads else []
        if "sync" in has:
            dep = [bk.book("sync", dep)]
        if "compute" in has:
            comps.append(bk.book("compute", dep))
            dep = [comps[-1]]
        last = dep
        j = i + g.inflight
        if loads and j < g.trips:
            loads.append(bk.book("load", [comps[i]] if comps else []))
    if "epilogue" in has:
        last = [bk.book("epilogue", last)]
    end = max((e.done for e in last), default=0.0)
    end = max([end, *bk.free.values()])
    for r, free in bk.free.items():
        if end > free + EPS:
            bk.add(r)("prologue_epilogue_tail", end - free)
    bk.wave.duration = end
    bk.wave.intervals = _merge(bk.wave.intervals)
    return bk.wave


def _merge(intervals: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    last: dict[str, int] = {}
    for iv in sorted(intervals, key=lambda i: (i.resource, i.start_us)):
        k = last.get(iv.resource)
        prev = out[k] if k is not None else None
        if (
            prev
            and prev.phase == iv.phase
            and prev.utilisation == iv.utilisation
            and iv.start_us <= prev.end_us + EPS
        ):
            out[k] = dataclasses.replace(prev, end_us=max(prev.end_us, iv.end_us))  # type: ignore[index]
            continue
        last[iv.resource] = len(out)
        out.append(iv)
    return out
