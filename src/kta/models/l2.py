"""L2: transaction-level simulation of a chain's effects on a MachineModel (docs/design/02
§4 "Four fidelity levels", "Sustained saturation and walls"; spec/modules/
formulation-solve-models.md "L2"; decisions D500–D505).

The kernel is a per-CTA transaction graph. Effects are whole-kernel counts, so each
transaction is divided over the CTAs (``cta_launch`` transactions) and the main-loop trips
(the loop overlap statement: the first two-group statement with ``trips``). Phases:

    prefix    serial operation classes and single-group non-overlap statements (kernel)
    load      the loop statement's first group     (default: pattern transfer)
    sync      pattern sync
    compute   the loop statement's second group    (default: pattern contract) + the rest
    epilogue  patterns store and reduce, once per CTA after the loop

Edges: load_i -> sync_i -> compute_i (data, after the producer's latency) and
compute_i -> load_{i+inflight} (buffer slot reuse). The resident CTAs of one wave run in
lockstep (D501): a wave is the graph with every amount times the resident count, booked in
topological order on FIFO resources, each resource independently (D502). SM-scoped
resources of a partial wave run at the occupied share. Waves follow each other.

Idle time of each resource is attributed by walking the binding dependency of the op that
ends the gap (D503): producer latency of a load = ``data``, of a sync = ``sync``; producer
busy or waiting for its own resources = ``other_unit``; wave start, drain, the serial prefix
and the tail wave = ``prologue_epilogue_tail``; idle that relaxing the binding residency wall
removes = ``wall`` (a what-if run, D504). Deterministic: no randomness, no clock.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from kta.models.l1 import US_PER_S, EvalEffects, L1Result, WallCheck, price
from kta.models.l2graph import EPS, Graph, Interval, Wave, build_graph, simulate_wave
from kta.models.machine import CHIP, MachineModel

SIM_VERSION = "1"  # bump on any change of semantics; part of every L2 model string
CATEGORIES = ("other_unit", "data", "sync", "prologue_epilogue_tail", "wall")
# ncu warp-state names for the idle categories of the bottleneck resource (D505); in us of
# bottleneck idle time, not in samples.
STALL_NCU = {
    "data": "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "sync": "smsp__pcsamp_warps_issue_stalled_barrier",
    "other_unit": "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "prologue_epilogue_tail": "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "wall": "smsp__pcsamp_warps_issue_stalled_not_selected",
}


@dataclass(frozen=True)
class WaveSpan:
    index: int
    kind: str  # full | tail
    ctas: int
    start_us: float
    end_us: float


@dataclass(frozen=True)
class L2Result:
    total_us: float
    feasible: bool
    bottleneck: str
    busy_us: dict[str, float]
    idle_us: dict[str, dict[str, float]]  # resource -> category -> us
    stalls: dict[str, float]  # ncu stall name -> us (bottleneck)
    metrics: dict[str, float]  # ncu metric name -> value
    waves: tuple[WaveSpan, ...]
    intervals: dict[str, tuple[Interval, ...]]  # wave kind -> merged busy intervals
    ctas: int
    trips: int
    inflight: int
    residency: int | None
    prefix_us: float
    interaction_us: float
    walls: tuple[WallCheck, ...]
    unpriced: dict[str, float]
    unmodelled: dict[str, float]
    l1_total_us: float
    model: str
    notes: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """The numbers a Report stores (spec/modules/formulation-solve-models.md "L2")."""
        return {
            "model": self.model,
            "total_us": self.total_us,
            "l1_total_us": self.l1_total_us,
            "feasible": self.feasible,
            "bottleneck": self.bottleneck,
            "busy_us": self.busy_us,
            "idle_us": self.idle_us,
            "stalls": self.stalls,
            "metrics": self.metrics,
            "ctas": self.ctas,
            "trips": self.trips,
            "inflight": self.inflight,
            "residency": self.residency,
            "waves": [dataclasses.asdict(w) for w in self.waves],
            "intervals": {k: [dataclasses.asdict(i) for i in v] for k, v in self.intervals.items()},
            "unpriced": self.unpriced,
            "unmodelled": self.unmodelled,
            "notes": list(self.notes),
        }


# ---- the kernel ---------------------------------------------------------------------


def moved(
    mm: MachineModel,
    peaks: Mapping[str, float] | None = None,
    walls: Mapping[str, float] | None = None,
) -> MachineModel:
    """``mm`` with resource peaks (chip-level rate per second) or wall limits replaced."""
    res = dict(mm.resources)
    for r, rate in (peaks or {}).items():
        if r not in res:
            raise KeyError(f"{mm.target}: no resource {r!r}")
        res[r] = dataclasses.replace(res[r], rate=float(rate), status="what_if")
    limits = dict(walls or {})
    names = {w.spec.name for w in mm.walls}
    unknown = set(limits) - names
    if unknown:
        raise KeyError(f"{mm.target}: no wall {sorted(unknown)}")
    ws = tuple(
        dataclasses.replace(w, limit=float(limits[w.spec.name])) if w.spec.name in limits else w
        for w in mm.walls
    )
    return dataclasses.replace(mm, resources=res, walls=ws)


def _shares(mm: MachineModel, resources: Iterable[str], resident: int, per_sm: int) -> dict:
    sm = mm.scope_counts.get("sm", 1.0)
    occupied = min(sm, math.ceil(resident / max(per_sm, 1)))
    out: dict[str, float] = {}
    for r in resources:
        spec = mm.resources.get(r)
        out[r] = 1.0 if spec is None or spec.scope == CHIP else occupied / sm
    return out


def _infeasible(l1: L1Result, mm: MachineModel) -> L2Result:
    cut = [w.name for w in l1.walls if not w.ok and w.cliff == "no_launch"]
    return L2Result(
        math.inf,
        False,
        "",
        {},
        {},
        {},
        {},
        (),
        {},
        0,
        0,
        0,
        None,
        0.0,
        0.0,
        l1.walls,
        dict(l1.unpriced),
        dict(l1.unmodelled),
        l1.total_us,
        f"l2:{SIM_VERSION}:{mm.version}",
        (f"no launch: wall {cut}",),
    )


def simulate(
    effects: EvalEffects,
    mm: MachineModel,
    features: Mapping[str, Any] | None = None,
    attribute_walls: bool = True,
) -> L2Result:
    """Simulate ``effects`` on ``mm`` (deterministic; the same inputs give the same result)."""
    l1 = price(effects, mm, features)
    if not l1.feasible:
        return _infeasible(l1, mm)
    feats = {**({"residency": l1.residency} if l1.residency else {}), **(features or {})}
    used = {r for _, _, times in l1.per_tx for r in times}
    eff = {r: mm.efficiency_of(r, feats) for r in used}
    g = build_graph(effects, l1, mm, eff)
    notes = list(g.notes)
    sm = round(mm.scope_counts.get("sm", 1.0))
    if l1.residency is None:
        notes.append("no residency wall applies: every CTA is resident at once")
    per_sm = l1.residency or math.ceil(g.ctas / sm)
    concurrent = min(g.ctas, per_sm * sm)
    full, tail = divmod(g.ctas, concurrent)
    resources = sorted({r for op in g.ops.values() for r in op.base_us})
    sims: dict[str, tuple[Wave, int, int]] = {}
    if full:
        sims["full"] = (
            simulate_wave(g, concurrent, _shares(mm, resources, concurrent, per_sm)),
            full,
            concurrent,
        )
    if tail:
        sims["tail"] = (simulate_wave(g, tail, _shares(mm, resources, tail, per_sm)), 1, tail)
    busy, idle, waves = _account(g, sims, l1.composed.interaction_us, used)
    total = g.prefix_us + sum(w.duration * n for w, n, _ in sims.values())
    total += l1.composed.interaction_us
    result = _result(l1, mm, g, total, busy, idle, waves, sims, notes)
    if attribute_walls:
        result = _attribute_walls(result, effects, mm, features, sm)
    return result


def _account(
    g: Graph, sims: Mapping[str, tuple[Wave, int, int]], inter_us: float, used: Iterable[str]
) -> tuple[dict, dict, tuple[WaveSpan, ...]]:
    busy: dict[str, float] = {r: us for r, us in g.prefix_busy_us.items()}
    idle: dict[str, dict[str, float]] = {}

    def add(r: str, cat: str, us: float) -> None:
        if us > EPS:
            slot = idle.setdefault(r, dict.fromkeys(CATEGORIES, 0.0))
            slot[cat] += us

    for r in used:
        add(r, "prologue_epilogue_tail", g.prefix_us - g.prefix_busy_us.get(r, 0.0))
        add(r, "other_unit", inter_us)
    waves: list[WaveSpan] = []
    t = g.prefix_us
    for kind, (w, n, ctas) in sims.items():
        for r in used:
            b = w.busy.get(r, 0.0)
            busy[r] = busy.get(r, 0.0) + n * b
            if r not in w.busy:
                add(r, "other_unit", n * w.duration)
            elif kind == "tail":
                add(r, "prologue_epilogue_tail", w.duration - b)
            else:
                for cat, us in w.idle.get(r, {}).items():
                    add(r, cat, n * us)
        for _ in range(n):
            waves.append(WaveSpan(len(waves), kind, ctas, t, t + w.duration))
            t += w.duration
    return busy, idle, tuple(waves)


def _result(
    l1: L1Result,
    mm: MachineModel,
    g: Graph,
    total: float,
    busy: dict,
    idle: dict,
    waves: tuple[WaveSpan, ...],
    sims: Mapping,
    notes: list[str],
) -> L2Result:
    bottleneck = max(busy, key=lambda r: (busy[r], r)) if busy else ""
    stalls = {STALL_NCU[c]: us for c, us in idle.get(bottleneck, {}).items() if us > 0}
    metrics: dict[str, float] = {"gpu__time_duration.sum": total * 1e3, "launch__grid_size": g.ctas}
    if mm.clock_hz:
        metrics["sm__cycles_elapsed.max"] = total / US_PER_S * mm.clock_hz
    peak = dict(l1.parts.busy_peak_us)
    for vec in l1.parts.serial_peak_us:
        for r, us in vec.items():
            peak[r] = peak.get(r, 0.0) + us
    for r, b in busy.items():
        spec = mm.resources.get(r)
        if spec is None or not spec.ncu or spec.rate is None:
            continue
        metrics[spec.ncu[0]] = peak.get(r, 0.0) / US_PER_S * spec.rate
        metrics[f"{spec.ncu[0]}.pct_of_peak_sustained_elapsed"] = 100.0 * b / total
    return L2Result(
        total_us=total,
        feasible=True,
        bottleneck=bottleneck,
        busy_us=busy,
        idle_us=idle,
        stalls=stalls,
        metrics=metrics,
        waves=waves,
        intervals={k: tuple(w.intervals) for k, (w, _, _) in sims.items()},
        ctas=g.ctas,
        trips=g.trips,
        inflight=g.inflight,
        residency=l1.residency,
        prefix_us=g.prefix_us,
        interaction_us=l1.composed.interaction_us,
        walls=l1.walls,
        unpriced=dict(l1.unpriced),
        unmodelled=dict(l1.unmodelled),
        l1_total_us=l1.total_us,
        model=f"l2:{SIM_VERSION}:{mm.version}",
        notes=tuple(notes),
    )


_WALL_FROM = ("data", "prologue_epilogue_tail", "other_unit", "sync")


def binding_wall(r: L2Result, sm: int) -> WallCheck | None:
    """The residency wall that limits how many CTAs run at once, if it binds (D504)."""
    if r.residency is None or r.ctas <= r.residency * sm:
        return None
    walls = [w for w in r.walls if w.residency == r.residency and w.usage > 0]
    return min(walls, key=lambda w: w.name) if walls else None


def _attribute_walls(
    r: L2Result, effects: EvalEffects, mm: MachineModel, features: Mapping[str, Any] | None, sm: int
) -> L2Result:
    w = binding_wall(r, sm)
    if w is None or r.residency is None:
        return r
    relaxed = simulate(
        effects,
        moved(mm, walls={w.name: (r.residency + 1) * w.usage}),
        features,
        attribute_walls=False,
    )
    idle = {res: dict(c) for res, c in r.idle_us.items()}
    for res, cats in idle.items():
        removed = sum(cats.values()) - sum(relaxed.idle_us.get(res, {}).values())
        for cat in _WALL_FROM:
            take = min(max(removed, 0.0), cats[cat])
            cats[cat] -= take
            cats["wall"] += take
            removed -= take
    stalls = {STALL_NCU[c]: us for c, us in idle.get(r.bottleneck, {}).items() if us > 0}
    return dataclasses.replace(
        r,
        idle_us=idle,
        stalls=stalls,
        notes=(*r.notes, f"wall {w.name} binds residency {r.residency}"),
    )
