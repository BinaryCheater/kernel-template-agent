"""GB10 microbenchmarks beyond the per-unit peaks, and the T5 benchmark format (WP11).

WP7's ``kta.measure.peaks`` measures one peak per unit. This module adds, first version:

- ``l2_sweep``: the ``ld.global.cg`` read kernel of ``peak_kernels.L2`` over working sets
  from 1 MiB to 64 MiB. Each point is an HwFact ``read_bw_ws_<MiB>mib`` (unit ``l2``,
  bytes/s); the derived HwFact ``l2_effective_bytes`` is the largest working set whose
  bandwidth stays at or above the midpoint between the smallest-set and the largest-set
  bandwidth (where the L2 stops holding the set; WP13 B5 found model error at that edge);
- ``dram_latency``: one thread chasing a data-dependent pseudo-random index through
  64 MiB, ``dram_latency_s`` = time / hops (a first version: TLB misses are included);
- ``launch_latency``: one empty CTA, ``launch_latency_s`` = the measured kernel time.

Every value is computed from Measurements taken by ``Measurer.measure`` under a
measurement Protocol on the GPU worker; nothing here returns a number without a run.

The T5 format (``Microbench``) is what a worker agent submits for "write a
microbenchmark": ``microbench.yaml`` (quantity, unit, entry, launch, tensors, shape, work
expression, value form) plus the CUDA source. ``repeat`` measures it ``n`` times and the
done-check compares the coefficient of variation with the noise floor (D404).
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from kta.backends.base import Backend, Param
from kta.core.errors import SchemaError
from kta.core.expr import Expr
from kta.core.schemas import Actor, Record
from kta.core.schemas.hardware import HwFact, Target
from kta.measure.peak_kernels import L2, SM_COUNT, PeakKernel
from kta.measure.peaks import build, workload
from kta.measure.runner import Measured, Measurer

log = logging.getLogger("kta.measure.microbench")
ACTOR = Actor(type="tool", name="kta measure microbench")
MIB = 1 << 20
SWEEP_MIB = (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64)
SWEEP_BYTES_PER_POINT = 1 << 30  # each point reads about 1 GiB, so times are comparable

CHASE = """\
extern "C" __global__ void chase(const float* __restrict__ X, float* __restrict__ Y, int n,
                                 int hops) {
  unsigned idx = 0;
  float acc = 0.f;
  for (int h = 0; h < hops; ++h) {
    float v = __ldcg(&X[idx]);
    acc += v;
    // The next index depends on the loaded value (never equal to the sentinel in practice,
    // which the compiler cannot know), so loads cannot overlap.
    idx = (idx * 1103515245u + 12345u + (__float_as_uint(v) == 0x7fbfffffu)) % (unsigned)n;
  }
  Y[0] = acc;
}
"""
EMPTY = """\
extern "C" __global__ void empty_cta(float* __restrict__ Y) {
  if (threadIdx.x == 0) Y[0] = 0.f;
}
"""


@dataclass(frozen=True)
class FactResult:
    facts: dict[str, str]  # quantity -> HwFact ID
    values: dict[str, float]
    measurements: list[str]


def _l2_point(mib: int, sm_count: int) -> PeakKernel:
    n4 = mib * MIB // 16
    reps = max(1, SWEEP_BYTES_PER_POINT // (mib * MIB))
    grid = sm_count * 4
    return PeakKernel(
        "l2",
        L2,
        "peak_l2",
        grid,
        256,
        [Param("X", "f32*"), Param("Y", "f32*"), Param("n4", "i32"), Param("reps", "i32")],
        [("X", "n")],
        [("Y", "m")],
        {"n": 4 * n4, "n4": n4, "reps": reps, "m": grid * 256, "grid": grid, "block": 256},
        lambda s: {"l2": 16.0 * s["n4"] * s["reps"]},
        f"ld.global.cg read of a {mib} MiB working set, {reps} passes (bytes read)",
    )


def _chase(n: int, hops: int) -> PeakKernel:
    return PeakKernel(
        "dram",
        CHASE,
        "chase",
        1,
        1,
        [Param("X", "f32*"), Param("Y", "f32*"), Param("n", "i32"), Param("hops", "i32")],
        [("X", "n")],
        [("Y", "m")],
        {"n": n, "hops": hops, "m": 1, "grid": 1, "block": 1},
        lambda s: {"dram": float(s["hops"])},
        f"dependent pseudo-random loads over {4 * n // MIB} MiB, {hops} hops (loads)",
    )


def _empty() -> PeakKernel:
    return PeakKernel(
        "launch",
        EMPTY,
        "empty_cta",
        1,
        32,
        [Param("Y", "f32*")],
        [],
        [("Y", "m")],
        {"m": 1, "grid": 1, "block": 32},
        lambda s: {"launch": 1.0},
        "one empty CTA (launch to completion)",
    )


def _measure(
    m: Measurer, backend: Backend, target: Target, target_id: str, k: PeakKernel, protocol: str
) -> tuple[Measured, str]:
    case_id, case = workload(m.store, m.clock, k, target_id)
    meas = m.measure(build(backend, k, target, m.ex), case, k.shape, protocol)
    return meas, case_id


def _put_fact(
    m: Measurer, fact: HwFact, refs: Mapping[str, Any], measurement: str | list[str]
) -> str:
    key = "measurement" if isinstance(measurement, str) else "measurements"
    rec = Record.new("HwFact", fact, ACTOR, m.clock.now(), refs={**refs, key: measurement})
    return m.store.put(rec)


def effective_capacity(points: list[tuple[int, float]]) -> int:
    """Largest working set (bytes) whose bandwidth is at or above the midpoint between the
    smallest-set and the largest-set bandwidth; ``points`` = (bytes, bytes/s) by size."""
    if len(points) < 3:
        raise SchemaError("an L2 sweep needs at least three working-set sizes")
    pts = sorted(points)
    mid = (pts[0][1] + pts[-1][1]) / 2.0
    held = [size for size, bw in pts if bw >= mid]
    return max(held) if held else pts[0][0]


def l2_sweep(
    m: Measurer,
    backend: Backend,
    target: Target,
    target_id: str,
    protocol_id: str,
    sm_count: int = SM_COUNT,
    sizes_mib: tuple[int, ...] = SWEEP_MIB,
) -> FactResult:
    refs = {"target": target_id, "protocol": protocol_id}
    facts: dict[str, str] = {}
    values: dict[str, float] = {}
    meas_ids: list[str] = []
    points: list[tuple[int, float]] = []
    for mib in sizes_mib:
        k = _l2_point(mib, sm_count)
        meas, case_id = _measure(m, backend, target, target_id, k, protocol_id)
        work = k.work(k.shape)["l2"]
        bw = work / (meas.body.median_us * 1e-6)
        q = f"read_bw_ws_{mib}mib"
        fact = HwFact(
            unit="l2",
            quantity=q,
            value=bw,
            ci_low=work / (meas.body.ci_high_us * 1e-6),
            ci_high=work / (meas.body.ci_low_us * 1e-6),
            method="measured",
        )
        facts[q] = _put_fact(m, fact, {**refs, "case": case_id}, meas.id)
        values[q] = bw
        meas_ids.append(meas.id)
        points.append((mib * MIB, bw))
    cap = effective_capacity(points)
    fact = HwFact(unit="l2", quantity="l2_effective_bytes", value=float(cap), method="measured")
    facts["l2_effective_bytes"] = _put_fact(m, fact, refs, sorted(meas_ids))
    values["l2_effective_bytes"] = float(cap)
    log.info("L2 sweep: effective capacity %d MiB", cap // MIB)
    return FactResult(facts, values, meas_ids)


def dram_latency(
    m: Measurer,
    backend: Backend,
    target: Target,
    target_id: str,
    protocol_id: str,
    buffer_mib: int = 64,
    hops: int = 1 << 16,
) -> FactResult:
    k = _chase(buffer_mib * MIB // 4, hops)
    meas, case_id = _measure(m, backend, target, target_id, k, protocol_id)
    per_hop = meas.body.median_us * 1e-6 / hops
    fact = HwFact(
        unit="dram",
        quantity="dram_latency_s",
        value=per_hop,
        ci_low=meas.body.ci_low_us * 1e-6 / hops,
        ci_high=meas.body.ci_high_us * 1e-6 / hops,
        method="measured",
    )
    refs = {"target": target_id, "protocol": protocol_id, "case": case_id}
    fid = _put_fact(m, fact, refs, meas.id)
    return FactResult({"dram_latency_s": fid}, {"dram_latency_s": per_hop}, [meas.id])


def launch_latency(
    m: Measurer, backend: Backend, target: Target, target_id: str, protocol_id: str
) -> FactResult:
    k = _empty()
    meas, case_id = _measure(m, backend, target, target_id, k, protocol_id)
    latency = meas.body.median_us * 1e-6
    fact = HwFact(
        unit="launch",
        quantity="launch_latency_s",
        value=latency,
        ci_low=meas.body.ci_low_us * 1e-6,
        ci_high=meas.body.ci_high_us * 1e-6,
        method="measured",
    )
    refs = {"target": target_id, "protocol": protocol_id, "case": case_id}
    fid = _put_fact(m, fact, refs, meas.id)
    return FactResult({"launch_latency_s": fid}, {"launch_latency_s": latency}, [meas.id])


Suite = Callable[[Measurer, Backend, Target, str, str], FactResult]
SUITE: dict[str, Suite] = {
    "l2_sweep": l2_sweep,
    "dram_latency": dram_latency,
    "launch_latency": launch_latency,
}


# ---- T5: a submitted microbenchmark ---------------------------------------------


@dataclass(frozen=True)
class Microbench:
    """``microbench.yaml`` of a T5 submission (D404)."""

    quantity: str
    unit: str
    source: str
    entry: str
    grid: int
    block: int
    params: list[Param]
    inputs: list[tuple[str, str]]
    outputs: list[tuple[str, str]]
    shape: dict[str, int]
    work: str  # expression over the shape variables: the work of one launch
    value: Literal["work_per_s", "s_per_work"]
    meaning: str

    @staticmethod
    def from_doc(doc: Mapping[str, Any], source: str) -> Microbench:
        need = ("quantity", "unit", "entry", "grid", "block", "params", "shape", "work", "value")
        missing = [k for k in need if k not in doc]
        if missing:
            raise SchemaError(f"microbench.yaml misses {missing}")
        if doc["value"] not in ("work_per_s", "s_per_work"):
            raise SchemaError("microbench.yaml value must be work_per_s or s_per_work")
        shape = {str(k): int(v) for k, v in dict(doc["shape"]).items()}
        Expr.parse(str(doc["work"]), set(shape))  # rejects names outside the shape
        return Microbench(
            quantity=str(doc["quantity"]),
            unit=str(doc["unit"]),
            source=source,
            entry=str(doc["entry"]),
            grid=int(doc["grid"]),
            block=int(doc["block"]),
            params=[Param(str(p["name"]), str(p["type"])) for p in doc["params"]],
            inputs=[(str(a), str(b)) for a, b in doc.get("inputs", [])],
            outputs=[(str(a), str(b)) for a, b in doc.get("outputs", [])],
            shape=shape,
            work=str(doc["work"]),
            value=doc["value"],
            meaning=str(doc.get("meaning", doc["quantity"])),
        )

    def kernel(self) -> PeakKernel:
        expr = Expr.parse(self.work, set(self.shape))
        return PeakKernel(
            self.unit,
            self.source,
            self.entry,
            self.grid,
            self.block,
            self.params,
            self.inputs,
            self.outputs,
            {**self.shape, "grid": self.grid, "block": self.block},
            lambda s: {self.unit: float(expr.eval(s))},
            self.meaning,
        )

    def value_of(self, median_us: float) -> float:
        work = float(Expr.parse(self.work, set(self.shape)).eval(self.shape))
        seconds = median_us * 1e-6
        return work / seconds if self.value == "work_per_s" else seconds / work


@dataclass(frozen=True)
class Repeats:
    measurements: list[str]
    values: list[float]
    cv: float
    case: str


def repeat(
    m: Measurer,
    backend: Backend,
    target: Target,
    target_id: str,
    mb: Microbench,
    protocol_id: str,
    n: int = 5,
) -> Repeats:
    """Build once, measure ``n`` times (separate jobs), and return the value spread."""
    k = mb.kernel()
    case_id, case = workload(m.store, m.clock, k, target_id)
    module = build(backend, k, target, m.ex)
    ids: list[str] = []
    values: list[float] = []
    for _ in range(n):
        meas = m.measure(module, case, k.shape, protocol_id)
        ids.append(meas.id)
        values.append(mb.value_of(meas.body.median_us))
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean if mean else float("inf")
    return Repeats(ids, values, cv, case_id)


def write_repeat_fact(
    m: Measurer, mb: Microbench, rep: Repeats, target_id: str, protocol_id: str, actor: Actor
) -> str:
    """The HwFact of an accepted T5 benchmark: the median of the repeats."""
    fact = HwFact(
        unit=mb.unit,
        quantity=mb.quantity,
        value=statistics.median(rep.values),
        ci_low=min(rep.values),
        ci_high=max(rep.values),
        method="measured",
    )
    refs = {"target": target_id, "protocol": protocol_id, "case": rep.case}
    rec = Record.new(
        "HwFact", fact, actor, m.clock.now(), refs={**refs, "measurements": rep.measurements}
    )
    return m.store.put(rec)
