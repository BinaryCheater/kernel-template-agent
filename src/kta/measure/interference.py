"""Pairwise interference microbenchmarks (docs/design/02 §4, D-T15; decision D469).

For two operation classes A and B, three kernels run the same loop: A alone, B alone,
and A and B interleaved in one instruction stream. The interference matrix
(``kta.core.schemas.hwmodel.InterferenceMatrix``) records the three times per pair;
``slowdown = together / max(alone)``: 1 when the two overlap fully, (a + b) / max(a, b)
when they serialise. A pair that measurably interferes is evidence for a shared resource
in both classes' reservation tables (or an InteractionSpec), never a silent fudge.

Only the job definitions (``plan``) and the pure matrix construction are exercised at
the code stage (D320); ``measure_pair`` runs the three kernels through the Measurer on
the GPU worker and is a pending real run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kta.backends.base import Backend, Param
from kta.core.errors import NotSupported
from kta.core.schemas.hardware import Target
from kta.core.schemas.hwmodel import InterferenceCell, InterferenceMatrix
from kta.measure.microbench import _measure  # pyright: ignore[reportPrivateUsage]
from kta.measure.peak_kernels import SM_COUNT, PeakKernel
from kta.measure.runner import Measurer

Mode = Literal["a", "b", "ab"]
GLOBAL_N4 = 1 << 18  # 4 MiB of float4: stays in L2


@dataclass(frozen=True)
class Probe:
    """Loop-body snippet that keeps one operation class busy (independent chains)."""

    op_class: str
    decl: str
    body: str
    sink: str  # expression folded into the output so the work is not dead code
    reads_global: bool = False


PROBES: dict[str, Probe] = {
    "ffma": Probe(
        "ffma",
        "float f0 = threadIdx.x * 1e-3f, f1 = f0 + 1.f, f2 = f0 + 2.f, f3 = f0 + 3.f;",
        "f0 = fmaf(f0, 0.9999f, 1e-4f); f1 = fmaf(f1, 0.9999f, 1e-4f);"
        " f2 = fmaf(f2, 0.9999f, 1e-4f); f3 = fmaf(f3, 0.9999f, 1e-4f);",
        "f0 + f1 + f2 + f3",
    ),
    "mufu": Probe(
        "mufu",
        "float u0 = threadIdx.x * 1e-6f, u1 = u0 + 0.5f;",
        "u0 = exp2f(-u0) * 0.5f; u1 = exp2f(-u1) * 0.5f;",
        "u0 + u1",
    ),
    "mma_sync_m16n8k16_f16": Probe(
        "mma_sync_m16n8k16_f16",
        "unsigned ha = 0x14001400u + (threadIdx.x & 1), hb = 0x14001400u;"
        " float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;",
        'asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},'
        ' {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)'
        ' : "r"(ha), "r"(hb));',
        "c0 + c1 + c2 + c3",
    ),
    "smem_access": Probe(
        "smem_access",
        "__shared__ uint4 sbuf[1024]; for (int i = threadIdx.x; i < 1024; i += blockDim.x)"
        " sbuf[i] = make_uint4(i, i, i, i); __syncthreads(); unsigned sx = 0;",
        "sx ^= sbuf[((r + (threadIdx.x >> 5)) * 32 + (threadIdx.x & 31)) & 1023].x;",
        "(float)sx",
    ),
    "global_load": Probe(
        "global_load",
        "float gx = 0.f; int gi = blockIdx.x * blockDim.x + threadIdx.x;",
        "{ float4 v = __ldcg(&X[(gi + r * gridDim.x * blockDim.x) & (n4 - 1)]);"
        " gx += v.x + v.w; }",
        "gx",
        reads_global=True,
    ),
    "warp_shuffle": Probe(
        "warp_shuffle",
        "float wx = threadIdx.x;",
        "wx += __shfl_xor_sync(0xffffffffu, wx, 1);",
        "wx",
    ),
}


@dataclass(frozen=True)
class InterferenceJob:
    a: str
    b: str
    mode: Mode
    kernel: PeakKernel


def _source(probes: list[Probe], entry: str) -> str:
    reads = any(p.reads_global for p in probes)
    head = "const float4* __restrict__ X, int n4, " if reads else ""
    decl = "\n  ".join(p.decl for p in probes)
    body = "\n    ".join(p.body for p in probes)
    sink = " + ".join(f"({p.sink})" for p in probes)
    return (
        f'extern "C" __global__ void {entry}({head}float* __restrict__ Y, int reps) {{\n'
        f"  {decl}\n"
        f"  for (int r = 0; r < reps; ++r) {{\n    {body}\n  }}\n"
        f"  Y[blockIdx.x * blockDim.x + threadIdx.x] = {sink};\n}}\n"
    )


def kernel(a: str, b: str, mode: Mode, sm_count: int = SM_COUNT, reps: int = 4096) -> PeakKernel:
    """The microbenchmark kernel of one pair in one mode (timing only, no unit work)."""
    missing = [c for c in {a, b} if c not in PROBES]
    if missing:
        raise NotSupported(f"no interference probe for operation classes {sorted(missing)}")
    probes = {"a": [PROBES[a]], "b": [PROBES[b]], "ab": [PROBES[a], PROBES[b]]}[mode]
    entry = f"intf_{mode}"
    grid, block = sm_count * 4, 256
    reads = any(p.reads_global for p in probes)
    params = [Param("Y", "f32*"), Param("reps", "i32")]
    shape = {"reps": reps, "m": grid * block, "grid": grid, "block": block}
    inputs: list[tuple[str, str]] = []
    if reads:
        params = [Param("X", "f32*"), Param("n4", "i32"), *params]
        shape |= {"n": 4 * GLOBAL_N4, "n4": GLOBAL_N4}
        inputs = [("X", "n")]
    return PeakKernel(
        unit=f"interference_{a}__{b}__{mode}",
        source=_source(probes, entry),
        entry=entry,
        grid=grid,
        block=block,
        params=params,
        inputs=inputs,
        outputs=[("Y", "m")],
        shape=shape,
        work=lambda s: {},
        meaning=f"interference probe: {a} and {b}, mode {mode}",
    )


def pairs(classes: list[str]) -> list[tuple[str, str]]:
    ordered = sorted(set(classes))
    return [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]]


def plan(classes: list[str], sm_count: int = SM_COUNT) -> list[InterferenceJob]:
    """Three jobs (A alone, B alone, A+B) per unordered pair of ``classes``."""
    return [
        InterferenceJob(a, b, mode, kernel(a, b, mode, sm_count))
        for a, b in pairs(classes)
        for mode in ("a", "b", "ab")
    ]


def cell(
    a: str, b: str, alone_a_s: float, alone_b_s: float, together_s: float, measurements: list[str]
) -> InterferenceCell:
    return InterferenceCell(
        a=a,
        b=b,
        status="measured",
        alone_a_s=alone_a_s,
        alone_b_s=alone_b_s,
        together_s=together_s,
        slowdown=together_s / max(alone_a_s, alone_b_s),
        measurements=measurements,
    )


def empty_matrix(target: str, arch: str | None, classes: list[str]) -> InterferenceMatrix:
    """Every pair unmeasured: the starting coverage (D-T14)."""
    return InterferenceMatrix(
        target=target,
        arch=arch,
        classes=sorted(set(classes)),
        cells=[InterferenceCell(a=a, b=b, status="unmeasured") for a, b in pairs(classes)],
    )


def measure_pair(
    m: Measurer, backend: Backend, target: Target, target_id: str, a: str, b: str, protocol: str
) -> InterferenceCell:
    """Run the three kernels of (a, b) on the GPU worker (pending real run, D320)."""
    times: dict[str, float] = {}
    ids: list[str] = []
    for mode in ("a", "b", "ab"):
        meas, _ = _measure(m, backend, target, target_id, kernel(a, b, mode), protocol)
        times[mode] = meas.body.median_us * 1e-6
        ids.append(meas.id)
    return cell(a, b, times["a"], times["b"], times["ab"], ids)
