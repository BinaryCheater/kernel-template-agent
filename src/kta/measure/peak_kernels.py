"""Microbenchmark kernels for per-unit peaks (docs/design/50 §6): one kernel per unit.

Each kernel does a known amount of work on one unit and as little as possible on the
others; its peak is that work divided by the median time under the measurement protocol.
Work is counted in the units the Templates' cost terms use:

- ``dram``: bytes read plus bytes written by a streaming copy of a buffer larger than L2
  (the protocol flushes L2 before every launch);
- ``l2``: bytes read with ``ld.global.cg`` (L2 only) from a 4 MiB buffer read 256 times;
  the first pass comes from DRAM (under 1% of the work);
- ``smem``: shared-memory wavefronts (128 bytes each): every warp-wide 16-byte load of 32
  consecutive vectors is 4 wavefronts;
- ``tensor``: FLOPs of ``mma.sync.m16n8k16`` FP16 x FP16 + FP32, 8 independent accumulators
  per warp;
- ``fma``: FP32 FMAs, 8 independent chains per thread; the same run gives ``issue`` in warp
  instructions (FFMA issues at one warp instruction per scheduler per cycle on sm_12x);
- ``launch``: CTAs of an empty kernel dispatched per second.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kta.backends.base import Param

DRAM = """\
extern "C" __global__ void peak_dram(const float4* __restrict__ X, float4* __restrict__ Y,
                                     int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) Y[i] = X[i];
}
"""
L2 = """\
extern "C" __global__ void peak_l2(const float4* __restrict__ X, float* __restrict__ Y, int n4,
                                   int reps) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc = 0.f;
  for (int r = 0; r < reps; ++r)
    for (int i = tid; i < n4; i += stride) {
      float4 v = __ldcg(&X[i]);
      acc += v.x + v.y + v.z + v.w;
    }
  Y[tid] = acc;
}
"""
SMEM = """\
extern "C" __global__ void peak_smem(unsigned* __restrict__ Y, int reps) {
  __shared__ uint4 buf[1024];
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) buf[i] = make_uint4(i, i + 1, i + 2, i + 3);
  __syncthreads();
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint4 acc = make_uint4(0, 0, 0, 0);
#pragma unroll 8
  for (int r = 0; r < reps; ++r) {
    uint4 v = buf[((r + warp) * 32 + lane) & 1023];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  Y[blockIdx.x * blockDim.x + threadIdx.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
"""
TENSOR = """\
#define CHAINS 8
extern "C" __global__ void peak_tensor(float* __restrict__ Y, int reps) {
  unsigned a0 = 0x14001400u + (threadIdx.x & 1), a1 = 0x14001400u, a2 = 0x14001400u,
           a3 = 0x14001400u, b0 = 0x14001400u, b1 = 0x14001400u;
  float c[CHAINS][4];
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  Y[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""
FMA = """\
#define CHAINS 8
extern "C" __global__ void peak_fma(float* __restrict__ Y, int reps) {
  float v[CHAINS];
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) v[j] = threadIdx.x * 1e-3f + j;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) v[j] = fmaf(v[j], 0.9999f, 1e-4f);
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) s += v[j];
  Y[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""
LAUNCH = """\
extern "C" __global__ void peak_launch(float* __restrict__ Y) {
  if (blockIdx.x == 0 && threadIdx.x == 0) Y[0] = 1.f;
}
"""

SM_COUNT = 48  # GB10 (WP2 probe); grids are multiples of it so every SM gets equal work
MMA_FLOPS = 2 * 16 * 8 * 16
CHAINS = 8


@dataclass(frozen=True)
class PeakKernel:
    unit: str
    source: str
    entry: str
    grid: int
    block: int
    params: list[Param]
    inputs: list[tuple[str, str]]  # (tensor name, element-count variable), f32
    outputs: list[tuple[str, str]]
    shape: dict[str, int]
    work: Callable[[dict[str, int]], dict[str, float]]  # unit -> work of one launch
    meaning: str


def _threads(s: dict[str, Any]) -> int:
    return int(s["grid"]) * int(s["block"])


def kernels(sm_count: int = SM_COUNT) -> list[PeakKernel]:
    n4 = 1 << 21  # 32 MiB per buffer: larger than the 24 MiB L2
    l2_n4 = 1 << 18  # 4 MiB
    grid4 = sm_count * 4
    return [
        PeakKernel(
            "dram",
            DRAM,
            "peak_dram",
            n4 // 256,
            256,
            [Param("X", "f32*"), Param("Y", "f32*"), Param("n4", "i32")],
            [("X", "n")],
            [("Y", "n")],
            {"n": 4 * n4, "n4": n4, "grid": n4 // 256, "block": 256},
            lambda s: {"dram": 2.0 * 16 * s["n4"]},
            "streaming float4 copy of 32 MiB (read + write bytes)",
        ),
        PeakKernel(
            "l2",
            L2,
            "peak_l2",
            grid4,
            256,
            [Param("X", "f32*"), Param("Y", "f32*"), Param("n4", "i32"), Param("reps", "i32")],
            [("X", "n")],
            [("Y", "m")],
            {
                "n": 4 * l2_n4,
                "n4": l2_n4,
                "reps": 256,
                "m": grid4 * 256,
                "grid": grid4,
                "block": 256,
            },
            lambda s: {"l2": 16.0 * s["n4"] * s["reps"]},
            "ld.global.cg of a 4 MiB buffer, 256 passes (bytes read)",
        ),
        PeakKernel(
            "smem",
            SMEM,
            "peak_smem",
            grid4,
            256,
            [Param("Y", "u32*"), Param("reps", "i32")],
            [],
            [("Y", "m")],
            {"reps": 8192, "m": grid4 * 256, "grid": grid4, "block": 256},
            lambda s: {"smem": 4.0 * (_threads(s) // 32) * s["reps"]},
            "warp-wide 16-byte shared loads, conflict free (128-byte wavefronts)",
        ),
        PeakKernel(
            "tensor",
            TENSOR,
            "peak_tensor",
            grid4,
            256,
            [Param("Y", "f32*"), Param("reps", "i32")],
            [],
            [("Y", "m")],
            {"reps": 2048, "m": grid4 * 256, "grid": grid4, "block": 256},
            lambda s: {"tensor": float(MMA_FLOPS * CHAINS * (_threads(s) // 32) * s["reps"])},
            "mma.sync m16n8k16 f16.f16.f32, 8 accumulators per warp (FLOPs)",
        ),
        PeakKernel(
            "fma",
            FMA,
            "peak_fma",
            grid4,
            256,
            [Param("Y", "f32*"), Param("reps", "i32")],
            [],
            [("Y", "m")],
            {"reps": 16384, "m": grid4 * 256, "grid": grid4, "block": 256},
            lambda s: {
                "fma": float(CHAINS * _threads(s) * s["reps"]),
                "issue": float(CHAINS * (_threads(s) // 32) * s["reps"]),
            },
            "FP32 FMA, 8 chains per thread (FMAs; warp instructions for issue)",
        ),
        PeakKernel(
            "launch",
            LAUNCH,
            "peak_launch",
            sm_count * 2048,
            32,
            [Param("Y", "f32*")],
            [],
            [("Y", "m")],
            {"m": 1, "grid": sm_count * 2048, "block": 32},
            lambda s: {"launch": float(s["grid"])},
            "empty kernel of 98304 CTAs (CTAs dispatched)",
        ),
    ]
