"""Real-run cases of WP3's measurement acceptance: simple kernels (decision D94 allows them).

- ``vadd``: f32 vector add, memory bound; the noise probe's first kernel.
- ``gemm``: FP16 tiled GEMM with f32 accumulation (16x16 shared-memory tiles), M, N, K as
  runtime scalars; the noise probe's second kernel and the checkers' correct kernel.
- ``fma``: a dependent FMA chain per element whose length is a macro; the variants with
  1000 and 1050 steps form the pair "known to differ by about 5%" for ``compare``. It is
  a timing-only problem: its reference is not defined, so it is never numerically checked.

The Problem, Contract and Case records are written with the tool actor; they describe
the selftest workloads and are not part of any user Case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kta.backends.base import Backend, BuiltModule, GeneratedKernel, LaunchSpec, Param
from kta.core.context import Clock
from kta.core.errors import ExecutionError
from kta.core.ids import record_id
from kta.core.jobs import JobRunner
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case, Contract, Problem, TensorSpec
from kta.core.schemas.hardware import Target
from kta.store.base import Store

ACTOR = Actor(type="tool", name="kta measure selftest")
TILE = 16

VADD = """\
extern "C" __global__ void vadd(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}
"""

GEMM_HEAD = """\
#include <cuda_fp16.h>
#define TILE 16
extern "C" __global__ void gemm(const __half* __restrict__ A, const __half* __restrict__ B,
                                __half* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    int ka = k0 + threadIdx.x, kb = k0 + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? __half2float(A[row * K + ka]) : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (kb < K && col < N) ? __half2float(B[kb * N + col]) : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
"""
GEMM_STORE = "  if (row < M && col < N) C[row * N + col] = __float2half(acc);\n}\n"
GEMM = GEMM_HEAD + GEMM_STORE

FMA = """\
#ifndef STEPS
#define STEPS 1000
#endif
extern "C" __global__ void fma_chain(const float* __restrict__ X, float* __restrict__ Y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float y = X[i];
#pragma unroll 8
  for (int k = 0; k < STEPS; ++k) y = fmaf(y, 0.999f, 0.001f);
  Y[i] = y;
}
"""
FMA_N = 1 << 20

F32_TOL = {"rtol": 1e-5, "atol": 1e-6, "rel_l2": 1e-6}
F16_TOL = {"rtol": 1e-2, "atol": 1e-2, "rel_l2": 1e-3}


def vadd_kernel(n: int) -> GeneratedKernel:
    params = [Param("A", "f32*"), Param("B", "f32*"), Param("C", "f32*"), Param("n", "i32")]
    return GeneratedKernel(
        "cuda", {"kernel.cu": VADD}, "vadd", LaunchSpec([(n + 255) // 256], [256], 0, params)
    )


def gemm_kernel(shape: dict[str, Any], source: str = GEMM) -> GeneratedKernel:
    m, n = int(shape["M"]), int(shape["N"])
    params = [
        Param("A", "f16*"),
        Param("B", "f16*"),
        Param("C", "f16*"),
        Param("M", "i32"),
        Param("N", "i32"),
        Param("K", "i32"),
    ]
    grid = [(n + TILE - 1) // TILE, (m + TILE - 1) // TILE]
    return GeneratedKernel(
        "cuda", {"kernel.cu": source}, "gemm", LaunchSpec(grid, [TILE, TILE], 0, params)
    )


def fma_kernel(steps: int) -> GeneratedKernel:
    params = [Param("X", "f32*"), Param("Y", "f32*"), Param("n", "i32")]
    source = f"#define STEPS {steps}\n{FMA}"
    return GeneratedKernel(
        "cuda", {"kernel.cu": source}, "fma_chain", LaunchSpec([FMA_N // 256], [256], 0, params)
    )


PROBLEMS: dict[str, Problem] = {
    "vadd": Problem(
        name="selftest_vadd_f32",
        reference_impl="C = A + B",
        inputs=[
            TensorSpec(name="A", shape=["n"], dtype="f32", layout="contiguous"),
            TensorSpec(name="B", shape=["n"], dtype="f32", layout="contiguous"),
        ],
        outputs=[TensorSpec(name="C", shape=["n"], dtype="f32", layout="contiguous")],
    ),
    "gemm": Problem(
        name="selftest_gemm_f16",
        reference_impl="C = A @ B",
        inputs=[
            TensorSpec(name="A", shape=["M", "K"], dtype="f16", layout="row_major"),
            TensorSpec(name="B", shape=["K", "N"], dtype="f16", layout="row_major"),
        ],
        outputs=[TensorSpec(name="C", shape=["M", "N"], dtype="f16", layout="row_major")],
        params={"accumulate": "f32"},
    ),
    "fma": Problem(
        name="selftest_fma_chain_timing",
        reference_impl="none: timing-only problem (the step count differs per variant)",
        inputs=[TensorSpec(name="X", shape=["n"], dtype="f32", layout="contiguous")],
        outputs=[TensorSpec(name="Y", shape=["n"], dtype="f32", layout="contiguous")],
    ),
}
CONTRACTS: dict[str, Contract] = {
    "vadd": Contract(
        tolerances={"f32": F32_TOL},
        input_dists=["normal", "zeros", "extreme"],
        timing_boundary="single kernel launch",
    ),
    "gemm": Contract(
        tolerances={"f16": F16_TOL},
        input_dists=["normal", "zeros", "extreme", "cancellation"],
        edge_shapes=[{"M": 1, "N": 64, "K": 64}, {"M": 37, "N": 45, "K": 23}],
        timing_boundary="single kernel launch",
    ),
    "fma": Contract(tolerances={}, input_dists=[], timing_boundary="single kernel launch"),
}
SHAPES: dict[str, dict[str, Any]] = {
    "vadd": {"n": 1 << 24},
    "gemm": {"M": 1024, "N": 1024, "K": 1024},
    "fma": {"n": FMA_N},
}


@dataclass(frozen=True)
class Workload:
    name: str
    problem_id: str
    contract_id: str
    case_id: str
    case: Case
    problem: Problem
    contract: Contract
    shape: dict[str, Any]


def workload(store: Store, clock: Clock, name: str, target_id: str) -> Workload:
    """Problem, Contract and Case records of a selftest workload (reused when present)."""
    problem, contract, shape = PROBLEMS[name], CONTRACTS[name], SHAPES[name]
    pid = _ensure(store, clock, "Problem", problem.model_dump(), {"body.name": problem.name}, {})
    cid = _ensure(store, clock, "Contract", contract.model_dump(), {}, {"problem": pid})
    case = Case(
        problem=pid,
        shape_dist=[{"shape": shape, "weight": 1.0}],
        objective="weighted_latency",
        target=target_id,
    )
    case_id = _ensure(
        store, clock, "Case", case.model_dump(), {}, {"problem": pid, "contract": cid}
    )
    return Workload(name, pid, cid, case_id, case, problem, contract, shape)


def _ensure(
    store: Store,
    clock: Clock,
    kind: str,
    body: dict[str, Any],
    where: dict[str, Any],
    refs: dict[str, str | list[str]],
) -> str:
    for rec in store.query(kind, {**where, "actor.name": ACTOR.name}, order="created_at"):
        if rec.body == body and rec.refs == refs:
            return record_id(rec)
    return store.put(Record.new(kind, body, ACTOR, clock.now(), refs=refs))


def build(backend: Backend, gk: GeneratedKernel, target: Target, ex: JobRunner) -> BuiltModule:
    result, module = backend.build(gk, target, ex)
    if not result.ok:
        raise ExecutionError(
            "nonzero", f"selftest kernel {gk.entry} did not build: {result.log_summary[-800:]}"
        )
    return module
