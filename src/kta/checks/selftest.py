"""Real-run acceptance of the checkers: a correct GEMM and three deliberately wrong variants.

All four are real modules built and run on the GPU worker (no simulated outputs):

- ``correct``: the selftest FP16 tiled GEMM (``kta.measure.selftest.GEMM``);
- ``oob_write``: the same, plus one thread writing one element past the end of C;
- ``missing_row``: the same, but the last row of C is never written;
- ``cublas``: a host shared library whose entry calls ``cublasGemmEx`` (correct
  numbers, but the work is done by a library).

Expected verdicts (spec/modules/measure-check.md "验收"): pass; sanitizer memcheck
failure; NaN coverage (and numeric) failure; library and kernel-trace failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from kta.backends import common
from kta.backends.base import Backend, BuiltModule, GeneratedKernel
from kta.checks.base import Checked, Checker, actor
from kta.checks.full import run_full
from kta.core.errors import ExecutionError
from kta.core.jobs import Job
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Build
from kta.core.schemas.hardware import Target
from kta.core.schemas.process import Report
from kta.measure import selftest as ms

log = logging.getLogger("kta.checks.selftest")
SHAPE = {"M": 128, "N": 128, "K": 128}
OOB_STORE = (
    "  if (row < M && col < N) C[row * N + col] = __float2half(acc);\n"
    "  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)\n"
    "    C[M * N] = __float2half(acc);  // one element past the end\n}\n"
)
MISSING_ROW_STORE = "  if (row < M - 1 && col < N) C[row * N + col] = __float2half(acc);\n}\n"
CUBLAS_SO = """\
#include <cublas_v2.h>
#include <cuda_fp16.h>
extern "C" void gemm(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) cublasCreate(&handle);
  const float alpha = 1.f, beta = 0.f;
  // Row-major C = A B is column-major C^T = B^T A^T.
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A,
               CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
"""
VARIANTS = ("correct", "oob_write", "missing_row", "cublas")
EXPECTED = {
    "correct": set[str](),
    "oob_write": {"sanitizer.memcheck"},
    "missing_row": {"anticheat.nan_coverage", "numeric"},
    "cublas": {"anticheat.libraries", "anticheat.kernels"},
}


def variant_kernel(variant: str, shape: dict[str, Any]) -> GeneratedKernel:
    store = {"correct": ms.GEMM_STORE, "oob_write": OOB_STORE, "missing_row": MISSING_ROW_STORE}
    if variant == "cublas":
        gk = ms.gemm_kernel(shape)
        return GeneratedKernel("cuda", {"kernel.cu": CUBLAS_SO}, "gemm", gk.launch)
    return ms.gemm_kernel(shape, ms.GEMM_HEAD + store[variant])


def build_shared_library(
    c: Checker, gk: GeneratedKernel, target: Target, flags: list[str]
) -> BuiltModule:
    """nvcc -shared ... -lcublas on the worker; a Build record with kind ``so``."""
    so_flags = [*flags, "-shared", "-Xcompiler", "-fPIC", "-lcublas"]
    key = common.build_key(gk, target, so_flags, None, c.ex)
    src = c.ex.put_blob(gk.files["kernel.cu"].encode(), "text/x-cuda")
    cmd = "{cuda_home}/bin/nvcc " + " ".join(so_flags) + " -o out/module.so kernel.cu"
    job = Job(
        kind="compile",
        argv=["sh", "-c", cmd],
        requires={"backend": "cuda"},
        inputs={"kernel.cu": src},
    )
    result = c.ex.run(job)
    so = result.outputs.get("module.so")
    log_text = c.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")
    body = Build(ok=result.ok and so is not None, log_summary=log_text[-1500:] or "ok")
    blobs = {
        "build_key": key.sha256,
        "log": result.stderr_blob,
        "src/kernel.cu": src,
        "entry_name": c.ex.put_blob(gk.entry.encode(), "text/plain"),
    }
    if so is not None:
        blobs["module"] = so
    refs: dict[str, str | list[str]] = {"env": result.env_manifest, "worker": result.worker}
    if (tid := common.latest_target_id(c.store, target)) is not None:
        refs["target"] = tid
    build_id = c.store.put(
        Record.new("Build", body, actor("checks selftest"), c.clock.now(), refs=refs, blobs=blobs)
    )
    if not body.ok:
        raise ExecutionError("nonzero", f"shared library did not build: {log_text[-800:]}")
    return BuiltModule(so or "", "so", {}, None, gk.entry, build_id)


@dataclass(frozen=True)
class VariantResult:
    variant: str
    check: Checked
    failed_checks: set[str]
    as_expected: bool


def run_variant(
    c: Checker, cuda: Backend, target: Target, flags: list[str], w: ms.Workload, variant: str
) -> VariantResult:
    def build_for(shape: dict[str, Any]) -> tuple[GeneratedKernel, BuiltModule]:
        gk = variant_kernel(variant, shape)
        if variant == "cublas":
            return gk, build_shared_library(c, gk, target, flags)
        return gk, ms.build(cuda, gk, target, c.ex)

    gk, module = build_for(SHAPE)
    chk = run_full(
        c,
        module,
        gk,
        w.problem,
        w.contract,
        w.case,
        case_id=w.case_id,
        shape=SHAPE,
        build_for=build_for,
    )
    failed = {str(f["check"]) for f in chk.body.failures}
    expected = EXPECTED[variant]
    ok = (chk.passed if not expected else expected <= failed) and not chk.checker_error
    log.info("variant %s: passed=%s failed=%s expected=%s", variant, chk.passed, failed, expected)
    return VariantResult(variant, chk, failed, ok)


def run_all(
    c: Checker,
    cuda: Backend,
    target: Target,
    target_id: str,
    flags: list[str],
    variants: tuple[str, ...] = VARIANTS,
) -> tuple[str, list[VariantResult]]:
    w = ms.workload(c.store, c.clock, "gemm", target_id)
    results = [run_variant(c, cuda, target, flags, w, v) for v in variants]
    data = {
        r.variant: {
            "check": r.check.id,
            "passed": r.check.passed,
            "failed_checks": sorted(r.failed_checks),
            "expected": sorted(EXPECTED[r.variant]),
            "as_expected": r.as_expected,
        }
        for r in results
    }
    summary = "; ".join(
        f"{r.variant}: {'pass' if r.check.passed else 'fail ' + ','.join(sorted(r.failed_checks))}"
        for r in results
    )
    body = Report(title="checker selftest", subject="checks", summary=summary, data=data)
    refs: dict[str, str | list[str]] = {"checks": [r.check.id for r in results], "case": w.case_id}
    rid = c.store.put(
        Record.new("Report", body, actor("checks selftest"), c.clock.now(), refs=refs)
    )
    return rid, results
