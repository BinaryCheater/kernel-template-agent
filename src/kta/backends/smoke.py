"""Real-run checks of WP2: vector add per backend, GPU lease deferral, timeout recovery.

Deliberately simple kernels (decision D94 allows simple first cases). Every result is
a record: the Build from the backend, a Check comparing the GPU output with A + B
computed on the control plane, and Reports for the two executor behaviours with the
job timings taken from their spans.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kta.backends import harness
from kta.backends.base import Backend, GeneratedKernel, LaunchSpec, Param
from kta.core.context import Clock
from kta.core.errors import ExecutionError
from kta.core.jobs import Job, JobKind, JobResult, JobRunner
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Build, Check
from kta.core.schemas.hardware import Target
from kta.core.schemas.process import Report
from kta.store.base import Ops, Store

log = logging.getLogger("kta.backends.smoke")
CHECKER = Actor(type="checker", name="kta smoke vector-add@1")
REPORTER = Actor(type="tool", name="kta backend selftest")
BUILDER = Actor(type="tool", name="kta backend selftest")
N = 1 << 20
BLOCK = 256

CUDA_VADD = """\
extern "C" __global__ void vadd(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}
"""
TILELANG_VADD = f"""\
import tilelang.language as T

N = {N}
BLOCK = {BLOCK}


@T.prim_func
def vadd(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
    with T.Kernel(T.ceildiv(N, BLOCK), threads=BLOCK) as bx:
        for i in T.Parallel(BLOCK):
            C[bx * BLOCK + i] = A[bx * BLOCK + i] + B[bx * BLOCK + i]
"""
C_SOURCE = "int kta_add(int a, int b) { return a + b; }\n"
CUDA_SPIN = """\
extern "C" __global__ void spin(int* flag) {
  // Never true: the buffer is zeroed. The volatile read keeps the loop alive.
  while (*(volatile int*)flag == 0) { }
}
"""
POINTERS = [Param("A", "f32*"), Param("B", "f32*"), Param("C", "f32*")]


def vadd_kernel(backend: str) -> GeneratedKernel:
    grid = [N // BLOCK]
    if backend == "cuda":
        launch = LaunchSpec(grid, [BLOCK], 0, [*POINTERS, Param("n", "i32")])
        return GeneratedKernel("cuda", {"kernel.cu": CUDA_VADD}, "vadd", launch)
    launch = LaunchSpec(grid, [BLOCK], 0, POINTERS)
    return GeneratedKernel("tilelang", {"kernel.py": TILELANG_VADD}, "vadd", launch)


@dataclass(frozen=True)
class SmokeResult:
    backend: str
    build_id: str
    build_ok: bool
    check_id: str | None = None
    passed: bool = False
    run: JobResult | None = None
    detail: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


class Smoke:
    def __init__(self, store: Store, ops: Ops, clock: Clock, ex: JobRunner, target: Target) -> None:
        self.store = store
        self.ops = ops
        self.clock = clock
        self.ex = ex
        self.target = target
        self.gpu = {"gpu": target.sm}

    def _inputs(self, seed: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal(N, dtype=np.float32)
        b = rng.standard_normal(N, dtype=np.float32)
        return a, b

    def vector_add(
        self, backend: Backend, seed: int, kind: JobKind = "run", iters: int = 0
    ) -> SmokeResult:
        gk = vadd_kernel(backend.name)
        build, module = backend.build(gk, self.target, self.ex)
        if not build.ok:
            return SmokeResult(
                backend.name, module.build_id, False, detail={"log": build.log_summary}
            )
        a, b = self._inputs(seed)
        nbytes = N * 4
        buffers = {
            "A": harness.Buffer(nbytes, a.tobytes()),
            "B": harness.Buffer(nbytes, b.tobytes()),
            "C": harness.Buffer(nbytes),
        }
        scalars: dict[str, int | float] = {"n": N} if backend.name == "cuda" else {}
        job = harness.run_job(
            self.ex,
            module,
            gk.launch,
            buffers,
            scalars,
            ["C"],
            self.gpu,
            kind=kind,
            iters=iters,
            warmup=10 if iters else 0,
        )
        run = harness.collect(self.ex, self.ex.run(job), ["C"])
        return self._check(backend.name, module.build_id, seed, a, b, run)

    def _check(
        self, backend: str, build_id: str, seed: int, a: Any, b: Any, run: harness.HarnessRun
    ) -> SmokeResult:
        failures: list[dict[str, Any]] = []
        detail: dict[str, Any] = {"n": N, "seed": seed}
        if not run.result.ok or "C" not in run.outputs:
            failures.append(
                {
                    "check": "run",
                    "error_kind": run.result.error_kind,
                    "exit_code": run.result.exit_code,
                }
            )
        else:
            got = np.frombuffer(run.outputs["C"], dtype=np.float32)
            want = a + b  # float32 addition rounds exactly as the GPU's IEEE fp32 add
            mismatch = int(np.count_nonzero(got != want))
            detail.update(max_abs_err=float(np.max(np.abs(got - want))), mismatches=mismatch)
            if mismatch:
                failures.append(
                    {
                        "check": "exact_equal",
                        "mismatches": mismatch,
                        "max_abs_err": detail["max_abs_err"],
                    }
                )
        params = self.ex.put_blob(_json(detail), "application/json")
        blobs = {
            "params": params,
            "stdout": run.result.stdout_blob,
            "stderr": run.result.stderr_blob,
        }
        if "C.bin" in run.result.outputs:
            blobs["output_C"] = run.result.outputs["C.bin"]
        refs: dict[str, str | list[str]] = {
            "env": run.result.env_manifest,
            "worker": run.result.worker,
            "build": build_id,
        }
        body = Check(passed=not failures, failures=failures)
        check_id = self.store.put(
            Record.new("Check", body, CHECKER, self.clock.now(), refs=refs, blobs=blobs)
        )
        log.info("vector add (%s): passed=%s %s", backend, not failures, detail)
        return SmokeResult(backend, build_id, True, check_id, not failures, run.result, detail)

    # ---- executor behaviour ----------------------------------------------------------

    def _span_times(self, since: Any, job_ids: list[str]) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for row in self.ops.spans(since=since, limit=10_000):
            jid = row.attrs.get("job_id")
            if jid in job_ids and row.kind in ("job_queue", "job_run"):
                entry = out.setdefault(str(jid), {})
                entry[f"{row.kind}_start"] = row.start_ts.isoformat()
                entry[f"{row.kind}_end"] = row.end_ts.isoformat() if row.end_ts else ""
        return out

    def lease_deferral(self, cuda: Backend, lease_probe: Any) -> str:
        """An exclusive measurement holds the GPU; shared and cpu jobs submitted meanwhile wait."""
        since = self.clock.now()
        gk = vadd_kernel("cuda")
        _, module = cuda.build(gk, self.target, self.ex)
        a, b = self._inputs(1)
        buffers = {
            "A": harness.Buffer(N * 4, a.tobytes()),
            "B": harness.Buffer(N * 4, b.tobytes()),
            "C": harness.Buffer(N * 4),
        }
        exclusive = harness.run_job(
            self.ex,
            module,
            gk.launch,
            buffers,
            {"n": N},
            ["C"],
            self.gpu,
            kind="measure",
            iters=20000,
            warmup=100,
        )
        shared = harness.run_job(
            self.ex, module, gk.launch, buffers, {"n": N}, ["C"], self.gpu, kind="run"
        )
        cpu = Job(kind="shell", argv=["sh", "-c", "echo cpu job"], requires={})
        ex_id = self.ex.submit(exclusive)
        deadline = time.monotonic() + 120
        while not lease_probe():  # wait until the exclusive job holds the lease
            if time.monotonic() > deadline:
                raise ExecutionError("infra", "exclusive job never took the GPU lease")
            time.sleep(0.05)  # kta:allowed polls real executor state, not a test clock
        sh_id, cpu_id = self.ex.submit(shared), self.ex.submit(cpu)
        results = {
            "exclusive": self.ex.wait(ex_id),
            "shared": self.ex.wait(sh_id),
            "cpu": self.ex.wait(cpu_id),
        }
        times = self._span_times(since, [ex_id, sh_id, cpu_id])
        ex_end = times[ex_id]["job_run_end"]
        deferred = all(times[j]["job_run_start"] >= ex_end for j in (sh_id, cpu_id))
        data = {
            "jobs": {"exclusive": ex_id, "shared": sh_id, "cpu": cpu_id},
            "spans": times,
            "ok": {k: r.ok for k, r in results.items()},
            "queue_s": {k: round(r.queue_s, 3) for k, r in results.items()},
            "deferred_until_exclusive_end": deferred,
        }
        summary = (
            f"gpu-exclusive job {ex_id} ran {results['exclusive'].wall_s:.2f}s; gpu-shared and cpu "
            f"jobs submitted during it started after it ended: {deferred}"
        )
        return self._report(
            "wp2-gpu-lease-deferral",
            "GPU lease defers shared and cpu jobs",
            summary,
            data,
            results["exclusive"],
        )

    def timeout_recovery(self, cuda: Backend, seed: int, timeout_s: int = 10) -> str:
        """A kernel that never ends is killed at its timeout; the GPU then runs vector add."""
        gk = GeneratedKernel(
            "cuda",
            {"kernel.cu": CUDA_SPIN},
            "spin",
            LaunchSpec([1], [32], 0, [Param("flag", "i32*")]),
        )
        build, module = cuda.build(gk, self.target, self.ex)
        if not build.ok:
            raise ExecutionError("nonzero", f"spin kernel did not build: {build.log_summary}")
        job = harness.run_job(
            self.ex,
            module,
            gk.launch,
            {"flag": harness.Buffer(4)},
            {},
            [],
            self.gpu,
            timeout_s=timeout_s,
        )
        spin = self.ex.run(job)
        after = self.vector_add(cuda, seed)
        data = {
            "spin": {
                "job_id": spin.job_id,
                "error_kind": spin.error_kind,
                "exit_code": spin.exit_code,
                "wall_s": round(spin.wall_s, 3),
                "timeout_s": timeout_s,
                "build": module.build_id,
            },
            "after": {
                "check": after.check_id,
                "passed": after.passed,
                "job_id": after.run.job_id if after.run else None,
            },
        }
        summary = (
            f"spin kernel ended as {spin.error_kind} after {spin.wall_s:.1f}s "
            f"(timeout {timeout_s}s); "
            f"vector add afterwards passed: {after.passed}"
        )
        refs_extra = [after.check_id] if after.check_id else []
        return self._report(
            "wp2-timeout-recovery",
            "GPU recovers after a timed-out kernel",
            summary,
            data,
            spin,
            refs_extra,
        )

    def cpu_compile(self, requires: dict[str, str]) -> str:
        """A C compile job on a CPU worker chosen by labels; returns the Build record ID."""
        source = self.ex.put_blob(C_SOURCE.encode(), "text/x-c")
        job = Job(
            kind="compile",
            argv=["sh", "-c", "cc -O2 -c hello.c -o out/hello.o && cc --version | head -1"],
            requires=requires,
            inputs={"hello.c": source},
        )
        result = self.ex.run(job)
        log_text = self.ex.read_blob(result.stdout_blob).decode("utf-8", "replace")
        obj = result.outputs.get("hello.o")
        body = Build(ok=result.ok and obj is not None, log_summary=log_text[-1500:])
        blobs = {"src/hello.c": source, "log": result.stderr_blob, "stdout": result.stdout_blob}
        if obj is not None:
            blobs["module"] = obj
        refs: dict[str, str | list[str]] = {"env": result.env_manifest, "worker": result.worker}
        rec = Record.new("Build", body, BUILDER, self.clock.now(), refs=refs, blobs=blobs)
        return self.store.put(rec)

    def _report(
        self,
        subject: str,
        title: str,
        summary: str,
        data: dict[str, Any],
        on: JobResult,
        checks: list[str] | None = None,
    ) -> str:
        refs: dict[str, str | list[str]] = {"env": on.env_manifest, "worker": on.worker}
        if checks:
            refs["checks"] = checks
        body = Report(title=title, subject=subject, summary=summary, data=data)
        return self.store.put(Record.new("Report", body, REPORTER, self.clock.now(), refs=refs))


def _json(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True).encode()
