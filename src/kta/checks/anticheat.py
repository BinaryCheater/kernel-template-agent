"""Anti-cheat checks (docs/design/50 §5).

One bench job runs the module under Nsight Systems (``nsys profile --trace=cuda``; the
allowed WP3 simplification instead of a CUPTI client, decision D221) with one normal
input set, then the same set again. From it:

- ``anticheat.kernels``: every kernel launched must be a declared one (a library call
  shows up as its own kernels, e.g. cuBLAS's ``*gemm*``);
- ``anticheat.nan_coverage``: outputs were NaN-filled before the launch, so a remaining
  NaN is an element the kernel never wrote;
- ``anticheat.determinism``: the second run of the same input gives the same bytes.

Without a job, ``anticheat.libraries`` reads the module's ELF dynamic section: it must
not link (or import symbols of) cuBLAS, cuDNN, PyTorch and similar libraries.
Contract and baseline identity are checked by ``full_check``, which knows the Case.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from kta.backends.base import BuiltModule, GeneratedKernel
from kta.checks import elf, reference
from kta.checks.base import Checked, Checker, Evidence, stderr_tail, write_check
from kta.core.context import Context
from kta.core.errors import CheckerError
from kta.core.ids import canonical_json
from kta.core.schemas.evaluation import Problem
from kta.measure import inputs as inp
from kta.measure import jobs

NAME = "anticheat_checker"
# Open tables: library name prefixes and symbol prefixes a kernel module must not use.
FORBIDDEN_LIBS = (
    "libcublas",
    "libcudnn",
    "libtorch",
    "libc10",
    "libcufft",
    "libcurand",
    "libcusparse",
    "libcusolver",
    "libnvinfer",
)
FORBIDDEN_SYMBOLS = (
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusparse",
    "cusolver",
    "_ZN2at",
    "_ZN3c10",
)
# nsys needs a writable TMPDIR; the job directory's .home is created by the worker.
NSYS = [
    "env",
    "TMPDIR={job}/.home",
    "{cuda_home}/bin/nsys",
    "profile",
    "--trace=cuda",
    "--sample=none",
    "--cpuctxsw=none",
    "--export=sqlite",
    "--force-overwrite=true",
    "--output=out/trace",
]
TIMEOUT_S = 900


def traced_kernels(db: bytes) -> list[str] | None:
    """Kernel names launched in an nsys sqlite export; None when CUDA was not traced."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.sqlite"
        path.write_bytes(db)
        con = sqlite3.connect(path)
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
            if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
                return [] if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else None
            rows = con.execute(
                "SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.shortName = s.id"
            ).fetchall()
        finally:
            con.close()
    return [str(r[0]) for r in rows]


def library_failures(data: bytes) -> list[dict[str, Any]]:
    info = elf.dynamic_info(data)
    libs = [n for n in info.needed if n.startswith(FORBIDDEN_LIBS)]
    syms = sorted({s for s in info.undefined if s.startswith(FORBIDDEN_SYMBOLS)})
    if not libs and not syms:
        return []
    return [{"check": "anticheat.libraries", "needed": libs, "symbols": syms[:20]}]


def _outputs(
    c: Checker, problem: Problem, shape: dict[str, Any], outputs: dict[str, str], suffix: str
) -> dict[str, np.ndarray[Any, Any]]:
    return {
        t.name: np.frombuffer(
            c.ex.read_blob(outputs[f"{t.name}{suffix}.bin"]), dtype=inp.np_dtype(t.dtype)
        )
        for t in problem.outputs
    }


def anticheat_check(
    c: Checker,
    module: BuiltModule,
    gk: GeneratedKernel,
    problem: Problem,
    shape: dict[str, Any],
    declared: list[str] | None = None,
) -> Checked:
    ev = Evidence()
    kernels_ok = set(declared or [module.entry])
    ev.failures += library_failures(c.ex.read_blob(module.sha256))
    rng = c.rng(f"checks.anticheat:{canonical_json(shape).decode()}")
    sets = [jobs.random_inputs(problem, shape, "normal", rng)]
    sm = reference.gpu_target(c).sm
    bench = jobs.bench_job(
        c.ex,
        c.store,
        [module],
        problem,
        shape,
        sets,
        "run",
        "check",
        {"gpu": sm},
        {"nan_fill": True, "repeat": 2},
        wrap=NSYS,
        timeout_s=TIMEOUT_S,
    )
    result = c.ex.run(bench.job)
    ev.job(result, "nsys")
    if result.error_kind == "infra":
        ev.error("anticheat", f"job {result.job_id}: {stderr_tail(c, result)}")
        return write_check(c, NAME, ev, module.build_id)
    if not result.ok:
        ev.failures.append(
            {
                "check": "anticheat.run",
                "error_kind": result.error_kind,
                "exit_code": result.exit_code,
                "detail": stderr_tail(c, result, 600),
            }
        )
        return write_check(c, NAME, ev, module.build_id)
    if "trace.sqlite" not in result.outputs:
        ev.error("anticheat.kernels", "nsys wrote no sqlite export")
    else:
        ev.blobs["trace"] = result.outputs["trace.sqlite"]
        try:
            names = traced_kernels(c.ex.read_blob(result.outputs["trace.sqlite"]))
        except sqlite3.DatabaseError as exc:
            raise CheckerError(f"unreadable nsys export: {exc}") from exc
        if names is None:
            ev.error("anticheat.kernels", "the nsys export has no CUDA activity")
        else:
            launched = sorted(set(names))
            ev.detail["kernels"] = {n: names.count(n) for n in launched}
            undeclared = [n for n in launched if n not in kernels_ok]
            missing = sorted(kernels_ok - set(launched))
            if undeclared or missing:
                ev.failures.append(
                    {
                        "check": "anticheat.kernels",
                        "undeclared": undeclared[:20],
                        "declared_not_launched": missing,
                    }
                )
    first = _outputs(c, problem, shape, result.outputs, ".s0")
    again = _outputs(c, problem, shape, result.outputs, ".s0.r1")
    for name, arr in first.items():
        nan = int(np.count_nonzero(np.isnan(arr.astype(np.float64))))
        if nan:
            ev.failures.append(
                {
                    "check": "anticheat.nan_coverage",
                    "output": name,
                    "unwritten": nan,
                    "elements": int(arr.size),
                }
            )
        differ = int(np.count_nonzero(arr.view(np.uint8) != again[name].view(np.uint8)))
        if differ:
            ev.failures.append(
                {"check": "anticheat.determinism", "output": name, "differing_bytes": differ}
            )
    return write_check(c, NAME, ev, module.build_id)


def check_anticheat(
    ctx: Context, module: BuiltModule, gk: GeneratedKernel, problem: Problem, shape: dict[str, Any]
) -> Checked:
    return anticheat_check(Checker.from_ctx(ctx), module, gk, problem, shape)
