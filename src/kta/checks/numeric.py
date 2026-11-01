"""Numeric check: the kernel's outputs against a GPU reference (docs/design/50 §3).

Per shape, one ``check`` job runs the module once per input distribution with outputs
NaN-filled, and one reference job computes the same inputs in FP64. Criterion per
element: ``|y - ref| <= atol + rtol * |ref|`` (NaN where the reference is finite fails),
and the relative L2 error ``||y - ref|| / ||ref||`` must stay within ``rel_l2`` when the
Contract gives one. Hidden shapes are reported by category only.

Library-relative acceptance (SYSDEBUG-2, D695): when the Contract's tolerance for a dtype
says ``relative_to: library``, the reference job also runs the statement in the inputs' own
dtype (the library: cuBLAS for ``A @ B``), and a candidate passes at ``level`` (``loose`` for
the quick check before measuring, ``strict`` for the full check of a round's best) when its
relative L2 and largest absolute error stay within ``factor`` times the library's own, with
small floors so a library that is exact does not make every rounding a failure. Non-finite
output where the reference is finite always fails. The fixed element-wise counts are kept
as information.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kta.backends.base import BuiltModule
from kta.checks import reference
from kta.checks.base import Checked, Checker, Evidence, stderr_tail, write_check
from kta.checks.inputs import TABLE
from kta.core.context import Context
from kta.core.errors import CheckerError, ConfigError, ExecutionError
from kta.core.ids import canonical_json
from kta.core.schemas.evaluation import Contract, Problem
from kta.measure import inputs as inp
from kta.measure import jobs

NAME = "numeric_checker"


def compare_output(
    got: np.ndarray[Any, Any], ref: np.ndarray[Any, Any], tol: dict[str, Any]
) -> dict[str, Any]:
    with np.errstate(invalid="ignore", over="ignore"):
        return _compare(got.astype(np.float64), ref, tol)


def _compare(
    g: np.ndarray[Any, Any], ref: np.ndarray[Any, Any], tol: dict[str, Any]
) -> dict[str, Any]:
    both_nan = np.isnan(g) & np.isnan(ref)
    same_inf = np.isinf(g) & np.isinf(ref) & (np.sign(g) == np.sign(ref))
    err = np.abs(g - ref)
    ok = (err <= float(tol["atol"]) + float(tol["rtol"]) * np.abs(ref)) | both_nan | same_inf
    finite = np.isfinite(ref)
    diff = np.where(finite & np.isfinite(g), g - ref, 0.0)
    ref_norm = float(np.linalg.norm(np.where(finite, ref, 0.0)))
    rel_l2 = float(np.linalg.norm(diff)) / ref_norm if ref_norm > 0 else float(np.linalg.norm(diff))
    bad_nonfinite = int(np.count_nonzero(~np.isfinite(g) & finite))
    return {
        "mismatches": int(np.count_nonzero(~ok)),
        "elements": int(g.size),
        "nan_count": int(np.count_nonzero(np.isnan(g) & ~np.isnan(ref))),
        "nonfinite_where_ref_finite": bad_nonfinite,
        "max_abs_err": float(np.max(np.where(finite & np.isfinite(g), err, 0.0), initial=0.0)),
        "rel_l2": rel_l2 if bad_nonfinite == 0 else None,  # undefined with non-finite output
    }


def relative_to_library(tol: dict[str, Any]) -> bool:
    return str(tol.get("relative_to", "")) == "library"


def library_verdict(
    stats: dict[str, Any], lib: dict[str, Any], tol: dict[str, Any], level: str
) -> tuple[bool, dict[str, Any]]:
    """Pass/fail of a candidate's ``stats`` against the library's ``lib`` at ``level``."""
    factor = float(tol.get(level, 2.0))
    l2_floor = float(tol.get("rel_l2_floor", 1e-6))
    abs_floor = float(tol.get("abs_floor", 1e-3))
    l2_limit = max(factor * float(lib["rel_l2"] or 0.0), l2_floor)
    abs_limit = max(factor * float(lib["max_abs_err"]), abs_floor)
    ok = (
        stats["nonfinite_where_ref_finite"] == 0
        and stats["rel_l2"] is not None
        and stats["rel_l2"] <= l2_limit
        and stats["max_abs_err"] <= abs_limit
    )
    detail = {
        "level": level,
        "factor": factor,
        "rel_l2_limit": l2_limit,
        "max_abs_limit": abs_limit,
        "library_rel_l2": lib["rel_l2"],
        "library_max_abs_err": lib["max_abs_err"],
        "library_mismatches": lib["mismatches"],
    }
    return ok, detail


def tolerance(contract: Contract, dtype: str) -> dict[str, Any]:
    if dtype not in contract.tolerances:
        raise ConfigError(f"the Contract has no tolerance for dtype {dtype}")
    return contract.tolerances[dtype]


def run_numeric(
    c: Checker,
    ev: Evidence,
    module: BuiltModule,
    problem: Problem,
    contract: Contract,
    shape: dict[str, Any],
    shown: str | None,
    dists: list[str],
    level: str | None = None,
) -> None:
    """Append the failures of ``module`` at ``shape``; ``shown`` replaces a hidden shape.

    ``level`` (``loose`` / ``strict``) applies library-relative acceptance to every output
    whose tolerance asks for it (D695); None keeps the fixed element-wise criterion."""
    where: dict[str, Any] = {"shape": shape} if shown is None else {"shape_class": shown}
    sets: list[dict[str, bytes]] = []
    for dist in dists:
        rng = c.rng(f"checks.numeric:{canonical_json(shape).decode()}:{dist}")
        sets.append(jobs.random_inputs(problem, shape, dist, rng, TABLE))
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
        {"nan_fill": True, "repeat": 1},
    )
    result = c.ex.run(bench.job)
    tag = f"numeric_{sum(k.endswith('_stdout') for k in ev.blobs)}"
    ev.job(result, tag)
    if result.error_kind == "infra":
        ev.error("numeric", f"bench job {result.job_id}: {stderr_tail(c, result)}")
        return
    if not result.ok:
        ev.failures.append(
            {
                "check": "numeric.run",
                **where,
                "error_kind": result.error_kind,
                "exit_code": result.exit_code,
                "detail": stderr_tail(c, result, 600),
            }
        )
        return
    lib_wanted = level is not None and any(
        relative_to_library(tolerance(contract, o.dtype)) for o in problem.outputs
    )
    try:
        ref = reference.reference(c, problem, shape, sets, library=lib_wanted)
    except (CheckerError, ExecutionError) as exc:
        ev.error("numeric.reference", str(exc))
        return
    ev.blobs[f"{tag}_reference_stderr"] = ref.result.stderr_blob
    for i, dist in enumerate(dists):
        for t in problem.outputs:
            raw = c.ex.read_blob(result.outputs[f"{t.name}.s{i}.bin"])
            got = np.frombuffer(raw, dtype=inp.np_dtype(t.dtype)).reshape(inp.dims(t, shape))
            tol = tolerance(contract, t.dtype)
            stats = compare_output(got, ref.outputs[i][t.name], tol)
            limit = tol.get("rel_l2")
            l2 = stats["rel_l2"]
            extra: dict[str, Any] = {}
            if level is not None and relative_to_library(tol) and ref.library is not None:
                lib = compare_output(ref.library[i][t.name], ref.outputs[i][t.name], tol)
                ok, extra = library_verdict(stats, lib, tol, level)
                failed = not ok
            else:
                failed = bool(stats["mismatches"]) or (
                    limit is not None and (l2 is None or l2 > float(limit))
                )
            if failed:
                ev.failures.append(
                    {
                        "check": "numeric",
                        **where,
                        "dist": dist,
                        "output": t.name,
                        **stats,
                        "rtol": tol["rtol"],
                        "atol": tol["atol"],
                        **({"rel_l2_limit": limit} if not extra else extra),
                    }
                )
            if shown is None:
                ev.detail.setdefault("results", []).append(
                    {"shape": shape, "dist": dist, "output": t.name, **stats, **extra}
                )


def dists_for(contract: Contract, extra: list[str] | None) -> list[str]:
    out: list[str] = list(dict.fromkeys([*contract.input_dists, *(extra or [])]))
    if not out:
        raise ConfigError("no input distributions: the Contract lists none")
    return out


def numeric_check(
    c: Checker,
    module: BuiltModule,
    problem: Problem,
    contract: Contract,
    shapes: list[dict[str, Any]],
    extra_dists: list[str] | None = None,
) -> Checked:
    """All ``shapes`` with one module (its launch must cover them)."""
    ev = Evidence()
    dists = dists_for(contract, extra_dists)
    ev.detail.update({"dists": dists, "ctx_seed": c.seed, "reference": "f64"})
    for shape in shapes:
        run_numeric(c, ev, module, problem, contract, shape, None, dists)
    return write_check(c, NAME, ev, module.build_id)


def check_numeric(
    ctx: Context,
    module: BuiltModule,
    problem: Problem,
    contract: Contract,
    shapes: list[dict[str, Any]],
) -> Checked:
    return numeric_check(Checker.from_ctx(ctx), module, problem, contract, shapes)
