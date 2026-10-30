"""Numeric check: the kernel's outputs against a GPU reference (docs/design/50 §3).

Per shape, one ``check`` job runs the module once per input distribution with outputs
NaN-filled, and one reference job computes the same inputs in FP64. Criterion per
element: ``|y - ref| <= atol + rtol * |ref|`` (NaN where the reference is finite fails),
and the relative L2 error ``||y - ref|| / ||ref||`` must stay within ``rel_l2`` when the
Contract gives one. Hidden shapes are reported by category only.
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
    g = got.astype(np.float64)
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
        "rel_l2": rel_l2 if bad_nonfinite == 0 else float("inf"),
    }


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
) -> None:
    """Append the failures of ``module`` at ``shape``; ``shown`` replaces a hidden shape."""
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
    try:
        ref = reference.reference(c, problem, shape, sets)
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
            if stats["mismatches"] or (limit is not None and stats["rel_l2"] > float(limit)):
                ev.failures.append(
                    {
                        "check": "numeric",
                        **where,
                        "dist": dist,
                        "output": t.name,
                        **stats,
                        "rtol": tol["rtol"],
                        "atol": tol["atol"],
                        "rel_l2_limit": limit,
                    }
                )
            if shown is None:
                ev.detail.setdefault("results", []).append(
                    {"shape": shape, "dist": dist, "output": t.name, **stats}
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
