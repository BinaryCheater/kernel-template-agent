"""WP7 fault injections: each must be stopped by the checks (spec/workplan.md WP7).

1. ``oob_write``: the generator's output is compiled with TileLang's boundary
   legalization switched off (``tl.disable_safe_memory_legalize``), so residue tiles load
   and store past the ends of A, B and C. Checked at a shape whose M is not a multiple of
   BM; expected: ``sanitizer.memcheck``.
2. ``racy_legality``: the family's glue ``g_no_combine`` (``split == 1``) is replaced by
   the wrong condition ``split >= 1``. The solver then admits split-K assignments whose
   CTAs store partial tiles to the same outputs with plain stores (a write-write race).
   The best admitted split > 1 candidate is built and checked; expected: ``numeric``.
3. ``cublas``: the generator is replaced by one that emits a host library calling
   ``cublasGemmEx`` (correct numbers, the work done by a library); expected:
   ``anticheat.libraries`` and ``anticheat.kernels``.

Every injection writes its aggregate Check; a Report (subject ``wp7_faults``) lists them.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from kta.backends.base import GeneratedKernel
from kta.checks.base import Checked
from kta.checks.full import run_full
from kta.checks.selftest import CUBLAS_SO, build_shared_library
from kta.core.errors import ExecutionError
from kta.core.registry import resolve
from kta.core.schemas import Record
from kta.core.schemas.knowledge import ExprSpec
from kta.core.schemas.process import Report
from kta.formulation import assemble
from kta.knowledge import FamilyPackage
from kta.measure.selftest import gemm_kernel
from kta.models import MachineModel, UnitModel
from kta.solve import record_solve, solve
from kta_wp7.pipeline import Env, Generator, builder, templates_of
from kta_wp7.records import ACTOR, CaseIds

log = logging.getLogger("kta.wp7.faults")
OOB_SHAPE = {"M": 200, "N": 256, "K": 256}
OOB_ASSIGNMENT = {"BM": 128, "BN": 128, "BK": 64, "S": 2, "P": 1, "threads": 128, "split": 1}
NO_LEGALIZE = "tl.disable_safe_memory_legalize"
WRONG_GLUE = "split >= 1"
RACE_SANITIZER_SHAPE = {"M": 256, "N": 256, "K": 512}
EXPECTED = {
    "oob_write": {"sanitizer.memcheck"},
    "racy_legality": {"numeric"},
    "cublas": {"anticheat.libraries", "anticheat.kernels"},
}


def _failed(chk: Checked) -> set[str]:
    return {str(f["check"]) for f in chk.body.failures}


def without_legalization(gk: GeneratedKernel) -> GeneratedKernel:
    """The same kernel with TileLang's boundary legalization switched off."""
    src = gk.files["kernel.py"]
    marker = "PASS_CONFIGS = {"
    if marker not in src:
        raise ExecutionError("nonzero", "generated kernel has no PASS_CONFIGS to modify")
    src = src.replace(marker, f"{marker}{NO_LEGALIZE!r}: True, ", 1)
    return replace(gk, files={**gk.files, "kernel.py": src})


def oob_write(e: Env, ids: CaseIds, gen: Generator) -> Checked:
    def make(shape: dict[str, Any]) -> GeneratedKernel:
        return without_legalization(gen(OOB_ASSIGNMENT, shape))

    build_for = builder(e, make)
    gk, module = build_for(OOB_SHAPE)
    return run_full(
        e.c,
        module,
        gk,
        ids.problem_body,
        ids.contract_body,
        ids.body,
        case_id=ids.case,
        shape=OOB_SHAPE,
        build_for=build_for,
    )


def wrong_legality(fp: FamilyPackage) -> FamilyPackage:
    glue = [
        ExprSpec(id=g.id, expr=WRONG_GLUE, meaning=f"INJECTED FAULT (WP7): {g.meaning}", cls=g.cls)
        if g.id == "g_no_combine"
        else g
        for g in fp.family.glue_constraints
    ]
    family = fp.family.model_copy(update={"glue_constraints": glue})
    return replace(fp, family=family)


def racy_legality(
    e: Env,
    ids: CaseIds,
    fp: FamilyPackage,
    gen: Generator,
    refs: dict[str, str],
    hw: MachineModel,
    model: UnitModel,
) -> tuple[Checked, dict[str, Any]]:
    """Solve the bucket with the correct and with the injected glue; check the racy pick."""
    templates = templates_of(e.store, fp)
    bucket = {"name": "s2", "shape": dict(ids.body.shape_dist[2]["shape"]), "weight": 1.0}
    info: dict[str, Any] = {"bucket": bucket, "wrong_glue": WRONG_GLUE}
    racy: dict[str, Any] = {}
    for label, family in (("correct", fp), ("injected", wrong_legality(fp))):
        f = assemble(family, family.chain, templates, ids.body, hw.hw)
        r = solve(
            f, ids.body, bucket, model, hw, shortlist=64, rng=e.ctx.rng(f"wp7.faults.{label}")
        )
        recs = record_solve(
            e.ctx, e.store, r, ACTOR, ids.case, refs["family"], refs["hw_model"], refs["model"]
        )
        split = [c for c in r.candidates if c.assignment["split"] > 1]
        info[label] = {
            "solve_report": recs.report,
            "feasible": r.feasible,
            "pruned_by_g_no_combine": r.pruned_by.get("glue:g_no_combine", 0),
            "admitted_split": len(split),
            "best": r.candidates[0].assignment,
        }
        if label == "injected":
            if not split:
                raise ExecutionError("nonzero", "the injected condition admitted no split-K point")
            racy = split[0].assignment
            info["racy_assignment"] = racy
    shape = bucket["shape"]
    build_for = builder(e, lambda s: gen(racy, s))
    gk, module = build_for(shape)
    chk = run_full(
        e.c,
        module,
        gk,
        ids.problem_body,
        ids.contract_body,
        ids.body,
        case_id=ids.case,
        shape=shape,
        build_for=build_for,
        sanitizer_shape=RACE_SANITIZER_SHAPE,
    )
    return chk, info


def cublas(e: Env, ids: CaseIds) -> Checked:
    flags = resolve_flags(e)

    def build_for(shape: dict[str, Any]) -> tuple[GeneratedKernel, Any]:
        gk = GeneratedKernel("cuda", {"kernel.cu": CUBLAS_SO}, "gemm", gemm_kernel(shape).launch)
        return gk, build_shared_library(e.c, gk, e.target, flags)

    shape = OOB_SHAPE
    gk, module = build_for(shape)
    return run_full(
        e.c,
        module,
        gk,
        ids.problem_body,
        ids.contract_body,
        ids.body,
        case_id=ids.case,
        shape=shape,
        build_for=build_for,
    )


def resolve_flags(e: Env) -> list[str]:
    return list(resolve("target", e.ctx).compile_flags("cuda"))


def run_faults(
    e: Env,
    ids: CaseIds,
    fp: FamilyPackage,
    refs: dict[str, str],
    hw: MachineModel,
    model: UnitModel,
) -> str:
    gen = Generator(fp)
    checks: dict[str, Checked] = {"oob_write": oob_write(e, ids, gen)}
    checks["racy_legality"], race_info = racy_legality(e, ids, fp, gen, refs, hw, model)
    checks["cublas"] = cublas(e, ids)
    data: dict[str, Any] = {"racy_legality_solve": race_info}
    for name, chk in checks.items():
        failed = _failed(chk)
        data[name] = {
            "check": chk.id,
            "passed": chk.passed,
            "failed_checks": sorted(failed),
            "expected": sorted(EXPECTED[name]),
            "caught": (not chk.passed) and EXPECTED[name] <= failed and not chk.checker_error,
        }
        log.info("fault %s: %s", name, data[name])
    summary = "; ".join(
        f"{n}: {'caught' if data[n]['caught'] else 'NOT caught'} "
        f"({', '.join(data[n]['failed_checks'])})"
        for n in checks
    )
    body = Report(title="wp7 fault injections", subject="wp7_faults", summary=summary, data=data)
    refs_out: dict[str, str | list[str]] = {
        "case": ids.case,
        "checks": [c.id for c in checks.values()],
    }
    return e.store.put(Record.new("Report", body, ACTOR, e.ctx.clock.now(), refs=refs_out))
