"""Top-k pipeline of WP7 before the orchestrator exists (decision D99): solve every shape
bucket with the WP5 solver, build the shortlist, re-rank with compile feedback, run
``full_check`` on the top k, measure those that pass and the baseline, and record the
rank correlation of predicted and measured latency. Every step writes records; the
summary is one Report (subject ``wp7_topk``).
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from kta.backends.base import Backend, BuiltModule, GeneratedKernel
from kta.checks.base import Checked, Checker
from kta.checks.full import BuildFor, run_full
from kta.core.context import Context
from kta.core.errors import ExecutionError, SchemaError
from kta.core.expr import Expr
from kta.core.jobs import JobRunner
from kta.core.registry import resolve
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Build
from kta.core.schemas.hardware import Target
from kta.core.schemas.knowledge import SassExpect, Template
from kta.core.schemas.process import Report
from kta.formulation import Formulation, assemble
from kta.knowledge import FamilyPackage
from kta.knowledge.pool import find_template
from kta.measure.baseline import measure_baseline
from kta.measure.runner import Measurer
from kta.models import MachineModel, UnitModel
from kta.solve import record_solve, rerank, shape_buckets, solve
from kta.solve.result import Candidate
from kta.store.base import Store
from kta_wp7.records import ACTOR, CaseIds
from kta_wp7.stats import spearman

log = logging.getLogger("kta.wp7")
# compute-sanitizer runs on a small, non-tile-multiple shape (docs/design/50 §3, D285).
SANITIZER_SHAPE = {"M": 200, "N": 136, "K": 72}


@dataclass(frozen=True)
class Env:
    ctx: Context
    store: Store
    ex: JobRunner
    m: Measurer
    c: Checker
    tilelang: Backend
    cuda: Backend
    target: Target
    target_id: str


def make_env(ctx: Context, target: Target, target_id: str) -> Env:
    store: Store = resolve("store", ctx)
    return Env(
        ctx,
        store,
        resolve("executor", ctx),
        Measurer.from_ctx(ctx),
        Checker.from_ctx(ctx),
        resolve("backend.tilelang", ctx),
        resolve("backend.cuda", ctx),
        target,
        target_id,
    )


class Generator:
    """The family's ``generate`` (loaded once from its package directory)."""

    def __init__(self, fp: FamilyPackage) -> None:
        spec = importlib.util.spec_from_file_location(f"kta_family_{fp.family.name}", fp.gen_path)
        if spec is None or spec.loader is None:
            raise SchemaError(f"cannot load {fp.gen_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.fn: Callable[[dict[str, Any], dict[str, Any]], GeneratedKernel] = getattr(
            module, fp.gen_function
        )

    def __call__(self, assignment: dict[str, Any], shape: dict[str, Any]) -> GeneratedKernel:
        return self.fn(dict(assignment), dict(shape))


def build(e: Env, gk: GeneratedKernel) -> tuple[Build, BuiltModule]:
    backend = e.tilelang if gk.backend == "tilelang" else e.cuda
    return backend.build(gk, e.target, e.ex)


def builder(e: Env, make: Callable[[dict[str, Any]], GeneratedKernel]) -> BuildFor:
    """``build_for`` of full_check: compile the same assignment for another shape."""

    def build_for(shape: dict[str, Any]) -> tuple[GeneratedKernel, BuiltModule]:
        gk = make(shape)
        result, module = build(e, gk)
        if not result.ok:
            raise ExecutionError(
                "nonzero", f"build for {shape} failed: {result.log_summary[-600:]}"
            )
        return gk, module

    return build_for


def templates_of(store: Store, fp: FamilyPackage) -> dict[str, Template]:
    out: dict[str, Template] = {}
    for item in fp.chain.items:
        hit = find_template(store, item.template)
        if hit is None:
            raise SchemaError(f"Template {item.template} is not in the store")
        out[item.template] = hit[1]
    return out


def sass_for(
    fp: FamilyPackage, templates: dict[str, Template], assignment: dict[str, Any]
) -> tuple[list[SassExpect], dict[str, Any]]:
    """The chain's SASS expectations whose ``when`` holds, evaluated over each item's
    Template variables (constants from bindings, family values otherwise)."""
    out: list[SassExpect] = []
    for item in fp.chain.items:
        t = templates[item.template]
        env = {
            v.name: assignment.get(str(item.bindings[v.name]), item.bindings[v.name])
            for v in t.variables
            if v.name in item.bindings
        }
        for e in t.expected_sass:
            if Expr.parse(e.when, set(env)).eval(env) is True:
                out.append(
                    SassExpect(
                        when="True",
                        must_contain=e.must_contain,
                        must_not_contain=e.must_not_contain,
                    )
                )
    return out, assignment


def check(
    e: Env,
    ids: CaseIds,
    gen: Generator,
    fp: FamilyPackage,
    templates: dict[str, Template],
    c: Candidate,
    gk: GeneratedKernel,
    module: BuiltModule,
    shape: dict[str, Any],
) -> Checked:
    sass, assignment = sass_for(fp, templates, c.assignment)
    return run_full(
        e.c,
        module,
        gk,
        ids.problem_body,
        ids.contract_body,
        ids.body,
        case_id=ids.case,
        shape=shape,
        build_for=builder(e, lambda s: gen(c.assignment, s)),
        sass=sass,
        assignment=assignment,
        sanitizer_shape=SANITIZER_SHAPE,
    )


def _prediction(e: Env, c: Candidate, cand_id: str, build_id: str, refs: dict[str, str]) -> str:
    rec = Record.new(
        "Prediction",
        c.prediction,
        ACTOR,
        e.ctx.clock.now(),
        refs={"candidate": cand_id, "build": build_id, **refs},
    )
    return e.store.put(rec)


@dataclass(frozen=True)
class Solved:
    formulation: Formulation
    templates: dict[str, Template]


def run_bucket(
    e: Env,
    ids: CaseIds,
    fp: FamilyPackage,
    gen: Generator,
    s: Solved,
    bucket: dict[str, Any],
    refs: dict[str, str],
    hw: MachineModel,
    model: UnitModel,
    k: int,
    shortlist: int,
    workers: int,
) -> dict[str, Any]:
    shape = dict(bucket["shape"])
    r = solve(
        s.formulation,
        ids.body,
        bucket,
        model,
        hw,
        shortlist=shortlist,
        rng=e.ctx.rng(f"wp7.solve:{bucket['name']}"),
    )
    recs = record_solve(
        e.ctx, e.store, r, ACTOR, ids.case, refs["family"], refs["hw_model"], refs["model"]
    )
    cand_ids = {c.key: cid for c, cid in zip(r.candidates, recs.candidates, strict=True)}
    kernels = {c.key: gen(c.assignment, shape) for c in r.candidates}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        built = dict(
            zip(kernels, pool.map(lambda key: build(e, kernels[key]), kernels), strict=True)
        )
    top = rerank(r.candidates, {key: b for key, (b, _) in built.items()}, model, hw, k)
    pred_refs = {"hw_model": refs["hw_model"], "model": refs["model"]}
    rows: list[dict[str, Any]] = []
    for c in top:
        module = built[c.key][1]
        row: dict[str, Any] = {
            "candidate": cand_ids[c.key],
            "assignment": c.assignment,
            "build": module.build_id,
            "prediction": _prediction(e, c, cand_ids[c.key], module.build_id, pred_refs),
            "predicted_us": c.prediction.total,
        }
        chk = check(e, ids, gen, fp, s.templates, c, kernels[c.key], module, shape)
        row.update(check=chk.id, passed=chk.passed, checker_error=chk.checker_error)
        if chk.passed:
            meas = e.m.measure(module, ids.body, shape, refs["protocol"])
            row.update(measurement=meas.id, median_us=meas.body.median_us)
        rows.append(row)
        log.info(
            "bucket %s %s: passed=%s %s",
            bucket["name"],
            c.assignment,
            chk.passed,
            row.get("median_us"),
        )
    base = None
    if ids.baseline is not None:
        bm = measure_baseline(e.m, ids.baseline, ids.body, shape, refs["protocol"], ids.case)
        base = {"measurement": bm.id, "median_us": bm.body.median_us}
    measured = [x for x in rows if "median_us" in x]
    rho = spearman([x["predicted_us"] for x in measured], [x["median_us"] for x in measured])
    return {
        "bucket": bucket["name"],
        "shape": shape,
        "solve_report": recs.report,
        "feasible": r.feasible,
        "exhaustive": r.exhaustive,
        "model_optimal": r.model_optimal,
        "shortlisted": len(r.candidates),
        "built_ok": sum(1 for b, _ in built.values() if b.ok),
        "builds": {cand_ids[key]: m.build_id for key, (_, m) in built.items()},
        "top": rows,
        "baseline": base,
        "spearman": rho,
        "n_measured": len(measured),
    }


def _line(b: dict[str, Any]) -> str:
    passed = sum(1 for x in b["top"] if x["passed"])
    best = min((x["median_us"] for x in b["top"] if "median_us" in x), default=float("nan"))
    base = b["baseline"]["median_us"] if b["baseline"] else float("nan")
    return (
        f"{b['bucket']}: {passed}/{len(b['top'])} passed, best {best:.1f} us, "
        f"baseline {base:.1f} us, spearman {b['spearman']}"
    )


def run_topk(
    e: Env,
    ids: CaseIds,
    fp: FamilyPackage,
    refs: dict[str, str],
    hw: MachineModel,
    model: UnitModel,
    k: int = 8,
    shortlist: int = 64,
    workers: int = 8,
) -> str:
    templates = templates_of(e.store, fp)
    s = Solved(assemble(fp, fp.chain, templates, ids.body, hw.hw), templates)
    gen = Generator(fp)
    buckets = [
        run_bucket(e, ids, fp, gen, s, b, refs, hw, model, k, shortlist, workers)
        for b in shape_buckets(ids.body, fp)
    ]
    summary = "; ".join(_line(b) for b in buckets)
    body = Report(
        title="wp7 top-k",
        subject="wp7_topk",
        summary=summary,
        data={"buckets": buckets, "k": k, "shortlist": shortlist, **refs},
    )
    out_refs: dict[str, str | list[str]] = {
        **refs,
        "case": ids.case,
        "checks": [x["check"] for b in buckets for x in b["top"]],
        "measurements": [x["measurement"] for b in buckets for x in b["top"] if "measurement" in x],
    }
    return e.store.put(Record.new("Report", body, ACTOR, e.ctx.clock.now(), refs=out_refs))
