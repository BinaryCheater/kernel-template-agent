"""Space verification items for a family (spec/modules/measure-check.md ``verify_space``).

checks may not import knowledge, so the orchestrator enumerates: one ``path`` item per
reachable structural path (the distinct values of the family's structure variables
among feasible points) and 4 random feasible assignments as ``sample`` items, on the
first shape of the card's Case. Each item is generated and built through the Executor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kta.checks.full import SpaceItem, SpaceReport, run_full, space_verify
from kta.core.errors import SchemaError
from kta.core.ids import record_id
from kta.core.schemas.process import TaskCard
from kta.formulation import assemble, evaluate_effects
from kta.formulation.blocks import INSTANCES
from kta.knowledge import FamilyPackage, load_family
from kta.models import min_work
from kta.models.l1 import price, sol_us
from kta.orchestrator import pipeline
from kta.orchestrator.pipeline import CaseView, Services, bucket_shape, case_view
from kta.solve import enumerate_space, shape_buckets
from kta.store.base import Store

SAMPLES = 4
MAX_POINTS = 100_000  # feasible points visited when looking for structural paths


def case_of(store: Store, card: TaskCard) -> CaseView:
    for rid in card.inputs:
        if store.exists(rid) and store.get(rid).kind == "Case":
            return case_view(store, rid)
    raise SchemaError("the task card names no Case to verify against")


def points(
    svc: Services, fp: FamilyPackage, cv: CaseView
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, mm = svc.machine(cv.case.target)
    f = assemble(fp, fp.chain, pipeline.templates_of(svc.store, fp), cv.case, mm.hw)
    shape = bucket_shape(shape_buckets(cv.case)[0])
    found: list[dict[str, Any]] = []
    enumerate_space(f, shape, mm.hw, found.append, max_nodes=MAX_POINTS)
    return shape, found


EXPRESSION_SAMPLES = 8
EXPRESSION_POINTS = 5_000
WORK_SLACK = 0.99  # rounding in ceil_div counts must not trip the conservation check


def expression_issues(
    svc: Services, fp: FamilyPackage, cv: CaseView
) -> tuple[list[str], list[str]]:
    """The search expression's hard errors and hints (D732).

    Errors: work below the problem's minimum on a resource at a sampled point (the
    expression would predict faster than the speed of light). Hints: an item with
    transactions and no ``p.instances`` (its counts must then cover the whole kernel), and
    family variables no cost term reads (the expression cannot rank them).
    """
    _, mm = svc.machine(cv.case.target)
    templates = pipeline.templates_of(svc.store, fp)
    f = assemble(fp, fp.chain, templates, cv.case, mm.hw)
    errors: list[str] = []
    hints: list[str] = []
    for item in fp.chain.items:
        txs = [t for t in f.transactions if t.source == item.key]
        if txs and INSTANCES not in item.bindings:
            hints.append(
                f"chain item {item.key}: bind {INSTANCES} to how many times its transactions run "
                "in the whole kernel (per-CTA items: the CTA count, e.g. "
                "ceil_div(case.M, BM) * ceil_div(case.N, BN); per-iteration items: that times "
                "the iterations per CTA); left unbound it counts one instance"
            )
    shape = bucket_shape(shape_buckets(cv.case)[0])
    found: list[dict[str, Any]] = []
    enumerate_space(f, shape, mm.hw, found.append, max_nodes=EXPRESSION_POINTS)
    try:
        sol = sol_us(min_work(cv.problem, shape), mm)
    except Exception:  # noqa: BLE001 - a problem without min_work has no conservation check
        sol = {}
    hints_absent: list[str] = []
    for a in found[:: max(1, len(found) // EXPRESSION_SAMPLES)][:EXPRESSION_SAMPLES]:
        r = price(evaluate_effects(f, a, shape, mm.hw), mm)
        busy: dict[str, float] = {}
        for _, _, times in r.per_tx:
            for res, t in times.items():
                busy[res] = busy.get(res, 0.0) + t
        low = [
            f"{res} {busy.get(res, 0.0):.4g} us < minimum {need:.4g} us "
            f"(1/{need / max(busy.get(res, 0.0), 1e-12):.0f} of the necessary work)"
            for res, need in sorted(sol.items())
            if need > 0 and 0.0 < busy.get(res, 0.0) < WORK_SLACK * need
        ]
        absent = [res for res, need in sorted(sol.items()) if need > 0 and res not in busy]
        if absent and not hints_absent:
            hints_absent = absent
        if low:
            errors.append(
                f"at {a} on {shape} the expression does less work than the problem needs: "
                + "; ".join(low)
                + ". Counts are probably per tile or per step: check p.instances and the "
                "count bindings of the items producing these resources"
            )
            break
    used = {v for t in f.transactions for e in (t.count, *t.amounts.values()) for v in e.free_vars}
    used |= {
        v for o in f.overlaps for e in (o.trips, o.inflight, o.executors) if e for v in e.free_vars
    }
    used |= {v for st in f.storage for v in st.bytes.free_vars}
    hints += [
        f"resource {res} has necessary work but no term in the expression (a missing term)"
        for res in hints_absent
    ]
    hints += [
        f"family variable {v} appears in no cost term: the expression cannot rank its values "
        "(a missing term, or a variable to fix)"
        for v in sorted(f.variables)
        if v not in used and v not in f.runtime_vars
    ]
    return errors, hints


def items(svc: Services, fp: FamilyPackage, cv: CaseView) -> list[SpaceItem]:
    shape, found = points(svc, fp, cv)
    if not found:
        raise SchemaError(f"family {fp.family.name} has no feasible point on {shape}")
    structure = [v.name for v in fp.family.variables if v.kind == "structure"]
    paths: dict[str, dict[str, Any]] = {}
    for a in found:
        key = json.dumps({k: a.get(k) for k in structure}, sort_keys=True, default=str)
        paths.setdefault(key, a)
    rng = svc.ctx.rng(f"space:{fp.family.name}@{fp.family.version}")
    samples = rng.sample(found, min(SAMPLES, len(found)))
    out: list[SpaceItem] = []
    for kind, i, a in [
        *(("path", i, a) for i, a in enumerate(paths.values())),
        *(("sample", i, a) for i, a in enumerate(samples)),
    ]:
        gk = pipeline.generate(fp, a, shape)
        _, module = pipeline.build(svc, gk)
        out.append(
            SpaceItem(
                path_id=f"{kind}{i}",
                assignment=a,
                gk=gk,
                module=module,
                problem=cv.problem,
                contract=cv.contract,
                kind="path" if kind == "path" else "sample",
                case=cv.case,
                case_id=cv.id,
                shape=shape,
                build_for=pipeline.builder(svc, fp, a),
            )
        )
    return out


def verify_family_space(svc: Services, workdir: Path, card: TaskCard) -> SpaceReport:
    fp = load_family(workdir)
    cv = case_of(svc.store, card)
    errors, _ = expression_issues(svc, fp, cv)
    if errors:
        raise SchemaError("search expression: " + " | ".join(errors))
    todo = items(svc, fp, cv)  # before the checker needs the executor
    return space_verify(svc.checker(), todo, [])


def rerun_failed_checks(svc: Services, workdir: Path, card: TaskCard) -> tuple[list[str], str]:
    """Rerun the full check of every failed (Check, Candidate) pair named on the card.

    The card lists each failed Check followed by its Candidate (solve_loop rule 1); the
    candidate's assignment is regenerated with the submitted family on its bucket's shape.
    """
    fp = load_family(workdir)
    cv = case_of(svc.store, card)
    recs = [svc.store.get(rid) for rid in card.inputs]
    checks = [r for r in recs if r.kind == "Check" and not r.body["passed"]]
    cands = [r for r in recs if r.kind == "Candidate"]
    if len(checks) != len(cands):
        return ["the card's failed Checks and Candidates do not pair up"], ""
    # A T6 from building names Builds, not Checks: verify_space rebuilds every path then.
    buckets = {str(b["name"]): b for b in shape_buckets(cv.case)}
    failures: list[str] = []
    reruns: list[str] = []
    for chk_rec, cand in zip(checks, cands, strict=True):
        a = dict(cand.body["assignment"])
        shape = bucket_shape(buckets.get(str(cand.body.get("bucket")), shape_buckets(cv.case)[0]))
        gk = pipeline.generate(fp, a, shape)
        _, module = pipeline.build(svc, gk)
        chk = run_full(
            svc.checker(),
            module,
            gk,
            cv.problem,
            cv.contract,
            cv.case,
            case_id=cv.id,
            shape=shape,
            build_for=pipeline.builder(svc, fp, a),
            assignment=a,
        )
        reruns.append(chk.id)
        if not chk.passed:
            failures.append(f"Check {record_id(chk_rec)[:12]} still fails: {chk.id[:12]}")
    return failures, ";".join(reruns)
