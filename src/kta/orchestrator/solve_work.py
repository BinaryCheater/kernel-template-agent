"""The work of each solve-loop state: real module calls whose results become facts.

Each handler reads the loop row, does the state's work (docs/design/10 §2.1) through the
knowledge, models, formulation, solve, backends, checks and measure modules, writes the
records those modules write, and returns the facts ``solve_loop.step`` decides on.
One family per instance: the best-ranked retrieved family (known issue, WP6 note).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

from kta.checks.full import run_full
from kta.checks.sass import sass_check
from kta.core.errors import SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Bound, Build, Measurement, Prediction
from kta.core.schemas.evaluation import Candidate as CandidateBody
from kta.formulation import assemble
from kta.knowledge import nearest_chains, retrieve_families, retrieve_templates, reuse_plan
from kta.measure.baseline import measure_baseline
from kta.models import bound, min_work
from kta.orchestrator import pipeline
from kta.orchestrator.pipeline import Services, bucket_shape, case_view
from kta.orchestrator.transition import Loop, Settings
from kta.solve import Candidate, rerank, shape_buckets, solve
from kta.solve.records import record_solve

ACTOR = Actor(type="tool", name="kta.orchestrator.solve_work")
Facts = dict[str, Any]


def _report(
    svc: Services,
    title: str,
    subject: str,
    summary: str,
    data: dict[str, Any],
    refs: dict[str, Any],
) -> str:
    body = {"title": title, "subject": subject, "summary": summary, "data": data}
    return svc.store.put(Record.new("Report", body, ACTOR, svc.ctx.clock.now(), refs=refs))


def work_new(svc: Services, loop: Loop) -> Facts:
    cv = case_view(svc.store, loop.subject)
    return {
        "case": cv.id,
        "problem": cv.case.problem,
        "contract": cv.contract_id,
        "baselines": [cv.baseline] if cv.baseline else [],
    }


def work_bounded(svc: Services, loop: Loop) -> Facts:
    cv = case_view(svc.store, loop.subject)
    hw_id, hw = svc.machine(cv.case.target)
    buckets = shape_buckets(cv.case)
    bounds: dict[str, str] = {}
    for b in buckets:
        shape = bucket_shape(b)
        body: Bound = bound(cv.case, shape, hw, min_work(cv.problem, shape))
        rec = Record.new(
            "Bound", body, ACTOR, svc.ctx.clock.now(), refs={"case": cv.id, "hw_model": hw_id}
        )
        bounds[str(b["name"])] = svc.store.put(rec)
    return {"bounds": bounds, "buckets": buckets, "hw_model": hw_id}


KNOWLEDGE_CONDITIONS = ("pool", "text", "zero")  # docs/design/50 §1: the experiment arms
# Aliases WP8 also accepts for the same three arms (agreed with WP8, D373).
KNOWLEDGE_ALIASES = {"from_zero": "zero", "text_knowledge": "text", "template_pool": "pool"}


def knowledge_of(condition: Mapping[str, Any] | None) -> str:
    raw = str(dict(condition or {}).get("knowledge", "pool"))
    knowledge = KNOWLEDGE_ALIASES.get(raw, raw)
    if knowledge not in KNOWLEDGE_CONDITIONS:
        raise SchemaError(
            f"experiment condition knowledge={raw!r}; one of {KNOWLEDGE_CONDITIONS} "
            f"or {sorted(KNOWLEDGE_ALIASES)}"
        )
    return knowledge


def pinned(svc: Services, loop: Loop) -> tuple[set[str] | None, str]:
    """(snapshot members or None, knowledge condition) of an experiment run (D373).

    ``loop.data.experiment`` is pinned by the experiment process (WP11): ``snapshot`` (a
    PoolSnapshot ID) limits retrieval to its members; ``condition.knowledge`` is one of
    ``pool`` (default), ``text`` or ``zero`` (or WP8's aliases). The ``text`` arm needs a
    snapshot (the text is rendered from it, WP8). Outside experiments: the whole pool.
    """
    exp: Mapping[str, Any] = loop.data.get("experiment") or {}
    knowledge = knowledge_of(exp.get("condition"))
    snap = exp.get("snapshot")
    if knowledge == "text" and not snap:
        raise SchemaError("the text-knowledge arm needs a pinned snapshot")
    if not snap:
        return None, knowledge
    rec = svc.store.get(str(snap))
    if rec.kind != "PoolSnapshot":
        raise SchemaError(f"experiment snapshot {str(snap)[:12]} is a {rec.kind}")
    return set(rec.body["members"]), knowledge


def allowed_cards(cards: list[Any], members: set[str] | None, knowledge: str) -> list[Any]:
    """Retrieved cards an experiment arm may see (D373): none for the ``text`` and
    ``zero`` arms (the solver gets no pool objects); snapshot members for ``pool``."""
    if knowledge != "pool":
        return []
    return [c for c in cards if members is None or c.id in members]


def work_retrieving(svc: Services, loop: Loop) -> Facts:
    cv = case_view(svc.store, loop.subject)
    _, hw = svc.hw_model(cv.case.target)
    members, knowledge = pinned(svc, loop)

    def allowed(cards: list[Any]) -> list[Any]:
        return allowed_cards(cards, members, knowledge)

    fams = allowed(retrieve_families(svc.store, cv.case, k=20, hw=hw.limits))[:5]
    if fams:
        return {"families": [c.id for c in fams]}
    chains = allowed(nearest_chains(svc.store, cv.case, k=10))[:3]
    data: dict[str, Any] = {"nearest_chains": [asdict(c) for c in chains]}
    inputs = [cv.id]
    if chains:
        plan = reuse_plan(svc.store, chains[0].id, cv.case, hw.limits)
        data["reuse_plan"] = plan.to_dict()
        inputs.append(chains[0].id)
        bad = [s.key for s in plan.steps if s.verdict != "keep"]
        goal = f"Build a family from chain {chains[0].name}; replace steps {', '.join(bad)}"
    else:
        cards = allowed(retrieve_templates(svc.store, case=cv.case, k=20, hw=hw.limits))[:5]
        data["templates"] = [asdict(c) for c in cards]
        inputs += [c.id for c in cards]
        goal = (
            "Build a family for this case from the retrieved Templates"
            if cards
            else f"Build a family for this case (knowledge condition: {knowledge})"
        )
    data["knowledge"] = knowledge
    if members is not None:
        data["snapshot"] = len(members)
    rid = _report(svc, "retrieval", cv.id, goal, data, {"case": cv.id})
    return {"families": [], "t1_goal": goal, "t1_inputs": [*inputs, rid]}


def work_solving(svc: Services, loop: Loop) -> Facts:
    s = Settings.from_config(svc.ctx.config)
    cv = case_view(svc.store, loop.subject)
    hw_id, hw = svc.machine(cv.case.target)
    model_id, model = svc.unit_model(cv.case.target)
    family = str(loop.data["families"][0])
    fp = pipeline.family_package(svc, family)
    f = assemble(fp, fp.chain, pipeline.templates_of(svc.store, fp), cv.case, hw.hw)
    out: dict[str, list[str]] = {}
    for b in loop.data["buckets"]:
        res = solve(
            f,
            cv.case,
            b,
            model,
            hw,
            shortlist=s.shortlist,
            time_limit_s=s.solve_time_limit_s,
            rng=svc.ctx.rng(f"solve:{loop.id}"),
        )
        recs = record_solve(svc.ctx, svc.store, res, ACTOR, cv.id, family, hw_id, model_id)
        out[str(b["name"])] = recs.candidates
    empty = [name for name, ids in out.items() if not ids]
    return {
        "candidates": out,
        "empty_buckets": empty,
        "families_tried": [family],
        "family": family,
        "model": model_id,
    }


def _candidate(svc: Services, cand_id: str) -> Candidate:
    body = CandidateBody.model_validate(svc.store.get(cand_id).body)
    preds = svc.store.query("Prediction", {"refs.candidate": cand_id}, limit=1)
    return Candidate(body.assignment, body.bucket, Prediction.model_validate(preds[0].body))


def _bucket(loop: Loop, name: str) -> Mapping[str, Any]:
    return next(b for b in loop.data["buckets"] if str(b["name"]) == name)


def _sass(svc: Services, loop: Loop, fp: Any, module: Any, assignment: dict[str, Any]) -> None:
    expect = pipeline.chain_sass(svc.store, fp, assignment)
    if expect and module.sass_blob:
        sass_check(svc.checker(), module, expect, assignment)


def work_building(svc: Services, loop: Loop) -> Facts:
    s = Settings.from_config(svc.ctx.config)
    cv = case_view(svc.store, loop.subject)
    _, hw = svc.machine(cv.case.target)
    _, model = svc.unit_model(cv.case.target)
    family = str(loop.data["families"][0])
    fp = pipeline.family_package(svc, family)
    top: dict[str, list[str]] = {}
    builds: dict[str, str] = {}
    for name, ids in loop.data["candidates"].items():
        shape = bucket_shape(_bucket(loop, name))
        cands = {cid: _candidate(svc, cid) for cid in ids}
        by_key: dict[str, Build] = {}
        for cid, c in cands.items():
            build, module = pipeline.build(svc, pipeline.generate(fp, c.assignment, shape))
            builds[cid] = module.build_id
            by_key[c.key] = build
            if build.ok:
                _sass(svc, loop, fp, module, c.assignment)
        kept = {c.key for c in rerank(list(cands.values()), by_key, model, hw, k=s.top_k)}
        top[name] = [cid for cid, c in cands.items() if c.key in kept]
    return {"top": top, "builds": builds, "family": family}


def _each_top(svc: Services, loop: Loop, fn: Callable[..., str]) -> dict[str, str]:
    cv = case_view(svc.store, loop.subject)
    fp = pipeline.family_package(svc, str(loop.data["families"][0]))
    out: dict[str, str] = {}
    for name, ids in loop.data["top"].items():
        shape = bucket_shape(_bucket(loop, name))
        for cid in ids:
            c = _candidate(svc, cid)
            gk = pipeline.generate(fp, c.assignment, shape)
            _, module = pipeline.build(svc, gk)
            out[cid] = fn(cv, fp, gk, module, shape, c.assignment)
    return out


def work_checking(svc: Services, loop: Loop) -> Facts:
    checker = svc.checker()
    small = pipeline.sanitizer_shape(case_view(svc.store, loop.subject))

    def check(cv: Any, fp: Any, gk: Any, module: Any, shape: Any, a: Any) -> str:
        return run_full(
            checker,
            module,
            gk,
            cv.problem,
            cv.contract,
            cv.case,
            case_id=cv.id,
            shape=shape,
            build_for=pipeline.builder(svc, fp, a),
            sass=pipeline.chain_sass(svc.store, fp, a),
            assignment=a,
            sanitizer_shape=small,
        ).id

    checks = _each_top(svc, loop, check)
    failed = [cid for cid in checks.values() if not svc.store.get(cid).body["passed"]]
    return {"checks": checks, "failed": failed}


def work_measuring(svc: Services, loop: Loop) -> Facts:
    protocol_id, _ = svc.protocol()
    m = svc.measurer()
    passed = {c for c, chk in loop.data["checks"].items() if svc.store.get(chk).body["passed"]}

    def meas(cv: Any, fp: Any, gk: Any, module: Any, shape: Any, a: Any) -> str:
        return m.measure(module, cv.case, shape, protocol_id).id

    top = {k: [c for c in v if c in passed] for k, v in loop.data["top"].items()}
    loop_passed = Loop(loop.id, loop.kind, loop.subject, loop.state, {**loop.data, "top": top})
    measured = _each_top(svc, loop_passed, meas)
    # The baseline is timed under the same protocol per bucket (D284, D362); judging reads
    # the newest Measurement with refs.case and refs.baseline.
    cv = case_view(svc.store, loop.subject)
    baselines: dict[str, str] = {}
    if cv.baseline is not None:
        for b in loop.data["buckets"]:
            got = measure_baseline(m, cv.baseline, cv.case, bucket_shape(b), protocol_id, cv.id)
            baselines[str(b["name"])] = got.id
    return {"measurements": measured, "protocol": protocol_id, "baseline_measurements": baselines}


def _baseline_us(svc: Services, case_id: str, baseline: str | None) -> float | None:
    if baseline is None:
        return None
    found = svc.store.query(
        "Measurement",
        {"refs.case": case_id, "refs.baseline": baseline},
        order="-created_at",
        limit=1,
    )
    return float(found[0].body["median_us"]) if found else None


def work_judging(svc: Services, loop: Loop) -> Facts:
    cv = case_view(svc.store, loop.subject)
    _, sig = svc.protocol()
    model_id = str(loop.data.get("model", ""))
    d = loop.data
    buckets: list[dict[str, Any]] = []
    rows: list[tuple[str, str, Prediction, Measurement]]
    for name, ids in d["top"].items():
        bnd = Bound.model_validate(svc.store.get(d["bounds"][name]).body)
        failed = [[d["checks"][c], c] for c in ids if d["checks"].get(c) in d.get("failed", [])]
        rows = []
        for cid in ids:
            if cid not in d.get("measurements", {}):
                continue
            pred = svc.store.query("Prediction", {"refs.candidate": cid}, limit=1)[0]
            meas = Measurement.model_validate(svc.store.get(d["measurements"][cid]).body)
            rows.append((cid, record_id(pred), Prediction.model_validate(pred.body), meas))
        if not rows and not failed:
            raise SchemaError(f"bucket {name}: no measurement and no failed check")
        if not rows:
            buckets.append(
                {
                    "name": name,
                    "failed_checks": failed,
                    "pairs": [],
                    "best_us": 0.0,
                    "predicted_us": 0.0,
                    "bound_us": 0.0,
                }
            )
            continue
        cid, pid, pred, meas = min(rows, key=lambda r: r[3].median_us)
        units = [
            {"unit": u, "bound_us": bnd.per_unit_us.get(u), "predicted_us": t, "measured_us": None}
            for u, t in sorted(pred.per_unit.items())
        ]
        buckets.append(
            {
                "name": name,
                "failed_checks": failed,
                "pairs": [(r[2].total, r[3].median_us) for r in rows],
                "evidence": [x for r in rows for x in (r[1], d["measurements"][r[0]])],
                "best_us": meas.median_us,
                "predicted_us": pred.total,
                "bound_us": bnd.bound_us,
                "reference_us": _baseline_us(svc, cv.id, cv.baseline),
                "units": units,
                "assignment": CandidateBody.model_validate(svc.store.get(cid).body).assignment,
                "measurement": d["measurements"][cid],
                "check": d["checks"][cid],
                "bound": d["bounds"][name],
                "prediction": pid,
            }
        )
    refs = svc.store.query("Reference", {"problem": cv.case.problem}, limit=5)
    return {
        "buckets": buckets,
        "references": [record_id(r) for r in refs],
        "family": str(d["families"][0]),
        "model": model_id,
        "significance_rel": sig,
    }


WORK: dict[str, Callable[[Services, Loop], Facts]] = {
    "new": work_new,
    "bounded": work_bounded,
    "retrieving": work_retrieving,
    "solving": work_solving,
    "building": work_building,
    "checking": work_checking,
    "measuring": work_measuring,
    "judging": work_judging,
}
