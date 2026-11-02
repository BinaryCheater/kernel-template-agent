"""The work of each solve-loop state: real module calls whose results become facts.

Each handler reads the loop row, does the state's work (docs/design/10 §2.1) through the
knowledge, models, formulation, solve, backends, checks and measure modules, writes the
records those modules write, and returns the facts ``solve_loop.step`` decides on.
One family per instance: the best-ranked retrieved family (known issue, WP6 note).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, cast

from kta.checks.full import run_full, run_quick
from kta.checks.sass import sass_check
from kta.core import hostlock
from kta.core.errors import ExecutionError, NotSupported, SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Bound, Build, Measurement, Prediction
from kta.core.schemas.evaluation import Candidate as CandidateBody
from kta.formulation import assemble
from kta.knowledge import retrieve_families
from kta.knowledge.findings import couplings
from kta.measure.baseline import measure_baseline
from kta.measure.profile import run_profile
from kta.models import bound, min_work
from kta.orchestrator import analysis, disciplines, pipeline, proofgate
from kta.orchestrator.pipeline import Services, bucket_shape, case_view
from kta.orchestrator.solve_baseline import baseline_us
from kta.orchestrator.solve_evaluate import checkpoint, each_top, paused, pending
from kta.orchestrator.transition import Loop, Settings
from kta.solve import (
    EXHAUSTIVE_LIMIT,
    Candidate,
    enumerate_space,
    rerank,
    shape_buckets,
    solve,
    space_size,
)
from kta.solve.plan import default_plan, plan_inputs
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

    fams = allowed(retrieve_families(svc.store, cv.case, k=20, hw=hw.limits, case_id=cv.id))
    if fams:
        ids = [c.id for c in fams]
        # D706: a family an agent just delivered goes first, so the round measures the
        # agent's change; the incumbent stays the baseline judging compares against.
        # Rollback (D527): otherwise the incumbent's family goes first while retrievable.
        tried = loop.data.get("try_family")
        incumbent = loop.data.get("prefer_family")
        prefer = tried if tried in ids else incumbent
        order = sorted(ids, key=lambda i: i != prefer)[:5]
        why = (
            "delivered by the agent task of the last round"
            if prefer == tried and tried
            else "the incumbent (best earlier round)"
            if prefer == incumbent and incumbent in ids
            else "best retrievable for this case (scope, assurance, similarity)"
        )
        out: dict[str, Any] = {
            "families": order,
            "family_choice": {"picked": order[0], "why": why, "retrieved": len(ids)},
        }
        if tried and tried not in ids:
            out["delivered_family_note"] = (
                f"delivered family {tried} is not retrievable for this case (scope, "
                "hardware or status); the round uses the best retrievable family"
            )
            out["family_choice"]["note"] = out["delivered_family_note"]
        return out
    # D717: no family of this case yet. Choosing the Templates, the chain and the search
    # space is the formulation the agent is asked to do; the system preselects nothing (no
    # nearest chain, no keep/replace verdicts). The agent reaches the knowledge (Templates,
    # reference chains) through `kta query` and its skills.
    inputs = [cv.id]
    goal = (
        "Build the first family for this case: choose the Templates and the search space "
        "yourself and say why"
    )
    if knowledge != "pool":
        goal += f" (knowledge condition: {knowledge})"
    data: dict[str, Any] = {"knowledge": knowledge, "preselected": None}
    if members is not None:
        data["snapshot"] = len(members)
    rid = _report(svc, "retrieval", cv.id, goal, data, {"case": cv.id})
    return {"families": [], "t1_goal": goal, "t1_inputs": [*inputs, rid]}


def work_solving(svc: Services, loop: Loop) -> Facts:
    s = Settings.of(svc.ctx.controls)
    cv = case_view(svc.store, loop.subject)
    hw_id, hw = svc.machine(cv.case.target)
    model_id, model = svc.unit_model(cv.case.target)
    family = str(loop.data["families"][0])
    fp = pipeline.family_package(svc, family)
    templates = pipeline.templates_of(svc.store, fp)
    f = assemble(fp, fp.chain, templates, cv.case, hw.hw)
    out: dict[str, list[str]] = {}
    gate: dict[str, dict[str, Any]] = {}
    solver: dict[str, dict[str, Any]] = {}
    arms: dict[str, dict[str, list[str]]] = {}
    for b in loop.data["buckets"]:
        rng = svc.ctx.rng(f"solve:{loop.id}")
        # On the compute host (long-run mode) the in-process solve is heavy work: it holds
        # the host lock shared, so no measurement runs beside it (WP23, D604).
        proj, plan_note = projection_inputs(fp, f, hw, b)
        with hostlock.heavy(svc.ctx.config.section(hostlock.HOST_SECTION), f"solve:{loop.id}"):
            res = solve(
                f,
                cv.case,
                b,
                model,
                hw,
                shortlist=s.shortlist,
                time_limit_s=s.solve_time_limit_s,
                rng=rng,
                projection=proj,
            )
        recs = record_solve(svc.ctx, svc.store, res, ACTOR, cv.id, family, hw_id, model_id)
        name = str(b["name"])
        # once per candidate (WP24): C and Γ at the assignment before anything is built
        kept, checks, claims = proofgate.candidate_gate(
            svc, family, fp, templates, hw.hw, b, recs.candidates
        )
        out[name] = kept
        dropped = [c for c in recs.candidates if c not in kept]
        gate[name] = {"checks": checks, "claims": claims, "dropped": dropped}
        top = res.candidates[0].prediction.total if res.candidates else None
        relaxed, dpre_note = proofgate.relaxed_prediction(
            svc, fp, templates, cv.case, b, model, hw, top,
            s.shortlist, s.solve_time_limit_s, svc.ctx.rng(f"relaxed:{loop.id}"),
        )  # fmt: skip
        solver[name] = {
            # SYSDEBUG-2 (D708): what the round was made of, kept with the judgement so the
            # case page can show family choice, formulation and proof gate per round
            "family_choice": loop.data.get("family_choice"),
            "formulation": formulation_summary(f),
            "proof_gate": {"checked": len(checks), "kept": len(kept), "dropped": len(dropped)},
            "relaxed_pred_us": relaxed,
            "delta_pre_note": dpre_note,
            "method": res.method,
            "model_optimal": res.model_optimal,
            "exhaustive": res.exhaustive,
            "mip_gap": res.mip_gap,
            # D715: how the round was solved and why (projection MIP or the fallback)
            "mip_status": res.mip_status,
            "plan": proj,
            "plan_note": plan_note,
            "notes": res.notes,
            "space_size": res.space_size,
            "feasible": res.feasible,
            "pruned_by": res.pruned_by,
            "shortlist": shortlist_view(res.candidates, recs.candidates, kept),
            "compiled": kept[: s.compile_count],
            "top_pred_us": res.candidates[0].prediction.total if res.candidates else None,
            "report": recs.report,
        }
        if recs.candidates:
            arms[name] = selection_arms(svc, loop, s, f, hw, b, family, cv.id)
    empty = [name for name, ids in out.items() if not ids]
    return {
        "candidates": out,
        "empty_buckets": empty,
        "families_tried": [family],
        "family": family,
        "model": model_id,
        "solver": solver,
        "arms": arms,
        "proof_gate": gate,
    }


def projection_inputs(
    fp: Any, f: Any, mm: Any, b: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    """The default FormulationPlan's fixed/menus/outer for the projection MIP (D715).

    Every round solves the family's formulation with the projection compiler: finite
    menus one-hot, products of menu variables and table terms tabulated, ``ceil`` as
    integer columns, the rest enumerated outside (02 §5.2). When no plan can be made the
    round falls back to enumeration, and the reason is kept with the round."""
    try:
        plan = default_plan(fp, f, mm, bucket_shape(b))
        return plan_inputs(plan, f), "default plan"
    except (NotSupported, SchemaError) as exc:
        return None, f"no plan, enumeration: {exc}"


SHORTLIST_SHOWN = 12


def _domain_text(v: Any) -> str:
    d: Any = v.domain
    if isinstance(d, list):
        return "{" + ",".join(map(str, cast(list[Any], d))) + "}"
    lo, hi = getattr(d, "lo", None), getattr(d, "hi", None)
    return f"{lo}..{hi}" if lo is not None else str(d)


def shortlist_view(cands: list[Any], ids: list[str], kept: list[str]) -> list[dict[str, Any]]:
    """The best-predicted candidates as the case page shows them (D716): assignment,
    predicted latency, the unit that bounds it, and whether the proof gate kept it."""
    out: list[dict[str, Any]] = []
    for c, cid in list(zip(cands, ids, strict=True))[:SHORTLIST_SHOWN]:
        units = dict(c.prediction.per_unit or {})
        out.append(
            {
                "id": cid,
                "assignment": dict(c.assignment),
                "pred_us": c.prediction.total,
                "bottleneck": max(units, key=lambda u: units[u]) if units else None,
                "kept": cid in kept,
            }
        )
    return out


def formulation_summary(f: Any) -> dict[str, Any]:
    """The assembled formulation in a few numbers (D708): its variables, constraints by
    kind and one row per chain block (Template, constraints, constants it fixes)."""
    kinds: dict[str, int] = {}
    for c in f.constraints:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    return {
        "family": f.family,
        "fingerprint": f.fingerprint[:12],
        "variables": sorted(f.variables),
        "runtime_vars": list(f.runtime_vars),
        "constraints": kinds,
        "linear": f.linear,
        "transactions": len(f.transactions),
        "missing_hw": list(f.missing_hw),
        # D716: the exported problem, one row per constraint with its source step
        "exprs": [
            {
                "source": c.source,
                "id": c.id,
                "kind": c.kind,
                "expr": c.expr.src,
                "meaning": c.meaning,
            }
            for c in f.constraints
        ],
        "domains": {k: _domain_text(v) for k, v in f.variables.items()},
        "blocks": [
            {
                "key": k,
                "template": b.template,
                "constraints": len(b.constraints),
                "fixed": dict(b.fixed),
                "uses": sorted(b.uses),
            }
            for k, b in f.blocks.items()
        ],
    }


def selection_arms(
    svc: Services, loop: Loop, s: Settings, f: Any, hw: Any, b: Any, family: str, case: str
) -> dict[str, list[str]]:
    """Candidate records of the control arms (12 §3 selection controls and exploration
    share, D524): ``random`` = exploration_k feasible points drawn uniformly by reservoir
    sampling over the enumerated space, chosen without the model; ``copied`` = the
    assignment of the newest Solution of this family on another case."""
    shape = bucket_shape(b)
    rng = svc.ctx.rng(f"explore:{loop.id}:{b['name']}")
    picked: list[dict[str, Any]] = []
    seen = [0]

    def visit(a: dict[str, Any]) -> None:
        seen[0] += 1
        if len(picked) < s.exploration_k:
            picked.append(dict(a))
        else:
            j = rng.randrange(seen[0])
            if j < s.exploration_k:
                picked[j] = dict(a)

    if s.exploration_k:
        limit = None if space_size(f) < EXHAUSTIVE_LIMIT else EXHAUSTIVE_LIMIT
        enumerate_space(f, shape, hw.hw, visit, max_nodes=limit, rng=rng)
    now, refs = svc.ctx.clock.now(), {"case": case, "family": family}

    def put(a: dict[str, Any], extra: Mapping[str, str]) -> str:
        body = CandidateBody(assignment=a, bucket=str(b["name"]))
        return svc.store.put(Record.new("Candidate", body, ACTOR, now, refs={**refs, **extra}))

    arms = {"random": [put(a, {}) for a in picked], "copied": []}
    if s.selection_controls:
        sols = svc.store.query("Solution", {"refs.family": family}, order="-created_at", limit=50)
        for sol in sols:
            if sol.refs.get("case") != case:
                origin = {"copied_from": record_id(sol)}
                arms["copied"] = [put(dict(sol.body["assignment"]), origin)]
                break
    return arms


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
    s = Settings.of(svc.ctx.controls)
    cv = case_view(svc.store, loop.subject)
    _, hw = svc.machine(cv.case.target)
    _, model = svc.unit_model(cv.case.target)
    family = str(loop.data["families"][0])
    fp = pipeline.family_package(svc, family)
    top: dict[str, list[str]] = {}
    builds: dict[str, str] = {}
    arms: dict[str, dict[str, str]] = {}
    controls: Mapping[str, Mapping[str, list[str]]] = loop.data.get("arms") or {}
    for name, ids in loop.data["candidates"].items():
        shape = bucket_shape(_bucket(loop, name))
        # D697: only the best-predicted compile_count of the shortlist get build feedback,
        # compiled build_width at a time (the Executor's cpu pool runs them side by side).
        cands = {cid: _candidate(svc, cid) for cid in ids[: s.compile_count]}
        by_key: dict[str, Build] = {}

        def one(
            item: tuple[str, Candidate], shape: dict[str, Any] = shape
        ) -> tuple[str, Candidate, Build, Any]:
            cid, c = item
            build, module = pipeline.build(svc, pipeline.generate(fp, c.assignment, shape))
            return cid, c, build, module

        with ThreadPoolExecutor(max_workers=max(1, s.build_width)) as pool:
            done = list(pool.map(one, cands.items()))
        for cid, c, build, module in done:
            builds[cid] = module.build_id
            by_key[c.key] = build
            if build.ok:
                _sass(svc, loop, fp, module, c.assignment)
        extra = controls.get(name, {})
        # The random arm takes exploration_k of the top_k measured points (12 §3).
        k = s.top_k - len(extra.get("random", []))
        kept = {c.key for c in rerank(list(cands.values()), by_key, model, hw, k=k)}
        top[name] = [cid for cid, c in cands.items() if c.key in kept]
        arms[name] = {cid: "model" for cid in top[name]}
        wanted = [
            (arm, cid)
            for arm, cids in extra.items()
            for cid in proofgate.control_gate(svc, family, fp, hw, _bucket(loop, name), cids)
        ]

        def control(
            item: tuple[str, str], shape: dict[str, Any] = shape
        ) -> tuple[str, str, Build, Any]:
            arm, cid = item
            a = CandidateBody.model_validate(svc.store.get(cid).body).assignment
            build, module = pipeline.build(svc, pipeline.generate(fp, a, shape))
            return arm, cid, build, module

        with ThreadPoolExecutor(max_workers=max(1, s.build_width)) as pool:
            controlled = list(pool.map(control, wanted))
        for arm, cid, build, module in controlled:
            builds[cid] = module.build_id
            if build.ok:
                top[name].append(cid)
                arms[name][cid] = arm
    return {"top": top, "builds": builds, "family": family, "arm_of": arms}


def work_checking(svc: Services, loop: Loop) -> Facts:
    checker = svc.checker()

    # D694: the gate before measuring is the quick check (loose level, the module as built);
    # the full check runs on each bucket's best measured candidate in judging.
    def check(cv: Any, fp: Any, gk: Any, module: Any, shape: Any, a: Any, cid: str) -> str:
        return run_quick(
            checker,
            module,
            gk,
            cv.problem,
            cv.contract,
            cv.case,
            case_id=cv.id,
            shape=shape,
            sass=pipeline.chain_sass(svc.store, fp, a),
            assignment=a,
        ).id

    checks = each_top(svc, loop, check, "checks")
    failed = [cid for cid in checks.values() if not svc.store.get(cid).body["passed"]]
    return {"checks": checks, "failed": failed, "pending": pending(loop, checks)}


def work_measuring(svc: Services, loop: Loop) -> Facts:
    round_ = int(loop.data.get("round", 1))
    if loop.data.get("measurement_round") != round_:
        updates: dict[str, Any] = {
            "protocol": svc.protocol()[0],
            "measurement_round": round_,
            "measurements": {},
            "baseline_measurements": {},
        }
        checkpoint(svc, loop, **updates)
        loop = replace(loop, data={**loop.data, **updates})
    protocol_id = str(loop.data["protocol"])
    m = svc.measurer()
    passed = {c for c, chk in loop.data["checks"].items() if svc.store.get(chk).body["passed"]}

    def meas(cv: Any, fp: Any, gk: Any, module: Any, shape: Any, a: Any, cid: str) -> str:
        return m.measure(module, cv.case, shape, protocol_id, candidate=cid).id

    top = {k: [c for c in v if c in passed] for k, v in loop.data["top"].items()}
    loop_passed = Loop(loop.id, loop.kind, loop.subject, loop.state, {**loop.data, "top": top})
    measured = each_top(svc, loop_passed, meas, "measurements")
    # Baselines use the same pinned protocol and current-round bucket identity.
    cv = case_view(svc.store, loop.subject)
    baselines: dict[str, str] = dict(loop.data.get("baseline_measurements", {}))
    if cv.baseline is not None:
        for b in loop.data["buckets"]:
            if paused(svc, loop):
                break
            if str(b["name"]) in baselines:
                continue
            got = measure_baseline(m, cv.baseline, cv.case, bucket_shape(b), protocol_id, cv.id)
            baselines[str(b["name"])] = got.id
            checkpoint(svc, loop, baseline_measurements=dict(baselines))
    incomplete = pending(loop_passed, measured) or (
        cv.baseline is not None
        and any(str(b["name"]) not in baselines for b in loop.data["buckets"])
    )
    return {
        "measurements": measured,
        "protocol": protocol_id,
        "baseline_measurements": baselines,
        "pending": incomplete,
    }


FINAL_TRIES = 3  # measured candidates, fastest first, given the full check per bucket


def _final_checked(
    svc: Services, loop: Loop, name: str, rows: list[tuple[str, str, Prediction, Measurement]]
) -> tuple[tuple[str, str, Prediction, Measurement] | None, list[list[str]]]:
    """The fastest measured candidate of a bucket that passes the full check at the strict
    level (D694), and the [check, candidate] pairs of those that did not. Results are kept
    in loop data ``final_checks`` so a resumed judging step does not check twice. The map is
    read from the stored loop, not the step's snapshot: buckets checkpoint one after another,
    and the snapshot would drop the entries of the buckets before this one (SYSDEBUG5)."""
    done: dict[str, str] = _final_map(svc, loop)
    cv = case_view(svc.store, loop.subject)
    family = str(loop.data["families"][0])
    fp = pipeline.family_package(svc, family)
    shape = bucket_shape(_bucket(loop, name))
    small = pipeline.sanitizer_shape(cv)
    failed: list[list[str]] = []
    for row in rows[:FINAL_TRIES]:
        cid = row[0]
        if cid not in done:
            a = CandidateBody.model_validate(svc.store.get(cid).body).assignment
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
                sass=pipeline.chain_sass(svc.store, fp, a),
                assignment=a,
                sanitizer_shape=small,
                level="strict",
            )
            done[cid] = chk.id
            checkpoint(svc, loop, final_checks=dict(done))
        if svc.store.get(done[cid]).body["passed"]:
            return row, failed
        failed.append([done[cid], cid])
    return None, failed


# D735: the per-unit counters read on each bucket's best candidate, by KTA resource. The
# tensor pipe, L2 sectors and shared-memory wavefronts read on GB10; DRAM reads n/a there.
NCU_UNITS = {
    "tensor": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l2": "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "smem": "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
}
NCU_STALLS = ("long_scoreboard", "barrier", "mio_throttle", "short_scoreboard",
              "math_pipe_throttle", "wait", "not_selected", "lg_throttle", "selected")  # fmt: skip
NCU_METRICS = [
    "gpu__time_duration.sum",
    *NCU_UNITS.values(),
    *(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio" for s in NCU_STALLS),
]


def _ncu(svc: Services, loop: Loop, name: str, cid: str) -> dict[str, Any]:
    """One ncu profile of the bucket's best candidate (checkpointed in loop data ``ncu`` so
    a resumed judging step does not profile twice); an error is reported, not raised."""
    done: dict[str, Any] = dict(svc.ops.get_loop(loop.id)["data"].get("ncu") or {})
    if done.get(name, {}).get("candidate") == cid:
        return dict(done[name])
    cv = case_view(svc.store, loop.subject)
    shape = bucket_shape(_bucket(loop, name))
    try:
        fp = pipeline.family_package(svc, str(loop.data["families"][0]))
        a = CandidateBody.model_validate(svc.store.get(cid).body).assignment
        _, module = pipeline.build(svc, pipeline.generate(fp, a, shape))
        pid, body = run_profile(svc.measurer(), module, cv.case, shape, NCU_METRICS)
        m = body.metrics
        out: dict[str, Any] = {
            "candidate": cid,
            "profile": pid,
            "time_us": m.get("gpu__time_duration.sum", 0.0) / 1e3 or None,
            "busy": {u: m[k] / 100 for u, k in NCU_UNITS.items() if k in m},
            "stalls": {
                s: m[k]
                for s in NCU_STALLS
                if (k := f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") in m
            },
        }
    except (NotSupported, ExecutionError, SchemaError) as exc:
        out = {"candidate": cid, "error": f"{type(exc).__name__}: {exc}"}
    done[name] = out
    checkpoint(svc, loop, ncu=done)
    return out


def _final_map(svc: Services, loop: Loop) -> dict[str, str]:
    """The full checks judging recorded so far (checkpointed in loop data)."""
    return dict(svc.ops.get_loop(loop.id)["data"].get("final_checks") or {})


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
        arm_of: Mapping[str, str] = d.get("arm_of", {}).get(name, {})
        arms: dict[str, Any] = {"model": [], "random": [], "copied": []}
        for cid in ids:
            if cid not in d.get("measurements", {}):
                continue
            if arm_of.get(cid, "model") != "model":  # control arms carry no prediction
                us = Measurement.model_validate(svc.store.get(d["measurements"][cid]).body)
                arms[arm_of[cid]].append(us.median_us)
                if arm_of[cid] == "copied":
                    arms["copied_from"] = svc.store.get(cid).refs.get("copied_from")
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
        arms["model"] = [(r[2].total, r[3].median_us) for r in rows]
        best = _final_checked(svc, loop, name, sorted(rows, key=lambda r: r[3].median_us))
        failed += best[1]
        if best[0] is None:
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
        cid, pid, pred, meas = best[0]
        ncu = _ncu(svc, loop, name, cid)
        solver = dict(d.get("solver", {}).get(name, {}))
        units = [
            {"unit": u, "bound_us": bnd.per_unit_us.get(u), "predicted_us": t, "measured_us": None}
            for u, t in sorted(pred.per_unit.items())
        ]
        buckets.append(
            {
                "name": name,
                "shape": bucket_shape(_bucket(loop, name)),  # for the task header (D707)
                "failed_checks": failed,
                "pairs": [(r[2].total, r[3].median_us) for r in rows],
                "evidence": [x for r in rows for x in (r[1], d["measurements"][r[0]])],
                "best_us": meas.median_us,
                "predicted_us": pred.total,
                "bound_us": bnd.bound_us,
                "reference_us": baseline_us(
                    svc.store, loop, name, bucket_shape(_bucket(loop, name)), cv.baseline
                ),
                "units": units,
                "assignment": CandidateBody.model_validate(svc.store.get(cid).body).assignment,
                "measurement": d["measurements"][cid],
                "check": d["checks"][cid],
                "bound": d["bounds"][name],
                "prediction": pid,
                "ci_us": (meas.ci_high_us - meas.ci_low_us) / 2,
                "top_pred_us": solver.get("top_pred_us"),
                "relaxed_pred_us": solver.get("relaxed_pred_us"),
                "delta_pre_note": solver.get("delta_pre_note"),
                "solver": solver,
                "unpriced": dict(pred.unpriced),
                "ncu": ncu,
                "arms": arms,
                "weight": float(_bucket(loop, name).get("weight", 1.0)),
            }
        )
    refs = svc.store.query("Reference", {"problem": cv.case.problem}, limit=5)
    family = str(d["families"][0])
    # D733: judging is the analysis; the last round's analysis is still in the loop data
    report = analysis.analyze(buckets, d.get("analysis"))
    templates = [i.template for i in pipeline.family_package(svc, family).chain.items]
    return {
        # D701: returned as a fact too, or the transition writes the loop data read before
        # judging and the checkpointed full checks are lost
        "final_checks": _final_map(svc, loop),
        "case_name": cv.problem.name,  # for the task header (D707)
        "basis": disciplines.basis(templates, family, model_id),
        "couplings": [
            [m.name for m in c.body.members]
            for c in couplings(svc.store, target=cv.case.target)
            if any(m.name in {t.split("@", 1)[0] for t in templates} for m in c.body.members)
        ],
        "buckets": buckets,
        "analysis": report,
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
