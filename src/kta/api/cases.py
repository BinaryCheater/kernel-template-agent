"""Case views: /cases, /cases/{id}/timeline, /lineage/{id} (docs/design/30 §1 Case, 结果详情).

Record conventions read here (spec/modules/api-web.md "Record conventions"):
Bound, Solution, Candidate and baseline Measurements carry ``refs.case``; a Solution has
``refs.family``, ``refs.measurement``, ``refs.check``, ``refs.bound``; a judgement is a
Report with ``data.kind == "judgement"`` and ``refs.case`` (``refs.solution`` when there is
one); loop steps are ``loop_step`` spans whose trace is the loop instance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from kta.api import opsread, pool, proofview
from kta.api.common import (
    Api,
    body,
    done_check_reports,
    envelope,
    incoming,
    iso,
    latest,
    ref_one,
    refs,
)
from kta.api.common import (
    record_ids as _ids,
)
from kta.core.ids import record_id
from kta.orchestrator.control import paused_scope
from kta.store.base import SpanRow

# Lifecycle states (D265) of a task still being worked on, i.e. neither terminal nor waiting
# for you.
LIVE_TASK = ("queued", "leased", "running", "submitted", "checking", "rejected", "expired")


def _loop_for(api: Api, case_id: str) -> opsread.LoopRow | None:
    return next(
        (
            lp
            for lp in reversed(opsread.loops(api.ops))
            if lp.kind == "solve" and lp.subject == case_id
        ),
        None,
    )


def _judgements(api: Api, case_id: str) -> list[str]:
    return [
        r
        for r in incoming(api, case_id, "case", "Report")
        if body(api, r).get("data", {}).get("kind") == "judgement"
    ]


def _measured_us(api: Api, measurement: str | None) -> float | None:
    return float(body(api, measurement)["median_us"]) if measurement else None


def _best(api: Api, case_id: str) -> dict[str, Any] | None:
    sols = incoming(api, case_id, "case", "Solution")
    if not sols:
        return None
    best = max(sols, key=lambda s: float(body(api, s)["pct_of_bound"]))
    meas = ref_one(api, best, "measurement")
    fam = ref_one(api, best, "family")
    return {
        "us": _measured_us(api, meas),
        "record": meas,
        "solution": best,
        "family": body(api, fam)["name"] if fam else None,
        "pct": float(body(api, best)["pct_of_bound"]),
    }


# Writers of measurement-fixture Cases (calibration peaks, self-tests): the frontend lists them
# apart from the research Cases (D661).
PROBE_WRITERS = ("kta measure",)


def purpose(api: Api, case_id: str, loop: opsread.LoopRow | None) -> str:
    """``solve`` for a research Case, ``probe`` for a measurement fixture never solved."""
    writer = str(api.store.get_json(case_id).get("actor", {}).get("name", ""))
    return "probe" if loop is None and writer.startswith(PROBE_WRITERS) else "solve"


def summary(api: Api, case_id: str) -> dict[str, Any]:
    case = body(api, case_id)
    loop = _loop_for(api, case_id)
    bound = latest(api, incoming(api, case_id, "case", "Bound"))
    if loop and len(loop.data.get("bounds", {})) > 1:
        bound = None  # A single bucket bound cannot stand for the whole case.
    best = _best(api, case_id)
    data = loop.data if loop else {}
    return {
        "id": case_id,
        "record": case_id,
        "name": (data.get("name") if data.get("name") != case_id[:12] else None)
        or body(api, case["problem"]).get("name", case["problem"]),
        "problem": case["problem"],
        "loop_id": loop.id if loop else None,
        "state": loop.state if loop else "not started",
        "round": data.get("round", 0),
        "blocked": data.get("blocked"),
        "paused": paused_scope(api.ops, [f"loop:{loop.id}", "process:solve", "global"]) is not None
        if loop
        else False,
        "bound_us": float(body(api, bound)["bound_us"]) if bound else None,
        "bound_record": bound,
        "best_us": best["us"] if best else None,
        "pct_of_bound": best["pct"] if best else None,
        "objective": case["objective"],
        "purpose": purpose(api, case_id, loop),
    }


def cases(api: Api) -> list[dict[str, Any]]:
    """GET /cases: summary plus per-round progress, the active task and last activity."""
    tasks = opsread.tasks(api.ops)
    out = []
    for cid in _ids(api, "Case"):
        s = summary(api, cid)
        rounds = _rounds(api, s["loop_id"], cid)
        bound = s["bound_us"]
        cands = [
            (api.store.get_json(c)["created_at"], c)
            for c in incoming(api, cid, "case", "Candidate")
        ]
        progress = []
        for r in rounds:
            lo, hi = r["steps"][0]["start"], r["steps"][-1]["end"] or "9999"
            best = [_best_measured(api, c) for at, c in cands if lo <= at <= hi]
            best_us = min((b for b in best if b is not None), default=None)
            progress.append(
                {
                    "n": r["n"],
                    "last_state": r["steps"][-1]["state"],
                    "open": r["steps"][-1]["end"] is None,
                    "rule": r["outcome"]["rule"] if r["outcome"] else None,
                    "tasks": [t["type"] for t in r["tasks"]],
                    "pct_of_bound": bound / best_us if bound and best_us else None,
                }
            )
        live = [t for t in tasks if t.work_unit == s["loop_id"] and t.state in LIVE_TASK]
        stamps = [st["end"] or st["start"] for r in rounds for st in r["steps"]]
        out.append(
            {
                **s,
                "rounds": progress,
                "active_task": {"type": live[-1].card_type, "state": live[-1].state}
                if live
                else None,
                "last_activity": max(stamps) if stamps else None,
            }
        )
    return out


def _best_measured(api: Api, cand: str) -> float | None:
    build = latest(api, incoming(api, cand, "candidate", "Build"))
    meas = latest(api, incoming(api, build, "build", "Measurement")) if build else None
    return _measured_us(api, meas)


def _rounds(api: Api, loop_id: str | None, case_id: str) -> list[dict[str, Any]]:
    if loop_id is None:
        return []
    steps: dict[int, list[SpanRow]] = defaultdict(list)
    for s in api.ops.spans(trace_id=loop_id, limit=20000):
        if s.kind == "loop_step":
            steps[int(s.attrs.get("round", 1))].append(s)
    tasks = [t for t in opsread.tasks(api.ops) if t.work_unit == loop_id]
    loop_data = next((lp.data for lp in opsread.loops(api.ops) if lp.id == loop_id), {})
    outcomes = {int(body(api, r)["data"].get("round", 0)): r for r in _judgements(api, case_id)}
    jobs = [s for s in api.ops.spans(trace_id=loop_id, limit=20000) if s.kind == "job_run"]
    events = _loop_events(api, loop_id)
    rounds = []
    for n in sorted(steps):
        ss = sorted(steps[n], key=lambda s: s.start_ts)
        lo, hi = ss[0].start_ts, ss[-1].end_ts
        mine = [t for t in tasks if lo <= t.created_at and (hi is None or t.created_at <= hi)]
        out = outcomes.get(n)
        rounds.append(
            {
                "n": n,
                "steps": [
                    {
                        "state": s.attrs.get("state", s.name),
                        "start": iso(s.start_ts),
                        "end": iso(s.end_ts),
                        "note": s.attrs.get("note"),
                        "work": _step_work(jobs, s.start_ts, s.end_ts),
                        "events": [
                            e for e in events if _within(e["at"], iso(s.start_ts), iso(s.end_ts))
                        ],
                    }
                    for s in ss
                ],
                "checks": _check_reasons(api, loop_data),
                "made": _made(
                    api,
                    refs(api, out).get("decomposition") if out else None,
                    loop_data if n == max(steps) else {},
                ),
                "tasks": [_round_task(api, t) for t in mine],
                "summary": _round_summary(api, case_id, iso(lo), iso(hi), loop_data),
                "outcome": {
                    "rule": body(api, out)["data"].get("rule"),
                    "label": body(api, out)["summary"],
                    "record": out,
                    # WP18: the round's gap class and its GapDecomposition record.
                    "gap_class": body(api, out)["data"].get("gap_class"),
                    "decomposition": refs(api, out).get("decomposition"),
                }
                if out
                else None,
            }
        )
    return rounds


def _made(api: Api, decomposition: Any, live: dict[str, Any]) -> dict[str, Any] | None:
    """How a round was made (SYSDEBUG-2 D708): which family and why, the formulation it
    assembled, and per bucket the solve and its proof gate. From the round's judgement
    (GapDecomposition inputs) once judged, else from the loop's current data."""
    per: dict[str, dict[str, Any]] = {}
    if isinstance(decomposition, str):
        for b in body(api, decomposition).get("buckets", []):
            per[str(b.get("name"))] = dict(b.get("inputs", {}).get("solver") or {})
    else:
        per = {str(k): dict(v) for k, v in (live.get("solver") or {}).items()}
    if not per:
        return None
    first = next(iter(per.values()))
    choice = first.get("family_choice") or live.get("family_choice")
    fam = (first.get("formulation") or {}).get("family")
    return {
        "family_choice": {
            **(choice or {}),
            **({"picked_ref": _family_ref(api, choice["picked"])} if choice else {}),
        }
        if choice
        else None,
        "family": fam,
        "formulation": first.get("formulation"),
        "solve": [
            {
                "bucket": name,
                "method": s.get("method"),
                "exhaustive": s.get("exhaustive"),
                "model_optimal": s.get("model_optimal"),
                "top_pred_us": s.get("top_pred_us"),
                "proof_gate": s.get("proof_gate"),
                "report": s.get("report"),
                # D715/D716: how it was solved, the shortlist and what was compiled
                "mip_status": s.get("mip_status"),
                "mip_gap": s.get("mip_gap"),
                "plan": s.get("plan"),
                "plan_note": s.get("plan_note"),
                "notes": s.get("notes") or [],
                "space_size": s.get("space_size"),
                "feasible": s.get("feasible"),
                "pruned_by": s.get("pruned_by") or {},
                "shortlist": s.get("shortlist") or [],
                "compiled": s.get("compiled") or [],
            }
            for name, s in per.items()
        ],
        "family_diff": _family_diff(api, choice["picked"]) if choice else None,
    }


DIFF_MAX_CHARS = 20_000


def _family_diff(api: Api, fid: str) -> dict[str, Any] | None:
    """The family's change against its previous version (D716), when it has one."""
    if not api.store.exists(fid):
        return None
    rec = api.store.get(fid)
    sha = rec.blobs.get("diff")
    prev = rec.refs.get("previous")
    if not sha:
        return {"previous": prev, "text": None} if prev else None
    text = api.store.blob_path(sha).read_text(encoding="utf-8", errors="replace")
    return {"previous": prev, "text": text[:DIFF_MAX_CHARS]}


def _round_summary(
    api: Api, case_id: str, lo: str | None, hi: str | None, d: dict[str, Any]
) -> dict[str, Any]:
    """What one round used and produced (D662): the families its Candidates name, the
    candidate funnel (built, checked, measured) and per bucket the best predicted and best
    measured latency. Candidates belong to the round they were created in."""
    cands = [
        c
        for c in incoming(api, case_id, "case", "Candidate")
        if lo and lo <= api.store.get_json(c)["created_at"] <= (hi or "9999")
    ]
    fams: list[str] = []
    built = build_failed = passed = failed = measured = 0
    buckets: dict[str, dict[str, Any]] = {}
    for c in cands:
        fam = ref_one(api, c, "family")
        if fam and fam not in fams:
            fams.append(fam)
        b = buckets.setdefault(
            body(api, c)["bucket"],
            {"candidates": 0, "predicted_best_us": None, "measured_best_us": None},
        )
        b["candidates"] += 1
        pred = latest(api, incoming(api, c, "candidate", "Prediction"))
        if pred:
            p = float(body(api, pred)["total"])
            b["predicted_best_us"] = min(p, b["predicted_best_us"] or p)
        # The loop data maps candidate -> Build/Check/Measurement (Build records do not
        # name their candidate); older candidates fall back to the record edges.
        build = d.get("builds", {}).get(c) or latest(api, incoming(api, c, "candidate", "Build"))
        if build is None:
            continue
        if not body(api, build)["ok"]:
            build_failed += 1
            continue
        built += 1
        # A candidate of the loop's current round counts only checks and measurements the
        # loop recorded for it: a cached Build carries Checks from earlier runs (as in
        # _candidates).
        current = c in d.get("builds", {})
        chk = d.get("checks", {}).get(c)
        checks = [chk] if chk else ([] if current else incoming(api, build, "build", "Check"))
        if checks:
            ok = all(body(api, x)["passed"] for x in checks)
            passed += ok
            failed += not ok
        meas = _measured_us(
            api,
            d.get("measurements", {}).get(c)
            or (None if current else latest(api, incoming(api, build, "build", "Measurement"))),
        )
        if meas is not None:
            measured += 1
            b["measured_best_us"] = min(meas, b["measured_best_us"] or meas)
    return {
        "families": [_family_ref(api, f) for f in fams],
        "candidates": len(cands),
        "built": built,
        "build_failed": build_failed,
        "check_passed": passed,
        "check_failed": failed,
        "measured": measured,
        "buckets": buckets,
    }


def _family_ref(api: Api, fid: str) -> dict[str, Any]:
    b = body(api, fid)
    return {"id": fid, "name": b.get("name", fid[:12]), "version": b.get("version")}


# Knowledge kinds a task may deliver (machines.yaml task_types: Family, Template, …).
DELIVERED = ("Family", "Chain", "Template", "Reconstruction", "Gap", "HwModel", "Report")


def _produced(api: Api, task_id: str) -> list[dict[str, Any]]:
    """Records a task wrote, knowledge kinds only, so a round shows what it changed."""
    with api.store.db.read() as con:
        rows = con.execute(
            "SELECT id, kind FROM records WHERE task = ? ORDER BY created_at",
            (task_id,),
        ).fetchall()
    out = []
    for rid, kind in rows:
        if kind not in DELIVERED:
            continue
        b = body(api, rid)
        label = b.get("name") or b.get("summary") or b.get("description") or kind
        version = b.get("version")
        out.append(
            {
                "id": rid,
                "kind": kind,
                "label": f"{label}@{version}" if version is not None else str(label)[:80],
            }
        )
    return out


def _round_task(api: Api, t: opsread.TaskRow) -> dict[str, Any]:
    """A task of one round: its lifecycle state (D265) and its rejections (D276)."""
    rejections = [
        {"at": r["created_at"], "record": r["id"], "reason": r["reason"]}
        for r in done_check_reports(api, t.id)
        if not r["passed"]
    ]
    ended = t.state in ("accepted", "cancelled", "escalated", "suspended")
    card = body(api, t.card_id)
    header = card.get("header") or {}
    with api.store.db.read() as con:
        run_ids = [
            r[0]
            for r in con.execute(
                "SELECT id FROM records WHERE task = ? AND kind = 'AgentRun' ORDER BY created_at",
                (t.id,),
            ).fetchall()
        ]
    return {
        # SYSDEBUG-2 (D708): what the agent is given and what it did, for the case page
        "inputs": list(card.get("inputs", [])),
        "header": header.get("text"),
        "situation": header.get("situation"),
        "runs": [
            {
                "id": rid,
                "exit": body(api, rid).get("exit"),
                "wall_s": body(api, rid).get("wall_s"),
                "model": body(api, rid).get("model"),
                # D716: context monitoring
                "tokens_in": body(api, rid).get("tokens_in"),
                "tokens_cached": body(api, rid).get("tokens_cached"),
                "tokens_out": body(api, rid).get("tokens_out"),
                "tool_calls": body(api, rid).get("tool_calls"),
                "session_mode": body(api, rid).get("session_mode"),
            }
            for rid in run_ids
        ],
        "id": t.id,
        "type": t.card_type,
        "card": t.card_id,
        "created_at": iso(t.created_at),
        "ended_at": iso(t.updated_at) if ended else None,
        "state": t.state,
        "goal": body(api, t.card_id).get("goal"),
        "rejections": rejections,
        "produced": _produced(api, t.id),
    }


def _candidates(api: Api, case_id: str) -> list[dict[str, Any]]:
    loop = _loop_for(api, case_id)
    d = loop.data if loop else {}
    current = {c for ids in d.get("candidates", {}).values() for c in ids}
    current.update(c for arms in d.get("arms", {}).values() for ids in arms.values() for c in ids)
    selected = {c for ids in d.get("top", {}).values() for c in ids}
    out = []
    for cand in incoming(api, case_id, "case", "Candidate"):
        if current and cand not in current:
            continue
        pred = latest(api, incoming(api, cand, "candidate", "Prediction"))
        build = d.get("builds", {}).get(cand) or latest(
            api, incoming(api, cand, "candidate", "Build")
        )
        meas = d.get("measurements", {}).get(cand)
        chk = d.get("checks", {}).get(cand)
        checks = (
            [chk]
            if chk
            else ([] if current else incoming(api, build, "build", "Check") if build else [])
        )
        if not current and build:
            meas = latest(api, incoming(api, build, "build", "Measurement"))
        proofs = [
            r
            for r in incoming(api, cand, "subject", "ProofResult")
            if body(api, r)["condition"] == "candidate"
        ]
        proof = latest(api, proofs)
        bucket = body(api, cand)["bucket"]
        arm = d.get("arm_of", {}).get(bucket, {}).get(cand)
        if arm is None:
            arm = next(
                (a for a, ids in d.get("arms", {}).get(bucket, {}).items() if cand in ids), "model"
            )
        out.append(
            {
                "assignment": body(api, cand)["assignment"],
                "bucket": bucket,
                "arm": arm,
                "selected": cand in selected,
                "build_record": build,
                "check_record": chk,
                "measurement_record": meas,
                "proof_record": proof,
                "claims": body(api, proof)["data"].get("claims") if proof else None,
                "record": cand,
                "predicted_us": float(body(api, pred)["total"]) if pred else None,
                "measured_us": _measured_us(api, meas),
                "build_ok": body(api, build)["ok"] if build else None,
                "check_passed": all(body(api, c)["passed"] for c in checks) if checks else None,
            }
        )
    out.sort(
        key=lambda c: (
            c["bucket"] or "",
            not c["selected"],
            c["predicted_us"] is None,
            c["predicted_us"] or 0,
        )
    )
    return [{"rank": i + 1, **c} for i, c in enumerate(out)]


def _references(api: Api, case_id: str) -> list[dict[str, Any]]:
    out = []
    loop = _loop_for(api, case_id)
    current = loop.data.get("baseline_measurements", {}) if loop else {}
    names = {mid: name for name, mid in current.items()}
    for m in incoming(api, case_id, "case", "Measurement"):
        if current and m not in names:
            continue
        base = ref_one(api, m, "baseline")
        if base is None:
            continue
        out.append(
            {
                "name": body(api, base).get("name") or body(api, base).get("title", "baseline"),
                "us": float(body(api, m)["median_us"]),
                "record": m,
                "bucket": names.get(m),
            }
        )
    return sorted(out, key=lambda r: r["us"])


def timeline(api: Api, case_id: str) -> dict[str, Any]:
    s = summary(api, case_id)
    case = body(api, case_id)
    refs_ = _references(api, case_id)
    best = _best(api, case_id)
    candidates = _candidates(api, case_id)
    buckets = []
    loop = _loop_for(api, case_id)
    d = loop.data if loop else {}
    for name, bound in d.get("bounds", {}).items():
        measured = [
            r["measured_us"]
            for r in candidates
            if r["bucket"] == name and r["measured_us"] is not None and r["check_passed"]
        ]
        buckets.append(
            {
                "bucket": name,
                "bound_us": float(body(api, bound)["bound_us"]),
                "bottleneck": body(api, bound).get("bottleneck"),
                "bound_record": bound,
                "best_us": min(measured) if measured else None,
                "references": {r["name"]: r["us"] for r in refs_ if r.get("bucket") == name},
                "solution": None,
            }
        )
    for sol in [] if buckets else incoming(api, case_id, "case", "Solution"):
        b = body(api, sol)
        bound = ref_one(api, sol, "bound")
        buckets.append(
            {
                "bucket": b.get("bucket") or "all",
                "bound_us": float(body(api, bound)["bound_us"]) if bound else s["bound_us"],
                "best_us": _measured_us(api, ref_one(api, sol, "measurement")),
                "references": {r["name"]: r["us"] for r in refs_},
                "solution": sol,
            }
        )
    if not buckets and s["bound_us"] is not None:
        buckets.append(
            {
                "bucket": "all",
                "bound_us": s["bound_us"],
                "best_us": None,
                "references": {r["name"]: r["us"] for r in refs_},
                "solution": None,
            }
        )
    gaps = incoming(api, case_id, "case", "Gap")
    rounds = _rounds(api, s["loop_id"], case_id)
    inputs = {
        "families": [pool.family_summary(api, f) for f in d.get("families", [])],
        "model": d.get("model"),
        "hw_model": d.get("hw_model"),
        "controls": d.get("controls"),
    }
    return {
        **s,
        "shapes": "; ".join(
            ", ".join(f"{k}={v}" for k, v in d.get("shape", {}).items()) for d in case["shape_dist"]
        ),
        "references": refs_,
        "best": best,
        "rounds": rounds,
        "buckets": buckets,
        "candidates": candidates,
        "gaps": [{"id": g, "description": body(api, g)["description"]} for g in gaps],
        "inputs": inputs,
        "standard": _standard(api, case_id),
        "anomalies": _anomalies(api, d, buckets, rounds),
    }


def lineage(api: Api, id: str, depth: int) -> dict[str, Any]:
    """GET /lineage/{id}: the dependency graph, plus the evidence chain when id is a Solution."""
    g = api.store.lineage(id, depth)
    out: dict[str, Any] = {"root": id, "nodes": g.nodes, "edges": [list(e) for e in g.edges]}
    if g.nodes[id] == "Solution":
        out["result"] = _result(api, id)
    return out


def _link(api: Api, stage: str, rid: str | None, title: str, fact: str) -> dict[str, Any] | None:
    if rid is None:
        return None
    env = envelope(api, rid)
    return {
        "stage": stage,
        "title": title,
        "fact": fact,
        "record": rid,
        "status": env["assurance"] or env["status"],
        "origin": env["origin"],
    }


def _result(api: Api, sol: str) -> dict[str, Any]:
    s = body(api, sol)
    case_id = ref_one(api, sol, "case")
    fam = ref_one(api, sol, "family")
    chain = ref_one(api, fam, "chain") if fam else None
    meas = ref_one(api, sol, "measurement")
    build = ref_one(api, meas, "build") if meas else None
    cand = ref_one(api, build, "candidate") if build else None
    pred = latest(api, incoming(api, cand, "candidate", "Prediction")) if cand else None
    checks = incoming(api, build, "build", "Check") if build else []
    sass = next(
        (c for c in checks if api.store.get_json(c)["actor"]["name"].startswith("check_sass")), None
    )
    numeric = ref_one(api, sol, "check")
    bound = ref_one(api, sol, "bound")
    judge = latest(api, [r for r in incoming(api, sol, "solution", "Report")])
    jdata = body(api, judge)["data"] if judge else {}
    m = body(api, meas) if meas else {}
    b = body(api, build) if build else {}
    bd = body(api, bound) if bound else {}
    links = [
        _link(
            api,
            "Templates",
            chain,
            f"{body(api, fam)['name']}@{body(api, fam)['version']} chain" if fam else "",
            " → ".join(i["template"] for i in body(api, chain)["items"]) if chain else "",
        ),
        _link(
            api,
            "Assignment",
            cand,
            "Chosen candidate",
            ", ".join(f"{k}={v}" for k, v in s["assignment"].items()),
        ),
        _link(
            api,
            "Prediction",
            pred,
            f"model {body(api, pred)['model']}" if pred else "",
            f"{body(api, pred)['total']} µs predicted" if pred else "",
        ),
        _link(
            api,
            "Build",
            build,
            api.store.get_json(build)["actor"]["name"] if build else "",
            f"{b.get('regs')} regs, {b.get('spills')} spills"
            + (f", {b['smem']} B smem" if b.get("smem") else ""),
        ),
        _link(
            api,
            "SASS check",
            sass,
            api.store.get_json(sass)["actor"]["name"] if sass else "",
            "expected instructions present" if sass and body(api, sass)["passed"] else "failed",
        ),
        _link(
            api,
            "Correctness",
            numeric,
            api.store.get_json(numeric)["actor"]["name"] if numeric else "",
            "passed" if numeric and body(api, numeric)["passed"] else "failed",
        ),
        _link(
            api,
            "Measurement",
            meas,
            f"{m.get('protocol')}, n={m.get('n')}",
            f"median {m.get('median_us')} µs [{m.get('ci_low_us')}, {m.get('ci_high_us')}]",
        ),
        _link(
            api,
            "Gap to bound",
            bound,
            f"bound {bd.get('bound_us')} µs ({bd.get('bottleneck')})",
            f"{s['pct_of_bound'] * 100:.0f}% of bound",
        ),
    ]
    return {
        "id": sol,
        "case_id": case_id,
        "case_name": summary(api, case_id)["name"] if case_id else "",
        "family": fam,
        "family_name": f"{body(api, fam)['name']}@{body(api, fam)['version']}" if fam else "",
        "bucket": s.get("bucket") or "all",
        "measured_us": m.get("median_us"),
        "bound_us": bd.get("bound_us"),
        "pct_of_bound": s["pct_of_bound"],
        "chain": [x for x in links if x is not None],
        "missing": [
            st
            for st, x in zip(
                [
                    "Templates",
                    "Assignment",
                    "Prediction",
                    "Build",
                    "SASS check",
                    "Correctness",
                    "Measurement",
                    "Gap to bound",
                ],
                links,
                strict=True,
            )
            if x is None
        ],
        "units": jdata.get("units", []),
        "explained_share": jdata.get("explained_share"),
        "verdict": body(api, judge)["summary"] if judge else None,
        "judgement": judge,
        "assurance": proofview.result_assurance(api, fam, cand, numeric, sass, build),
        "refs": refs(api, sol),
    }


# ---- what each step ran, and why checks failed (SYSDEBUG-2, D700) ----------------------


def _within(at: str | None, lo: str | None, hi: str | None) -> bool:
    return bool(at and lo and lo <= at <= (hi or "9999"))


def _step_work(jobs: list[SpanRow], lo: Any, hi: Any) -> list[dict[str, Any]]:
    """The Executor jobs that ran inside one step, per kind: how many, how they ended, the
    time they took and how many ran at once (the largest overlap)."""
    mine = [j for j in jobs if j.start_ts >= lo and (hi is None or j.start_ts <= hi)]
    kinds: dict[str, list[SpanRow]] = defaultdict(list)
    for j in mine:
        kinds[j.name.split(" ", 1)[0]].append(j)
    out = []
    for kind, js in sorted(kinds.items()):
        durs = [(j.end_ts - j.start_ts).total_seconds() for j in js if j.end_ts]
        edges = sorted([(j.start_ts, 1) for j in js] + [(j.end_ts, -1) for j in js if j.end_ts])
        now = peak = 0
        for _, d in edges:
            now += d
            peak = max(peak, now)
        out.append(
            {
                "kind": kind,
                "count": len(js),
                "failed": sum(j.status == "fail" for j in js),
                "busy_s": round(sum(durs), 1),
                "mean_s": round(sum(durs) / len(durs), 2) if durs else None,
                "max_s": round(max(durs), 1) if durs else None,
                "max_parallel": peak,
            }
        )
    return out


def _loop_events(api: Api, loop_id: str) -> list[dict[str, Any]]:
    """Blocks of the loop (event log) and the human actions that advanced or retried it."""
    out: list[dict[str, Any]] = []
    for e in api.ops.events_for("loops", loop_id):
        b = e.change.get("blocked")
        if b:
            out.append(
                {
                    "at": iso(e.at),
                    "kind": "blocked",
                    "text": f"stopped in {b.get('state')}: {b.get('error')}: "
                    f"{str(b.get('message', ''))[:300]}",
                }
            )
    for rec in api.store.query("HumanAction", order="-created_at", limit=2000):
        action = str(rec.body.get("action", ""))
        if not action.startswith("advance ") or loop_id not in action:
            continue
        out.append(
            {
                "at": iso(rec.created_at),
                "kind": "human",
                "text": f"{rec.actor.name}: {action.split(' ', 1)[1]} — "
                f"{rec.body.get('reason', '')}",
            }
        )
    for rec in api.store.query(
        "Report", {"subject": "measurement_interfered"}, order="-created_at", limit=2000
    ):
        out.append(
            {
                "at": iso(rec.created_at),
                "kind": "rejected_measurement",
                "text": f"measurement not counted: {rec.body.get('summary')}",
                "record": record_id(rec),
            }
        )
    return sorted(out, key=lambda e: str(e["at"]))


def _check_reasons(api: Api, d: dict[str, Any]) -> dict[str, Any]:
    """The round's checks grouped by failure reason: quick checks (gate before measuring)
    and full checks (a bucket's best), each reason with the candidates it hit, one
    example's numbers and the records."""
    out: dict[str, Any] = {}
    for level, key in (("quick", "checks"), ("full", "final_checks")):
        mapping: dict[str, str] = d.get(key) or {}
        groups: dict[str, dict[str, Any]] = {}
        passed = 0
        for chk in mapping.values():
            b = body(api, chk)
            if b.get("passed"):
                passed += 1
                continue
            seen: set[str] = set()
            for f in b.get("failures", []):
                name = str(f.get("check", "?"))
                if f.get("dist"):
                    name += f" · {f['dist']}"
                g = groups.setdefault(
                    name, {"reason": name, "candidates": 0, "example": f, "records": []}
                )
                if name not in seen:
                    g["candidates"] += 1
                    seen.add(name)
                if len(g["records"]) < 5 and chk not in g["records"]:
                    g["records"].append(chk)
        out[level] = {
            "total": len(mapping),
            "passed": passed,
            "reasons": sorted(groups.values(), key=lambda g: -g["candidates"]),
        }
    return out


def _standard(api: Api, case_id: str) -> dict[str, Any] | None:
    """The correctness standard the case is checked against: its Contract, and who wrote it."""
    cref = refs(api, case_id).get("contract")
    if not isinstance(cref, str):
        return None
    rec = api.store.get(cref)
    return {
        "record": cref,
        "written_by": rec.actor.name,
        "written_at": iso(rec.created_at),
        **rec.body,
    }


def _anomalies(
    api: Api, d: dict[str, Any], buckets: list[dict[str, Any]], rounds: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Facts on the page that contradict each other or stop progress, said plainly."""
    out: list[dict[str, Any]] = []
    for b in buckets:
        for name, us in (b.get("references") or {}).items():
            if us is not None and b.get("bound_us") and us < b["bound_us"]:
                out.append(
                    {
                        "level": "fail",
                        "text": f"{b['bucket']}: {name} measured {us:.1f} µs, below the lower "
                        f"bound {b['bound_us']:.1f} µs — the bound is not a bound (hardware "
                        "peak too low)",
                        "record": b.get("bound_record"),
                    }
                )
    for r in rounds:
        q = (r.get("checks") or {}).get("quick") or {}
        if q.get("total") and not q.get("passed"):
            out.append(
                {
                    "level": "fail",
                    "text": f"round {r['n']}: none of {q['total']} candidates passed the check "
                    "before measuring, so none was measured",
                }
            )
        for st in r["steps"]:
            for w in st.get("work", []):
                if w["kind"] == "compile" and w["count"] > 4 and w["max_parallel"] <= 1:
                    out.append(
                        {
                            "level": "warn",
                            "text": f"round {r['n']}, step {st['state']}: {w['count']} compiles "
                            f"ran one at a time ({w['busy_s']:.0f} s in all)",
                        }
                    )
    return out
