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

from kta.api import opsread
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
from kta.store.base import SpanRow


# Lifecycle states (D265) of a task still being worked on, i.e. neither terminal nor waiting
# for you.
LIVE_TASK = ("queued", "leased", "running", "submitted", "checking", "rejected", "expired")


def _loop_for(api: Api, case_id: str) -> opsread.LoopRow | None:
    return next(
        (lp for lp in opsread.loops(api.ops) if lp.kind == "solve" and lp.subject == case_id), None
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


def summary(api: Api, case_id: str) -> dict[str, Any]:
    case = body(api, case_id)
    loop = _loop_for(api, case_id)
    bound = latest(api, incoming(api, case_id, "case", "Bound"))
    best = _best(api, case_id)
    data = loop.data if loop else {}
    return {
        "id": case_id,
        "record": case_id,
        "name": data.get("name") or case["problem"],
        "problem": case["problem"],
        "loop_id": loop.id if loop else None,
        "state": loop.state if loop else "not started",
        "round": data.get("round", 0),
        "bound_us": float(body(api, bound)["bound_us"]) if bound else None,
        "bound_record": bound,
        "best_us": best["us"] if best else None,
        "pct_of_bound": best["pct"] if best else None,
        "objective": case["objective"],
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


def _ids(api: Api, kind: str) -> list[str]:
    with api.store.db.read() as con:
        rows = con.execute(
            "SELECT id FROM records WHERE kind = ? ORDER BY created_at", (kind,)
        ).fetchall()
    return [r[0] for r in rows]


def _rounds(api: Api, loop_id: str | None, case_id: str) -> list[dict[str, Any]]:
    if loop_id is None:
        return []
    steps: dict[int, list[SpanRow]] = defaultdict(list)
    for s in api.ops.spans(trace_id=loop_id, limit=20000):
        if s.kind == "loop_step":
            steps[int(s.attrs.get("round", 1))].append(s)
    tasks = [t for t in opsread.tasks(api.ops) if t.work_unit == loop_id]
    outcomes = {int(body(api, r)["data"].get("round", 0)): r for r in _judgements(api, case_id)}
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
                    }
                    for s in ss
                ],
                "tasks": [_round_task(api, t) for t in mine],
                "outcome": {
                    "rule": body(api, out)["data"].get("rule"),
                    "label": body(api, out)["summary"],
                    "record": out,
                }
                if out
                else None,
            }
        )
    return rounds


def _round_task(api: Api, t: opsread.TaskRow) -> dict[str, Any]:
    """A task of one round: its lifecycle state (D265) and its rejections (D276)."""
    rejections = [
        {"at": r["created_at"], "record": r["id"], "reason": r["reason"]}
        for r in done_check_reports(api, t.id)
        if not r["passed"]
    ]
    ended = t.state in ("accepted", "cancelled", "escalated", "suspended")
    return {
        "id": t.id,
        "type": t.card_type,
        "card": t.card_id,
        "created_at": iso(t.created_at),
        "ended_at": iso(t.updated_at) if ended else None,
        "state": t.state,
        "rejections": rejections,
    }


def _candidates(api: Api, case_id: str) -> list[dict[str, Any]]:
    out = []
    for cand in incoming(api, case_id, "case", "Candidate"):
        pred = latest(api, incoming(api, cand, "candidate", "Prediction"))
        build = latest(api, incoming(api, cand, "candidate", "Build"))
        meas = latest(api, incoming(api, build, "build", "Measurement")) if build else None
        checks = [c for c in incoming(api, build, "build", "Check")] if build else []
        out.append(
            {
                "assignment": body(api, cand)["assignment"],
                "record": cand,
                "predicted_us": float(body(api, pred)["total"]) if pred else None,
                "measured_us": _measured_us(api, meas),
                "build_ok": body(api, build)["ok"] if build else None,
                "check_passed": all(body(api, c)["passed"] for c in checks) if checks else None,
            }
        )
    out.sort(key=lambda c: (c["predicted_us"] is None, c["predicted_us"] or 0))
    return [{"rank": i + 1, **c} for i, c in enumerate(out)]


def _references(api: Api, case_id: str) -> list[dict[str, Any]]:
    out = []
    for m in incoming(api, case_id, "case", "Measurement"):
        base = ref_one(api, m, "baseline")
        if base is None:
            continue
        out.append(
            {"name": body(api, base)["name"], "us": float(body(api, m)["median_us"]), "record": m}
        )
    return sorted(out, key=lambda r: r["us"])


def timeline(api: Api, case_id: str) -> dict[str, Any]:
    s = summary(api, case_id)
    case = body(api, case_id)
    refs_ = _references(api, case_id)
    best = _best(api, case_id)
    buckets = []
    for sol in incoming(api, case_id, "case", "Solution"):
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
    return {
        **s,
        "shapes": "; ".join(
            ", ".join(f"{k}={v}" for k, v in d.get("shape", {}).items()) for d in case["shape_dist"]
        ),
        "references": refs_,
        "best": best,
        "rounds": _rounds(api, s["loop_id"], case_id),
        "buckets": buckets,
        "candidates": _candidates(api, case_id),
        "gaps": [{"id": g, "description": body(api, g)["description"]} for g in gaps],
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
        "refs": refs(api, sol),
    }
