"""Solve loop: pure transition function (docs/design/10 §2.1, machines.yaml ``solve``).

Facts each state's work reports (``kta.orchestrator.solve_work``):

- new: ``case``, ``problem``, ``contract``, ``baselines`` (record IDs)
- bounded: ``bounds`` {bucket: Bound ID}, ``buckets`` [bucket dicts]
- retrieving: ``families`` [Family IDs]; when empty, ``t1_inputs`` and ``t1_goal``
- solving: ``candidates`` {bucket: [Candidate IDs]}; ``empty_buckets``; ``family``
- building: ``top`` {bucket: [Candidate IDs]}; ``builds`` {Candidate ID: Build ID}
- checking: ``checks`` {Candidate ID: Check ID}; ``failed`` [Check IDs]
- measuring: ``measurements`` {Candidate ID: Measurement ID}; ``baseline_measurements``
- judging: ``buckets`` [per-bucket judgement inputs, see ``judge``]; ``references``
- waiting: ``tasks`` {task ID: lifecycle state}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kta.core.context import Context
from kta.orchestrator.transition import Draft, Loop, Settings, TaskRequest, Transition, hold

# Which state a waiting instance resumes at, by the type of the accepted task (D261).
RESUME_AT = {"T1": "retrieving", "T2": "retrieving", "T6": "retrieving", "T4": "solving"}
LINEAR = {
    "new": "bounded",
    "bounded": "retrieving",
    "checking": "measuring",
    "measuring": "judging",
}
DONE_TASK_STATES = frozenset({"accepted"})
STUCK_TASK_STATES = frozenset({"escalated", "cancelled", "suspended"})


def _wait(loop: Loop, rule: int | None, task: TaskRequest, note: str, **data: Any) -> Transition:
    return Transition(
        loop.state, "waiting", "creates_task", rule, data={**data}, tasks=(task,), note=note
    )


def step(loop: Loop, ctx: Context) -> Transition:
    s = Settings.from_config(ctx.config)
    f = loop.facts
    if loop.state in LINEAR:
        return Transition(loop.state, LINEAR[loop.state], "normal", data=dict(f))
    if loop.state == "retrieving":
        if f.get("families"):
            return Transition(loop.state, "solving", "normal", data={"families": f["families"]})
        task = TaskRequest("T1", str(f["t1_goal"]), list(f["t1_inputs"]))
        return _wait(loop, None, task, "no legal family in the pool")
    if loop.state == "solving":
        if f.get("empty_buckets"):
            goal = f"Extend a family so buckets {', '.join(f['empty_buckets'])} have candidates"
            task = TaskRequest("T1", goal, [loop.subject, *f.get("families_tried", [])])
            return _wait(loop, None, task, "no feasible candidate", candidates=f["candidates"])
        data = {"candidates": f["candidates"], "model": f.get("model")}
        return Transition(loop.state, "building", "normal", data=data)
    if loop.state == "building":
        if not any(f.get("top", {}).values()):
            goal = "Fix the family: no shortlisted candidate built"
            task = TaskRequest(
                "T6", goal, [f["family"], loop.subject, *f.get("builds", {}).values()]
            )
            return _wait(loop, None, task, "no candidate built")
        return Transition(
            loop.state, "checking", "normal", data={"top": f["top"], "builds": f["builds"]}
        )
    if loop.state == "judging":
        return judge(loop, s)
    if loop.state == "waiting":
        return resume(loop, s)
    return hold(loop, f"{loop.state} has no outgoing transition")


def resume(loop: Loop, s: Settings) -> Transition:
    """Resume when the awaited task is accepted; a new round starts (D261)."""
    states: Mapping[str, str] = loop.facts.get("tasks", {})
    waiting_on = loop.data.get("waiting_on")
    state = states.get(str(waiting_on), "")
    if state in STUCK_TASK_STATES:
        return hold(loop, f"task {waiting_on} is {state}", escalated=True)
    if state not in DONE_TASK_STATES:
        return hold(loop, f"waiting for task {waiting_on} ({state or 'unknown'})")
    rounds = int(loop.data.get("round", 1))
    if rounds >= s.max_rounds:
        return hold(loop, f"round limit {s.max_rounds} reached", escalated=True)
    task_type = str(loop.data.get("waiting_type", "T1"))
    to = RESUME_AT.get(task_type, "retrieving")
    data = {"round": rounds + 1, "waiting_on": None, "waiting_type": None, "escalated": False}
    return Transition(loop.state, to, "resume", data=data, note=f"{task_type} accepted")


def _spearman(pred: Sequence[float], meas: Sequence[float]) -> float:
    def ranks(xs: Sequence[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rp, rm = ranks(pred), ranks(meas)
    n = len(rp)
    mp, mm = sum(rp) / n, sum(rm) / n
    cov = sum((a - mp) * (b - mm) for a, b in zip(rp, rm, strict=True))
    vp = sum((a - mp) ** 2 for a in rp) ** 0.5
    vm = sum((b - mm) ** 2 for b in rm) ** 0.5
    return cov / (vp * vm) if vp > 0 and vm > 0 else 0.0


def bucket_metrics(b: Mapping[str, Any]) -> dict[str, Any]:
    """Ranking agreement, model error and explained gap share of one bucket.

    ``b`` has ``pairs`` [(predicted_us, measured_us)] for the measured top k, ``best_us``,
    ``predicted_us`` (of the best), ``bound_us`` and ``reference_us`` (None if no baseline).
    """
    pairs = [(float(p), float(m)) for p, m in b.get("pairs", [])]
    rho = _spearman([p for p, _ in pairs], [m for _, m in pairs]) if len(pairs) >= 2 else None
    errors = sorted(abs(p - m) / m for p, m in pairs if m > 0)
    median_err = errors[len(errors) // 2] if errors else None
    best, bound, pred = float(b["best_us"]), float(b["bound_us"]), float(b["predicted_us"])
    gap = best - bound
    explained = 1.0 if gap <= 0 else max(0.0, min(1.0, (pred - bound) / gap))
    return {
        "rho": rho,
        "n_ranked": len(pairs),
        "median_rel_error": median_err,
        "explained_share": explained,
        "pct_of_bound": 100.0 * bound / best if best > 0 else 0.0,
    }


def _model_off(m: Mapping[str, Any], s: Settings) -> bool:
    rho, err = m["rho"], m["median_rel_error"]
    bad_rank = rho is not None and m["n_ranked"] >= s.rank_n and rho < s.rank_min
    return bad_rank or (err is not None and err > s.rel_error_max)


def _met(b: Mapping[str, Any], m: Mapping[str, Any], s: Settings, sig: float) -> bool:
    ref = b.get("reference_us")
    beats = ref is not None and float(b["best_us"]) <= float(ref) * (1 + sig)
    return beats or m["explained_share"] >= s.explained_min


def judge(loop: Loop, s: Settings) -> Transition:
    """Rules 1-4 in order; every judgement writes a Report (kind judgement)."""
    f = loop.facts
    buckets: list[Mapping[str, Any]] = list(f.get("buckets", []))
    metrics = [bucket_metrics(b) for b in buckets]
    sig = float(f.get("significance_rel", 0.0))
    # Each failed check is a [Check ID, Candidate ID] pair; T6 reruns them (space.py).
    failed = [i for b in buckets for pair in b.get("failed_checks", []) for i in pair]
    rounds = int(loop.data.get("round", 1))
    if failed:
        rule, task = (
            1,
            TaskRequest(
                "T6",
                "Fix the family: correctness checks failed",
                [str(f["family"]), loop.subject, *failed],
            ),
        )
    elif any(_model_off(m, s) for m in metrics):
        ids = [i for b in buckets for i in b.get("evidence", [])]
        rule, task = (
            2,
            TaskRequest(
                "T4", "Diagnose the cost model: ranking or error off", [str(f["model"]), *ids]
            ),
        )
    elif buckets and all(_met(b, m, s, sig) for b, m in zip(buckets, metrics, strict=True)):
        rule, task = 3, None
    else:
        refs = list(f.get("references", []))
        task = (
            TaskRequest(
                "T2", "Decompose this reference to explain the gap", [refs[0], loop.subject]
            )
            if refs
            else TaskRequest(
                "T1",
                "Extend the family: the gap is not explained",
                [str(f["family"]), loop.subject],
            )
        )
        rule = 4
    units = [{"bucket": b["name"], **u} for b in buckets for u in b.get("units", [])]
    share = min((m["explained_share"] for m in metrics), default=0.0)
    data: dict[str, Any] = {
        "kind": "judgement",
        "round": rounds,
        "rule": rule,
        "units": units,
        "explained_share": share,
        "buckets": [{"name": b["name"], **m} for b, m in zip(buckets, metrics, strict=True)],
        "settings": {
            "rank_min": s.rank_min,
            "rel_error_max": s.rel_error_max,
            "explained_min": s.explained_min,
            "significance_rel": sig,
        },
    }
    records: list[Draft] = []
    suspicious = False
    if rule == 3:
        for b, m in zip(buckets, metrics, strict=True):
            body = {
                "bucket": b["name"],
                "assignment": b["assignment"],
                "pct_of_bound": m["pct_of_bound"],
            }
            refs = {
                "case": loop.subject,
                "family": str(f["family"]),
                "measurement": b["measurement"],
                "check": b["check"],
                "bound": b["bound"],
            }
            records.append(Draft("Solution", body, refs))
            ref = b.get("reference_us")
            suspicious |= m["pct_of_bound"] > s.suspicious_pct_of_bound or (
                ref is not None and float(ref) > s.suspicious_vs_reference * float(b["best_us"])
            )
    data["suspicious"] = suspicious
    summary = f"round {rounds}: rule {rule}" + (f", creates {task.type}" if task else ", accepted")
    report = Draft(
        "Report",
        {"title": "judgement", "subject": loop.subject, "summary": summary, "data": data},
        {"case": loop.subject},
        {"solution": tuple(range(len(records)))} if records else {},
        data_key="judgement",
    )
    records.append(report)
    upd = {"suspicious": suspicious}
    if task is None:
        return Transition(
            "judging", "accepted", "normal", 3, data=upd, records=tuple(records), note=summary
        )
    return Transition(
        "judging",
        "waiting",
        "creates_task",
        rule,
        data=upd,
        tasks=(task,),
        records=tuple(records),
        note=summary,
    )
