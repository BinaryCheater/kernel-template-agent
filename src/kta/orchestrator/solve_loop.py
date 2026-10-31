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
from kta.orchestrator import disciplines, gap
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
    s = Settings.of(ctx.controls)
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
        data = {
            "candidates": f["candidates"],
            "model": f.get("model"),
            "solver": f.get("solver", {}),
            "arms": f.get("arms", {}),
        }
        return Transition(loop.state, "building", "normal", data=data)
    if loop.state == "building":
        if not any(f.get("top", {}).values()):
            goal = "Fix the family: no shortlisted candidate built"
            task = TaskRequest(
                "T6", goal, [f["family"], loop.subject, *f.get("builds", {}).values()]
            )
            return _wait(loop, None, task, "no candidate built")
        return Transition(
            loop.state,
            "checking",
            "normal",
            data={"top": f["top"], "builds": f["builds"], "arm_of": f.get("arm_of", {})},
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
        flag = bool(s.to_user["on_round_limit"])
        return hold(loop, f"round limit {s.max_rounds} reached", escalated=flag)
    task_type = str(loop.data.get("waiting_type", "T1"))
    to = RESUME_AT.get(task_type, "retrieving")
    data = {"round": rounds + 1, "waiting_on": None, "waiting_type": None, "escalated": False}
    if task_type == "T4":  # escalation.case_to_knowledge counts calibrations per case
        data["calibrations"] = int(loop.data.get("calibrations", 0)) + 1
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


def model_off(m: Mapping[str, Any], s: Settings) -> bool:
    rho, err = m["rho"], m["median_rel_error"]
    bad_rank = rho is not None and m["n_ranked"] >= s.rank_n and rho < s.rank_min
    return bad_rank or (err is not None and err > s.rel_error_max)


def _task(loop: Loop, f: Mapping[str, Any], cls: str, route: Mapping[str, Any]) -> TaskRequest:
    """The task of a waiting route; inputs are fixed per class (the first is the subject)."""
    buckets: list[Mapping[str, Any]] = list(f.get("buckets", []))
    refs = list(f.get("references", []))
    task_type, goal = gap.task_for(route, bool(refs))
    if cls == "correctness":
        failed = [i for b in buckets for pair in b.get("failed_checks", []) for i in pair]
        inputs = [str(f["family"]), loop.subject, *failed]  # T6 reruns them (space.py)
    elif cls == "epsilon":
        inputs = [str(f["model"]), *[i for b in buckets for i in b.get("evidence", [])]]
    elif task_type == "T2":
        inputs = [refs[0], loop.subject]
    else:
        inputs = [str(f["family"]), loop.subject]
    return TaskRequest(task_type, goal, inputs)


def _decide(loop: Loop, s: Settings, decs: list[dict[str, Any]], d: Mapping[str, Any]) -> str:
    """The route name: correctness; accept (unless the model is off); the null-round
    limit; model error after calibration; model off; else the decomposition's class."""
    cls = decs[0]["case_class"] if decs else "correctness"
    if cls == "correctness":
        return cls
    off = any(x["model_off"] for x in decs)
    if not off and decs and all(x["accepted"] for x in decs) and not d.get("rolled_back"):
        return "accept"
    if d.get("null_rounds", 0) >= s.null_round_limit:
        return "null_rounds"
    k2k = s.case_to_knowledge
    calibrated = int(loop.data.get("calibrations", 0)) >= int(k2k["calibrations_first"])
    err = max((x["median_rel_error"] or 0.0 for x in decs), default=0.0)
    if (off or cls == "epsilon") and calibrated and err > float(k2k["topk_model_error_rel"]):
        return "model_after_calibration"
    return "epsilon" if off else cls


def _decomposition(loop: Loop, s: Settings) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-bucket decompositions and the discipline verdict of this round."""
    f = loop.facts
    table = gap.routing()
    buckets: list[Mapping[str, Any]] = list(f.get("buckets", []))
    decs: list[dict[str, Any]] = []
    for b in buckets:
        m = bucket_metrics(b) if b.get("pairs") or b.get("best_us") else {}
        failed = bool(b.get("failed_checks"))
        if failed and not b.get("pairs"):
            decs.append({"name": b["name"], "cls": "correctness", "model_off": False,
                         "accepted": False, "median_rel_error": None, "terms": {}})  # fmt: skip
            continue
        dec = gap.decompose(b, s.improvement_band)
        off = model_off(m, s)
        dec.update(
            name=b["name"],
            cls=gap.classify(dec, table.order, failed, off),
            model_off=off,
            accepted=not failed and gap.accepts(dec, b.get("reference_us"), s.accept),
            median_rel_error=m.get("median_rel_error"),
            rho=m.get("rho"),
            reference_us=b.get("reference_us"),
            selection=gap.selection(b.get("arms") or {}, s.exploration_k),
        )
        decs.append(dec)
    cls = gap.case_class([x["cls"] for x in decs], (*table.order, "none"))
    for x in decs:
        x["case_class"] = cls
    weights = [float(b.get("weight", 1.0)) for b in buckets]
    w = sum(weights) or 1.0
    t = sum(wt * float(x.get("t_meas_us", 0.0)) for wt, x in zip(weights, decs, strict=True)) / w
    band = sum(wt * float(x.get("band_us", 0.0)) for wt, x in zip(weights, decs, strict=True)) / w
    now = dict(f.get("basis") or {})
    names, count = disciplines.mechanisms(loop.data.get("basis"), now, f.get("couplings", ()))
    verdict = disciplines.credit(loop.data, t, band, names, count, str(f.get("family", "")), s)
    verdict.update(t_us=t, band_us=band, basis=now)
    return decs, verdict


def judge(loop: Loop, s: Settings) -> Transition:
    """Decompose the gap, apply the disciplines and route (routing.yaml); every judgement
    writes a GapDecomposition and a Report (kind judgement)."""
    f = loop.facts
    table = gap.routing()
    buckets: list[Mapping[str, Any]] = list(f.get("buckets", []))
    metrics = [bucket_metrics(b) for b in buckets]
    sig = float(f.get("significance_rel", 0.0))
    rounds = int(loop.data.get("round", 1))
    decs, verdict = _decomposition(loop, s)
    name = _decide(loop, s, decs, verdict)
    if name == "accept":
        route: Mapping[str, Any] = {"rule": 3, "to": "accepted", "escalate": "none"}
    else:
        route = table.escalations.get(name) or table.classes[name]
    rule = int(route["rule"])
    task = _task(loop, f, name, route) if route["to"] == "waiting" else None
    units = [{"bucket": b["name"], **u} for b in buckets for u in b.get("units", [])]
    share = min((m["explained_share"] for m in metrics), default=0.0)
    data: dict[str, Any] = {
        "kind": "judgement",
        "round": rounds,
        "rule": rule,
        "gap_class": name,
        "units": units,
        "explained_share": share,
        "buckets": [{"name": b["name"], **m} for b, m in zip(buckets, metrics, strict=True)],
        "settings": {
            "rank_min": s.rank_min,
            "rel_error_max": s.rel_error_max,
            "explained_min": s.explained_min,
            "significance_rel": sig,
            "accept": dict(s.accept),
            "improvement_band": dict(s.improvement_band),
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
    outcome = f"creates {task.type}" if task else str(route["to"])
    summary = f"round {rounds}: rule {rule} ({name}), {outcome}"
    links: dict[str, tuple[int, ...]] = {"solution": tuple(range(len(records)))} if records else {}
    records.append(_gap_draft(loop, table, name, route, decs, verdict, task))
    links["decomposition"] = (len(records) - 1,)
    report = Draft(
        "Report",
        {"title": "judgement", "subject": loop.subject, "summary": summary, "data": data},
        {"case": loop.subject},
        links,
        data_key="judgement",
    )
    records.append(report)
    upd = _carry(loop, verdict, suspicious, route)
    if task is None:
        to = "accepted" if rule == 3 else "closed"
        return Transition(
            "judging", to, "normal", rule, data=upd, records=tuple(records), note=summary
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


def _carry(
    loop: Loop, verdict: Mapping[str, Any], suspicious: bool, route: Mapping[str, Any]
) -> dict[str, Any]:
    """Loop data the next round's disciplines read (D525-D527)."""
    rolled = list(loop.data.get("rolled_back_mechanisms", []))
    if verdict.get("rolled_back"):
        rolled += list(verdict["mechanisms"])
    out: dict[str, Any] = {
        "suspicious": suspicious,
        "basis": verdict["basis"],
        "incumbent": verdict["incumbent"],
        "null_rounds": verdict["null_rounds"],
        "rolled_back_mechanisms": rolled,
        # A rolled-back round's family is not preferred next round (work_retrieving).
        "prefer_family": verdict["incumbent"].get("family"),
    }
    if route["escalate"] != "none":
        out["escalated"] = route["escalate"]
    return out


def _gap_draft(
    loop: Loop,
    table: gap.Routing,
    name: str,
    route: Mapping[str, Any],
    decs: list[dict[str, Any]],
    verdict: Mapping[str, Any],
    task: TaskRequest | None,
) -> Draft:
    f = loop.facts
    body = {
        "round": int(loop.data.get("round", 1)),
        "gap_class": decs[0]["case_class"] if decs else "correctness",
        "route": {
            "name": name,
            "rule": int(route["rule"]),
            "to": str(route["to"]),
            "task": task.type if task else None,
            "escalate": str(route["escalate"]),
            "improve": str(route.get("improve", "none")),
        },
        "routing": table.id,
        "buckets": [_bucket_body(x, b) for x, b in zip(decs, f.get("buckets", []), strict=True)],
        "discipline": {k: v for k, v in verdict.items() if k not in ("basis", "incumbent")},
    }
    refs: dict[str, str | list[str]] = {"case": loop.subject}
    if f.get("family"):
        refs["family"] = str(f["family"])
    return Draft("GapDecomposition", body, refs, data_key="decomposition")


INPUT_KEYS = ("best_us", "ci_us", "predicted_us", "top_pred_us", "bound_us", "reference_us",
              "pairs", "solver", "relaxed_pred_us", "delta_pre_note", "irreducible_us",
              "unpriced")  # fmt: skip


def _bucket_body(x: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """One bucket of a GapDecomposition: the inputs (canaries replay them), terms, class."""
    inputs = {k: b[k] for k in INPUT_KEYS if b.get(k) is not None}
    inputs["pairs"] = [list(p) for p in inputs.get("pairs", [])]
    inputs["failed_checks"] = len(b.get("failed_checks", []))
    keep = ("t_meas_us", "t_sol_us", "gap_us", "band_us", "l_rep_bound_us", "terms",
            "model_off", "accepted", "rho", "median_rel_error", "selection")  # fmt: skip
    return {"name": str(x["name"]), "class": x["cls"], "inputs": inputs,
            **{k: x[k] for k in keep if k in x}}  # fmt: skip
