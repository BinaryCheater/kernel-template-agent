"""The improvement loop (docs/design/10 §2.3), run by the summarize process.

``summarize(runner, batch, since, until)`` counts, for one batch window:

- every Gap created in the window, by ``attribution`` (the six classes of 10 §2.3 plus
  ``other``, an open table: unknown values are counted under their own name);
- every judgement Report in the window, by the class its rule implies (D406): rule 1
  (a check failed) -> ``verification``, rule 2 (model off) -> ``model``, rule 4 (gap not
  explained) -> ``space``; rule 3 (accepted) counts nothing. The loss of a judgement is
  its unexplained gap share ``1 - explained_share``.

It writes one Report (``title = improvement``, D345: ``data.batch``,
``data.attribution``, ``data.items``) and acts on the items:

- knowledge items: each ``space``/``granularity`` Gap that names a missing Template or
  Pattern gets one T3 (booked to ``learn``); ``model`` items are left to the calibrate
  process, which turns model error into T4s (and is named in the item);
- system items (``intent_sass``, ``verification``, ``process``, ``other``) wait for you:
  the inbox lists each until a HumanAction ``approve_improvement`` (or ``reject_improvement``)
  names the Report in ``refs.improvement`` (10 §2.3 step 4).

Step 5 (checking an improvement on the development cases) is not done here: it needs
real runs and a frozen development set (WP11 known issue).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any

from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.orchestrator import canary
from kta.orchestrator.pipeline import Services
from kta.orchestrator.transition import TaskRequest

if TYPE_CHECKING:  # the runner imports the inbox, which imports this module
    from kta.orchestrator.runner import Runner

ACTOR = Actor(type="system", name="kta.orchestrator.improve")
TITLE = "improvement"
CLASSES = ("space", "model", "intent_sass", "verification", "process", "granularity")
KNOWLEDGE = ("space", "granularity", "model")
RULE_CLASS = {1: "verification", 2: "model", 4: "space"}
T3_MISSING = ("template", "pattern")
DECIDED = ("approve_improvement", "reject_improvement")


def _in(rec: Record, since: datetime | None, until: datetime) -> bool:
    return (since is None or rec.created_at > since) and rec.created_at <= until


def collect(svc: Services, since: datetime | None, until: datetime) -> dict[str, Any]:
    gaps = [
        g
        for g in svc.store.query("Gap", None, order="created_at", limit=10000)
        if _in(g, since, until)
    ]
    judgements = [
        j
        for j in svc.store.query("Report", {"title": "judgement"}, order="created_at", limit=10000)
        if _in(j, since, until)
    ]
    decs = [
        d
        for d in svc.store.query("GapDecomposition", None, order="created_at", limit=10000)
        if _in(d, since, until)
    ]
    count: Counter[str] = Counter({c: 0 for c in CLASSES})
    loss: dict[str, float] = {}
    cases: dict[str, set[str]] = {}
    for g in gaps:
        count[str(g.body["attribution"])] += 1
    for d in decs:  # WP18: the round's routed class (routing.yaml `improve`)
        cls = str(d.body["route"].get("improve", "none"))
        if cls == "none":
            continue
        count[cls] += 1
        loss[cls] = loss.get(cls, 0.0) + _loss(d.body)
        cases.setdefault(cls, set()).add(str(d.refs.get("case")))
        if d.body["route"].get("escalate") == "knowledge":
            cases.setdefault("escalated:" + cls, set()).add(str(d.refs.get("case")))
    for j in judgements:
        if "gap_class" in j.body["data"]:  # counted through its GapDecomposition
            continue
        cls = RULE_CLASS.get(int(j.body["data"].get("rule", 0)))
        if cls is None:
            continue
        count[cls] += 1
        loss[cls] = loss.get(cls, 0.0) + 1.0 - float(j.body["data"].get("explained_share", 0.0))
    return {
        "gaps": gaps,
        "judgements": judgements,
        "decompositions": decs,
        "count": dict(count),
        "loss": loss,
        "cases": {k: sorted(v) for k, v in cases.items()},
    }


def _loss(body: dict[str, Any]) -> float:
    """Share of T the routed class's term takes, averaged over buckets (1 for checks)."""
    name = str(body["route"].get("name"))
    shares = [
        abs(float(b["terms"][name]["us"] or 0.0)) / float(b["t_meas_us"])
        for b in body["buckets"]
        if name in b.get("terms", {}) and float(b.get("t_meas_us") or 0.0) > 0
    ]
    return sum(shares) / len(shares) if shares else 1.0


def _items(runner: Runner, batch: str, found: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    same_cases = runner.settings.case_to_knowledge["same_residual_cases"]
    for cls, n in sorted(found["count"].items()):
        if not n:
            continue
        kind = "knowledge" if cls in KNOWLEDGE else "system"
        gaps = [record_id(g) for g in found["gaps"] if g.body["attribution"] == cls]
        item: dict[str, Any] = {
            "key": f"{batch}:{cls}",
            "title": f"{n} x {cls}",
            "kind": kind,
            "class": cls,
            "gaps": gaps,
            "loss": found["loss"].get(cls, 0.0),
            "tasks": [],
        }
        many = len(found["cases"].get(cls, ())) >= int(same_cases)
        if many or found["cases"].get("escalated:" + cls):
            # 12 §4 escalate case -> knowledge: the knowledge loop owns this class now.
            item["escalate"] = "knowledge"
            item["cases"] = found["cases"].get(cls, [])
        if cls == "model":
            item["description"] = "model error: the calibrate process creates T4 diagnoses"
        elif kind == "knowledge":
            item["description"] = (
                "missing knowledge: T3 extraction per Gap naming a Template/Pattern"
            )
            item["tasks"] = _t3s(runner, batch, found, cls)
        else:
            item["description"] = (
                f"system-level ({cls}): needs your approval, then a construction task"
            )
        items.append(item)
    return items


def _t3s(runner: Runner, batch: str, found: dict[str, Any], cls: str) -> list[str]:
    out: list[str] = []
    for g in found["gaps"]:
        if g.body["attribution"] != cls or g.body["missing"] not in T3_MISSING:
            continue
        gid = record_id(g)
        goal = f"Extract the missing {g.body['missing']}: {g.body['description']}"[:300]
        out.append(runner.queue.create(TaskRequest("T3", goal, [gid]), f"improve:{batch}", "learn"))
    return out


def summarize(runner: Runner, batch: str, since: datetime | None, until: datetime) -> str:
    found = collect(runner.svc, since, until)
    items = _items(runner, batch, found)
    canaries = canary.run(runner.svc, batch, found["decompositions"], runner.settings, until)
    body = Report(
        title=TITLE,
        subject=batch,
        summary=(
            f"batch {batch}: {len(found['gaps'])} Gaps, {len(found['judgements'])} judgements; "
            f"{sum(1 for i in items if i['kind'] == 'system')} system items wait for approval"
        ),
        data={
            "batch": batch,
            "since": since.isoformat() if since else None,
            "until": until.isoformat(),
            "attribution": found["count"],
            "loss": found["loss"],
            "items": items,
            "rule_classes": {str(k): v for k, v in RULE_CLASS.items()},
        },
    )
    refs: dict[str, Any] = {}
    if found["gaps"]:
        refs["gaps"] = [record_id(g) for g in found["gaps"]]
    if found["judgements"]:
        refs["judgements"] = [record_id(j) for j in found["judgements"]]
    if found["decompositions"]:
        refs["decompositions"] = [record_id(d) for d in found["decompositions"]]
    if canaries:
        refs["canaries"] = canaries
    tasks = [t for i in items for t in i["tasks"]]
    if tasks:
        body.data["tasks"] = tasks
    return runner.svc.store.put(Record.new("Report", body, ACTOR, until, refs))


def pending_items(svc: Services) -> list[tuple[str, dict[str, Any]]]:
    """(improvement Report ID, item) of system items in Reports you have not decided."""
    decided = {
        str(h.refs.get("improvement"))
        for h in svc.store.query("HumanAction", None, order="created_at", limit=10000)
        if str(h.body.get("action", "")).startswith(DECIDED)
    }
    out: list[tuple[str, dict[str, Any]]] = []
    for rec in svc.store.query("Report", {"title": TITLE}, order="created_at", limit=1000):
        rid = record_id(rec)
        if rid in decided:
            continue
        out += [(rid, i) for i in rec.body["data"].get("items", []) if i.get("kind") == "system"]
    return out
