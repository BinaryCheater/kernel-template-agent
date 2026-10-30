"""Views of what is running: /runs/active, /inbox, /budget (docs/design/30 §1 "运行中").

Everything is derived from the ops tables, open spans and TaskCard records; the record
conventions read here are listed in spec/modules/api-web.md ("Record conventions").
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from kta.api import opsread
from kta.api.common import Api, body, config_int, done_check_reports, iso, machine_limit
from kta.orchestrator import quotas
from kta.store.base import SpanRow

RECENT = timedelta(hours=2)
JOB_KINDS = {"compile", "check", "measure", "profile", "solve", "job_run", "job_queue"}
# Task states shown as finished in "just finished"; ops rows hold lifecycle names (D265).
FINISHED = ("accepted", "escalated", "suspended", "cancelled")
TERMINAL_TASK = ("accepted", "cancelled")


def _last_rejection(api: Api, task_id: str) -> str | None:
    raw = api.ops.get_pointer(f"task:{task_id}")
    rid = json.loads(raw).get("last_rejection") if raw else None
    if not rid:
        return None
    return next((r["reason"] for r in done_check_reports(api, task_id) if r["id"] == rid), None)


def task_view(api: Api, t: opsread.TaskRow, sessions: dict[str, SpanRow]) -> dict[str, Any]:
    """An ops task row; its state is the lifecycle state the orchestrator stored (D265)."""
    card = body(api, t.card_id)
    sess = sessions.get(t.id)
    now = api.now()
    started = sess.start_ts if sess else (t.updated_at if t.state == "running" else None)
    return {
        "id": t.id,
        "card": t.card_id,
        "type": t.card_type,
        "goal": card.get("goal", ""),
        "state": t.state,
        "slot": t.lease_owner if t.state in ("leased", "running") else None,
        "loop_id": t.work_unit,
        "created_at": iso(t.created_at),
        "state_since": iso(t.updated_at),
        "retries": max(0, t.attempts - 1),
        "max_retries": machine_limit("task", "max_retries", 2),
        "budget_wall_s": float(card.get("budget", {}).get("wall_s", 0)),
        "used_wall_s": (now - started).total_seconds() if started else 0.0,
        "queue_s": ((started or now) - t.created_at).total_seconds(),
        "priority": t.priority,
        "last_rejection": _last_rejection(api, t.id) if t.state != "accepted" else None,
    }


def loop_view(api: Api, row: opsread.LoopRow) -> dict[str, Any]:
    d = row.data
    return {
        "id": row.id,
        "machine": row.kind,
        "subject": row.subject,
        "subject_name": d.get("name") or _subject_name(api, row.subject),
        "state": row.state,
        "round": d.get("round"),
        "max_rounds": machine_limit(row.kind, "max_rounds", 0) or None,
        "started_at": d.get("started_at") or iso(row.updated_at),
        "state_since": d.get("state_since") or iso(row.updated_at),
        "waiting_on": d.get("waiting_on"),
        "escalated": bool(d.get("escalated")),
        "trace_id": row.id,
    }


def _subject_name(api: Api, subject: str) -> str:
    b = body(api, subject)
    return str(b.get("name") or b.get("problem") or subject[:12])


def _sessions(spans: list[SpanRow]) -> dict[str, SpanRow]:
    return {
        str(s.attrs["task_id"]): s
        for s in spans
        if s.kind == "agent_session" and "task_id" in s.attrs
    }


def _pool(lane: str) -> str:
    if lane.startswith("gpu:"):
        return "gpu-exclusive"
    parts = lane.split(":")
    return parts[2] if len(parts) == 3 else lane


def job_view(s: SpanRow) -> dict[str, Any]:
    """A job from its open queue or run span (conventions §4.1: queue and run are two spans)."""
    queued = s.kind == "job_queue"
    host = s.lane.split(":")[1] if ":" in s.lane else s.lane
    return {
        "id": str(s.attrs.get("job_id", s.span_id)),
        "kind": str(s.attrs.get("job_kind", s.kind)),
        "name": s.name,
        "pool": str(s.attrs.get("pool", _pool(s.lane))),
        "worker": str(s.attrs.get("worker", host)),
        "state": "queued" if queued else "running",
        "queued_at": str(s.attrs.get("queued_at", iso(s.start_ts))),
        "started_at": None if queued else iso(s.start_ts),
        "task_id": s.attrs.get("task_id"),
        "loop_id": s.trace_id,
    }


def _workers(api: Api, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in opsread.workers(api.ops):
        rid = _record_id(api, w.name)
        rec = body(api, rid) if rid else {}
        labels: dict[str, str] = dict(rec.get("labels", {}))
        pools = []
        for item in filter(None, labels.pop("pools", "").split(",")):
            name, _, cap = item.partition(":")
            mine = [j for j in jobs if j["pool"] == name and j["worker"] in (w.name, w.host)]
            pools.append(
                {
                    "pool": name,
                    "capacity": int(cap or 1),
                    "running": sum(j["state"] == "running" for j in mine),
                    "queued": sum(j["state"] == "queued" for j in mine),
                }
            )
        out.append(
            {
                "name": w.name,
                "host": w.host,
                "transport": rec.get("transport"),
                "labels": labels,
                "commit": rec.get("commit"),
                "last_seen": iso(w.heartbeat),
                "health": "ok" if w.healthy else "down",
                "record": rid,
                "pools": pools,
            }
        )
    return out


def _record_id(api: Api, worker: str) -> str | None:
    with api.store.db.read() as con:
        row = con.execute(
            "SELECT id FROM records WHERE kind = 'Worker' "
            "AND json_extract(json, '$.body.name') = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (worker,),
        ).fetchone()
    return str(row[0]) if row else None


def _leases(api: Api, open_spans: list[SpanRow]) -> list[dict[str, Any]]:
    out = []
    for lock in opsread.locks(api.ops):
        if not lock.name.startswith("gpu:") or lock.until < api.now():
            continue
        hold = next(
            (s for s in open_spans if s.kind == "lock_hold" and s.attrs.get("lock") == lock.name),
            None,
        )
        out.append(
            {
                "gpu": lock.name[4:],
                "holder_job": lock.holder,
                "since": iso(hold.start_ts) if hold else None,
                "max_s": 600,
                "until": iso(lock.until),
            }
        )
    return out


def budgets(api: Api) -> list[dict[str, Any]]:
    """GET /budget: one tree per resource key of the global scope (11 §3.1)."""
    rows = {b.scope: b for b in opsread.budgets(api.ops)}
    root = rows.get("global")
    if root is None:
        return []
    trees = []
    for key, limit in root.limit.items():
        children = [
            {
                "key": s,
                "label": s.split(":", 1)[1],
                "unit": key.rpartition("_")[2],
                "limit": b.limit.get(key, 0),
                "used": b.used.get(key, 0),
                "reserved": b.reserved.get(key, 0),
            }
            for s, b in rows.items()
            if s.startswith("process:") and key in b.limit
        ]
        name, _, unit = key.rpartition("_")
        trees.append(
            {
                "key": f"global:{key}",
                "label": f"{name or key} time today",
                "unit": unit or key,
                "limit": limit,
                "used": root.used.get(key, 0),
                "reserved": root.reserved.get(key, 0),
                "children": children,
            }
        )
    return trees


# Inbox trigger -> the operations offered next to it (names of /actions/capabilities).
INBOX_ACTIONS = {
    "Task over retry limit": ["retry", "cancel"],
    "Checker crashed": ["cancel"],
    "improvement_pending": ["approve-improvement", "reject-improvement"],
}


def inbox(api: Api) -> list[dict[str, Any]]:
    """WP6's pending items (kta.orchestrator.inbox, D270), with the task and controls."""
    rows = {t.id: t for t in opsread.tasks(api.ops)}
    out: list[dict[str, Any]] = []
    for item in api.orchestration.inbox():
        kind, _, key = str(item["id"]).partition(":")
        task = rows.get(key) if kind == "task" else None
        out.append(
            {
                **item,
                "task": task.id if task else None,
                "at": iso(task.updated_at) if task else None,
                "detail": _last_rejection(api, task.id) if task else None,
                "actions": INBOX_ACTIONS.get(str(item["trigger"]), []),
            }
        )
    return out


def active(api: Api) -> dict[str, Any]:
    now = api.now()
    spans = opsread.open_spans(api.ops)
    sessions = _sessions(spans)
    tasks = [task_view(api, t, sessions) for t in opsread.tasks(api.ops)]
    live = [t for t in tasks if t["state"] not in TERMINAL_TASK]
    jobs = [
        job_view(s)
        for s in spans
        if s.kind in JOB_KINDS and s.lane.startswith(("worker:", "gpu:")) and s.parent_id is None
    ]
    loops = [loop_view(api, r) for r in opsread.loops(api.ops)]
    finished = _finished(api, tasks, loops, now)
    return {
        "as_of": iso(now),
        "agent_slots": [f"agent:{i + 1}" for i in range(config_int(api, "agents", "slots"))],
        "loops": loops,
        "tasks": live,
        "jobs": jobs,
        "leases": _leases(api, spans),
        "workers": _workers(api, jobs),
        "budgets": budgets(api),
        "inbox_count": len(api.orchestration.inbox()),
        "recently_finished": finished,
    }


def _finished(
    api: Api, tasks: list[dict[str, Any]], loops: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    out = [
        {
            "id": t["id"],
            "label": f"{t['type']} {t['goal']} → {t['state']}",
            "status": "ok" if t["state"] == "accepted" else "fail",
            "ended_at": t["state_since"],
            "link": "/pipeline",
        }
        for t in tasks
        if t["state"] in FINISHED
    ]
    terminal = {"accepted", "covered"}
    out += [
        {
            "id": lp["id"],
            "label": f"{lp['machine']} {lp['subject_name']} → {lp['state']}",
            "status": "ok",
            "ended_at": lp["state_since"],
            "link": f"/cases/{lp['subject']}" if lp["machine"] == "solve" else "/coverage",
        }
        for lp in loops
        if lp["state"] in terminal
    ]
    cutoff = iso(now - RECENT) or ""
    return sorted(
        (x for x in out if (x["ended_at"] or "") >= cutoff),
        key=lambda x: x["ended_at"],
        reverse=True,
    )


def quota_view(api: Api) -> dict[str, Any]:
    """GET /quotas: process shares configured and in force (WP11 D401, as ``kta quotas``)."""
    base = quotas.shares(api.config)
    active = quotas.active_experiments(api.ops)
    return {
        "configured": base,
        "in_force": quotas.effective(base, bool(active)),
        "experiments": active,
    }
