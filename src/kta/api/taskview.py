"""GET /tasks/{id} (design/30 task detail) and GET /blobs/{sha256}.

A task is the ops row plus its card; what the agent did is read from the records written
in the task (``records.task``: AgentRun, Submission, Check, Report, knowledge objects),
the spans tagged with ``task_id`` and their descendants (session, model and tool segments,
``kta`` calls, jobs), and the HumanActions on its card. Nothing is inferred from logs.
"""

from __future__ import annotations

from typing import Any

from kta.api import opsread, runs
from kta.api.common import Api, body, iso
from kta.core.errors import NotSupported
from kta.store.base import SpanRow
from kta.store.sqlite import RecordNotFound

BLOB_TEXT_MAX_BYTES = 256 * 1024
# Span kinds shown as the agent's calls (the kta commands and parsed tool calls).
CALL_KINDS = ("tool_call", "kta", "compile", "check", "measure", "profile", "job_run", "job_queue")


def _records_of_task(api: Api, task_id: str) -> list[dict[str, Any]]:
    with api.store.db.read() as con:
        rows = con.execute(
            "SELECT id, kind FROM records WHERE task = ? ORDER BY created_at, rowid", (task_id,)
        ).fetchall()
    out: list[dict[str, Any]] = []
    for rid, kind in rows:
        rec = api.store.get_json(rid)
        out.append(
            {
                "id": rid,
                "kind": kind,
                "created_at": rec["created_at"],
                "actor": rec["actor"]["name"],
                "status": api.store.status(rid)[0],
                "origin": rec["origin"],
                "body": rec["body"],
                "blobs": rec.get("blobs", {}),
                "refs": rec.get("refs", {}),
            }
        )
    return out


def _span(s: SpanRow) -> dict[str, Any]:
    dur = (s.end_ts - s.start_ts).total_seconds() if s.end_ts else None
    return {
        "span_id": s.span_id,
        "parent_id": s.parent_id,
        "trace_id": s.trace_id,
        "kind": s.kind,
        "lane": s.lane,
        "name": s.name,
        "start": iso(s.start_ts),
        "end": iso(s.end_ts),
        "dur_s": dur,
        "status": s.status,
        "attrs": s.attrs,
    }


def _time_split(spans: list[SpanRow], runs_: list[dict[str, Any]]) -> dict[str, Any]:
    """Time composition: from AgentRun records when present, else from closed spans."""
    if runs_:
        return {
            "wall_s": sum(float(r["body"]["wall_s"]) for r in runs_),
            "tool_s": sum(float(r["body"]["tool_s"]) for r in runs_),
            "queue_s": sum(float(r["body"]["queue_s"]) for r in runs_),
            "source": "AgentRun",
        }
    by_kind: dict[str, float] = {}
    for s in spans:
        if s.end_ts is not None:
            by_kind[s.kind] = by_kind.get(s.kind, 0.0) + (s.end_ts - s.start_ts).total_seconds()
    return {
        "wall_s": by_kind.get("agent_session", 0.0),
        "tool_s": sum(v for k, v in by_kind.items() if k in CALL_KINDS),
        "queue_s": by_kind.get("job_queue", 0.0),
        "source": "spans",
    }


def _human_actions(api: Api, task_id: str, card: str) -> list[dict[str, Any]]:
    ids = set(api.store.neighbors(card, "card", direction="in"))
    with api.store.db.read() as con:
        rows = con.execute(
            "SELECT id FROM records WHERE task = ? AND kind = 'HumanAction'", (task_id,)
        ).fetchall()
    ids |= {r[0] for r in rows}
    out = []
    for rid in ids:
        if api.store.kind_of(rid) != "HumanAction":
            continue
        rec = api.store.get_json(rid)
        out.append({"id": rid, "at": rec["created_at"], "by": rec["actor"]["name"], **rec["body"]})
    return sorted(out, key=lambda x: x["at"])


def task_detail(api: Api, task_id: str) -> dict[str, Any]:
    row = next((t for t in opsread.tasks(api.ops) if t.id == task_id), None)
    if row is None:
        raise RecordNotFound(f"no task {task_id}")
    spans = opsread.spans_of_task(api.ops, task_id)
    sessions = {
        str(s.attrs["task_id"]): s
        for s in spans
        if s.kind == "agent_session" and s.end_ts is None and "task_id" in s.attrs
    }
    view = runs.task_view(api, row, sessions)
    loop = next((lp for lp in opsread.loops(api.ops) if lp.id == row.work_unit), None)
    records = _records_of_task(api, task_id)
    agent_runs = [r for r in records if r["kind"] == "AgentRun"]
    checks = [r for r in records if r["kind"] == "Check"]
    return {
        "task": view,
        "card": {"id": row.card_id, **body(api, row.card_id)},
        "loop": runs.loop_view(api, loop) if loop else None,
        "spans": [_span(s) for s in spans],
        "calls": [_span(s) for s in spans if s.kind in CALL_KINDS],
        "records": records,
        "agent_runs": agent_runs,
        "submissions": [r for r in records if r["kind"] == "Submission"],
        "checks": checks,
        "rejections": [c for c in checks if not c["body"].get("passed", True)],
        "time": _time_split(spans, agent_runs),
        "human_actions": _human_actions(api, task_id, row.card_id),
    }


def blob_text(api: Api, sha256: str) -> dict[str, Any]:
    """A stored blob as text, for prompts, notes and logs; binary blobs are refused."""
    path = api.store.blob_path(sha256)
    media = api.store.blob_media(sha256)
    size = path.stat().st_size
    with path.open("rb") as f:
        data = f.read(BLOB_TEXT_MAX_BYTES)
    if b"\0" in data[:8192] or media in ("application/gzip", "application/x-tar"):
        raise NotSupported(f"blob {sha256[:12]} is binary ({media}, {size} bytes)")
    return {
        "sha256": sha256,
        "media": media,
        "size": size,
        "truncated": size > BLOB_TEXT_MAX_BYTES,
        "text": data.decode("utf-8", errors="replace"),
    }
