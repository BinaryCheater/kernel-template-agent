"""Pipeline views over spans: /trace, /trace/summary, /trace/critical-path (docs/design/30 §1.2)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from kta.api import opsread
from kta.api.common import Api, iso
from kta.store.base import SpanRow

# Waiting spans are drawn light and do not count as busy time (conventions §4.1).
WAIT_KINDS = {"job_queue", "lock_wait", "human_wait"}


def span_json(s: SpanRow) -> dict[str, Any]:
    return {
        "span_id": s.span_id,
        "parent_id": s.parent_id,
        "trace_id": s.trace_id,
        "kind": s.kind,
        "lane": s.lane,
        "name": s.name,
        "start_ts": iso(s.start_ts),
        "end_ts": iso(s.end_ts),
        "status": s.status,
        "attrs": s.attrs,
    }


def window(
    api: Api,
    start: datetime,
    end: datetime,
    lanes: list[str] | None = None,
    trace: str | None = None,
    kinds: list[str] | None = None,
) -> dict[str, Any]:
    spans = opsread.spans_overlapping(api.ops, start, end, trace)
    if lanes:
        spans = [s for s in spans if any(s.lane == x or s.lane.startswith(f"{x}:") for x in lanes)]
    if kinds:
        spans = [s for s in spans if s.kind in kinds]
    return {
        "from": iso(start),
        "to": iso(end),
        "spans": [span_json(s) for s in spans],
        "traces": traces(api, spans),
    }


def traces(api: Api, spans: list[SpanRow]) -> list[dict[str, Any]]:
    """Label each trace: a loop instance (case or reference) or a bare trace id."""
    loops = {lp.id: lp for lp in opsread.loops(api.ops)}
    out = []
    for tid in sorted({s.trace_id for s in spans}):
        lp = loops.get(tid)
        if lp is not None:
            kind = "Case" if lp.kind == "solve" else "Reference"
            out.append(
                {
                    "id": tid,
                    "label": f"{kind} {lp.data.get('name', lp.subject[:12])}",
                    "kind": lp.kind,
                }
            )
        else:
            human = all(s.lane == "human" for s in spans if s.trace_id == tid)
            out.append(
                {
                    "id": tid,
                    "label": "Human" if human else f"trace {tid[-8:]}",
                    "kind": "human" if human else "other",
                }
            )
    return out


def _clip(s: SpanRow, start: datetime, end: datetime) -> float:
    a = max(start, s.start_ts)
    b = min(end, s.end_ts or end)
    return max(0.0, (b - a).total_seconds())


def _union(intervals: list[tuple[datetime, datetime]]) -> float:
    total = 0.0
    cur: tuple[datetime, datetime] | None = None
    for a, b in sorted(intervals):
        if cur is None or a > cur[1]:
            if cur is not None:
                total += (cur[1] - cur[0]).total_seconds()
            cur = (a, b)
        else:
            cur = (cur[0], max(cur[1], b))
    if cur is not None:
        total += (cur[1] - cur[0]).total_seconds()
    return total


def summary(api: Api, start: datetime, end: datetime) -> dict[str, Any]:
    """Busy share per lane, time per kind, and the queue share of job time in the window."""
    spans = opsread.spans_overlapping(api.ops, start, end)
    length = max(1e-9, (end - start).total_seconds())
    by_lane: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    by_kind: dict[str, float] = defaultdict(float)
    for s in spans:
        by_kind[s.kind] += _clip(s, start, end)
        if s.parent_id is None and s.kind not in WAIT_KINDS:
            by_lane[s.lane].append((max(start, s.start_ts), min(end, s.end_ts or end)))
    lanes = {lane: _union(iv) / length for lane, iv in by_lane.items()}
    gpu = [v for k, v in lanes.items() if k.startswith("gpu:")]
    queue = by_kind.get("job_queue", 0.0)
    work = sum(
        v for k, v in by_kind.items() if k in {"compile", "check", "measure", "profile", "job_run"}
    )
    session = by_kind.get("agent_session", 0.0)
    return {
        "from": iso(start),
        "to": iso(end),
        "lane_busy": lanes,
        "kind_seconds": dict(by_kind),
        "gpu_idle": 1 - max(gpu) if gpu else None,
        "queue_share": queue / (queue + work) if queue + work else None,
        "agent_think_share": by_kind.get("agent_model", 0.0) / session if session else None,
    }


def critical_path(api: Api, trace_id: str) -> dict[str, Any]:
    """The chain of top-level spans that determined when the trace finished.

    Walk back from the span that ended last: each step takes the span that ended last
    before the current one started (its likely predecessor). Waiting spans are kept,
    because waiting on the critical path is exactly what the page should show.
    """
    spans = [s for s in api.ops.spans(trace_id=trace_id, limit=20000) if s.parent_id is None]
    now = api.now()
    if not spans:
        return {"trace_id": trace_id, "path": [], "total_s": 0.0}
    end_of = {s.span_id: s.end_ts or now for s in spans}
    cur = max(spans, key=lambda s: end_of[s.span_id])
    path = [cur]
    while True:
        before = [s for s in spans if end_of[s.span_id] <= cur.start_ts and s not in path]
        if not before:
            break
        cur = max(before, key=lambda s: end_of[s.span_id])
        path.append(cur)
    path.reverse()
    total = (end_of[path[-1].span_id] - path[0].start_ts).total_seconds()
    return {"trace_id": trace_id, "path": [span_json(s) for s in path], "total_s": total}
