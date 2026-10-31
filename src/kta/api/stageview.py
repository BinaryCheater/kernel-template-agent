"""``GET /trace/stages``: pipeline efficiency over time (WP23, D612).

Everything is derived from spans the system already writes, so the view never holds
numbers of its own:

- stage depth, entries, exits and wait: ``loop_step`` spans (one per loop state; the
  lane ``process:solve`` is a solve loop, ``process:learn`` an expand loop), mapped to
  stages by ``orchestrator/stage_queues.yaml``; a state no stage names counts as ``other``;
- open measurement windows: ``lock_hold`` spans on ``gpu:*`` lanes held by ``window:*``;
- GPU busy: the union of ``job_run`` spans on ``worker:<w>:gpu-shared|gpu-exclusive``;
- host-lock waits: ``lock_wait`` spans on ``host:<worker>`` lanes.

``now`` adds the current depth and capacity per stage and the held loops with the reason
(loop data ``deferred``, written by the solve process when backpressure holds a loop).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from kta.api import opsread
from kta.api.common import Api, controls, iso
from kta.orchestrator import stagequeue
from kta.orchestrator.machines import machine
from kta.store.base import SpanRow

KIND_OF_LANE = {"process:solve": "solve", "process:learn": "expand"}
OTHER = "other"
MAX_BUCKETS = 720


def _clip(s: SpanRow, start: datetime, end: datetime, now: datetime) -> tuple[float, float]:
    a = max(s.start_ts, start)
    b = min(s.end_ts or now, end)
    return a.timestamp(), b.timestamp()


def _union_s(intervals: list[tuple[float, float]]) -> float:
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_a is not None and cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_a is not None and cur_b is not None:
        total += cur_b - cur_a
    return total


def _stage(s: SpanRow) -> str | None:
    kind = KIND_OF_LANE.get(s.lane)
    if s.kind != "loop_step" or kind is None:
        return None
    if machine(kind).terminal(s.name):
        return None
    return stagequeue.stage_of(kind, s.name) or OTHER


def series(
    spans: list[SpanRow], start: datetime, end: datetime, bucket_s: float, now: datetime
) -> dict[str, Any]:
    n = max(1, min(MAX_BUCKETS, int((end - start).total_seconds() // bucket_s) or 1))
    edges = [start + timedelta(seconds=bucket_s * i) for i in range(n + 1)]
    names = [s.id for s in stagequeue.stages()] + [OTHER]
    depth = {k: [0.0] * n for k in names}
    entries = {k: [0] * n for k in names}
    exits = {k: [0] * n for k in names}
    waits: dict[str, list[list[float]]] = {k: [[] for _ in range(n)] for k in names}
    gpu: list[list[tuple[float, float]]] = [[] for _ in range(n)]
    lock_wait_s, lock_waits = [0.0] * n, [0] * n
    window = next(s.id for s in stagequeue.stages() if s.window)

    def idx(t: datetime) -> int | None:
        i = int((t - start).total_seconds() // bucket_s)
        return i if 0 <= i < n else None

    for sp in spans:
        st = _stage(sp)
        if (
            sp.kind == "lock_hold"
            and sp.lane.startswith("gpu:")
            and str(sp.attrs.get("holder", "")).startswith("window:")
        ):
            st = window
        for i in range(n):
            a, b = _clip(sp, edges[i], edges[i + 1], now)
            if b <= a:
                continue
            if st is not None:
                depth[st][i] += (b - a) / bucket_s  # time-averaged depth in the bucket
            if sp.kind == "job_run" and sp.lane.endswith((":gpu-shared", ":gpu-exclusive")):
                gpu[i].append((a, b))
        if st is not None:
            i0 = idx(sp.start_ts)
            if i0 is not None:
                entries[st][i0] += 1
            if sp.end_ts is not None and (i1 := idx(sp.end_ts)) is not None:
                exits[st][i1] += 1
                waits[st][i1].append((sp.end_ts - sp.start_ts).total_seconds())
        if sp.kind == "lock_wait" and sp.lane.startswith("host:"):
            i0 = idx(sp.start_ts)
            if i0 is not None:
                lock_waits[i0] += 1
                lock_wait_s[i0] += ((sp.end_ts or now) - sp.start_ts).total_seconds()
    return {
        "buckets": [iso(t) for t in edges[:-1]],
        "bucket_s": bucket_s,
        "stages": {
            k: {
                "depth": [round(x, 3) for x in depth[k]],
                "entries": entries[k],
                "throughput_per_h": [x * 3600 / bucket_s for x in exits[k]],
                "mean_wait_s": [round(sum(w) / len(w), 3) if w else None for w in waits[k]],
            }
            for k in names
        },
        "gpu_busy": [round(_union_s(iv) / bucket_s, 4) for iv in gpu],
        "host_lock": {"waits": lock_waits, "wait_s": [round(x, 3) for x in lock_wait_s]},
    }


def current(api: Api) -> dict[str, Any]:
    cap = controls(api).stage_capacity
    rows = [
        {"id": lp.id, "kind": lp.kind, "state": lp.state, "data": lp.data, "subject": lp.subject}
        for lp in opsread.loops(api.ops)
        if not machine(lp.kind).terminal(lp.state)
    ]
    depth = stagequeue.depths(rows)
    return {
        "stages": [
            {
                "id": s.id,
                "does": s.does,
                "pool": s.pool,
                "depth": depth.get(s.id, 0),
                "capacity": cap.get(s.id),
                "window": s.window,
            }
            for s in stagequeue.stages()
        ],
        "held": [
            {
                "loop": r["id"],
                "kind": r["kind"],
                "state": r["state"],
                "reason": r["data"]["deferred"],
            }
            for r in rows
            if r["data"].get("deferred")
        ],
    }


def stages_view(api: Api, start: datetime, end: datetime, bucket_s: float) -> dict[str, Any]:
    spans = opsread.spans_overlapping(api.ops, start, end, None, limit=200_000)
    out = series(spans, start, end, bucket_s, api.now())
    return {"from": iso(start), "to": iso(end), **out, "now": current(api)}


__all__ = ["current", "series", "stages_view"]
