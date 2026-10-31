"""Idempotent resource accounting, including runs that finish after cancellation."""

from __future__ import annotations

from kta.core.ids import record_id
from kta.store.base import Store
from kta.store.sqlite_ops import SqliteOps


def reconcile(store: Store, ops: SqliteOps, task_id: str, scopes: list[str]) -> None:
    for run in store.query("AgentRun", {"actor.task": task_id}, limit=100_000):
        for scope in scopes:
            ops.settle(
                scope,
                {},
                {"agent_min": float(run.body["wall_s"]) / 60},
                receipt="agent-run:" + record_id(run),
            )
    if not scopes:
        return
    spans = ops.spans(limit=100_000)
    owned = {s.span_id for s in spans if s.attrs.get("task_id") == task_id}
    # Include API/Executor descendants even when they do not repeat task_id.
    changed = True
    while changed:
        before = len(owned)
        owned.update(s.span_id for s in spans if s.parent_id in owned)
        changed = len(owned) != before
    for s in spans:
        if s.span_id not in owned or s.kind != "job_run" or s.end_ts is None:
            continue
        if not s.lane.endswith((":gpu-shared", ":gpu-exclusive")):
            continue
        for scope in scopes:
            elapsed = float(s.attrs.get("wall_s", (s.end_ts - s.start_ts).total_seconds()))
            ops.settle(
                scope,
                {},
                {"gpu_min": elapsed / 60},
                receipt="gpu-job:" + s.span_id,
            )
