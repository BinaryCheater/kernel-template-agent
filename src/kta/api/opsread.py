"""Read-only listings of the operations tables that the Ops interface does not offer yet.

``SqliteOps`` has per-row reads (``task``, ``get_loop``, ``budget``) but no listings, and
``spans`` filters only by start time. The API needs listings and a time-window overlap
query, so this module reads the tables of WP1's schema directly, through the same
connection and lock. It never writes. Requested as Ops methods in deliveries/WP9a.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kta.store import SqliteOps, SqliteStore
from kta.store.base import SpanRow
from kta.store.db import parse_ts, ts


@dataclass(frozen=True)
class LoopRow:
    id: str
    kind: str
    subject: str
    state: str
    data: dict[str, Any]
    updated_at: datetime


@dataclass(frozen=True)
class TaskRow:
    id: str
    card_id: str
    card_type: str
    state: str
    priority: int
    lease_owner: str | None
    lease_until: datetime | None
    attempts: int
    work_unit: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class WorkerRow:
    name: str
    host: str
    healthy: bool
    heartbeat: datetime


@dataclass(frozen=True)
class LockRow:
    name: str
    holder: str
    until: datetime


@dataclass(frozen=True)
class BudgetRow:
    scope: str
    limit: dict[str, float]
    used: dict[str, float]
    reserved: dict[str, float]


def loops(ops: SqliteOps) -> list[LoopRow]:
    with ops.db.read() as con:
        rows = con.execute(
            "SELECT id, kind, subject, state, data_json, updated_at FROM loops ORDER BY id"
        ).fetchall()
    return [LoopRow(r[0], r[1], r[2], r[3], json.loads(r[4]), parse_ts(r[5])) for r in rows]


def tasks(ops: SqliteOps) -> list[TaskRow]:
    with ops.db.read() as con:
        rows = con.execute(
            "SELECT id, card_id, card_type, state, priority, lease_owner, lease_until, attempts, "
            "work_unit, created_at, updated_at FROM tasks ORDER BY created_at, id"
        ).fetchall()
    return [
        TaskRow(
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            parse_ts(r[6]) if r[6] else None,
            r[7],
            r[8],
            parse_ts(r[9]),
            parse_ts(r[10]),
        )
        for r in rows
    ]


def workers(ops: SqliteOps) -> list[WorkerRow]:
    with ops.db.read() as con:
        rows = con.execute("SELECT name, host, healthy, heartbeat FROM workers ORDER BY name")
        return [WorkerRow(r[0], r[1], bool(r[2]), parse_ts(r[3])) for r in rows.fetchall()]


def locks(ops: SqliteOps) -> list[LockRow]:
    with ops.db.read() as con:
        rows = con.execute("SELECT name, holder, until FROM locks ORDER BY name").fetchall()
    return [LockRow(r[0], r[1], parse_ts(r[2])) for r in rows]


def budgets(ops: SqliteOps) -> list[BudgetRow]:
    with ops.db.read() as con:
        rows = con.execute(
            "SELECT scope, limit_json, used_json, reserved_json FROM budgets ORDER BY scope"
        ).fetchall()
    return [BudgetRow(r[0], *(json.loads(x) for x in r[1:])) for r in rows]


def spans_overlapping(
    ops: SqliteOps, start: datetime, end: datetime, trace_id: str | None = None, limit: int = 20000
) -> list[SpanRow]:
    """Spans that overlap [start, end], including ones still running."""
    sql = (
        "SELECT span_id, parent_id, trace_id, kind, lane, name, start_ts, end_ts, status, "
        "attrs_json FROM spans WHERE start_ts <= ? AND (end_ts IS NULL OR end_ts >= ?)"
    )
    params: list[Any] = [ts(end), ts(start)]
    if trace_id is not None:
        sql += " AND trace_id = ?"
        params.append(trace_id)
    sql += " ORDER BY start_ts, span_id LIMIT ?"
    params.append(limit)
    with ops.db.read() as con:
        rows = con.execute(sql, params).fetchall()
    return [_span(r) for r in rows]


def open_spans(ops: SqliteOps) -> list[SpanRow]:
    with ops.db.read() as con:
        rows = con.execute(
            "SELECT span_id, parent_id, trace_id, kind, lane, name, start_ts, end_ts, status, "
            "attrs_json FROM spans WHERE end_ts IS NULL ORDER BY start_ts"
        ).fetchall()
    return [_span(r) for r in rows]


def _span(r: Any) -> SpanRow:
    return SpanRow(
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
        r[5],
        parse_ts(r[6]),
        parse_ts(r[7]) if r[7] else None,
        r[8],
        json.loads(r[9]),
    )


def records_after(store: SqliteStore, rowid: int, limit: int = 200) -> list[tuple[int, str, str]]:
    """(rowid, id, kind) of records inserted after ``rowid``; feeds the SSE stream."""
    with store.db.read() as con:
        rows = con.execute(
            "SELECT rowid, id, kind FROM records WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (rowid, limit),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def last_record_rowid(store: SqliteStore) -> int:
    with store.db.read() as con:
        row = con.execute("SELECT COALESCE(MAX(rowid), 0) FROM records").fetchone()
    return int(row[0])


def origin_counts(store: SqliteStore) -> dict[str, int]:
    with store.db.read() as con:
        rows = con.execute("SELECT origin, COUNT(*) FROM records GROUP BY origin").fetchall()
    return {r[0]: int(r[1]) for r in rows}


def spans_of_task(ops: SqliteOps, task_id: str) -> list[SpanRow]:
    """Spans tagged with ``attrs.task_id`` and all their descendants (session internals)."""
    with ops.db.read() as con:
        rows = con.execute(
            "WITH RECURSIVE tree(id) AS ("
            " SELECT span_id FROM spans WHERE json_extract(attrs_json, '$.task_id') = ?"
            " UNION SELECT s.span_id FROM spans s JOIN tree t ON s.parent_id = t.id) "
            "SELECT span_id, parent_id, trace_id, kind, lane, name, start_ts, end_ts, status, "
            "attrs_json FROM spans WHERE span_id IN (SELECT id FROM tree) "
            "ORDER BY start_ts, span_id",
            (task_id,),
        ).fetchall()
    return [_span(r) for r in rows]


def pointer_history(ops: SqliteOps, name: str) -> list[tuple[datetime, dict[str, Any]]]:
    """(time, change) of every update of an ops pointer, oldest first (from ops_events)."""
    with ops.db.read() as con:
        rows = con.execute(
            "SELECT at, change_json FROM ops_events WHERE table_name = 'pointers' AND row_id = ? "
            "ORDER BY seq",
            (name,),
        ).fetchall()
    return [(parse_ts(r[0]), json.loads(r[1])) for r in rows]
