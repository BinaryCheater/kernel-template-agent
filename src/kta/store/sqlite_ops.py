"""SQLite implementation of Ops: the mutable operations tables (docs/design/20 §4).

Every change is logged to ``ops_events`` in the same transaction, so the frontend and
debugging can replay what happened. None of this is evidence: conclusions are records.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection, Mapping
from datetime import datetime, timedelta
from typing import Any

from kta.core.context import Clock
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.ids import UlidFactory
from kta.core.trace import SpanStart, SpanStatus
from kta.store.base import OpsEvent, SpanRow, Task
from kta.store.db import Database, parse_ts, ts

FINAL_TASK_STATES = frozenset({"done", "failed", "cancelled"})
TASK_COLUMNS = (
    "id, card_id, card_type, state, priority, lease_owner, lease_until, attempts, work_unit"
)


def _task(row: Any) -> Task:
    return Task(
        id=row[0],
        card_id=row[1],
        card_type=row[2],
        state=row[3],
        priority=int(row[4]),
        lease_owner=row[5],
        lease_until=parse_ts(row[6]) if row[6] else None,
        attempts=int(row[7]),
        work_unit=row[8],
    )


class SqliteOps:
    def __init__(self, db: Database, clock: Clock, ids: UlidFactory) -> None:
        self.db = db
        self.clock = clock
        self.ids = ids

    def _now(self) -> datetime:
        return self.clock.now()

    def _log(
        self, con: sqlite3.Connection, table: str, row_id: str, change: Mapping[str, Any]
    ) -> None:
        con.execute(
            "INSERT INTO ops_events (at, table_name, row_id, change_json) VALUES (?, ?, ?, ?)",
            (ts(self._now()), table, row_id, json.dumps(change, sort_keys=True, default=str)),
        )

    # ---- tasks ------------------------------------------------------------------

    def enqueue(self, card_id: str, priority: int, work_unit: str) -> str:
        task_id = self.ids.new()
        now = ts(self._now())
        with self.db.write() as con:
            row = con.execute(
                "SELECT json_extract(json, '$.body.type') FROM records "
                "WHERE id = ? AND kind = 'TaskCard'",
                (card_id,),
            ).fetchone()
            if row is None:
                raise ConfigError(f"enqueue: {card_id} is not a TaskCard record")
            con.execute(
                "INSERT INTO tasks (id, card_id, card_type, state, priority, attempts, work_unit, "
                "created_at, updated_at) VALUES (?, ?, ?, 'queued', ?, 0, ?, ?, ?)",
                (task_id, card_id, row[0], priority, work_unit, now, now),
            )
            self._log(con, "tasks", task_id, {"state": "queued", "card_id": card_id})
        return task_id

    def lease(self, worker: str, kinds: list[str], seconds: int) -> Task | None:
        """Highest priority, oldest ready task of the given card types; expired leases count."""
        if not kinds:
            return None
        now = self._now()
        marks = ", ".join("?" * len(kinds))
        with self.db.write() as con:
            row = con.execute(
                f"SELECT id FROM tasks WHERE card_type IN ({marks}) AND (state = 'queued' OR "
                "(state = 'leased' AND lease_until < ?)) ORDER BY priority DESC, created_at, id "
                "LIMIT 1",
                [*kinds, ts(now)],
            ).fetchone()
            if row is None:
                return None
            until = ts(now + timedelta(seconds=seconds))
            con.execute(
                "UPDATE tasks SET state = 'leased', lease_owner = ?, lease_until = ?, "
                "attempts = attempts + 1, updated_at = ? WHERE id = ?",
                (worker, until, ts(now), row[0]),
            )
            self._log(con, "tasks", row[0], {"state": "leased", "owner": worker, "until": until})
            got = con.execute(f"SELECT {TASK_COLUMNS} FROM tasks WHERE id = ?", (row[0],))
            return _task(got.fetchone())

    def renew(self, task_id: str, worker: str, seconds: int) -> bool:
        now = self._now()
        until = ts(now + timedelta(seconds=seconds))
        with self.db.write() as con:
            cur = con.execute(
                "UPDATE tasks SET lease_until = ?, updated_at = ? WHERE id = ? "
                "AND state = 'leased' AND lease_owner = ? AND lease_until >= ?",
                (until, ts(now), task_id, worker, ts(now)),
            )
            if cur.rowcount != 1:
                return False
            self._log(con, "tasks", task_id, {"renewed_until": until, "owner": worker})
            return True

    def finish(self, task_id: str, state: str) -> None:
        if state not in FINAL_TASK_STATES:
            raise ConfigError(f"finish: state must be one of {sorted(FINAL_TASK_STATES)}")
        with self.db.write() as con:
            row = con.execute("SELECT state FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise ConfigError(f"finish: no task {task_id}")
            if row[0] in FINAL_TASK_STATES:
                raise ConfigError(f"finish: task {task_id} is already {row[0]}")
            con.execute(
                "UPDATE tasks SET state = ?, lease_owner = NULL, lease_until = NULL, "
                "updated_at = ? WHERE id = ?",
                (state, ts(self._now()), task_id),
            )
            self._log(con, "tasks", task_id, {"state": state})

    def task(self, task_id: str) -> Task:
        with self.db.read() as con:
            row = con.execute(f"SELECT {TASK_COLUMNS} FROM tasks WHERE id = ?", (task_id,))
            found = row.fetchone()
        if found is None:
            raise ConfigError(f"no task {task_id}")
        return _task(found)

    def update_task(
        self,
        task_id: str,
        expected: Collection[str],
        state: str,
        owner: str | None = None,
        lease_s: int | None = None,
        count_attempt: bool = False,
    ) -> bool:
        """Compare-and-set a task's lifecycle state (added by WP6, decision D265).

        Succeeds only while the task is in one of ``expected``. ``owner``/``lease_s`` set a
        lease; a state without a lease clears it. ``count_attempt`` adds one attempt.
        """
        now = self._now()
        until = ts(now + timedelta(seconds=lease_s)) if lease_s is not None else None
        marks = ", ".join("?" * len(expected))
        with self.db.write() as con:
            cur = con.execute(
                "UPDATE tasks SET state = ?, lease_owner = ?, lease_until = ?, "
                f"attempts = attempts + ?, updated_at = ? WHERE id = ? AND state IN ({marks})",
                (state, owner, until, int(count_attempt), ts(now), task_id, *expected),
            )
            if cur.rowcount != 1:
                return False
            change: dict[str, Any] = {"state": state}
            if owner is not None:
                change.update(owner=owner, until=until)
            self._log(con, "tasks", task_id, change)
            return True

    def list_tasks(
        self, work_unit: str | None = None, states: Collection[str] | None = None
    ) -> list[Task]:
        """Tasks, oldest first (added by WP6)."""
        sql, args = f"SELECT {TASK_COLUMNS} FROM tasks WHERE 1 = 1", list[Any]()
        if work_unit is not None:
            sql += " AND work_unit = ?"
            args.append(work_unit)
        if states is not None:
            sql += f" AND state IN ({', '.join('?' * len(states))})"
            args += list(states)
        with self.db.read() as con:
            rows = con.execute(sql + " ORDER BY created_at, id", args).fetchall()
        return [_task(r) for r in rows]

    # ---- locks and pointers -----------------------------------------------------

    def acquire(self, lock: str, holder: str, seconds: int) -> bool:
        """Take a free or expired lock, or extend one this holder already has."""
        now = self._now()
        until = ts(now + timedelta(seconds=seconds))
        with self.db.write() as con:
            cur = con.execute(
                "INSERT INTO locks (name, holder, until) VALUES (?, ?, ?) ON CONFLICT(name) DO "
                "UPDATE SET holder = excluded.holder, until = excluded.until "
                "WHERE locks.holder = excluded.holder OR locks.until < ?",
                (lock, holder, until, ts(now)),
            )
            if cur.rowcount != 1:
                return False
            self._log(con, "locks", lock, {"holder": holder, "until": until})
            return True

    def release(self, lock: str, holder: str) -> None:
        with self.db.write() as con:
            row = con.execute("SELECT holder FROM locks WHERE name = ?", (lock,)).fetchone()
            if row is None:
                return
            if row[0] != holder:
                raise PermissionDenied(f"lock {lock} is held by {row[0]}, not {holder}")
            con.execute("DELETE FROM locks WHERE name = ?", (lock,))
            self._log(con, "locks", lock, {"released_by": holder})

    def get_pointer(self, name: str) -> str | None:
        with self.db.read() as con:
            row = con.execute("SELECT value FROM pointers WHERE name = ?", (name,)).fetchone()
        return None if row is None else str(row[0])

    def set_pointer(self, name: str, expected: str | None, value: str) -> bool:
        """Compare-and-set: succeeds only if the pointer still has ``expected``."""
        with self.db.write() as con:
            if expected is None:
                cur = con.execute(
                    "INSERT INTO pointers (name, value) VALUES (?, ?) ON CONFLICT(name) DO NOTHING",
                    (name, value),
                )
            else:
                cur = con.execute(
                    "UPDATE pointers SET value = ? WHERE name = ? AND value = ?",
                    (value, name, expected),
                )
            if cur.rowcount != 1:
                return False
            self._log(con, "pointers", name, {"from": expected, "to": value})
            return True

    # ---- budgets ----------------------------------------------------------------

    def set_budget(self, scope: str, limit: Mapping[str, float]) -> None:
        with self.db.write() as con:
            con.execute(
                "INSERT INTO budgets (scope, limit_json, used_json, reserved_json) "
                "VALUES (?, ?, '{}', '{}') ON CONFLICT(scope) DO UPDATE SET "
                "limit_json = excluded.limit_json",
                (scope, json.dumps(dict(limit), sort_keys=True)),
            )
            self._log(con, "budgets", scope, {"limit": dict(limit)})

    def _budget(self, con: sqlite3.Connection, scope: str) -> tuple[dict[str, float], ...]:
        row = con.execute(
            "SELECT limit_json, used_json, reserved_json FROM budgets WHERE scope = ?", (scope,)
        ).fetchone()
        if row is None:
            raise ConfigError(f"no budget for scope {scope!r} (set_budget first)")
        return tuple(json.loads(x) for x in row)

    def reserve(self, scope: str, amount: Mapping[str, float]) -> bool:
        """Reserve ``amount`` if used + reserved + amount stays within every limit."""
        with self.db.write() as con:
            limit, used, reserved = self._budget(con, scope)
            for key, value in amount.items():
                if value < 0:
                    raise ConfigError(f"reserve: negative amount for {key}")
                total = used.get(key, 0.0) + reserved.get(key, 0.0) + value
                if key in limit and total > limit[key]:
                    return False
            for key, value in amount.items():
                reserved[key] = reserved.get(key, 0.0) + value
            con.execute(
                "UPDATE budgets SET reserved_json = ? WHERE scope = ?",
                (json.dumps(reserved, sort_keys=True), scope),
            )
            self._log(con, "budgets", scope, {"reserved": dict(amount)})
            return True

    def settle(self, scope: str, reserved: Mapping[str, float], used: Mapping[str, float]) -> None:
        """Release a reservation and book what was actually used."""
        with self.db.write() as con:
            _, have_used, have_reserved = self._budget(con, scope)
            for key, value in reserved.items():
                left = have_reserved.get(key, 0.0) - value
                if left < -1e-9:
                    raise ConfigError(f"settle: {key} releases more than was reserved")
                have_reserved[key] = max(left, 0.0)
            for key, value in used.items():
                have_used[key] = have_used.get(key, 0.0) + value
            con.execute(
                "UPDATE budgets SET used_json = ?, reserved_json = ? WHERE scope = ?",
                (json.dumps(have_used, sort_keys=True), json.dumps(have_reserved), scope),
            )
            self._log(con, "budgets", scope, {"released": dict(reserved), "used": dict(used)})

    def budget(self, scope: str) -> dict[str, dict[str, float]]:
        with self.db.read() as con:
            limit, used, reserved = self._budget(con, scope)
        return {"limit": limit, "used": used, "reserved": reserved}

    # ---- workers and loops ------------------------------------------------------

    def heartbeat(self, name: str, host: str, healthy: bool) -> None:
        with self.db.write() as con:
            con.execute(
                "INSERT INTO workers (name, host, healthy, heartbeat) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET host = excluded.host, "
                "healthy = excluded.healthy, heartbeat = excluded.heartbeat",
                (name, host, int(healthy), ts(self._now())),
            )
            self._log(con, "workers", name, {"host": host, "healthy": healthy})

    def worker_health(self, name: str) -> bool | None:
        """The last reported health of a worker; None when it never reported."""
        with self.db.read() as con:
            row = con.execute("SELECT healthy FROM workers WHERE name = ?", (name,)).fetchone()
        return None if row is None else bool(row[0])

    def record_event(self, table: str, row_id: str, change: Mapping[str, Any]) -> None:
        """Append an event that changes no table row (e.g. a forced lease release, D192)."""
        with self.db.write() as con:
            self._log(con, table, row_id, change)

    def put_loop(
        self, id: str, kind: str, subject: str, state: str, data: Mapping[str, Any]
    ) -> None:
        with self.db.write() as con:
            con.execute(
                "INSERT INTO loops (id, kind, subject, state, data_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET state = excluded.state, "
                "data_json = excluded.data_json, updated_at = excluded.updated_at",
                (id, kind, subject, state, json.dumps(dict(data), sort_keys=True), ts(self._now())),
            )
            self._log(con, "loops", id, {"state": state})

    def get_loop(self, id: str) -> dict[str, Any]:
        with self.db.read() as con:
            row = con.execute(
                "SELECT kind, subject, state, data_json, updated_at FROM loops WHERE id = ?", (id,)
            ).fetchone()
        if row is None:
            raise ConfigError(f"no loop {id}")
        return {
            "id": id,
            "kind": row[0],
            "subject": row[1],
            "state": row[2],
            "data": json.loads(row[3]),
            "updated_at": parse_ts(row[4]),
        }

    def list_loops(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Loop rows in ``get_loop`` form, by ID (added by WP6)."""
        sql = "SELECT id FROM loops" + (" WHERE kind = ?" if kind else "") + " ORDER BY id"
        with self.db.read() as con:
            ids = [r[0] for r in con.execute(sql, (kind,) if kind else ()).fetchall()]
        return [self.get_loop(i) for i in ids]

    # ---- events -----------------------------------------------------------------

    def events(self, since: int) -> list[OpsEvent]:
        with self.db.read() as con:
            rows = con.execute(
                "SELECT seq, at, table_name, row_id, change_json FROM ops_events WHERE seq > ? "
                "ORDER BY seq",
                (since,),
            ).fetchall()
        return [OpsEvent(r[0], parse_ts(r[1]), r[2], r[3], json.loads(r[4])) for r in rows]

    # ---- spans (kta.core.trace.SpanSink) ----------------------------------------

    def start_span(self, span: SpanStart) -> None:
        with self.db.write() as con:
            con.execute(
                "INSERT INTO spans (span_id, parent_id, trace_id, kind, lane, name, start_ts, "
                "end_ts, status, attrs_json) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'running', ?)",
                (
                    span.span_id,
                    span.parent_id,
                    span.trace_id,
                    span.kind,
                    span.lane,
                    span.name,
                    ts(span.start_ts),
                    json.dumps(span.attrs, sort_keys=True, default=str),
                ),
            )
            self._log(con, "spans", span.span_id, {"status": "running", "kind": span.kind})

    def end_span(
        self, span_id: str, end_ts: datetime, status: SpanStatus, attrs: Mapping[str, Any]
    ) -> None:
        with self.db.write() as con:
            row = con.execute("SELECT attrs_json FROM spans WHERE span_id = ?", (span_id,))
            found = row.fetchone()
            if found is None:
                raise ConfigError(f"end_span: no span {span_id}")
            merged = {**json.loads(found[0]), **attrs}
            con.execute(
                "UPDATE spans SET end_ts = ?, status = ?, attrs_json = ? WHERE span_id = ?",
                (ts(end_ts), status, json.dumps(merged, sort_keys=True, default=str), span_id),
            )
            self._log(con, "spans", span_id, {"status": status})

    def spans(
        self, trace_id: str | None = None, since: datetime | None = None, limit: int = 1000
    ) -> list[SpanRow]:
        sql = (
            "SELECT span_id, parent_id, trace_id, kind, lane, name, start_ts, end_ts, status, "
            "attrs_json FROM spans WHERE 1 = 1"
        )
        params: list[Any] = []
        if trace_id is not None:
            sql += " AND trace_id = ?"
            params.append(trace_id)
        if since is not None:
            sql += " AND start_ts >= ?"
            params.append(ts(since))
        sql += " ORDER BY start_ts, span_id LIMIT ?"
        params.append(limit)
        with self.db.read() as con:
            rows = con.execute(sql, params).fetchall()
        return [
            SpanRow(
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
            for r in rows
        ]

    def prune_spans(self, before: datetime) -> int:
        """Drop finished spans that started before ``before`` (retention: 90 days)."""
        with self.db.write() as con:
            cur = con.execute(
                "DELETE FROM spans WHERE start_ts < ? AND end_ts IS NOT NULL", (ts(before),)
            )
            self._log(con, "spans", "*", {"pruned_before": ts(before), "rows": cur.rowcount})
            return cur.rowcount
