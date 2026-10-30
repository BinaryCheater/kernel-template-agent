"""Shared pieces of the learn, calibrate, experiment and summarize processes (WP11).

Each process tick returns a ``ProcReport`` (what it did, by key) and records every action
that does work as a span on lane ``process:<name>`` (kind ``process_action``), so the
pipeline page shows the processes side by side (docs/design/11 §7, conventions §4.1).
State a process keeps between ticks is in ops pointers ``<process>:<key>`` (JSON),
updated by compare-and-set; evidence is always a record.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from kta.core.errors import ConfigError, KtaError
from kta.core.ids import UlidFactory
from kta.core.trace import SpanStart
from kta.orchestrator.runner import Runner
from kta.store.base import Task
from kta.store.sqlite_ops import SqliteOps

log = logging.getLogger("kta.orchestrator.processes")


@dataclass
class ProcReport:
    process: str
    done: dict[str, list[str]] = field(default_factory=lambda: dict[str, list[str]]())
    blocked: dict[str, str] = field(default_factory=lambda: dict[str, str]())

    def add(self, key: str, *ids: str) -> None:
        self.done.setdefault(key, []).extend(ids)

    def __str__(self) -> str:
        parts = [f"{k}={len(v)}" for k, v in sorted(self.done.items())]
        parts += [f"blocked {k}: {v}" for k, v in sorted(self.blocked.items())]
        return f"{self.process}: " + (", ".join(parts) or "idle")


def get_state(ops: SqliteOps, key: str) -> dict[str, Any]:
    raw = ops.get_pointer(key)
    return dict(json.loads(raw)) if raw else {}


def put_state(ops: SqliteOps, key: str, value: Mapping[str, Any]) -> None:
    old = ops.get_pointer(key)
    if not ops.set_pointer(key, old, json.dumps(dict(value), sort_keys=True)):
        raise ConfigError(f"{key} changed concurrently")


def update_state(ops: SqliteOps, key: str, **changes: Any) -> dict[str, Any]:
    new = {**get_state(ops, key), **changes}
    put_state(ops, key, new)
    return new


@contextmanager
def action(
    runner: Runner, process: str, name: str, trace: str, **attrs: Any
) -> Generator[dict[str, Any]]:
    """A span for one piece of work of a process; the yielded dict is added to its attrs."""
    ops, now = runner.svc.ops, runner.ctx.clock.now
    span_id = UlidFactory(runner.ctx.clock).new()
    ops.start_span(
        SpanStart(span_id, None, trace, "process_action", f"process:{process}", name, now(), attrs)
    )
    out: dict[str, Any] = {}
    try:
        yield out
    except BaseException as exc:
        ops.end_span(span_id, now(), "fail", {**out, "error": f"{type(exc).__name__}: {exc}"})
        raise
    ops.end_span(span_id, now(), "ok", out)


def process_of(runner: Runner, task: Task) -> str:
    return str(runner.queue.info(task.id).get("process", "solve"))


def check_submitted(runner: Runner, rep: ProcReport, owns: Callable[[Task], bool]) -> None:
    """Run the done-checks of submitted tasks this process owns."""
    for t in runner.svc.ops.list_tasks(states=("submitted",)):
        if owns(t):
            runner.queue.check(t.id)
            rep.add("checked", t.id)


def live_tasks(runner: Runner, process: str) -> list[tuple[Task, dict[str, Any]]]:
    """Unfinished tasks booked to ``process``, with their bookkeeping."""
    out: list[tuple[Task, dict[str, Any]]] = []
    for t in runner.svc.ops.list_tasks():
        if runner.queue.machine.terminal(t.state):
            continue
        info = runner.queue.info(t.id)
        if info.get("process") == process:
            out.append((t, info))
    return out


def due(ops: SqliteOps, key: str, now: datetime, every_s: float) -> bool:
    """True when the pointer ``key`` (an ISO time) is unset or older than ``every_s``."""
    raw = ops.get_pointer(key)
    return raw is None or datetime.fromisoformat(raw) + timedelta(seconds=every_s) <= now


def mark(ops: SqliteOps, key: str, now: datetime) -> None:
    old = ops.get_pointer(key)
    if not ops.set_pointer(key, old, now.isoformat()):
        raise ConfigError(f"{key} changed concurrently")


def guarded(
    rep: ProcReport, ops: SqliteOps, key: str, now: datetime, retry_s: float
) -> Callable[[Callable[[], None]], None]:
    """Run work unless it failed less than ``retry_s`` ago; a failure is recorded in the
    pointer ``key`` (JSON: at, error) and in the report, never silently dropped."""

    def run(fn: Callable[[], None]) -> None:
        state = get_state(ops, key)
        if (
            state.get("at")
            and datetime.fromisoformat(state["at"]) + timedelta(seconds=retry_s) > now
        ):
            rep.blocked[key] = str(state.get("error"))
            return
        try:
            fn()
        except KtaError as exc:  # recorded and retried after retry_s; the process goes on
            log.warning("%s failed: %s", key, exc)
            put_state(ops, key, {"at": now.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
            rep.blocked[key] = str(exc)
            return
        if state:
            put_state(ops, key, {})

    return run
