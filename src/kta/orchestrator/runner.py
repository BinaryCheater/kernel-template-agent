"""Applying transitions: loop creation, the loop-instance lock, spans, records and tasks.

``advance`` is one iteration of a process main loop for one instance (spec "状态机"):
take the instance lock -> run the state's work -> ``step`` -> write the result -> release.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from kta.core.controls import Controls
from kta.core.errors import ExecutionError, KtaError
from kta.core.ids import UlidFactory
from kta.core.schemas import Actor, Record
from kta.core.trace import SpanStart
from kta.orchestrator import budget, control, expand_work, solve_work
from kta.orchestrator.inbox import inbox
from kta.orchestrator.machines import machine
from kta.orchestrator.pipeline import Services
from kta.orchestrator.solve_work import knowledge_of
from kta.orchestrator.step import step
from kta.orchestrator.tasks import TaskQueue
from kta.orchestrator.transition import Draft, Loop, Settings, Transition
from kta.store import controls as controls_store

log = logging.getLogger("kta.orchestrator")
LOCK_S = 3600
PROCESS_OF = {"solve": "solve", "expand": "learn"}
Work = Callable[[Services, Loop], dict[str, Any]]


def _waiting(svc: Services, loop: Loop) -> dict[str, Any]:
    return {"tasks": expand_work.task_states(svc, loop)}


WORK: dict[str, dict[str, Work]] = {
    "solve": {**solve_work.WORK, "waiting": _waiting},
    "expand": expand_work.WORK,
}


def loop_of(row: Mapping[str, Any]) -> Loop:
    return Loop(row["id"], row["kind"], row["subject"], row["state"], dict(row["data"]))


class Runner:
    def __init__(self, svc: Services, holder: str) -> None:
        self.svc, self.holder = svc, holder
        self.ctx = svc.ctx
        self.ids = UlidFactory(self.ctx.clock)
        self.queue = TaskQueue(self.ctx, svc.store, svc.ops, svc)

    def controls(self) -> Controls:
        """The controls version in force (WP21): read at each start and step."""
        return controls_store.ensure(self.svc.store, self.svc.ops, self.ctx.clock.now())

    @property
    def settings(self) -> Settings:
        return Settings.of(self.controls())

    # ---- instances --------------------------------------------------------------

    def start(
        self,
        kind: str,
        subject: str,
        name: str,
        extra: Mapping[str, Any] | None = None,
        distinct: bool = False,
    ) -> str:
        """The live instance for ``subject``, or a new one in the initial state.

        Starting again clears a ``blocked`` mark (the human asked to retry the work).
        ``distinct`` always starts a new instance (experiment runs, WP11); ``extra`` joins
        its loop data (``process`` books its tasks to that process).
        """
        for row in [] if distinct else self.svc.ops.list_loops(kind):
            if row["subject"] == subject and not machine(kind).terminal(row["state"]):
                data = {**row["data"], "blocked": None, "infra_failures": 0}
                self.svc.ops.put_loop(row["id"], kind, subject, row["state"], data)
                return str(row["id"])
        loop_id = self.ids.new()
        now = self.ctx.clock.now().isoformat()
        m = machine(kind)
        data: dict[str, Any] = {
            "name": name,
            "round": 1,
            "started_at": now,
            "state_since": now,
            "escalated": False,
            "waiting_on": None,
            **(extra or {}),
        }
        data["state_span"] = self._open_span(loop_id, kind, m.initial, 1, "started")
        self.svc.ops.put_loop(loop_id, kind, subject, m.initial, data)
        process = PROCESS_OF[kind]
        budget.ensure_scope(self.svc.ops, f"loop:{loop_id}", self.settings.loop_limit())
        log.info(
            "loop %s started for %s (%s)", loop_id, subject, process, extra={"loop_id": loop_id}
        )
        return loop_id

    def _open_span(self, loop_id: str, kind: str, state: str, rounds: int, note: str) -> str:
        span_id = self.ids.new()
        self.svc.ops.start_span(
            SpanStart(
                span_id,
                None,
                loop_id,
                "loop_step",
                f"process:{PROCESS_OF[kind]}",
                state,
                self.ctx.clock.now(),
                {"state": state, "round": rounds, "note": note},
            )
        )
        return span_id

    # ---- one step ---------------------------------------------------------------

    def advance(self, loop_id: str) -> Transition | None:
        """One step of one instance; None when locked, terminal or blocked."""
        lock = f"loop:{loop_id}"
        if not self.svc.ops.acquire(lock, self.holder, LOCK_S):
            return None
        try:
            loop = loop_of(self.svc.ops.get_loop(loop_id))
            if machine(loop.kind).terminal(loop.state) or loop.data.get("blocked"):
                return None
            scopes = [f"loop:{loop_id}", f"process:{PROCESS_OF[loop.kind]}", "global"]
            if control.paused_scope(self.svc.ops, scopes) is not None:
                return None  # D364: a paused scope stops new steps; the state is kept
            snap = self.controls()
            svc = self.svc.with_controls(snap)
            facts = self._work(loop, svc)
            if facts is None:
                return None
            t = step(replace(loop, facts=facts), svc.ctx)
            # the version this step decided under goes with the loop's data (WP21)
            self.apply(loop, replace(t, data={**t.data, "controls": snap.version_id}))
            return t
        finally:
            self.svc.ops.release(lock, self.holder)

    def _work(self, loop: Loop, svc: Services) -> dict[str, Any] | None:
        """The state's work; a failure blocks the instance (D264), infra errors retry."""
        fn = WORK[loop.kind].get(loop.state)
        if fn is None:
            return {}
        try:
            return fn(svc, loop)
        except ExecutionError as exc:
            if exc.kind in ("infra", "timeout"):
                n = int(loop.data.get("infra_failures", 0)) + 1
                if n < Settings.of(svc.ctx.controls).infra_retries:
                    self._put(loop, loop.state, {"infra_failures": n})
                    log.warning("loop %s %s: infra error %d: %s", loop.id, loop.state, n, exc)
                    return None
            self._block(loop, exc)
        except KtaError as exc:
            self._block(loop, exc)
        except ValueError as exc:  # pydantic validation of stored or submitted data
            self._block(loop, exc)
        return None

    def _block(self, loop: Loop, exc: Exception) -> None:
        blocked = {
            "error": type(exc).__name__,
            "message": str(exc)[:2000],
            "at": self.ctx.clock.now().isoformat(),
            "state": loop.state,
        }
        self._put(loop, loop.state, {"blocked": blocked})
        log.error("loop %s blocked in %s: %s", loop.id, loop.state, exc, extra={"loop_id": loop.id})

    def _put(self, loop: Loop, state: str, update: Mapping[str, Any]) -> None:
        self.svc.ops.put_loop(loop.id, loop.kind, loop.subject, state, {**loop.data, **update})

    # ---- applying a transition --------------------------------------------------

    def apply(self, loop: Loop, t: Transition) -> None:
        if t.to is None:
            self._put(loop, loop.state, {**t.data, "last_note": t.note})
            return
        edge = machine(loop.kind).edge(t.src, t.to, t.kind, t.rule)
        extra = {r.type for r in t.tasks} - set(edge.creates)
        if extra:
            raise KtaError(f"{loop.kind}: {t.src}->{t.to} may not create {sorted(extra)}")
        if t.tasks and len(inbox(self.svc)) >= self.settings.attention_limit:
            self._put(loop, loop.state, {"deferred": "attention limit reached"})
            return
        data: dict[str, Any] = {**loop.data, **t.data, "deferred": None, "infra_failures": 0}
        self._write_drafts(loop, t.records, data)
        if t.tasks:
            process = str(loop.data.get("process") or PROCESS_OF[loop.kind])
            ids = [self.queue.create(r, loop.id, process) for r in t.tasks]
            exp: Mapping[str, Any] = loop.data.get("experiment") or {}
            for task_id in ids if exp else []:
                # The agent side reads these to pick the batch and apply the arm (D372, D373).
                self.queue.annotate(
                    task_id,
                    batch=exp.get("batch"),
                    arm={  # the shape WP8 reads (D373)
                        "name": exp.get("arm"),
                        "knowledge": knowledge_of(exp.get("condition")),
                        "condition": exp.get("condition") or {},
                        "snapshot": exp.get("snapshot"),
                        "protocol": exp.get("protocol"),
                    },
                )
            data.update(waiting_on=ids[0], waiting_tasks=ids, waiting_type=t.tasks[0].type)
        now = self.ctx.clock.now()
        if loop.data.get("state_span"):
            self.svc.ops.end_span(
                str(loop.data["state_span"]), now, "ok", {"to": t.to, "rule": t.rule}
            )
        rounds = int(data.get("round", 1))
        data["state_span"] = self._open_span(loop.id, loop.kind, t.to, rounds, t.note)
        data["state_since"] = now.isoformat()
        if machine(loop.kind).terminal(t.to):
            self.svc.ops.end_span(str(data["state_span"]), now, "ok", {})
        self._put(Loop(loop.id, loop.kind, loop.subject, loop.state, data), t.to, {})
        log.info(
            "loop %s: %s -> %s (%s%s) %s",
            loop.id,
            t.src,
            t.to,
            t.kind,
            f" rule {t.rule}" if t.rule else "",
            t.note,
            extra={"loop_id": loop.id},
        )

    def _write_drafts(self, loop: Loop, drafts: tuple[Draft, ...], data: dict[str, Any]) -> None:
        actor = Actor(type="system", name=f"kta.orchestrator.{loop.kind}_loop")
        written: list[str] = []
        for d in drafts:
            refs: dict[str, Any] = dict(d.refs)
            refs.update({k: [written[i] for i in idx] for k, idx in d.links.items()})
            rec = Record.new(d.kind, dict(d.body), actor, self.ctx.clock.now(), refs=refs)
            written.append(self.svc.store.put(rec))
            if d.data_key:
                data[d.data_key] = written[-1]
