"""Task cards, the queue and the task lifecycle (docs/design/10 §3, machines.yaml ``task``).

Every state change is a compare-and-set on the ops ``tasks`` row (D265) and must be an
edge of the ``task`` machine. Per-task bookkeeping that is not evidence (reservation,
span) lives in the ops pointer ``task:<id>``. Checks and acceptance write records.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kta.core.context import Context
from kta.core.controls import Controls
from kta.core.errors import ConfigError, KtaError, PermissionDenied, SchemaError
from kta.core.ids import UlidFactory
from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.process import Submission, TaskCard
from kta.core.trace import SpanStart
from kta.orchestrator import budget, control
from kta.orchestrator.done_checks import DONE_CHECKS, CheckInput, Outcome, run_check
from kta.orchestrator.machines import Machine, machine
from kta.orchestrator.outputs import write_outputs
from kta.orchestrator.pipeline import Services
from kta.orchestrator.transition import Settings, TaskRequest
from kta.store import controls as controls_store
from kta.store.base import Store, Task
from kta.store.sqlite_ops import SqliteOps

UNASSIGNED_PROFILE = "unassigned"  # when no profile is approved yet (D271, D366)
# The ops pointer kta.agents.profile.approve_profile moves (orchestrator may not import agents).
CURRENT_PROFILE = "current_profile"
LIVE = ("queued", "leased", "running", "submitted", "checking", "rejected", "expired")
# run_check turns bad submission data (ValueError, SchemaError) into failed outcomes, so any
# KtaError left is the checker or its infrastructure failing: the task is suspended.
CHECKER_FAILURES = (KtaError,)


@dataclass(frozen=True)
class Verdict:
    task_id: str
    state: str  # accepted | queued (retry) | escalated | suspended
    report_id: str
    outcomes: list[Outcome]


def submission_digest(sub: Record) -> str:
    """Content hash of a submission: its file names and blob hashes (12 §3)."""
    pairs = sorted((str(k), str(v)) for k, v in sub.blobs.items())
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


class TaskQueue:
    def __init__(
        self, ctx: Context, store: Store, ops: SqliteOps, svc: Services | None = None
    ) -> None:
        self.ctx, self.store, self.ops = ctx, store, ops
        self.svc = svc or Services(ctx, store, ops)  # one executor per process (D184)
        self.machine: Machine = machine("task")
        self.ids = UlidFactory(ctx.clock)

    def controls(self) -> Controls:
        """The controls version in force (stored on first use, WP21)."""
        return controls_store.ensure(self.store, self.ops, self.ctx.clock.now())

    @property
    def settings(self) -> Settings:
        return Settings.of(self.controls())

    # ---- bookkeeping ----------------------------------------------------------

    def info(self, task_id: str) -> dict[str, Any]:
        raw = self.ops.get_pointer(f"task:{task_id}")
        return json.loads(raw) if raw else {}

    def _set_info(self, task_id: str, **changes: Any) -> None:
        old = self.ops.get_pointer(f"task:{task_id}")
        new: dict[str, Any] = {**(json.loads(old) if old else {}), **changes}
        if not self.ops.set_pointer(f"task:{task_id}", old, json.dumps(new, sort_keys=True)):
            raise ConfigError(f"task {task_id} bookkeeping changed concurrently")

    def _move(
        self,
        task: Task,
        to: str,
        kind: str = "normal",
        owner: str | None = None,
        lease_s: int | None = None,
        count_attempt: bool = False,
    ) -> None:
        self.machine.edge(task.state, to, kind)  # raises if the lifecycle has no such edge
        if not self.ops.update_task(task.id, [task.state], to, owner, lease_s, count_attempt):
            raise ConfigError(f"task {task.id} is no longer {task.state}")

    def annotate(self, task_id: str, **changes: Any) -> None:
        """Add bookkeeping the agent side reads: ``batch`` and ``arm`` (D372, D373)."""
        self._set_info(task_id, **changes)

    def get(self, task_id: str) -> Task:
        return self.ops.task(task_id)

    def card(self, task: Task) -> TaskCard:
        return TaskCard.model_validate(self.store.get(task.card_id).body)

    # ---- creation -------------------------------------------------------------

    def create(
        self,
        req: TaskRequest,
        loop_id: str | None,
        process: str,
        priority: int = 0,
        human: Actor | None = None,
        reason: str = "",
    ) -> str:
        """Write the card (created) and enqueue it (queued).

        Without a loop the task is its own work unit ``human:<id>`` (D278, D365). The card
        names the approved profile (pointer ``current_profile``), else ``unassigned``.
        """
        snap = self.controls()
        s = Settings.of(snap)
        loop_id = loop_id or f"human:{self.ids.new()}"
        profile = self.ops.get_pointer(CURRENT_PROFILE) or UNASSIGNED_PROFILE
        body = TaskCard.model_validate(
            {
                "type": req.type,
                "goal": req.goal,
                "inputs": list(req.inputs),
                "done_checks": list(DONE_CHECKS[req.type]),
                "budget": s.card_budget(req.type),
                "profile": profile,
                "work_unit": loop_id,
                **({"header": req.header.model_dump(mode="json")} if req.header else {}),
            }
        )
        refs: dict[str, RefValue] = {"inputs": sorted(set(req.inputs))} if req.inputs else {}
        if req.header and req.header.plan:
            refs["plan"] = req.header.plan  # the default FormulationPlan the header shows
        if snap.version_id:
            refs["controls"] = snap.version_id  # the version the card's budget came from
        actor = human or Actor(type="system", name=f"kta.orchestrator ({process})")
        card_id = self.store.put(Record.new("TaskCard", body, actor, self.ctx.clock.now(), refs))
        task_id = self.ops.enqueue(card_id, priority, loop_id)
        span_id = self.ids.new()
        self.ops.start_span(
            SpanStart(
                span_id,
                None,
                loop_id,
                "task",
                "process:" + process,
                f"{req.type} {req.goal}"[:80],
                self.ctx.clock.now(),
                {"task_id": task_id, "card": card_id, "type": req.type},
            )
        )
        self._set_info(task_id, span=span_id, process=process)
        if human is not None:
            self._human(self.get(task_id), "create", reason or req.goal, human)
        return task_id

    # ---- lease, start, submit -------------------------------------------------

    def lease(self, task_id: str, owner: str) -> Task:
        """queued -> leased, reserving the card budget; BudgetRefused keeps it queued."""
        task = self.get(task_id)
        amount = budget.scope_amounts(self.card(task).budget)
        info = self.info(task_id)
        process = str(info.get("process", "solve"))
        stop = control.paused_scope(self.ops, control.task_scopes(task.id, task.work_unit, process))
        if stop is not None:
            raise control.Paused(f"task {task_id}: {stop} is paused")
        scopes = budget.reserve_chain(
            self.ops, budget.ancestors(str(info.get("process", "solve")), task.work_unit), amount
        )
        try:
            self._move(
                task, "leased", owner=owner, lease_s=self.settings.lease_s, count_attempt=True
            )
        except ConfigError:
            budget.settle_chain(self.ops, scopes, amount, {})
            raise
        self._set_info(task_id, reserved=amount, scopes=scopes)
        return self.get(task_id)

    def start(self, task_id: str, owner: str) -> None:
        task = self.get(task_id)
        if task.lease_owner != owner:
            raise ConfigError(f"task {task_id} is leased by {task.lease_owner}, not {owner}")
        self._move(task, "running", owner=owner, lease_s=self.settings.lease_s)

    def submit(
        self, task_id: str, owner: str, files: Mapping[str, bytes], note: str, actor: Actor
    ) -> str:
        """running -> submitted: the files become blobs of a Submission record."""
        task = self.get(task_id)
        if task.state != "running" or task.lease_owner != owner:
            raise ConfigError(f"task {task_id} is {task.state} for {task.lease_owner}; not {owner}")
        blobs = {
            name: self.store.put_blob(data, "application/octet-stream")
            for name, data in sorted(files.items())
        }
        sub = Submission(files=sorted(files), note=note)
        rec = Record.new(
            "Submission",
            sub,
            actor.model_copy(update={"task": task_id}),
            self.ctx.clock.now(),
            refs={"card": task.card_id},
            blobs=blobs,
        )
        sub_id = self.store.put(rec)
        self._move(task, "submitted", owner=owner, lease_s=self.settings.lease_s)
        self._set_info(task_id, submission=sub_id)
        return sub_id

    # ---- checking -------------------------------------------------------------

    def _workdir(self, task_id: str, sub_id: str, submission: Record) -> Path:
        root = self.ctx.data_dir / "work" / "checks" / task_id / sub_id[:16]
        if root.exists():
            shutil.rmtree(root)
        for name in submission.body["files"]:
            target = (root / name).resolve()
            if root.resolve() not in target.parents:
                raise SchemaError(f"submission file {name!r} leaves the work directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.store.blob_path(submission.blobs[name]), target)
        return root

    def check(self, task_id: str) -> Verdict:
        """submitted -> checking -> accepted | rejected (-> queued | escalated) | suspended."""
        task = self.get(task_id)
        if task.state == "submitted":
            self._move(task, "checking")
        elif task.state != "checking":
            raise ConfigError(f"task {task_id} is {task.state}; only submitted tasks are checked")
        task = self.get(task_id)
        card = self.card(task)
        sub_id = str(self.info(task_id)["submission"])
        sub = self.store.get(sub_id)
        inp = CheckInput(
            self.svc,
            self.ctx,
            self.store,
            task_id,
            task.card_id,
            card,
            sub_id,
            sub,
            self._workdir(task_id, sub_id, sub),
        )
        outcomes: list[Outcome] = []
        digest = submission_digest(sub)
        rejected = list(self.info(task_id).get("rejected_digests", []))
        if self.settings.duplicate_rejection and digest in rejected:
            # 12 §3: identical (by hash) to a rejected attempt; no check is run again.
            outcomes.append(Outcome("duplicate_proposal", False, [f"same as rejected {digest}"]))
            return self._reject(task, sub_id, outcomes, "duplicate of a rejected submission")
        try:
            for name in card.done_checks:
                outcomes.append(run_check(name, inp))
        except CHECKER_FAILURES as exc:
            report = self._report(task, sub_id, outcomes, f"checker error: {exc}")
            self._move(task, "suspended")
            self._set_info(task_id, suspended=str(exc))
            return Verdict(task_id, "suspended", report, outcomes)
        passed = all(o.passed for o in outcomes)
        if passed:
            # Writing into the pool is the last check: a refused record rejects the task.
            try:
                write_outputs(inp, Actor.model_validate(sub.actor.model_dump()))
            except ValueError as exc:  # SchemaError or pydantic validation of the outputs
                outcomes.append(Outcome("outputs_written", False, [f"{exc}"]))
                passed = False
        if passed:
            report = self._report(task, sub_id, outcomes, "passed")
            self._move(task, "accepted")
            self._finish(task_id, "accepted")
            return Verdict(task_id, "accepted", report, outcomes)
        self._set_info(task_id, rejected_digests=[*rejected, digest])
        return self._reject(task, sub_id, outcomes, "failed")

    def _reject(self, task: Task, sub_id: str, outcomes: list[Outcome], summary: str) -> Verdict:
        report = self._report(task, sub_id, outcomes, summary)
        self._move(task, "rejected")
        self._set_info(task.id, last_rejection=report)
        return Verdict(task.id, self._after_rejection(task.id), report, outcomes)

    def _report(self, task: Task, sub_id: str, outcomes: list[Outcome], summary: str) -> str:
        data = {
            "task": task.id,
            "passed": all(o.passed for o in outcomes)
            and bool(outcomes)
            and not summary.startswith("checker"),
            "checks": [o.to_dict() for o in outcomes],
        }
        body = {"title": "done_checks", "subject": task.card_id, "summary": summary, "data": data}
        actor = Actor(type="checker", name="kta.orchestrator.done_checks", task=task.id)
        refs: dict[str, RefValue] = {"card": task.card_id, "submission": sub_id}
        return self.store.put(Record.new("Report", body, actor, self.ctx.clock.now(), refs))

    def _after_rejection(self, task_id: str) -> str:
        """rejected -> queued (retry) or escalated (over the retry limit)."""
        task = self.get(task_id)
        self._settle(task_id)
        if task.attempts > self.settings.max_retries:
            self._move(task, "escalated")
            return "escalated"
        self._move(task, "queued", kind="retry")
        return "queued"

    def recheck(self, task_id: str, reason: str, human: Actor | None = None) -> Verdict:
        """suspended -> checking after a checker was fixed (human)."""
        task = self.get(task_id)
        self._human(task, "recheck", reason, human)
        self._move(task, "checking", kind="human")
        return self.check(task_id)

    # ---- human actions and expiry ---------------------------------------------

    def _human(self, task: Task, action: str, reason: str, human: Actor | None = None) -> str:
        """The HumanAction naming the operator (the API passes its Actor, D365)."""
        body = {"action": f"task {action} {task.id}", "reason": reason}
        who = human or Actor(type="human", name=f"kta task {action}")
        if who.type != "human":
            raise PermissionDenied(f"task {action} needs a human actor, not {who.type}")
        actor = who.model_copy(update={"task": task.id})
        return self.store.put(
            Record.new(
                "HumanAction", body, actor, self.ctx.clock.now(), refs={"card": task.card_id}
            )
        )

    def retry(
        self, task_id: str, reason: str, human: Actor | None = None, note: str | None = None
    ) -> str:
        """escalated -> queued; ``note`` reaches the next TASK.md verbatim (D365)."""
        task = self.get(task_id)
        action = self._human(task, "retry", reason, human)
        self._move(task, "queued", kind="human")
        if note:
            self._set_info(task_id, human_note=note)
        return action

    def set_priority(
        self, task_id: str, priority: int, reason: str, human: Actor | None = None
    ) -> str:
        """Change the priority of a task that is not finished (D367)."""
        task = self.get(task_id)
        if self.machine.terminal(task.state):
            raise ConfigError(f"task {task_id} is already {task.state}")
        action = self._human(task, f"priority {priority}", reason, human)
        if not self.ops.set_task_priority(task_id, priority):
            raise ConfigError(f"no task {task_id}")
        return action

    def retry_context(self, task_id: str) -> tuple[list[tuple[str, str]], str | None]:
        """For the next agent step: the failures of the last rejection (done check, text)
        and a human note, if any. The note is handed over once."""
        info = self.info(task_id)
        failures: list[tuple[str, str]] = []
        last = info.get("last_rejection")
        if last:
            for c in self.store.get(str(last)).body["data"].get("checks", []):
                if not c["passed"]:
                    failures += [(str(c["name"]), str(f)) for f in c["failures"]] or [
                        (str(c["name"]), "failed")
                    ]
        note = info.get("human_note")
        if note:
            self._set_info(task_id, human_note=None)
        return failures, str(note) if note else None

    def cancel(self, task_id: str, reason: str, human: Actor | None = None) -> str:
        task = self.get(task_id)
        if self.machine.terminal(task.state):
            raise ConfigError(f"task {task_id} is already {task.state}")
        action = self._human(task, "cancel", reason, human)
        self._move(task, "cancelled", kind="any")
        self._settle(task_id)
        self._finish(task_id, "cancelled")
        return action

    def expire(self, now_tasks: Collection[Task] | None = None) -> list[str]:
        """Lease lost: leased -> queued; running or submitted-but-unchecked stays (11 §6).

        Running tasks whose lease ran out expire and are handled as rejected.
        """
        now = self.ctx.clock.now()
        moved: list[str] = []
        for t in (
            now_tasks
            if now_tasks is not None
            else self.ops.list_tasks(states=("leased", "running"))
        ):
            if t.lease_until is None or t.lease_until >= now:
                continue
            if t.state == "leased":
                self._settle(t.id)
                self._move(t, "queued")
            else:
                self._move(t, "expired")
                self._move(self.get(t.id), "rejected")
                self._after_rejection(t.id)
            moved.append(t.id)
        return moved

    def abandon(self, task_id: str, reason: str) -> str:
        """An agent step ended without submitting: running -> expired -> rejected, then
        retry or escalation as for any rejection (docs/design/10 §9, D369)."""
        task = self.get(task_id)
        self._move(task, "expired")
        self._move(self.get(task_id), "rejected")
        self._set_info(task_id, last_abandon=reason)
        return self._after_rejection(task_id)

    def _settle(self, task_id: str) -> None:
        info = self.info(task_id)
        if not info.get("scopes"):
            return
        # Human and CLI submissions run no agent, so they book no agent time (D268).
        budget.settle_chain(self.ops, info["scopes"], info["reserved"], {})
        self._set_info(task_id, scopes=[], reserved={})

    def _finish(self, task_id: str, state: str) -> None:
        self._settle(task_id)
        span = self.info(task_id).get("span")
        if span:
            status = "ok" if state == "accepted" else "cancelled"
            self.ops.end_span(str(span), self.ctx.clock.now(), status, {"state": state})
