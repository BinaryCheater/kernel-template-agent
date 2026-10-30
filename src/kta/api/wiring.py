"""The operations the API delegates to the orchestrator (WP6) and the agents module (WP8).

The API writes records and ops rows itself only where the store alone defines the effect
(reject, flag, deprecate, confirm, approve a protocol or improvement, budget limits). Task
control, pausing, priorities, re-verification, the inbox, the profile pointer, batch
environments and task tokens belong to WP6 and WP8. ``Orchestration`` names each such call.

- ``Wired`` calls the merged WP6/WP8 functions (decision D355). The calls WP6 does not offer
  yet (priority, pause, resume, re-verification) still raise NotSupported and are listed by
  ``missing``; ``partial`` names what a wired call does not do yet.
- ``Unwired`` raises NotSupported for everything (D341); it is the default of a bare ``Api``
  in tests. An endpoint never reports an effect that did not happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from kta.agents import batches, guard
from kta.agents.config import AgentsConfig
from kta.agents.profile import approve_profile
from kta.core.context import Context
from kta.core.errors import NotSupported, PermissionDenied
from kta.core.schemas import Actor
from kta.orchestrator.inbox import inbox
from kta.orchestrator.pipeline import Services
from kta.orchestrator.tasks import TaskQueue
from kta.orchestrator.transition import TaskRequest
from kta.store import SqliteOps, SqliteStore


@dataclass(frozen=True)
class TaskSpec:
    """POST /actions/create-task, validated (inputs exist in the store)."""

    type: str
    goal: str
    inputs: list[str]
    priority: int


class Orchestration(Protocol):
    def missing(self) -> dict[str, str]:
        """Operation name -> what is missing; empty when every operation is wired."""
        ...

    def partial(self) -> dict[str, str]:
        """Operation name -> what a wired operation does not do yet."""
        ...

    def cancel_task(self, task_id: str, reason: str, human: Actor) -> dict[str, Any]: ...
    def retry_task(self, task_id: str, note: str, human: Actor) -> dict[str, Any]: ...
    def create_task(self, spec: TaskSpec, reason: str, human: Actor) -> dict[str, Any]: ...
    def set_priority(
        self, task_id: str, priority: int, reason: str, human: Actor
    ) -> dict[str, Any]: ...
    def pause(self, scope: str, reason: str, human: Actor) -> dict[str, Any]: ...
    def resume(self, scope: str, reason: str, human: Actor) -> dict[str, Any]: ...
    def request_reverify(self, record: str, reason: str, human: Actor) -> dict[str, Any]: ...
    def approve_profile(self, profile: str, reason: str, human: Actor) -> dict[str, Any]: ...
    def verify_batch(self, batch: str) -> dict[str, Any]: ...
    def verify_task_token(self, token: str, now: datetime) -> str: ...
    def inbox(self) -> list[dict[str, Any]]: ...


# What each operation waits for, by owner and function (spec/modules/orchestrator-cli.md,
# spec/modules/agents.md). Shown verbatim by GET /actions/capabilities.
NEEDS: dict[str, str] = {
    "cancel": "WP6 TaskQueue.cancel",
    "retry": "WP6 TaskQueue.retry",
    "create-task": "WP6 TaskQueue.create",
    "priority": "WP6 a priority update on a queued task (TaskQueue)",
    "pause": "WP6 pause semantics for the global, process, loop and task scopes",
    "resume": "WP6 pause semantics for the global, process, loop and task scopes",
    "reverify": "WP6 creating a re-verification task for a suspect record",
    "approve-profile": "WP8 kta.agents.profile.approve_profile (current_profile pointer)",
    "env-verify": "WP8 kta.agents.batches.verify (D322)",
    "write-record": "WP8 task tokens (kta.agents.guard.verify_task_token, D305)",
    "inbox": "WP6 kta.orchestrator.inbox.inbox",
}
# Still raising NotSupported after the WP6/WP8 code-stage merge (528caab); the WP6 session is
# adding them (human actors, priority, pause/resume, re-verification, agent stop).
NOT_YET = ("priority", "pause", "resume", "reverify")
PARTIAL: dict[str, str] = {
    "cancel": "the running agent process is not stopped yet, and WP6 names the HumanAction "
    "actor 'kta task cancel' instead of the operator (WP6 follow-up)",
    "retry": "the note is stored in the HumanAction but not yet passed to the next TASK.md, "
    "and WP6 names the actor 'kta task retry' (WP6 follow-up)",
}
HUMAN_PRIORITY = 100  # tasks you start come first (docs/design/11 §4.2; as `kta task create`)
DEFAULT_TOKEN_KEY = AgentsConfig.__dataclass_fields__["token_key_file"].default


def _no(name: str) -> NotSupported:
    return NotSupported(f"{name}: not wired yet; needs {NEEDS[name]}")


class Unwired:
    """Every delegated operation is unavailable (D341)."""

    def missing(self) -> dict[str, str]:
        return dict(NEEDS)

    def partial(self) -> dict[str, str]:
        return {}

    def cancel_task(self, task_id: str, reason: str, human: Actor) -> dict[str, Any]:
        raise _no("cancel")

    def retry_task(self, task_id: str, note: str, human: Actor) -> dict[str, Any]:
        raise _no("retry")

    def create_task(self, spec: TaskSpec, reason: str, human: Actor) -> dict[str, Any]:
        raise _no("create-task")

    def set_priority(
        self, task_id: str, priority: int, reason: str, human: Actor
    ) -> dict[str, Any]:
        raise _no("priority")

    def pause(self, scope: str, reason: str, human: Actor) -> dict[str, Any]:
        raise _no("pause")

    def resume(self, scope: str, reason: str, human: Actor) -> dict[str, Any]:
        raise _no("resume")

    def request_reverify(self, record: str, reason: str, human: Actor) -> dict[str, Any]:
        raise _no("reverify")

    def approve_profile(self, profile: str, reason: str, human: Actor) -> dict[str, Any]:
        raise _no("approve-profile")

    def verify_batch(self, batch: str) -> dict[str, Any]:
        raise _no("env-verify")

    def verify_task_token(self, token: str, now: datetime) -> str:
        raise _no("write-record")

    def inbox(self) -> list[dict[str, Any]]:
        raise _no("inbox")


class Wired(Unwired):
    """WP6's task queue and inbox, WP8's profile pointer, batch verify and task tokens."""

    def __init__(self, ctx: Context, store: SqliteStore, ops: SqliteOps) -> None:
        self.ctx, self.store, self.ops = ctx, store, ops
        self.svc = Services(ctx, store, ops)
        self.queue = TaskQueue(ctx, store, ops, self.svc)

    def missing(self) -> dict[str, str]:
        return {k: NEEDS[k] for k in NOT_YET}

    def partial(self) -> dict[str, str]:
        return dict(PARTIAL)

    def _task(self, task_id: str) -> dict[str, Any]:
        t = self.queue.get(task_id)
        return {"task": t.id, "state": t.state, "card": t.card_id}

    def cancel_task(self, task_id: str, reason: str, human: Actor) -> dict[str, Any]:
        self.queue.cancel(task_id, reason)
        return self._task(task_id)

    def retry_task(self, task_id: str, note: str, human: Actor) -> dict[str, Any]:
        self.queue.retry(task_id, note)
        return self._task(task_id)

    def create_task(self, spec: TaskSpec, reason: str, human: Actor) -> dict[str, Any]:
        # A task you start has its own work unit (as `kta task create` without --loop).
        unit = f"human:{self.queue.ids.new()}"
        req = TaskRequest(spec.type, spec.goal, list(spec.inputs))
        task_id = self.queue.create(req, unit, "human", priority=spec.priority)
        return {**self._task(task_id), "work_unit": unit}

    def approve_profile(self, profile: str, reason: str, human: Actor) -> dict[str, Any]:
        action = approve_profile(
            self.store, self.ops, profile, human.name, reason, self.ctx.clock.now()
        )
        return {"action": action, "current": self.ops.get_pointer("current_profile")}

    def verify_batch(self, batch: str) -> dict[str, Any]:
        batches.check_name(batch)
        result = batches.verify(self.store, batches.batches_root(self.ctx.data_dir) / batch)
        return {
            "batch": batch,
            "state": "ok" if result.ok else "failed",
            "problems": result.problems,
        }

    def verify_task_token(self, token: str, now: datetime) -> str:
        raw = self.ctx.config.section("agents").get("token_key_file", DEFAULT_TOKEN_KEY)
        path = Path(str(raw)).expanduser()
        if not path.exists():
            # No key on this host means no token was ever issued here; never create one.
            raise PermissionDenied(f"no task token key at {path}; no task token can be valid")
        return guard.verify_task_token(guard.ensure_key(path), token, now, self.ops)

    def inbox(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in inbox(self.svc)]
