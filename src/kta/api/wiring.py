"""The operations the API delegates to the orchestrator (WP6) and the agents module (WP8).

The API writes records and ops rows itself only where the store alone defines the effect
(reject, flag, deprecate, confirm, approve a protocol or improvement, budget limits). Task
control, pausing, priorities, re-verification, the profile pointer, batch environments and
task tokens belong to WP6 and WP8. ``Orchestration`` names each such call; ``Unwired`` is
the implementation used until those modules are merged into this branch: every call
raises ``NotSupported`` naming the missing piece, so an endpoint never reports an effect
that did not happen (decision D341). ``missing`` lets the frontend disable the controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from kta.core.errors import NotSupported
from kta.core.schemas import Actor


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


# What each operation waits for, by owner and function (spec/modules/orchestrator-cli.md,
# spec/modules/agents.md). Shown verbatim by GET /actions/capabilities.
NEEDS: dict[str, str] = {
    "cancel": "WP6 TaskQueue.cancel and the runner stopping the agent process",
    "retry": "WP6 TaskQueue.retry with the note passed to the next TASK.md",
    "create-task": "WP6 TaskQueue.create for a human-created task (work unit and process)",
    "priority": "a priority update on the ops task row (WP1 Ops / WP6 TaskQueue)",
    "pause": "WP6 pause semantics for the global, loop and task scopes",
    "resume": "WP6 pause semantics for the global, loop and task scopes",
    "reverify": "WP6 creating a re-verification task for a suspect record",
    "approve-profile": "WP8 kta.agents.profile.approve_profile (current_profile pointer)",
    "env-verify": "WP8 `kta agents env verify` (D322)",
    "write-record": "WP8 task tokens (kta.agents.guard.verify_task_token, D305)",
}


class Unwired:
    """Every delegated operation is unavailable until WP6/WP8 are merged (D341)."""

    def missing(self) -> dict[str, str]:
        return dict(NEEDS)

    def _no(self, name: str) -> NotSupported:
        return NotSupported(f"{name}: not wired yet; needs {NEEDS[name]}")

    def cancel_task(self, task_id: str, reason: str, human: Actor) -> dict[str, Any]:
        raise self._no("cancel")

    def retry_task(self, task_id: str, note: str, human: Actor) -> dict[str, Any]:
        raise self._no("retry")

    def create_task(self, spec: TaskSpec, reason: str, human: Actor) -> dict[str, Any]:
        raise self._no("create-task")

    def set_priority(
        self, task_id: str, priority: int, reason: str, human: Actor
    ) -> dict[str, Any]:
        raise self._no("priority")

    def pause(self, scope: str, reason: str, human: Actor) -> dict[str, Any]:
        raise self._no("pause")

    def resume(self, scope: str, reason: str, human: Actor) -> dict[str, Any]:
        raise self._no("resume")

    def request_reverify(self, record: str, reason: str, human: Actor) -> dict[str, Any]:
        raise self._no("reverify")

    def approve_profile(self, profile: str, reason: str, human: Actor) -> dict[str, Any]:
        raise self._no("approve-profile")

    def verify_batch(self, batch: str) -> dict[str, Any]:
        raise self._no("env-verify")

    def verify_task_token(self, token: str, now: datetime) -> str:
        raise self._no("write-record")
