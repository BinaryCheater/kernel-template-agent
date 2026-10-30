"""Sessions: one per work unit; when to resume and when to start fresh.

The session state of a work unit is an ops pointer ``agent-session/<work unit>`` holding
JSON (decision D306): it is operational state that the AgentRun records can rebuild,
not evidence. Rules (spec/modules/agents.md "会话", docs/design/10 §4.2), in order:

1. no session yet, or the last one is closed → new;
2. the adapter cannot resume → new;
3. the last step ended abnormally (no session id, start failure, wall, idle, guard) → new;
4. ``resume_scope`` is ``never`` (X3's fresh arm) → new;
5. context above ``reset_context_ratio`` of the window → new;
6. the same done check failed ``same_failure_rule`` times in a row for the same reason
   → new ("fresh view");
7. otherwise resume.

The policy values come from the controls registry (``kta.agents.policy.loop_policy``).

A new session's prompt asks the worker to read NOTES.md first (message ``new_session``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from kta.core.errors import ConfigError
from kta.store.base import Ops

POINTER_PREFIX = "agent-session/"
ABNORMAL_EXITS = frozenset({"start_failed", "wall", "idle", "gpu_guard", "refused", "infra"})
SET_ATTEMPTS = 5


@dataclass(frozen=True)
class SessionState:
    session: str | None = None
    worker: str | None = None  # the agent host the work unit is pinned to
    batch: str | None = None  # the batch root the work unit lives in (D322)
    status: Literal["active", "closed"] = "active"
    runs: list[str] = field(default_factory=lambda: list[str]())  # AgentRun record IDs
    last_exit: str | None = None
    context_ratio: float | None = None
    failures: list[str] = field(default_factory=lambda: list[str]())  # rejection signatures

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> SessionState:
        return cls(**json.loads(raw))


@dataclass(frozen=True)
class Decision:
    mode: Literal["start", "resume"]
    reason: str


def pointer(work_unit: str) -> str:
    return POINTER_PREFIX + work_unit


def load(ops: Ops, work_unit: str) -> tuple[SessionState | None, str | None]:
    raw = ops.get_pointer(pointer(work_unit))
    return (SessionState.from_json(raw) if raw else None), raw


def save(ops: Ops, work_unit: str, expected: str | None, state: SessionState) -> None:
    if not ops.set_pointer(pointer(work_unit), expected, state.to_json()):
        raise ConfigError(
            f"session state of {work_unit} changed concurrently (one writer per work unit)"
        )


def decide(
    state: SessionState | None,
    supports_resume: bool,
    loop_policy: dict[str, Any],
    failure: str | None = None,
) -> Decision:
    if state is None or state.session is None or state.status == "closed":
        return Decision("start", "new work unit" if state is None else "no open session")
    if not supports_resume:
        return Decision("start", "adapter cannot resume")
    if state.last_exit in ABNORMAL_EXITS:
        return Decision("start", f"last step ended abnormally ({state.last_exit})")
    if loop_policy["resume_scope"] == "never":
        return Decision("start", "resume scope is never: a fresh session every step")
    ratio = float(loop_policy["reset_context_ratio"])
    if state.context_ratio is not None and state.context_ratio > ratio:
        return Decision(
            "start", f"context at {state.context_ratio:.0%} of the window (> {ratio:.0%})"
        )
    n = int(loop_policy["same_failure_rule"])
    recent = state.failures[max(0, len(state.failures) - (n - 1)) :]
    if failure is not None and recent == [failure] * (n - 1):
        return Decision("start", f"same check failed {n} times in a row for the same reason")
    return Decision("resume", "continue the work unit's session")


def after_run(
    state: SessionState | None,
    *,
    session: str,
    worker: str,
    run_id: str,
    batch: str | None = None,
    exit: str,
    context_ratio: float | None,
    failure: str | None,
    started_new: bool,
) -> SessionState:
    prev = state or SessionState()
    failures = [] if started_new else list(prev.failures)
    if failure is not None:
        failures.append(failure)
    return SessionState(
        session=session or None,
        worker=worker,
        batch=batch or prev.batch,
        status="active",
        runs=[*prev.runs, run_id],
        last_exit=exit,
        context_ratio=context_ratio,
        failures=failures[-5:],
    )


def closed(state: SessionState) -> SessionState:
    return SessionState(**{**asdict(state), "status": "closed"})
