"""Operation endpoints (spec/modules/api-web.md "操作"): POST /actions/*.

Every operation writes a HumanAction (operate.py). ``reason`` is required for reject,
deprecate and budget (design/30 §3); a missing or blank one is a 400, as is any other
malformed body (app.py maps request validation errors to 400).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from kta.api import operate
from kta.api.common import Api, iso
from kta.api.wiring import HUMAN_PRIORITY, NEEDS, TaskSpec
from kta.core.ids import record_id

Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Id = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopeBody(_Body):
    scope: Id
    reason: str = ""


class TaskBody(_Body):
    task: Id
    reason: str = ""


class RetryBody(_Body):
    task: Id
    note: Reason


class BudgetBody(_Body):
    scope: Id
    limit: dict[str, float]
    reason: Reason


class PriorityBody(_Body):
    task: Id
    priority: int
    reason: str = ""


class CreateTaskBody(_Body):
    type: Literal["T1", "T2", "T3", "T4", "T5", "T6"]
    goal: Reason
    inputs: list[Id]
    priority: int | None = None  # default HUMAN_PRIORITY: tasks you start come first
    reason: str = ""


class RecordReasonBody(_Body):
    record: Id
    reason: Reason


class FlagBody(_Body):
    record: Id
    reason: str = ""


class _OneOf(_Body):
    """Exactly one of the subject fields is given."""

    reason: str = ""

    def subject(self) -> tuple[str, str]:
        given = [(k, v) for k, v in self.model_dump().items() if k != "reason" and v]
        return given[0]

    @model_validator(mode="after")
    def _exactly_one(self) -> _OneOf:
        given = [k for k, v in self.model_dump().items() if k != "reason" and v]
        if len(given) != 1:
            raise ValueError(f"give exactly one subject, got {given or 'none'}")
        return self


class ApproveBody(_OneOf):
    protocol: str | None = None
    improvement: str | None = None
    profile: str | None = None


class ConfirmBody(_OneOf):
    pattern: str | None = None
    structure: str | None = None


# Operation -> (endpoint, records written, delegated operation or None). The frontend reads
# it from GET /actions/capabilities to label and enable its controls.
CATALOGUE: list[tuple[str, str, str, str | None]] = [
    ("reject", "/actions/reject", "HumanAction, StatusChange (rejected; dependents stale)", None),
    ("flag", "/actions/flag", "HumanAction, StatusChange (suspect), re-verification task", None),
    ("deprecate", "/actions/deprecate", "HumanAction, StatusChange (deprecated)", None),
    ("approve-protocol", "/actions/approve", "HumanAction (protocol frozen)", None),
    ("approve-improvement", "/actions/approve", "HumanAction", None),
    ("approve-profile", "/actions/approve", "HumanAction, profile pointer", "approve-profile"),
    ("confirm", "/actions/confirm", "HumanAction", None),
    ("budget", "/actions/budget", "HumanAction, budget limit", None),
    ("cancel", "/actions/cancel", "HumanAction, task cancelled", "cancel"),
    ("retry", "/actions/retry", "HumanAction, task queued again", "retry"),
    ("create-task", "/actions/create-task", "HumanAction, TaskCard, queued task", "create-task"),
    ("priority", "/actions/priority", "HumanAction, task priority", "priority"),
    ("pause", "/actions/pause", "HumanAction", "pause"),
    ("resume", "/actions/resume", "HumanAction", "resume"),
    ("env-verify", "/agents/batches/{batch}/verify", "nothing (a lock check)", "env-verify"),
    ("write-record", "/write/record", "the agent's record, as its task", "write-record"),
]  # fmt: skip


def capabilities(api: Api) -> list[dict[str, Any]]:
    missing, partial = api.orchestration.missing(), api.orchestration.partial()
    out: list[dict[str, Any]] = []
    for name, endpoint, writes, delegated in CATALOGUE:
        gap = missing.get(delegated) if delegated else None
        row: dict[str, Any] = {
            "name": name, "endpoint": endpoint, "writes": writes, "wired": gap is None, "needs": gap
        }  # fmt: skip
        if delegated in partial:
            row["partial"] = partial[delegated]
        out.append(row)
    if "reverify" in missing:
        out[1]["partial"] = f"the re-verification task is not created yet: {NEEDS['reverify']}"
    return out


def action_log(api: Api, limit: int) -> list[dict[str, Any]]:
    """HumanAction records, newest first, with the verb and parameters split out."""
    out: list[dict[str, Any]] = []
    for rec in api.store.query("HumanAction", order="-created_at", limit=limit):
        data = rec.to_json_dict()
        out.append(
            {
                "id": record_id(data),
                "at": iso(rec.created_at),
                "by": rec.actor.name,
                "actor_type": rec.actor.type,
                "task": rec.actor.task,
                "action": rec.body["action"],
                "reason": rec.body["reason"],
                "refs": rec.refs,
                "origin": rec.origin,
                **operate.parse_action(rec.body["action"]),
            }
        )
    return out


def router(api: Api) -> APIRouter:
    r = APIRouter(prefix="/actions")

    @r.get("/capabilities")
    def get_capabilities() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return capabilities(api)

    @r.get("/log")
    def get_log(limit: int = Query(100, le=1000)) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return action_log(api, limit)

    @r.post("/pause")
    def pause(b: ScopeBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.pause(api, b.scope, b.reason, resume=False)

    @r.post("/resume")
    def resume(b: ScopeBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.pause(api, b.scope, b.reason, resume=True)

    @r.post("/cancel")
    def cancel(b: TaskBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.cancel(api, b.task, b.reason)

    @r.post("/retry")
    def retry(b: RetryBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.retry(api, b.task, b.note)

    @r.post("/budget")
    def budget(b: BudgetBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.set_budget(api, b.scope, b.limit, b.reason)

    @r.post("/priority")
    def priority(b: PriorityBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.priority(api, b.task, b.priority, b.reason)

    @r.post("/create-task")
    def create_task(b: CreateTaskBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        prio = HUMAN_PRIORITY if b.priority is None else b.priority
        spec = TaskSpec(b.type, b.goal, list(b.inputs), prio)
        return operate.create_task(api, spec, b.reason)

    @r.post("/reject")
    def reject(b: RecordReasonBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.reject(api, b.record, b.reason)

    @r.post("/deprecate")
    def deprecate(b: RecordReasonBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.deprecate(api, b.record, b.reason)

    @r.post("/flag")
    def flag(b: FlagBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return operate.flag(api, b.record, b.reason)

    @r.post("/approve")
    def approve(b: ApproveBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        what, record = b.subject()
        return operate.approve(api, what, record, b.reason)

    @r.post("/confirm")
    def confirm(b: ConfirmBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        what, record = b.subject()
        return operate.confirm(api, what, record, b.reason)

    return r
