"""The settings page's endpoints (WP21, D486): the controls registry and the release gate.

Reading is open. Writing needs the operator token (header ``X-KTA-Operator-Token``, the
content of ``[api] operator_token_file``; ``kta controls token`` creates it): a controls
version (``POST /controls``) and the release of one held agent call (``POST /release``).
Each write names the version the author read, carries a reason, and writes a HumanAction
next to its own record, so it shows in the action log. Nothing else is editable here.
"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, ConfigDict, StringConstraints

from kta.api import operate
from kta.api.common import Api, controls, iso
from kta.core.controls import definitions
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.schemas.process import TaskCard
from kta.orchestrator import release
from kta.store import controls as controls_store

TOKEN_HEADER = "X-KTA-Operator-Token"
DEFAULT_TOKEN_FILE = "~/.config/kta/operator-token"
AGENT_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# What a held call's reader is shown besides the card; both arrive with WP19.
PULL_LOG_NOTE = "pull logs are recorded from WP19 (access discipline); none exist yet"
PLAN_NOTE = "FormulationPlan records arrive with WP19; none exist yet"


class ControlsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected: str | None  # the version the author read (null: none stored yet)
    changes: dict[str, Any]
    reason: Reason


class ReleaseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Reason
    reason: Reason


def token_file(api: Api) -> Path:
    raw = api.config.section("api").get("operator_token_file", DEFAULT_TOKEN_FILE)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError("[api] operator_token_file must be a path")
    return Path(raw).expanduser()


def check_operator(api: Api, token: str | None) -> None:
    path = token_file(api)
    if not path.is_file():
        raise PermissionDenied(f"no operator token at {path}: create it with `kta controls token`")
    expected = path.read_text(encoding="utf-8").strip()
    if not expected or not token or not hmac.compare_digest(expected, token.strip()):
        raise PermissionDenied(f"writing needs the operator token in {TOKEN_HEADER}")


def controls_view(api: Api) -> dict[str, Any]:
    """GET /controls: every knob with its definition and the value in force."""
    c = controls(api)
    defs = definitions()
    return {
        "version": {
            "id": c.version_id,
            "version": c.version,
            "stored": c.version_id is not None,
            "definitions_hash": defs.hash,
            "stored_under_hash": c.definitions_hash,
        },
        "knobs": [{**k.to_dict(), "value": c.value(k.id), "initial": k.value} for k in defs.knobs],
        "experiments": [e.to_dict() for e in defs.experiments],
        "write": {"header": TOKEN_HEADER, "token_configured": token_file(api).is_file()},
    }


def versions(api: Api, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "id": rid,
            "version": rec.body["version"],
            "at": iso(rec.created_at),
            "author": rec.actor.name,
            "author_type": rec.actor.type,
            "reason": rec.body["reason"],
            "changes": rec.body["changes"],
            "previous": rec.refs.get("previous"),
        }
        for rid, rec in controls_store.history(api.store, limit)
    ]


def write_controls(api: Api, b: ControlsBody) -> dict[str, Any]:
    human = operate.operator(api)
    snap = controls_store.propose(
        api.store, api.ops, b.changes, human, b.reason, api.now(), b.expected
    )
    assert snap.version_id is not None
    action = operate.write_action(
        api, "set_controls", b.reason, {"controls": snap.version_id}, {"version": snap.version}
    )
    return {"id": snap.version_id, "version": snap.version, "action": action}


def _held_detail(api: Api, row: dict[str, Any]) -> dict[str, Any]:
    card = TaskCard.model_validate(api.store.get(str(row["card"])).body)
    return {
        **row,
        "card_body": card.model_dump(mode="json"),
        "pull_log": None,
        "pull_log_note": PULL_LOG_NOTE,
        "plan": None,
        "plan_note": PLAN_NOTE,
    }


def release_view(api: Api) -> dict[str, Any]:
    """GET /release: the stage in force, every stage with its exit criteria, held calls."""
    c = controls(api)
    stage = release.stage_of(c)
    in_flight = [t.id for t in api.ops.list_tasks(states=release.IN_FLIGHT)]
    held = [_held_detail(api, r) for r in release.held_calls(api.ops, c, AGENT_TYPES)]
    return {
        "stage": stage.to_dict(),
        "stages": [s.to_dict() for s in release.stages()],
        "controls": c.version_id,
        "in_flight": in_flight,
        "held": held,
    }


def write_release(api: Api, b: ReleaseBody) -> dict[str, Any]:
    task = api.ops.task(b.task)
    stage = release.stage_of(controls(api))
    if stage.agent_calls != "gated":
        raise ConfigError(f"stage {stage.id} does not gate agent calls; nothing to release")
    action = release.release_call(
        api.store, api.ops, task, operate.operator(api), b.reason, api.now()
    )
    return {"task": task.id, "attempt": task.attempts + 1, "action": action}


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/controls")
    def get_controls() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return controls_view(api)

    @r.get("/controls/versions")
    def get_versions(limit: int = Query(100, le=1000)) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return versions(api, limit)

    @r.post("/controls")
    def post_controls(  # pyright: ignore[reportUnusedFunction]
        b: ControlsBody, token: str | None = Header(None, alias=TOKEN_HEADER)
    ) -> dict[str, Any]:
        check_operator(api, token)
        return write_controls(api, b)

    @r.get("/release")
    def get_release() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return release_view(api)

    @r.post("/release")
    def post_release(  # pyright: ignore[reportUnusedFunction]
        b: ReleaseBody, token: str | None = Header(None, alias=TOKEN_HEADER)
    ) -> dict[str, Any]:
        check_operator(api, token)
        return write_release(api, b)

    return r
