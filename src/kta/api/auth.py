"""POST /write/record: records written through the API by an agent's ``kta`` command.

Only a request carrying a valid task token (``Authorization: Bearer kta1....``, issued by
WP8's guard while the task is live) may write, and only as that task: the record's actor
must name the token's task. The token check is delegated (wiring.py) and raises
NotSupported until WP8's guard is merged, so nothing is written before then.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header

from kta.api.common import Api
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.schemas import Record

WRITER_TYPES = ("agent", "tool")


def bearer(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise PermissionDenied("token endpoints need Authorization: Bearer <task token>")
    return authorization.removeprefix("Bearer ").strip()


def write_record(api: Api, token: str, data: dict[str, Any]) -> str:
    task_id = api.orchestration.verify_task_token(token, api.now())
    rec = Record.model_validate(data)
    if rec.actor.task != task_id:
        raise PermissionDenied(f"the record's actor.task must be the token's task {task_id}")
    if rec.actor.type not in WRITER_TYPES:
        raise PermissionDenied(f"a task may write records as {WRITER_TYPES}, not {rec.actor.type}")
    if rec.kind in ("HumanAction", "StatusChange"):
        raise PermissionDenied(f"a task may not write {rec.kind} records")
    if rec.origin != "real":
        raise ConfigError("records written through the API have origin real")
    return api.store.put(rec)


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.post("/write/record")
    def post_record(  # pyright: ignore[reportUnusedFunction]
        record: dict[str, Any], authorization: str | None = Header(None)
    ) -> dict[str, str]:
        return {"id": write_record(api, bearer(authorization), record)}

    return r
