"""GET /events: server-sent events for ops table changes, span starts/ends and new records.

The stream polls the ops_events log (WP1 writes one row per ops change, spans included) and
the records table. Each event carries its cursor so a reconnecting client resumes with
``?since=<ops seq>&records=<rowid>``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from kta.api import opsread
from kta.api.common import Api, iso

POLL_S = 1.0
KEEPALIVE_POLLS = 15


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream(api: Api, request: Request, since: int, records: int | None) -> AsyncIterator[str]:
    seq = since
    rowid = opsread.last_record_rowid(api.store) if records is None else records
    idle = 0
    yield _sse("hello", {"seq": seq, "records": rowid, "now": iso(api.now())})
    while not await request.is_disconnected():
        sent = False
        for ev in api.ops.events(seq):
            seq = ev.seq
            sent = True
            yield _sse(
                "ops",
                {
                    "seq": ev.seq,
                    "at": iso(ev.at),
                    "table": ev.table_name,
                    "row": ev.row_id,
                    "change": ev.change,
                },
            )
        for rid, record_id, kind in opsread.records_after(api.store, rowid):
            rowid = rid
            sent = True
            yield _sse("record", {"records": rid, "id": record_id, "kind": kind})
        idle = 0 if sent else idle + 1
        if idle >= KEEPALIVE_POLLS:
            idle = 0
            yield ": keepalive\n\n"
        await asyncio.sleep(POLL_S)


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/events")
    async def events(
        request: Request, since: int = 0, records: int | None = None
    ) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        return StreamingResponse(
            stream(api, request, since, records),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return r
