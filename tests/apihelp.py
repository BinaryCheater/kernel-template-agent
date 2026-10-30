"""Helpers for API tests: a store seeded with the WP9a fixture world, and an in-process
ASGI request (no HTTP client dependency; the app is called as the server would call it)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kta.api.common import Api
from kta.api.wiring import Wired
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.registry import Registry
from kta.core.schemas import Actor, Record
from kta.store import CodeState, open_ops, open_store
from kta_devseed.knowledge import Seeder, seed_knowledge
from kta_devseed.world import WINDOW_MIN, SeedClock, seed_world

END = datetime(2026, 10, 30, 14, 30, tzinfo=UTC)
TEST_ACTOR = Actor(type="system", name="test")


def seeded_api(tmp: Path, config: dict[str, Any] | None = None, wired: bool = True) -> Api:
    clock = SeedClock(END - timedelta(minutes=WINDOW_MIN + 60))
    ctx = Context(
        env="test",
        config=config_from_mapping({"agents": {"slots": 2, **(config or {})}}),
        data_dir=tmp,
        registry=Registry(),
        clock=clock,
    )
    store = open_store(ctx, CodeState("0" * 40, False))
    ops = open_ops(ctx)
    seeder = Seeder(store, clock)
    seed_knowledge(seeder)
    seed_world(seeder, ops, END)
    if not wired:
        return Api(ctx=ctx, store=store, ops=ops)
    return Api(ctx=ctx, store=store, ops=ops, orchestration=Wired(ctx, store, ops))


def put(api: Api, kind: str, body: dict[str, Any], refs: dict[str, Any] | None = None) -> str:
    rec = Record.new(kind, body, TEST_ACTOR, api.now(), refs or {}, origin="fixture")
    return api.store.put(rec)


def request(
    app: Any,
    method: str,
    url: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    path, _, query = url.partition("?")
    payload = json.dumps(body).encode() if body is not None else b""
    head = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    head += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": head,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 7310),
    }
    inbox = [{"type": "http.request", "body": payload, "more_body": False}]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return inbox.pop(0) if inbox else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(data) if data else None
