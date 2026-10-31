"""Read endpoints (spec/modules/api-web.md "接口 / 读"). Thin: parse parameters, call a view."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query

from kta.api import cases, opsread, pool, runs, trace
from kta.api.common import Api, envelope, iso, machines_view
from kta.core.errors import ConfigError
from kta.core.ids import record_id

MAX_WINDOW = timedelta(days=7)


def _time(value: str | None, default: datetime) -> datetime:
    if value is None:
        return default
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError(f"bad timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ConfigError(f"timestamp {value!r} needs a timezone")
    return parsed


def _list(value: str | None) -> list[str] | None:
    return [x for x in value.split(",") if x] if value else None


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/meta")
    def meta() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Environment and record origins; the frontend banners non-real data from this."""
        return {"env": api.env, "now": iso(api.now()), "origins": opsread.origin_counts(api.store)}

    @r.get("/machines")
    def get_machines() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return machines_view(api)

    @r.get("/records/{id}")
    def get_record(id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return envelope(api, id)

    @r.get("/records")
    def list_records(
        kind: str,
        where: str | None = None,
        order: str | None = None,  # pyright: ignore[reportUnusedFunction]
        limit: int = Query(100, le=1000),
    ) -> list[dict[str, Any]]:
        filters = json.loads(where) if where else None
        out = []
        for rec in api.store.query(kind, filters, order, limit):
            data = rec.to_json_dict()
            out.append({"id": record_id(data), **data})
        return out

    @r.get("/search")
    def search(
        q: str, kinds: str | None = None, limit: int = Query(20, le=200)
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return [h.__dict__ for h in api.store.search(q, _list(kinds), limit)]

    @r.get("/lineage/{id}")
    def lineage(id: str, depth: int = Query(10, le=50)) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return cases.lineage(api, id, depth)

    @r.get("/cases")
    def list_cases() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return cases.cases(api)

    @r.get("/cases/{id}/timeline")
    def case_timeline(id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return cases.timeline(api, id)

    @r.get("/families")
    def list_families(snapshot: str | None = None) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return pool.families(api, snapshot)

    @r.get("/families/{id}/chain")
    def family_chain(id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return pool.chain(api, id)

    @r.get("/snapshots")
    def list_snapshots() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return pool.snapshots(api)

    @r.get("/pool/diff")
    def pool_diff(a: str, b: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return pool.diff(api, a, b)

    @r.get("/coverage")
    def coverage(snapshot: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return pool.coverage(api, snapshot)

    @r.get("/model/residuals")
    def residuals(model: str | None = None) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return pool.residuals(api, model)

    @r.get("/runs/active")
    def active() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return runs.active(api)

    @r.get("/budget")
    def budget() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return runs.budgets(api)

    @r.get("/inbox")
    def inbox() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return runs.inbox(api)

    @r.get("/trace")
    def get_trace(
        from_: str | None = Query(None, alias="from"),
        to: str | None = None,  # pyright: ignore[reportUnusedFunction]
        lanes: str | None = None,
        trace_id: str | None = Query(None, alias="trace"),
        kinds: str | None = None,
    ) -> dict[str, Any]:
        end = _time(to, api.now())
        start = _time(from_, end - timedelta(hours=1))
        if end - start > MAX_WINDOW or end <= start:
            raise ConfigError("trace window must be positive and at most 7 days")
        return trace.window(api, start, end, _list(lanes), trace_id, _list(kinds))

    @r.get("/trace/summary")
    def trace_summary(
        from_: str | None = Query(None, alias="from"), to: str | None = None
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        end = _time(to, api.now())
        return trace.summary(api, _time(from_, end - timedelta(hours=1)), end)

    @r.get("/trace/critical-path/{trace_id}")
    def critical_path(trace_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return trace.critical_path(api, trace_id)

    return r
