"""Read endpoints added by WP9b: task detail, blobs, agents, pool, experiments, improvement."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from kta.api import agentsview, proofview, runs, studies, taskview
from kta.api.common import Api
from kta.store.sqlite import RecordNotFound


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/tasks/{id}")
    def task(id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return taskview.task_detail(api, id)

    @r.get("/blobs/{sha256}")
    def blob(sha256: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return taskview.blob_text(api, sha256)

    @r.get("/agents")
    def agents() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return agentsview.agents(api)

    @r.post("/agents/batches/{batch}/verify")
    def verify_batch(batch: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        if batch not in {b["batch"] for b in agentsview.batches(api)}:
            raise RecordNotFound(f"no batch environment {batch!r}")
        return api.orchestration.verify_batch(batch)

    @r.get("/quotas")
    def get_quotas() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return runs.quota_view(api)

    @r.get("/pool")
    def pool(snapshot: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return studies.pool(api, snapshot)

    @r.get("/proofs/{id}")
    def proofs(id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        api.store.get(id)  # 404 for an unknown record
        return proofview.proofs_of(api, id)

    @r.get("/experiments")
    def experiments() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return studies.experiments(api)

    @r.get("/improvements")
    def improvements() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return studies.improvements(api)

    return r
