"""The KTA API application (spec/modules/api-web.md).

The process holds the store's connection; it listens on the control host only
([api] host/port in config). It serves the read endpoints and SSE (WP9a), and the
operation endpoints, the task-token write endpoint and the WP9b read views.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kta.api import actions, auth, events, read, views
from kta.api.common import Api
from kta.api.wiring import Wired
from kta.core.context import Context
from kta.core.errors import ConfigError, KtaError, NotSupported, PermissionDenied, SchemaError
from kta.core.registry import resolve
from kta.store.sqlite import RecordNotFound


def open_api(ctx: Context) -> Api:
    store, ops = resolve("store", ctx), resolve("ops", ctx)
    return Api(ctx=ctx, store=store, ops=ops, orchestration=Wired(ctx, store, ops))


def create_app(api: Api) -> FastAPI:
    app = FastAPI(title="KTA API", docs_url="/docs", redoc_url=None)
    app.include_router(read.router(api))
    app.include_router(events.router(api))
    app.include_router(views.router(api))
    app.include_router(actions.router(api))
    app.include_router(auth.router(api))

    @app.exception_handler(KtaError)
    async def kta_error(_: Request, exc: KtaError) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(
            {"error": type(exc).__name__, "detail": str(exc)}, status_code=status_for(exc)
        )

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        # A missing required reason or a malformed body is the caller's error: 400 (api-web.md).
        problems = [
            f"{'.'.join(str(x) for x in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors()
        ]
        return JSONResponse({"error": "BadRequest", "detail": "; ".join(problems)}, status_code=400)

    return app


def status_for(exc: KtaError) -> int:
    if isinstance(exc, RecordNotFound):
        return 404
    if isinstance(exc, NotSupported):
        return 501
    if isinstance(exc, PermissionDenied):
        return 403
    if isinstance(exc, ConfigError | SchemaError):
        return 400
    return 500
