"""The KTA API application (spec/modules/api-web.md).

The process holds the store's connection; it listens on the control host only
([api] host/port in config). Phase B of WP9a serves the read endpoints and SSE; the
operation and write endpoints come with WP9b.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from kta.api import events, read
from kta.api.common import Api
from kta.core.context import Context
from kta.core.errors import ConfigError, KtaError, NotSupported, PermissionDenied, SchemaError
from kta.core.registry import resolve
from kta.store.sqlite import RecordNotFound


def open_api(ctx: Context) -> Api:
    return Api(store=resolve("store", ctx), ops=resolve("ops", ctx), config=ctx.config, env=ctx.env)


def create_app(api: Api) -> FastAPI:
    app = FastAPI(title="KTA API", docs_url="/docs", redoc_url=None)
    app.include_router(read.router(api))
    app.include_router(events.router(api))

    @app.exception_handler(KtaError)
    async def kta_error(_: Request, exc: KtaError) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(
            {"error": type(exc).__name__, "detail": str(exc)}, status_code=status_for(exc)
        )

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
