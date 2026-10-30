"""Client mode (D301, D370): with ``KTA_TASK_TOKEN`` set, ``kta`` runs inside an agent
workspace and reaches the store only through the control-plane API (``KTA_API``).

The API offers one token endpoint so far, ``POST /write/record`` (WP9b), so only commands
that write a record without files work here: ``kta note``. Every other command refuses
with NotSupported naming the endpoint it would need, instead of opening a local store.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any

from kta.core.context import SystemClock
from kta.core.errors import ConfigError, ExecutionError, NotSupported, PermissionDenied
from kta.core.schemas import Actor, Record

TOKEN_VAR = "KTA_TASK_TOKEN"
# Command -> the API endpoint it needs (none exists yet except write/record).
NEEDS = {
    "submit": "an upload-and-submit endpoint (Submission blobs, running -> submitted)",
    "show": "a token-scoped record read endpoint",
    "query": "a token-scoped retrieval endpoint",
    "why": "a token-scoped lineage endpoint",
    "card": "a token-scoped card endpoint",
    "validate-template": "a validation endpoint",
    "validate-family": "a validation endpoint with file upload",
    "instantiate": "a generation endpoint",
    "build": "a build endpoint (Executor job on the agent's behalf)",
    "sass": "a build endpoint",
}


def client_mode() -> bool:
    return bool(os.environ.get(TOKEN_VAR))


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    base = os.environ.get("KTA_API", "").rstrip("/")
    if not base:
        raise ConfigError("client mode needs KTA_API (the control-plane API URL)")
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload, default=str).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ[TOKEN_VAR]}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # KTA_API is ours
            return dict(json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionDenied(f"API refused {path}: {exc.read()[:300]!r}") from exc
        raise ExecutionError("infra", f"API {path}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ExecutionError("infra", f"API {path} unreachable: {exc.reason}") from exc


def note(args: argparse.Namespace) -> dict[str, Any]:
    task = os.environ.get("KTA_TASK") or args.task
    if not task:
        raise ConfigError("kta note needs KTA_TASK")
    body = {"title": "note", "subject": task, "summary": args.text, "data": {}}
    actor = Actor(type="agent", name="kta note", task=task)
    rec = Record.new("Report", body, actor, SystemClock().now())
    return post("/write/record", rec.to_json_dict())


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "note":
        return note(args)
    need = NEEDS.get(args.command, "an API endpoint for this command")
    raise NotSupported(f"kta {args.command} in client mode ({TOKEN_VAR} set) needs {need} (WP9b)")
