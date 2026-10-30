"""Client mode (D301, D370): with ``KTA_TASK_TOKEN`` set, ``kta`` runs inside an agent
workspace and reaches the store only through the control-plane API (``KTA_API``).

Every agent-facing command maps to one token endpoint of the API (WP9b, ``/agent/...``
and ``/write/record``). Files are sent as ``{relative posix path: base64}``. The trace
parent is forwarded, so each call is a span under the agent session. Commands without an
endpoint refuse with NotSupported instead of opening a local store.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from kta.core.context import SystemClock
from kta.core.errors import (
    ConfigError,
    ExecutionError,
    NotSupported,
    PermissionDenied,
    SchemaError,
)
from kta.core.schemas import Actor, Record

TOKEN_VAR = "KTA_TASK_TOKEN"
NOTE_FILE = "DELIVERY.md"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024  # the API's limit


def client_mode() -> bool:
    return bool(os.environ.get(TOKEN_VAR))


def call(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    base = os.environ.get("KTA_API", "").rstrip("/")
    if not base:
        raise ConfigError("client mode needs KTA_API (the control-plane API URL)")
    headers = {"Authorization": f"Bearer {os.environ[TOKEN_VAR]}"}
    parent = os.environ.get("KTA_TRACE_PARENT")
    if parent:
        headers["X-KTA-Trace-Parent"] = parent
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, default=str).encode()
    req = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:  # KTA_API is ours
            return dict(json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500].decode("utf-8", "replace")
        if exc.code in (401, 403):
            raise PermissionDenied(f"API refused {path}: {detail}") from exc
        if exc.code in (400, 404):
            raise SchemaError(f"API {path}: HTTP {exc.code}: {detail}") from exc
        if exc.code == 501:
            raise NotSupported(f"API {path}: {detail}") from exc
        raise ExecutionError("infra", f"API {path}: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ExecutionError("infra", f"API {path} unreachable: {exc.reason}") from exc


def encode_dir(root: Path, skip: str | None = None) -> dict[str, str]:
    files = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    files.pop(skip or "", None)
    if sum(len(v) for v in files.values()) > MAX_UPLOAD_BYTES:
        raise ConfigError(f"{root} is larger than the API's 32 MiB upload limit")
    return {k: base64.b64encode(v).decode() for k, v in files.items()}


def _q(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def note(args: argparse.Namespace) -> dict[str, Any]:
    task = os.environ.get("KTA_TASK") or args.task
    if not task:
        raise ConfigError("kta note needs KTA_TASK")
    body = {"title": "note", "subject": task, "summary": args.text, "data": {}}
    actor = Actor(type="agent", name="kta note", task=task)
    rec = Record.new("Report", body, actor, SystemClock().now())
    return call("POST", "/write/record", rec.to_json_dict())


def _family_body(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "family": args.family,
        "assignment": json.loads(args.assignment),
        "shape": json.loads(args.shape),
    }


def _instantiate(args: argparse.Namespace) -> dict[str, Any]:
    out = call("POST", "/agent/instantiate", _family_body(args))
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    for name, text in dict(out["files"]).items():
        (target / name).write_text(str(text), encoding="utf-8")
    (target / "launch.json").write_text(json.dumps(out["launch"], indent=1), "utf-8")
    return {**out, "dir": str(target)}


def _validate_template(args: argparse.Namespace) -> dict[str, Any]:
    doc = yaml.safe_load(Path(args.file).read_text(encoding="utf-8"))
    return call("POST", "/agent/validate-template", {"template": doc})


def _submit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.dir)
    note_path = root / NOTE_FILE
    if not note_path.is_file():
        raise ConfigError(f"the delivery note {note_path} is missing")
    payload = {"files": encode_dir(root, NOTE_FILE), "note": note_path.read_text("utf-8")}
    return call("POST", "/agent/submit", payload)


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    """Run one agent-facing command through the API (D370)."""
    cmd = args.command
    if cmd == "note":
        return note(args)
    if cmd == "show":
        return call("GET", f"/agent/records/{_q(args.id)}")
    if cmd == "card":
        return call("GET", f"/agent/cards/{_q(args.id)}")
    if cmd == "why":
        return call("GET", f"/agent/why/{_q(args.id)}?depth={int(args.depth)}")
    if cmd == "query":
        body: dict[str, Any] = {"what": args.what, "k": args.k}
        if args.term:
            body["term"] = args.term
        if args.case or args.near:
            body["case"] = args.case or args.near
        return call("POST", "/agent/query", body)
    if cmd == "validate-template":
        return _validate_template(args)
    if cmd == "validate-family":
        return call("POST", "/agent/validate-family", {"files": encode_dir(Path(args.dir))})
    if cmd == "instantiate":
        return _instantiate(args)
    if cmd in ("build", "sass"):
        return call("POST", f"/agent/{cmd}", _family_body(args))
    if cmd == "submit":
        return _submit(args)
    raise NotSupported(f"kta {cmd} has no API endpoint for agents ({TOKEN_VAR} set, D370)")
