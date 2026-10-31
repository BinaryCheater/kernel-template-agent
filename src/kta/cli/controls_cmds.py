"""``kta controls``: the controls registry (WP21). Values change only through the
settings page or ``POST /controls`` (a version with an author and a reason); the
command line reads them and creates the operator token those writes need."""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path
from typing import Any

from kta.cli.exec_cmds import ContextFactory, _out
from kta.core.controls import definitions
from kta.core.errors import ConfigError
from kta.orchestrator import release
from kta.orchestrator.pipeline import Services
from kta.store import controls as controls_store

DEFAULT_TOKEN_FILE = "~/.config/kta/operator-token"  # the same default as kta.api.controls


def cmd_show(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    svc = Services.open(make_ctx("controls-show"))  # stores version 1 if none exists yet
    c = svc.ctx.controls
    rows = [{"id": k.id, "value": c.value(k.id), "status": k.status} for k in definitions().knobs]
    stage = release.stage_of(c)
    text = "\n".join(
        [f"controls v{c.version} {c.version_id}  stage {stage.id} ({stage.name})"]
        + [f"{r['id']:40} {r['value']}  [{r['status']}]" for r in rows]
    )
    _out(args, {"version": c.version, "id": c.version_id, "knobs": rows}, text)
    return 0


def cmd_history(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    svc = Services.open(make_ctx("controls-history"))
    out: list[dict[str, Any]] = []
    for rid, rec in controls_store.history(svc.store, args.limit):
        out.append(
            {
                "id": rid,
                "version": rec.body["version"],
                "author": rec.actor.name,
                "reason": rec.body["reason"],
                "changes": rec.body["changes"],
            }
        )
    text = "\n".join(
        f"v{v['version']} {v['id'][:12]} {v['author']}: {v['reason']} ({len(v['changes'])} changes)"
        for v in out
    )
    _out(args, {"versions": out}, text or "(none)")
    return 0


def cmd_token(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    """Create the operator token file (mode 600) if it does not exist; never prints it."""
    ctx = make_ctx("controls-token")
    raw = ctx.config.section("api").get("operator_token_file", DEFAULT_TOKEN_FILE)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError("[api] operator_token_file must be a path")
    path = Path(raw).expanduser()
    created = not path.exists()
    if created:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(secrets.token_urlsafe(32) + "\n")
    note = "created" if created else "exists"
    _out(args, {"path": str(path), "created": created}, f"operator token {note}: {path}")
    return 0


def add_controls_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    ctl = sub.add_parser("controls", help="the controls registry").add_subparsers(
        dest="controls_command", required=True
    )
    for name, fn, h in (
        ("show", cmd_show, "the knob values in force and the release stage"),
        ("history", cmd_history, "stored controls versions, newest first"),
        ("token", cmd_token, "create the operator token the settings page needs to write"),
    ):
        p = ctl.add_parser(name, help=h)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=lambda a, fn=fn: fn(a, make_ctx))
        if name == "history":
            p.add_argument("--limit", type=int, default=20)
