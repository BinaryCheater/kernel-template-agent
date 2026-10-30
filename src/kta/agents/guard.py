"""Permissions: task tokens, the agent environment whitelist, the kta-agent user.

Task token (decision D305): ``kta1.<task id>.<expiry unix s>.<hmac>``, HMAC-SHA256 over
the first three parts with a key kept on the control host (``[agents] token_key_file``,
mode 600, created on first use). It is valid only before its expiry *and* while the task
is leased (:func:`verify_task_token`), so it dies with the task. The API (WP9) and the
agent-side ``kta`` (WP6) verify it with :func:`verify_token`; agents never see the key.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol

from kta.core.errors import ConfigError, PermissionDenied
from kta.store.base import Task

TOKEN_VAR = "KTA_TASK_TOKEN"
TOKEN_PREFIX = "kta1"
KEY_BYTES = 32
LIVE_TASK_STATES = frozenset({"leased", "running"})
# Environment entries the agent process may receive (spec/modules/agents.md "环境变量").
# Values come from the runner; anything else in the runner's own environment is dropped.
ENV_WHITELIST: frozenset[str] = frozenset(
    {
        TOKEN_VAR,
        "KTA_TRACE_PARENT",
        "KTA_API",  # control-plane API the agent-side kta talks to (WP6/WP9)
        "KTA_ENV",
        "KTA_WP",
        "KTA_TASK",
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "CODEX_HOME",
        "PI_CODING_AGENT_DIR",
    }
)
BASE_PATH = ("/usr/local/bin", "/usr/bin", "/bin")


class TaskLookup(Protocol):
    """The part of the ops store token checks need (SqliteOps.task, D152)."""

    def task(self, task_id: str) -> Task: ...


def ensure_key(path: Path) -> bytes:
    """Read the token key, creating it (mode 600) on first use."""
    path = path.expanduser()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(KEY_BYTES))
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionDenied(f"token key {path} must not be readable by others (mode {mode:o})")
    key = path.read_bytes()
    if len(key) < KEY_BYTES:
        raise ConfigError(f"token key {path} is too short")
    return key


def _mac(key: bytes, body: str) -> str:
    return hmac.new(key, body.encode("ascii"), hashlib.sha256).hexdigest()


def issue_token(key: bytes, task_id: str, expires: datetime) -> str:
    if "." in task_id:
        raise ConfigError(f"task id {task_id!r} cannot contain '.'")
    body = f"{TOKEN_PREFIX}.{task_id}.{int(expires.timestamp())}"
    return f"{body}.{_mac(key, body)}"


def verify_token(key: bytes, token: str, now: datetime) -> str:
    """The task id of a valid, unexpired token; PermissionDenied otherwise."""
    parts = token.split(".")
    if len(parts) != 4 or parts[0] != TOKEN_PREFIX or not parts[2].isdigit():
        raise PermissionDenied("malformed task token")
    body = ".".join(parts[:3])
    if not hmac.compare_digest(_mac(key, body), parts[3]):
        raise PermissionDenied("task token signature does not match")
    if int(parts[2]) < now.timestamp():
        raise PermissionDenied("task token expired")
    return parts[1]


def verify_task_token(key: bytes, token: str, now: datetime, ops: TaskLookup) -> str:
    """As verify_token, and the task must still be live (leased or running)."""
    task_id = verify_token(key, token, now)
    state = ops.task(task_id).state
    if state not in LIVE_TASK_STATES:
        raise PermissionDenied(f"task {task_id} is {state}; its token is no longer valid")
    return task_id


def agent_env(values: Mapping[str, str], path_dirs: list[str]) -> dict[str, str]:
    """Whitelisted environment; PATH holds only kta's bin dir and the base tool dirs."""
    unknown = sorted(set(values) - ENV_WHITELIST)
    if unknown:
        raise PermissionDenied(f"environment entries outside the whitelist: {unknown}")
    env = dict(values)
    env["PATH"] = ":".join([*path_dirs, *BASE_PATH])
    return env


def api_key_env(name: str, secrets_file: Path) -> dict[str, str]:
    """``{name: key}`` from secrets.env for API-auth providers (conventions §8)."""
    path = secrets_file.expanduser()
    if not path.exists():
        raise PermissionDenied(f"{path} not found; API keys live only there")
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == name:
            return {name: v.strip().strip('"').strip("'")}
    raise PermissionDenied(f"{name} is not set in {path}")
