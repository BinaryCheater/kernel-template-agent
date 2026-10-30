"""Pausing scopes (D364): ``global``, ``process:<name>``, ``loop:<id>``, ``task:<id>``.

A pause is the ops pointer ``paused:<scope>`` (JSON: paused, reason, by, at). It is
operational state, not evidence; the HumanAction that asked for it is the record. A
paused scope stops new work below it: processes skip their tick, the runner does not
advance a paused loop, and a task under a paused scope is not leased. Work already
running finishes (docs/design/11 §3.4, like an exhausted quota).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime

from kta.core.errors import ConfigError, KtaError
from kta.store.sqlite_ops import SqliteOps

SCOPE = re.compile(r"^(global|process:[a-z_]+|loop:\S+|task:\S+)$")


class Paused(KtaError):
    """New work under a paused scope is refused; it stays queued or waiting."""


def _key(scope: str) -> str:
    if not SCOPE.match(scope):
        raise ConfigError(f"not a pausable scope: {scope!r} (global, process:, loop:, task:)")
    return f"paused:{scope}"


def set_paused(
    ops: SqliteOps, scope: str, paused: bool, reason: str, by: str, at: datetime
) -> None:
    key = _key(scope)
    current = ops.get_pointer(key)
    value = json.dumps(
        {"paused": paused, "reason": reason, "by": by, "at": at.isoformat()}, sort_keys=True
    )
    if not ops.set_pointer(key, current, value):
        raise ConfigError(f"{scope} pause state changed concurrently; try again")


def is_paused(ops: SqliteOps, scope: str) -> bool:
    raw = ops.get_pointer(_key(scope))
    return bool(raw) and bool(json.loads(raw)["paused"])


def paused_scope(ops: SqliteOps, scopes: Iterable[str]) -> str | None:
    """The first paused scope among ``scopes``, or None."""
    return next((s for s in scopes if is_paused(ops, s)), None)


def task_scopes(task_id: str, work_unit: str, process: str) -> list[str]:
    return [f"task:{task_id}", f"loop:{work_unit}", f"process:{process}", "global"]
