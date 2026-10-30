"""Human operations (docs/design/30 §3): every one writes a HumanAction record.

HumanAction has only ``action`` and ``reason`` (spec/schemas.md §4.9), so the action
string carries the parameters: ``<verb>`` or ``<verb> <compact JSON>``, e.g.
``budget {"limit":{"agent_min":90.0},"scope":"global"}`` (decision D342). The subject is a
ref (``subject``, ``protocol``, ``profile``, ...), so the action shows up in the subject's
lineage.

Operations whose effect the store alone defines are carried out here. The others go
through ``api.orchestration`` (wiring.py), which raises NotSupported until WP6/WP8 are
merged; their parameters are validated first, so a bad request is still a 400 or 404.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

from kta.api import opsread
from kta.api.common import Api, envelope
from kta.api.wiring import TaskSpec
from kta.core.errors import ConfigError
from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.process import HumanAction, StatusChange
from kta.store import stale
from kta.store.sqlite import RecordNotFound
from kta.store.status import check_change

DEFAULT_OPERATOR = "user"
# Records about decisions themselves; rejecting or flagging them has no meaning.
NOT_JUDGED = frozenset({"HumanAction", "StatusChange"})
BUDGET_KEY = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*_[a-z]+$")
BUDGET_SCOPE = re.compile(r"^(global|(process|loop|task):[A-Za-z0-9_.:-]+)$")


def operator(api: Api) -> Actor:
    """The single user (design/30 §4: no login); the name comes from ``[api] operator``."""
    name = api.config.section("api").get("operator", DEFAULT_OPERATOR)
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("[api] operator must be a non-empty string")
    return Actor(type="human", name=name.strip())


def action_text(verb: str, params: Mapping[str, Any] | None = None) -> str:
    if not params:
        return verb
    return f"{verb} {json.dumps(params, sort_keys=True, separators=(',', ':'))}"


def write_action(
    api: Api,
    verb: str,
    reason: str,
    refs: dict[str, RefValue] | None = None,
    params: Mapping[str, Any] | None = None,
) -> str:
    body = HumanAction(action=action_text(verb, params), reason=reason)
    return api.store.put(Record.new("HumanAction", body, operator(api), api.now(), refs))


def _judged(api: Api, record: str) -> str:
    """The kind of a record a human may judge; 404 when it does not exist."""
    kind = api.store.kind_of(record)
    if kind in NOT_JUDGED:
        raise ConfigError(f"a {kind} records a decision; it cannot be judged itself")
    return kind


def _status_change(api: Api, record: str, to: str, reason: str) -> StatusChange:
    change = StatusChange.model_validate({"subject": record, "to_status": to, "reason": reason})
    check_change(api.store.status(record), change, "human")
    return change


def reject(api: Api, record: str, reason: str) -> dict[str, Any]:
    """Status -> rejected; everything depending on it becomes stale (design/20 §6)."""
    kind = _judged(api, record)
    _status_change(api, record, "rejected", reason)
    action = write_action(api, "reject", reason, {"subject": record}, {"kind": kind})
    marked = stale.reject(api.store, record, reason, operator(api), api.now())
    return {"action": action, "subject": envelope(api, record), "stale": marked}


def deprecate(api: Api, record: str, reason: str) -> dict[str, Any]:
    """Status -> deprecated: not used for new work; existing results keep their status."""
    kind = _judged(api, record)
    change = _status_change(api, record, "deprecated", reason)
    action = write_action(api, "deprecate", reason, {"subject": record}, {"kind": kind})
    api.store.put(Record.new("StatusChange", change, operator(api), api.now()))
    return {"action": action, "subject": envelope(api, record), "stale": []}


def flag(api: Api, record: str, reason: str) -> dict[str, Any]:
    """Status -> suspect, then ask the orchestrator for a re-verification (D343)."""
    kind = _judged(api, record)
    change = _status_change(api, record, "suspect", reason)
    action = write_action(api, "flag", reason, {"subject": record}, {"kind": kind})
    api.store.put(Record.new("StatusChange", change, operator(api), api.now()))
    gap = api.orchestration.missing().get("reverify")
    if gap is None:
        reverify = api.orchestration.request_reverify(record, reason, operator(api))
        followups = [{"what": "re-verification task", "state": "created", **reverify}]
    else:
        followups = [{"what": "re-verification task", "state": "not_supported", "needs": gap}]
    return {"action": action, "subject": envelope(api, record), "followups": followups}


APPROVABLE = {"protocol": ("Protocol",), "improvement": ("Report",)}
CONFIRMABLE = {"pattern": ("Pattern",), "structure": ("Structure",)}


def _expect_kind(api: Api, record: str, kinds: tuple[str, ...], what: str) -> None:
    kind = api.store.kind_of(record)
    if kind not in kinds:
        raise ConfigError(f"{record[:12]} is a {kind}, not a {what}")


def approve(api: Api, what: str, record: str, reason: str) -> dict[str, Any]:
    """Approve a protocol (it is frozen for experiments), an improvement item or a profile."""
    if what == "profile":
        _expect_kind(api, record, ("AgentProfile",), "AgentProfile")
        return api.orchestration.approve_profile(record, reason, operator(api))
    _expect_kind(api, record, APPROVABLE[what], what)
    if what == "improvement" and api.store.get_json(record)["body"]["title"] != "improvement":
        raise ConfigError(f"Report {record[:12]} is not an improvement report")
    action = write_action(api, f"approve_{what}", reason, {what: record})
    return {"action": action, "subject": envelope(api, record)}


def confirm(api: Api, what: str, record: str, reason: str) -> dict[str, Any]:
    """Confirm a new Pattern or Structure (design/21 admission needs the user)."""
    _expect_kind(api, record, CONFIRMABLE[what], what)
    action = write_action(api, f"confirm_{what}", reason, {what: record})
    return {"action": action, "subject": envelope(api, record)}


def set_budget(api: Api, scope: str, limit: Mapping[str, float], reason: str) -> dict[str, Any]:
    """Change limits of an existing budget scope; keys not given keep their limit."""
    if not BUDGET_SCOPE.match(scope):
        raise ConfigError(f"budget scope {scope!r} is not global or process:/loop:/task:<id>")
    rows = {b.scope: b for b in opsread.budgets(api.ops)}
    if scope not in rows:
        raise RecordNotFound(f"no budget scope {scope!r}")
    if not limit:
        raise ConfigError("budget: give at least one limit")
    for key, value in limit.items():
        if not BUDGET_KEY.match(key):
            raise ConfigError(f"budget key {key!r} is not <resource>_<unit>, e.g. agent_min")
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"budget limit {key} must be a finite number >= 0")
    old = {k: float(v) for k, v in rows[scope].limit.items()}
    new = {**old, **{k: float(v) for k, v in limit.items()}}
    params = {"scope": scope, "limit": dict(limit), "previous": old}
    action = write_action(api, "budget", reason, None, params)
    api.ops.set_budget(scope, new)
    return {"action": action, "scope": scope, "limit": new, "previous": old}


def _task_row(api: Api, task_id: str) -> opsread.TaskRow:
    for t in opsread.tasks(api.ops):
        if t.id == task_id:
            return t
    raise RecordNotFound(f"no task {task_id}")


def cancel(api: Api, task_id: str, reason: str) -> dict[str, Any]:
    _task_row(api, task_id)
    return api.orchestration.cancel_task(task_id, reason, operator(api))


def retry(api: Api, task_id: str, note: str) -> dict[str, Any]:
    _task_row(api, task_id)
    return api.orchestration.retry_task(task_id, note, operator(api))


def priority(api: Api, task_id: str, value: int, reason: str) -> dict[str, Any]:
    _task_row(api, task_id)
    return api.orchestration.set_priority(task_id, value, reason, operator(api))


def create_task(api: Api, spec: TaskSpec, reason: str) -> dict[str, Any]:
    if not spec.goal.strip():
        raise ConfigError("create-task: the goal is empty")
    for rid in spec.inputs:
        if not api.store.exists(rid):
            raise RecordNotFound(f"input record {rid} does not exist")
    return api.orchestration.create_task(spec, reason, operator(api))


def pause(api: Api, scope: str, reason: str, resume: bool) -> dict[str, Any]:
    if scope != "global" and not scope.startswith(("process:", "loop:", "task:")):
        raise ConfigError(f"pause scope {scope!r} is not global or process:/loop:/task:<id>")
    human = operator(api)
    if resume:
        return api.orchestration.resume(scope, reason, human)
    return api.orchestration.pause(scope, reason, human)


def parse_action(text: str) -> dict[str, Any]:
    """Split a HumanAction ``action`` into its verb and parameters (inverse of action_text).

    Actions written by other modules (``task cancel <id>``, ``approve_profile``) have no
    JSON part; their remaining words are returned as ``args``.
    """
    verb, _, rest = text.partition(" ")
    if rest.startswith("{"):
        try:
            return {"verb": verb, "params": json.loads(rest), "args": ""}
        except json.JSONDecodeError:
            return {"verb": verb, "params": None, "args": rest}
    return {"verb": verb, "params": None, "args": rest}
