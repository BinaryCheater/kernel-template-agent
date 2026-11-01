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
import logging
import math
import re
import threading
from collections.abc import Mapping
from typing import Any

from kta.api import opsread
from kta.api.common import Api, envelope
from kta.api.wiring import TaskSpec
from kta.core.errors import ConfigError, NotSupported
from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.process import HumanAction, StatusChange
from kta.store import stale
from kta.store.sqlite import RecordNotFound
from kta.store.status import check_change

DEFAULT_OPERATOR = "user"
log = logging.getLogger(__name__)

# Loops a human advance is working on now (loop id -> mode). One advance per loop at a time;
# the runner's loop lock would refuse a second one anyway, this makes it visible (D664).
_ADVANCING: dict[str, str] = {}
_ADVANCING_LOCK = threading.Lock()


def advancing() -> dict[str, str]:
    with _ADVANCING_LOCK:
        return dict(_ADVANCING)


def advance(api: Api, loop_id: str, mode: str, reason: str) -> dict[str, Any]:
    """Advance a loop by hand: one step, one round, continuously, or retry after a block.

    The HumanAction is written before this returns; steps and rounds then run on a
    background thread (a build or a measurement can take minutes), and their progress
    shows up as loop-step spans, records and journal lines like any other step."""
    if mode not in ("run",) and loop_id in advancing():
        raise ConfigError(f"loop {loop_id} is already advancing ({advancing()[loop_id]})")
    human = operator(api)
    out = api.orchestration.advance_action(loop_id, mode, reason, human)
    if mode == "run":
        return {**out, "started": False}

    def work() -> None:
        try:
            taken = api.orchestration.advance_work(loop_id, mode, f"human:{human.name}")
            log.info("advance %s %s: %d step(s)", loop_id, mode, len(taken))
        except Exception:  # kta:allowed a background step must not die silently
            log.exception("advance %s %s failed", loop_id, mode)
        finally:
            with _ADVANCING_LOCK:
                _ADVANCING.pop(loop_id, None)

    with _ADVANCING_LOCK:
        _ADVANCING[loop_id] = mode
    threading.Thread(target=work, name=f"advance-{loop_id}", daemon=True).start()
    return {**out, "started": True}


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
    """Status -> suspect, then ask the orchestrator to re-verify it (D343): WP6 starts a new
    solve loop for each Case whose Solution rests on the record (D368)."""
    kind = _judged(api, record)
    change = _status_change(api, record, "suspect", reason)
    action = write_action(api, "flag", reason, {"subject": record}, {"kind": kind})
    api.store.put(Record.new("StatusChange", change, operator(api), api.now()))
    gap = api.orchestration.missing().get("reverify")
    if gap is not None:
        followups = [{"what": "re-verification", "state": "not_supported", "needs": gap}]
    elif kind not in REVERIFIABLE:
        why = f"WP6 re-verifies a Solution or a {'/'.join(REVERIFIABLE[1:])} (D368), not a {kind}"
        followups = [{"what": "re-verification", "state": "not_supported", "needs": why}]
    else:
        try:
            done = api.orchestration.request_reverify(record, reason, operator(api))
            followups = [{"what": "re-verification", "state": "started", **done}]
        except NotSupported as exc:
            # The flag stands; WP6 re-verifies only what a Solution rests on (D368).
            followups = [{"what": "re-verification", "state": "not_supported", "needs": str(exc)}]
    return {"action": action, "subject": envelope(api, record), "followups": followups}


# What WP6 can re-verify: a Solution, or a record a Solution rests on (D368). Other kinds
# are flagged only, so no re-verification request is written for them.
REVERIFIABLE = ("Solution", "Measurement", "Check", "Bound")
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
    if what == "improvement":
        return decide_improvement(api, record, reason, approve=True)
    _expect_kind(api, record, APPROVABLE[what], what)
    action = write_action(api, f"approve_{what}", reason, {what: record})
    return {"action": action, "subject": envelope(api, record)}


def decide_improvement(api: Api, record: str, reason: str, approve: bool) -> dict[str, Any]:
    """Approve or reject the system items of an improvement Report (WP11 D406): the
    summarize process reads ``approve_improvement``/``reject_improvement`` with refs.improvement."""
    _expect_kind(api, record, APPROVABLE["improvement"], "improvement")
    if api.store.get_json(record)["body"]["title"] != "improvement":
        raise ConfigError(f"Report {record[:12]} is not an improvement report")
    verb = "approve_improvement" if approve else "reject_improvement"
    action = write_action(api, verb, reason, {"improvement": record})
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
    # TaskQueue.create writes the card and your HumanAction (WP6 D365, D358).
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

    Actions written by other modules have no JSON part: WP6's ``task cancel <id>`` gives verb
    ``cancel`` and args ``<id>``; WP8's ``approve_profile`` gives that verb alone.
    """
    verb, _, rest = text.partition(" ")
    if verb == "task" and rest:  # WP6's TaskQueue writes ``task <verb> <task id>``
        verb, _, rest = rest.partition(" ")
    if rest.startswith("{"):
        try:
            return {"verb": verb, "params": json.loads(rest), "args": ""}
        except json.JSONDecodeError:
            return {"verb": verb, "params": None, "args": rest}
    return {"verb": verb, "params": None, "args": rest}
