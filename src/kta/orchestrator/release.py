"""Staged release S0–S4 (docs/design/02 §6.3, D-T22; WP21, D485).

``stages.yaml`` defines each stage: the processes that tick, the task types an agent may
take, and how agent calls run (``none`` | ``gated`` | ``automatic``). The stage in force
is the controls knob ``release.stage``, so a stage change is a controls version with an
author and a reason, recorded with every run after it.

The gate sits where an agent call starts (the agent pool picks the next task): a task is
*held* while its stage forbids its type or agent calls, or, in a gated stage, while
another agent call is in flight or you have not released this attempt. A release is the
ops pointer ``release:task:<id>`` (the attempt it covers) plus a HumanAction record; it
covers one attempt, so a retry needs a new release. Processes outside the stage skip
their ticks. Held work is visible (``held_calls``), never dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from importlib import resources
from typing import Any, Literal, cast

import yaml

from kta.core.controls import Controls
from kta.core.errors import ConfigError
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import HumanAction
from kta.orchestrator.control import Paused
from kta.store.base import Store, Task
from kta.store.sqlite_ops import SqliteOps

AgentCalls = Literal["none", "gated", "automatic"]
IN_FLIGHT = ("leased", "running")
ALWAYS = ("maintain",)  # returns lost leases and prunes spans; safe in every stage


class Held(Paused):
    """An agent call the release stage does not allow now; the task stays queued."""


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    runs: str
    processes: tuple[str, ...]
    task_types: tuple[str, ...]
    agent_calls: AgentCalls
    exit: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "runs": self.runs,
            "processes": list(self.processes),
            "task_types": list(self.task_types),
            "agent_calls": self.agent_calls,
            "exit": list(self.exit),
        }


@cache
def stages() -> tuple[Stage, ...]:
    raw: Any = yaml.safe_load(
        resources.files("kta.orchestrator").joinpath("stages.yaml").read_text("utf-8")
    )
    if not isinstance(raw, dict) or cast(dict[str, Any], raw).get("format") != 1:
        raise ConfigError("stages.yaml must be a mapping with format: 1")
    out: list[Stage] = []
    for s in cast(list[dict[str, Any]], cast(dict[str, Any], raw)["stages"]):
        calls = str(s["agent_calls"])
        if calls not in ("none", "gated", "automatic"):
            raise ConfigError(f"stage {s['id']}: agent_calls {calls!r} unknown")
        out.append(
            Stage(
                id=str(s["id"]),
                name=str(s["name"]),
                runs=str(s["runs"]),
                processes=tuple(str(p) for p in s["processes"]),
                task_types=tuple(str(t) for t in s["task_types"]),
                agent_calls=calls,
                exit=tuple(str(e) for e in cast(list[Any], s["exit"] or [])),
            )
        )
    return tuple(out)


def stage_of(controls: Controls) -> Stage:
    sid = controls.release_stage
    for s in stages():
        if s.id == sid:
            return s
    raise ConfigError(f"release.stage {sid!r} is not defined in stages.yaml")


def process_allowed(controls: Controls, name: str) -> bool:
    return name in ALWAYS or name in stage_of(controls).processes


# ---- releases of single agent calls -----------------------------------------------------


def _key(task_id: str) -> str:
    return f"release:task:{task_id}"


def released_attempt(ops: SqliteOps, task_id: str) -> int | None:
    raw = ops.get_pointer(_key(task_id))
    return int(json.loads(raw)["attempt"]) if raw else None


def hold_reason(ops: SqliteOps, task: Task, controls: Controls) -> str | None:
    """Why ``task`` may not start an agent call now, or None when it may."""
    stage = stage_of(controls)
    if stage.agent_calls == "none":
        return f"stage {stage.id} ({stage.name}) runs no agent calls"
    if task.card_type not in stage.task_types:
        return f"stage {stage.id} does not run {task.card_type} tasks"
    if stage.agent_calls == "automatic":
        return None
    busy = [t.id for t in ops.list_tasks(states=IN_FLIGHT) if t.id != task.id]
    if busy:
        return f"stage {stage.id} runs one agent call at a time; {busy[0]} is in flight"
    if released_attempt(ops, task.id) != task.attempts + 1:
        return f"stage {stage.id}: attempt {task.attempts + 1} waits for your release"
    return None


def admit(ops: SqliteOps, task: Task, controls: Controls) -> None:
    reason = hold_reason(ops, task, controls)
    if reason is not None:
        raise Held(f"task {task.id} held: {reason}")


def release_call(
    store: Store, ops: SqliteOps, task: Task, human: Actor, reason: str, at: datetime
) -> str:
    """Release the next attempt of ``task`` for one agent call; returns the HumanAction ID."""
    if human.type != "human":
        raise ConfigError("only a human releases an agent call")
    if task.state != "queued":
        raise ConfigError(f"task {task.id} is {task.state}; only a queued task is released")
    attempt = task.attempts + 1
    params = json.dumps({"attempt": attempt, "task": task.id}, sort_keys=True)
    body = HumanAction(action=f"release_agent_call {params}", reason=reason)
    action = store.put(Record.new("HumanAction", body, human, at, refs={"card": task.card_id}))
    key = _key(task.id)
    value = json.dumps(
        {"attempt": attempt, "action": action, "by": human.name, "at": at.isoformat()}
    )
    if not ops.set_pointer(key, ops.get_pointer(key), value):
        raise ConfigError(f"release of {task.id} changed concurrently; try again")
    return action


def held_calls(
    ops: SqliteOps, controls: Controls, agent_types: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Queued agent tasks and why each is held (empty reason: it may start)."""
    out: list[dict[str, Any]] = []
    for t in ops.list_tasks(states=("queued",)):
        if t.card_type not in agent_types:
            continue
        out.append(
            {
                "task": t.id,
                "card": t.card_id,
                "type": t.card_type,
                "work_unit": t.work_unit,
                "attempt": t.attempts + 1,
                "released_attempt": released_attempt(ops, t.id),
                "held": hold_reason(ops, t, controls),
            }
        )
    return out
