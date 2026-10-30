"""The budget hierarchy above loops and tasks (docs/design/11 §3.1, §3.4, §4.2; WP11).

``budget.py`` reserves and settles along ``task < loop < process < global``. This module
keeps the upper two levels right while processes run:

- **Quotas.** Process shares are the controls knob ``budget.process_quotas`` (defaults of
  11 §3.1; changed through a controls version, WP21). While no experiment batch runs, the
  experiment share is handed to the other processes in proportion to their shares
  (11 §3.1); ``apply_quotas`` rewrites the process limits from the global limit
  whenever that changes (D401).
- **Periods.** ``[budget] period = "day" | "week"`` makes the global budget periodic: at a
  period change ``roll_period`` writes a Report (title ``budget_period``) with each
  scope's final use, then books the negative of that use so the period starts at zero.
  Reservations of running tasks are kept (D402).
- **Fair turns.** ``order_queued`` orders queued tasks by the process whose use is
  smallest relative to its share, then priority, then age (11 §4.2). Your tasks (process
  ``human``) always come first.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from kta.core.config import Config
from kta.core.controls import Controls, definitions
from kta.core.errors import ConfigError
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.orchestrator.budget import has_scope
from kta.store.base import Store, Task
from kta.store.sqlite_ops import SqliteOps

RESOURCES = ("agent_min", "gpu_min")
ACTIVE_EXPERIMENTS = "experiment:active"  # pointer: JSON list of running experiment Protocols
PERIOD = "budget:period"
ACTOR = Actor(type="system", name="kta.orchestrator.quotas")


def shares(controls: Controls) -> dict[str, float]:
    """Process shares in the controls (the registry validates >= 0 and a sum <= 1)."""
    return controls.process_quotas


def effective(base: Mapping[str, float], experiment_running: bool) -> dict[str, float]:
    """Shares in force: without a running experiment batch its share is redistributed."""
    out = dict(base)
    if experiment_running:
        return out
    spare = out.get("experiment", 0.0)
    others = {k: v for k, v in out.items() if k != "experiment"}
    total = sum(others.values())
    out["experiment"] = 0.0
    if spare and total:
        for k, v in others.items():
            out[k] = v + spare * v / total
    return out


def active_experiments(ops: SqliteOps) -> list[str]:
    raw = ops.get_pointer(ACTIVE_EXPERIMENTS)
    return list(json.loads(raw)) if raw else []


def set_active_experiments(ops: SqliteOps, protocols: Sequence[str]) -> bool:
    """Record the running experiment Protocols; True when the set changed."""
    old = ops.get_pointer(ACTIVE_EXPERIMENTS)
    new = json.dumps(sorted(set(protocols)))
    if old == new or (old is None and new == "[]"):
        return False
    if not ops.set_pointer(ACTIVE_EXPERIMENTS, old, new):
        raise ConfigError(f"{ACTIVE_EXPERIMENTS} changed concurrently")
    return True


def apply_quotas(ops: SqliteOps, controls: Controls) -> dict[str, float]:
    """Process limits = global limit x effective share; no-op without a global limit."""
    in_force = effective(shares(controls), bool(active_experiments(ops)))
    if not has_scope(ops, "global"):
        return in_force
    limit = ops.budget("global")["limit"]
    for process, share in in_force.items():
        ops.set_budget(f"process:{process}", {k: v * share for k, v in limit.items()})
    return in_force


def period_key(config: Config, now: datetime) -> str | None:
    period = str(config.section("budget").get("period", "none"))
    if period == "none":
        return None
    if period == "day":
        return now.date().isoformat()
    if period == "week":
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"
    raise ConfigError(f"[budget] period must be none, day or week, not {period!r}")


def roll_period(store: Store, ops: SqliteOps, config: Config, now: datetime) -> str | None:
    """Close the previous period when it changed; returns the Report ID if it did."""
    key = period_key(config, now)
    if key is None:
        return None
    old = ops.get_pointer(PERIOD)
    if old == key:
        return None
    if not ops.set_pointer(PERIOD, old, key):
        return None  # another process rolled it
    if old is None:
        return None  # first period: nothing to close
    processes = definitions().knob("budget.process_quotas").fields
    scopes = ["global", *(f"process:{p}" for p in processes)]
    usage: dict[str, Any] = {}
    for scope in scopes:
        if not has_scope(ops, scope):
            continue
        b = ops.budget(scope)
        usage[scope] = b
        ops.settle(scope, {}, {k: -v for k, v in b["used"].items()})
    body = Report(
        title="budget_period",
        subject=old,
        summary=f"budget period {old} closed; {key} starts at zero use",
        data={"period": old, "next": key, "scopes": usage},
    )
    return store.put(Record.new("Report", body, ACTOR, now))


def order_queued(
    tasks: Sequence[Task], process_of: Callable[[Task], str], ops: SqliteOps, controls: Controls
) -> list[Task]:
    """Queued tasks in the order the agent pool should try them (11 §4.2)."""
    in_force = effective(shares(controls), bool(active_experiments(ops)))

    def load(process: str) -> float:
        share = in_force.get(process, 0.0)
        scope = f"process:{process}"
        if share <= 0 or not has_scope(ops, scope):
            return float("inf") if share <= 0 else 0.0
        b = ops.budget(scope)
        spent = sum(b["used"].get(k, 0.0) + b["reserved"].get(k, 0.0) for k in RESOURCES)
        return spent / share

    def key(t: Task) -> tuple[int, float, int, str]:  # task IDs are ULIDs: by age
        p = process_of(t)
        return (0 if p == "human" else 1, load(p), -t.priority, t.id)

    return sorted(tasks, key=key)
