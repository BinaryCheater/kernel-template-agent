"""Staged queues with work-in-progress limits and backpressure (11 §4.3, WP23 D610–D612).

``stage_queues.yaml`` maps loop states to stages; ``pipeline.stage_capacity`` (controls,
set by X12) bounds each stage. ``plan`` is pure: given the live loops it decides which
to advance this tick, which are held and why, and which measurement-ready loops go into
one exclusive window (``measure.batch_window``).

Rules:

- Backpressure: a loop in stage S is advanced only while the next stage is below its
  capacity; depths are updated as the plan admits loops, so one tick never overfills a
  stage. The last stage (judge) has no downstream and is never held.
- Dequeue priority: loops that unblock waiting loops first (another loop's ``waiting_on``
  names this loop or a task it created), then the more downstream stage (finishing work
  frees capacity), then the oldest ``state_since``.
- Windows: the ready-to-measure loops are measured together once ``max_items`` are ready
  or the oldest has waited ``window_s``, and only while the open windows are below the
  ``measure`` capacity.

Holds are recorded by the caller as loop data ``deferred`` and an ops event ``stages``;
the depths, waits and throughput over time are derived from the ``loop_step`` spans
(``kta.api.stageview``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from importlib import resources
from typing import Any, cast

import yaml

from kta.core.errors import ConfigError


@dataclass(frozen=True)
class Stage:
    id: str
    does: str
    pool: str
    states: dict[str, tuple[str, ...]]
    window: bool = False


@cache
def stages() -> tuple[Stage, ...]:
    raw = yaml.safe_load(
        resources.files("kta.orchestrator").joinpath("stage_queues.yaml").read_text("utf-8")
    )
    out = tuple(
        Stage(
            id=str(s["id"]),
            does=str(s["does"]),
            pool=str(s["pool"]),
            states={str(k): tuple(str(x) for x in v) for k, v in dict(s["states"]).items()},
            window=bool(s.get("window", False)),
        )
        for s in cast(list[dict[str, Any]], raw["stages"])
    )
    seen: dict[tuple[str, str], str] = {}
    for st in out:
        for kind, names in st.states.items():
            for n in names:
                if (kind, n) in seen:
                    raise ConfigError(
                        f"stage_queues.yaml: {kind}.{n} in {seen[kind, n]} and {st.id}"
                    )
                seen[kind, n] = st.id
    return out


def stage_of(kind: str, state: str) -> str | None:
    for st in stages():
        if state in st.states.get(kind, ()):
            return st.id
    return None


def downstream(stage_id: str) -> str | None:
    ids = [s.id for s in stages()]
    i = ids.index(stage_id)
    return ids[i + 1] if i + 1 < len(ids) else None


def depths(rows: Sequence[Mapping[str, Any]], open_windows: int = 0) -> dict[str, int]:
    d = {s.id: 0 for s in stages()}
    for r in rows:
        sid = stage_of(str(r["kind"]), str(r["state"]))
        if sid is not None and not r["data"].get("blocked"):
            d[sid] += 1
    for s in stages():
        if s.window:
            d[s.id] = open_windows
    return d


def unblocks(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """For each loop, how many other live loops wait on it or on a task it created."""
    owner: dict[str, str] = {}
    for r in rows:
        owner[str(r["id"])] = str(r["id"])
        tasks = cast(list[Any], r["data"].get("waiting_tasks") or [])
        for t in tasks:
            owner[str(t)] = str(r["id"])
    count: dict[str, int] = {}
    for r in rows:
        for key in ("waiting_on_loop", "waiting_on"):
            target = owner.get(str(r["data"].get(key) or ""))
            if target is not None and target != str(r["id"]):
                count[target] = count.get(target, 0) + 1
    return count


@dataclass
class Plan:
    advance: list[str] = field(default_factory=lambda: list[str]())
    window: list[str] = field(default_factory=lambda: list[str]())  # measured in one window
    held: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    depth: dict[str, int] = field(default_factory=lambda: dict[str, int]())


def _age_s(row: Mapping[str, Any], now: datetime) -> float:
    since = row["data"].get("state_since")
    return (now - datetime.fromisoformat(str(since))).total_seconds() if since else 0.0


def plan(
    rows: Sequence[Mapping[str, Any]],
    capacity: Mapping[str, int],
    window: Mapping[str, Any],
    now: datetime,
    open_windows: int = 0,
    kinds: tuple[str, ...] = ("solve",),
    process: str = "solve",
) -> Plan:
    """What one tick of ``process`` may advance (pure). Loops of other kinds or processes
    (expand loops, experiment runs) fill the stages but are not moved here."""
    order = [s.id for s in stages()]
    depth = depths(rows, open_windows)
    ub = unblocks(rows)
    out = Plan(depth=dict(depth))

    def key(r: Mapping[str, Any]) -> tuple[int, int, float, str]:
        sid = stage_of(str(r["kind"]), str(r["state"]))
        pos = order.index(sid) if sid is not None else -1
        return (-ub.get(str(r["id"]), 0), -pos, -_age_s(r, now), str(r["id"]))

    mine = [
        r
        for r in rows
        if r["kind"] in kinds
        and r["data"].get("process", "solve") == process
        and not r["data"].get("blocked")
    ]
    ready: list[Mapping[str, Any]] = []
    for r in sorted(mine, key=key):
        rid, sid = str(r["id"]), stage_of(str(r["kind"]), str(r["state"]))
        if sid is None:
            out.advance.append(rid)
            continue
        nxt = downstream(sid)
        if nxt is not None and next(s for s in stages() if s.id == nxt).window:
            ready.append(r)
            continue
        if nxt is not None and depth[nxt] >= int(capacity.get(nxt, 1)):
            out.held[rid] = f"backpressure: {nxt} full ({depth[nxt]}/{capacity.get(nxt, 1)})"
            continue
        out.advance.append(rid)
        depth[sid] -= 1
        if nxt is not None:
            depth[nxt] += 1
    _windows(ready, capacity, window, now, depth, out)
    return out


def _windows(
    ready: list[Mapping[str, Any]],
    capacity: Mapping[str, int],
    window: Mapping[str, Any],
    now: datetime,
    depth: dict[str, int],
    out: Plan,
) -> None:
    if not ready:
        return
    win = next(s.id for s in stages() if s.window)
    after = downstream(win)
    full = int(window["max_items"]) <= len(ready)
    due = max(_age_s(r, now) for r in ready) >= float(window["window_s"])
    if depth[win] >= int(capacity.get(win, 1)):
        reason = f"backpressure: {win} full ({depth[win]}/{capacity.get(win, 1)})"
    elif not (full or due):
        reason = f"collecting for a measurement window ({len(ready)}/{window['max_items']})"
    else:
        room = int(window["max_items"])
        if after is not None:
            room = min(room, max(0, int(capacity.get(after, 1)) - depth[after]))
        if room == 0 and after is not None:
            reason = f"backpressure: {after} full ({depth[after]}/{capacity.get(after, 1)})"
        else:
            take, rest = ready[:room], ready[room:]
            out.window = [str(r["id"]) for r in take]
            depth[win] += 1
            for r in rest:
                out.held[str(r["id"])] = "waiting for the next measurement window"
            return
    for r in ready:
        out.held[str(r["id"])] = reason
