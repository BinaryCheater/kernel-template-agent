"""The learn process (docs/design/11 §1): expand loops, the learning backlog, pool snapshots.

One tick:

1. runs the done-checks of submitted tasks booked to ``learn`` (expand-loop tasks, and
   the T3s the summarize process creates);
2. starts an expand loop for every Reference on the backlog (``kta learn add``, pointer
   ``learn:backlog``) that has no live one;
3. advances every live expand loop once (the expand work itself is WP10b's);
4. writes a new PoolSnapshot when knowledge was accepted since the current one (or no
   snapshot exists yet) and moves ``current_pool`` by compare-and-set. Only this process
   writes snapshots (11 §5); on a conflict it retries next tick (D400).

A snapshot's members are all usable records of the knowledge kinds (D400): the pool as
retrieval sees it at that moment. Experiments pin a snapshot ID in their Protocol and are
not affected by later snapshots.
"""

from __future__ import annotations

import json
from typing import cast

from kta.core.errors import ConfigError
from kta.core.schemas import Actor
from kta.knowledge import pool
from kta.orchestrator.machines import machine
from kta.orchestrator.outputs import RESULT_TITLE
from kta.orchestrator.proc_common import ProcReport, action, check_submitted, process_of
from kta.orchestrator.runner import Runner
from kta.store.base import Store
from kta.store.snapshot import CURRENT_POOL, SnapshotConflict, snapshot
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps

BACKLOG = "learn:backlog"
KNOWLEDGE_KINDS = ("Pattern", "Structure", "Template", "Chain", "Family", "Reference")
ACTOR = Actor(type="system", name="kta.orchestrator.learn")


def backlog(ops: SqliteOps) -> list[str]:
    raw = ops.get_pointer(BACKLOG)
    return list(json.loads(raw)) if raw else []


def add_to_backlog(store: Store, ops: SqliteOps, reference_id: str) -> list[str]:
    """Put a Reference on the learning backlog (idempotent); returns the backlog."""
    if store.get(reference_id).kind != "Reference":
        raise ConfigError(f"{reference_id} is not a Reference")
    old = ops.get_pointer(BACKLOG)
    items = backlog(ops)
    if reference_id not in items:
        items.append(reference_id)
        if not ops.set_pointer(BACKLOG, old, json.dumps(items)):
            raise ConfigError(f"{BACKLOG} changed concurrently; try again")
    return items


def _start_backlog(runner: Runner, rep: ProcReport) -> None:
    ops = runner.svc.ops
    old = ops.get_pointer(BACKLOG)
    items = backlog(ops)
    if not items:
        return
    for ref in items:
        name = str(runner.svc.store.get(ref).body.get("name", ref[:12]))
        rep.add("started", runner.start("expand", ref, f"expand {name}"))
    if not ops.set_pointer(BACKLOG, old, "[]"):
        raise ConfigError(f"{BACKLOG} changed concurrently")


def members(store: Store) -> list[str]:
    out: list[str] = []
    for kind in KNOWLEDGE_KINDS:
        out += [
            rid for rid, _ in pool.with_ids(pool.records(store, kind)) if pool.usable(store, rid)
        ]
    return sorted(out)


def needs_snapshot(store: Store, ops: SqliteOps) -> bool:
    current = ops.get_pointer(CURRENT_POOL)
    if current is None:
        return True
    since = store.get(current).created_at
    for rec in store.query("Report", {"title": RESULT_TITLE}, order="-created_at", limit=20):
        if rec.created_at > since and rec.refs.get("outputs"):
            return True
    return False


def make_snapshot(runner: Runner, rep: ProcReport) -> None:
    svc = runner.svc
    if not needs_snapshot(svc.store, svc.ops):
        return
    parent = svc.ops.get_pointer(CURRENT_POOL)
    with action(runner, "learn", "pool snapshot", parent or "pool") as out:
        found = members(svc.store)
        prev = set(svc.store.get(parent).body["members"]) if parent else set[str]()
        if parent and set(found) == prev:
            out["unchanged"] = True
            return
        note = f"{len(found)} members; +{len(set(found) - prev)} -{len(prev - set(found))}"
        try:
            sid = snapshot(
                cast(SqliteStore, svc.store),
                svc.ops,
                found,
                parent,
                note,
                ACTOR,
                runner.ctx.clock.now(),
            )
        except SnapshotConflict as exc:  # another writer moved the pointer: retry next tick
            rep.blocked["snapshot"] = str(exc)
            out["conflict"] = True
            return
        out["snapshot"] = sid
        rep.add("snapshots", sid)


def learn_tick(runner: Runner) -> ProcReport:
    rep = ProcReport("learn")
    expand = {str(r["id"]) for r in runner.svc.ops.list_loops("expand")}
    check_submitted(
        runner, rep, lambda t: t.work_unit in expand or process_of(runner, t) == "learn"
    )
    _start_backlog(runner, rep)
    m = machine("expand")
    for row in runner.svc.ops.list_loops("expand"):
        if m.terminal(str(row["state"])):
            continue
        if runner.advance(str(row["id"])) is not None:
            rep.add("advanced", str(row["id"]))
    make_snapshot(runner, rep)
    return rep
