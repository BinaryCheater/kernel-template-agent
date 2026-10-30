"""``kta runner``, ``kta profile approve``, ``kta unit close``: the agent runner (D369).

The runner is the layer-9 process that joins the orchestrator's task queue (layer 7) and
the agents module (layer 8): it leases the next queued agent task for a slot (budget
reservation, pause), moves it to running, calls ``kta.agents.runner.run_step`` with the
last rejection and any human note, and then checks a submitted task or abandons one the
agent left without submitting. One step at a time per process; one process per name
(ops lock ``process:agent``), stopped by ``kta runner stop``.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import threading
from typing import Any

from kta.agents.cards import RetryNote
from kta.agents.config import agents_config
from kta.agents.profile import approve_profile
from kta.agents.runner import Deps, run_step
from kta.agents.units import close_work_unit
from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.core.errors import ConfigError
from kta.core.ids import UlidFactory
from kta.orchestrator import control, processes
from kta.orchestrator.budget import BudgetRefused
from kta.orchestrator.pipeline import Services
from kta.orchestrator.tasks import UNASSIGNED_PROFILE, TaskQueue
from kta.store.base import Task

log = logging.getLogger("kta.cli.agent_runner")
NAME = "agent"
AGENT_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")


def next_task(q: TaskQueue) -> Task | None:
    """Highest priority, oldest queued task with an approved profile (D366)."""
    queued = [t for t in q.ops.list_tasks(states=("queued",)) if t.card_type in AGENT_TYPES]
    queued.sort(key=lambda t: -t.priority)  # stable: oldest first within a priority
    for t in queued:
        if q.card(t).profile != UNASSIGNED_PROFILE:
            return t
    return None


def step_once(svc: Services, q: TaskQueue, slot: int, batch: str | None) -> dict[str, Any]:
    """Lease, run one agent step, then check or abandon. Returns what happened."""
    task = next_task(q)
    if task is None:
        return {"task": None}
    owner = f"agent:{slot}"
    try:
        q.lease(task.id, owner)
    except (BudgetRefused, control.Paused) as exc:
        return {"task": task.id, "skipped": str(exc)}
    q.start(task.id, owner)
    failures, note = q.retry_context(task.id)
    deps = Deps(svc.store, svc.ops, svc.executor, agents_config(svc.ctx))
    retry = RetryNote(failures) if failures else None
    result = run_step(
        svc.ctx, q.get(task.id), retry=retry, note=note, deps=deps, lane=owner, batch=batch
    )
    after = q.get(task.id).state
    out: dict[str, Any] = {"task": task.id, "run": result.run_id, "after_step": after}
    if after == "submitted":
        out["verdict"] = q.check(task.id).state
    elif after == "running":
        out["verdict"] = q.abandon(task.id, f"agent step {result.run_id} ended without kta submit")
    # A task cancelled during the step stays cancelled; its agent process was not stopped
    # early because the Executor cannot cancel a running job yet (Known issues).
    return out


def cmd_runner_start(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, NAME, "kta runner start") as ctx:
        svc = Services.open(ctx)
        q = TaskQueue(ctx, svc.store, svc.ops, svc)
        started = ctx.clock.now()
        holder = f"{NAME}:{UlidFactory(ctx.clock).new()}"
        lock = f"process:{NAME}"
        if not svc.ops.acquire(lock, holder, 3600):
            raise ConfigError("an agent runner is already running (lock process:agent)")
        pause = threading.Event()
        steps: list[dict[str, Any]] = []
        try:
            while args.ticks is None or len(steps) < args.ticks:
                if processes.stop_requested(svc, NAME, started):
                    break
                svc.ops.acquire(lock, holder, 3600)
                if control.paused_scope(svc.ops, [f"process:{NAME}", "global"]) is None:
                    steps.append(step_once(svc, q, args.slot, args.batch))
                    log.info("agent runner step: %s", steps[-1])
                else:
                    steps.append({"paused": True})
                pause.wait(args.interval)
        finally:
            svc.ops.release(lock, holder)
    _out(args, {"steps": steps}, "\n".join(str(s) for s in steps) or "(no steps)")
    return 0


def cmd_runner_stop(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    processes.request_stop(Services.open(make_ctx("runner-stop")), NAME)
    _out(args, {"process": NAME, "stop": True}, "stop requested for the agent runner")
    return 0


def cmd_profile_approve(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "profile-approve", f"kta profile approve {args.profile}") as ctx:
        svc = Services.open(ctx)
        action = approve_profile(
            svc.store, svc.ops, args.profile, getpass.getuser(), args.reason, ctx.clock.now()
        )
    _out(args, {"profile": args.profile, "action": action}, f"current_profile -> {args.profile}")
    return 0


def cmd_unit_close(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, "unit-close", f"kta unit close {args.work_unit}") as ctx:
        svc = Services.open(ctx)
        report = close_work_unit(
            ctx, svc.store, svc.ops, svc.executor, agents_config(ctx), args.work_unit
        )
    _out(args, {"work_unit": args.work_unit, "archive": report}, f"archived: {report}")
    return 0


def add_agent_runner_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser], make_ctx: ContextFactory
) -> None:
    def leaf(parent: Any, name: str, fn: Any, h: str) -> argparse.ArgumentParser:
        p = parent.add_parser(name, help=h)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=lambda a: fn(a, make_ctx))
        return p

    runner = sub.add_parser("runner", help="the agent runner (leases tasks, runs agent steps)")
    rc = runner.add_subparsers(dest="runner_command", required=True)
    s = leaf(rc, "start", cmd_runner_start, "run agent steps in the foreground until stopped")
    s.add_argument("--slot", type=int, default=1)
    s.add_argument("--batch", help="batch root (default [agents] batch)")
    s.add_argument("--interval", type=float, default=10.0)
    s.add_argument("--ticks", type=int)
    leaf(rc, "stop", cmd_runner_stop, "ask the running agent runner to stop")
    profile = sub.add_parser("profile", help="agent profiles")
    pc = profile.add_subparsers(dest="profile_command", required=True)
    a = leaf(pc, "approve", cmd_profile_approve, "point current_profile at a profile (human)")
    a.add_argument("profile")
    a.add_argument("--reason", required=True)
    unit = sub.add_parser("unit", help="work units")
    uc = unit.add_subparsers(dest="unit_command", required=True)
    leaf(uc, "close", cmd_unit_close, "archive and remove a work unit").add_argument("work_unit")
