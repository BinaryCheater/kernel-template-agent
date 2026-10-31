"""``kta runner``, ``kta profile approve``, ``kta unit close``: the agent runner (D369).

The runner is the layer-9 process that joins the orchestrator's task queue (layer 7) and
the agents module (layer 8): it leases the next queued agent task for a slot (budget
reservation, pause), moves it to running, calls ``kta.agents.runner.run_step`` with the
last rejection and any human note, and then checks a submitted task or abandons one the
agent left without submitting. One step at a time per process; one process per slot
(ops lock ``process:agent:<slot>``, D375); ``kta runner stop`` stops every slot.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import threading
from collections.abc import Callable
from typing import Any

from kta.agents.cards import RetryNote
from kta.agents.config import agents_config
from kta.agents.profile import approve_profile
from kta.agents.runner import Deps, run_step
from kta.agents.units import close_work_unit
from kta.cli.exec_cmds import ContextFactory, _out, command
from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import UlidFactory
from kta.core.jobs import Job, JobResult
from kta.orchestrator import control, processes, release
from kta.orchestrator.budget import BudgetRefused
from kta.orchestrator.pipeline import Services
from kta.orchestrator.quotas import order_queued
from kta.orchestrator.tasks import UNASSIGNED_PROFILE, TaskQueue
from kta.store.base import Task

log = logging.getLogger("kta.cli.agent_runner")
NAME = "agent"
AGENT_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")
CANCEL_POLL_S = 5.0


def next_task(q: TaskQueue) -> Task | None:
    """The first queued agent task with an approved profile (D366) that the release stage
    lets start (WP21, D485), in the agent pool's order: your tasks first, then by process
    quota load, priority and age (D372). Held tasks stay queued (``release.held_calls``)."""
    controls = q.controls()
    queued = [t for t in q.ops.list_tasks(states=("queued",)) if t.card_type in AGENT_TYPES]
    ordered = order_queued(
        queued, lambda t: str(q.info(t.id).get("process", "solve")), q.ops, controls
    )
    return next(
        (
            t
            for t in ordered
            if q.card(t).profile != UNASSIGNED_PROFILE
            and release.hold_reason(q.ops, t, controls) is None
        ),
        None,
    )


class TrackingRunner:
    """The Executor as the agent step sees it, remembering the step's jobs so that a
    cancelled task's agent job can be cancelled (D374). Everything else is delegated."""

    def __init__(self, ex: Any) -> None:
        self.ex = ex
        self.jobs: list[str] = []
        self._lock = threading.Lock()

    def submit(self, job: Job) -> str:
        job_id = str(self.ex.submit(job))
        with self._lock:
            self.jobs.append(job_id)
        return job_id

    def wait(self, job_id: str) -> JobResult:
        return self.ex.wait(job_id)

    def run(self, job: Job) -> JobResult:
        return self.wait(self.submit(job))

    def read_blob(self, sha256: str) -> bytes:
        return self.ex.read_blob(sha256)

    def put_blob(self, data: bytes, media: str) -> str:
        return self.ex.put_blob(data, media)

    def cancel_all(self) -> list[str]:
        """Cancel every job of this step; NotSupported when the Executor cannot cancel."""
        cancel = getattr(self.ex, "cancel", None)
        if cancel is None:
            raise NotSupported("the Executor cannot cancel a running job (WP2 cancel)")
        with self._lock:
            jobs = list(self.jobs)
        for job_id in jobs:
            cancel(job_id)
        return jobs


def watch_cancel(
    q: TaskQueue, task_id: str, jobs: TrackingRunner, done: threading.Event, poll_s: float
) -> dict[str, Any]:
    """Until ``done``: when the task is cancelled, cancel its agent jobs (D374)."""
    while not done.wait(poll_s):
        if q.get(task_id).state == "cancelled":
            try:
                return {"cancelled_jobs": jobs.cancel_all()}
            except NotSupported as exc:
                log.warning("task %s cancelled but its agent job runs on: %s", task_id, exc)
                return {"cancel_unsupported": str(exc)}
    return {}


StepFn = Callable[..., Any]


def step_once(
    svc: Services,
    q: TaskQueue,
    slot: int,
    batch: str | None,
    step: StepFn = run_step,
    poll_s: float = CANCEL_POLL_S,
) -> dict[str, Any]:
    """Lease, run one agent step, then check or abandon. Returns what happened.

    The batch is the task's own (an experiment run's, D372), else ``batch`` (the command
    line), else ``[agents] batch``. A watcher cancels the agent job when the task is
    cancelled meanwhile (D374).
    """
    task = next_task(q)
    if task is None:
        return {"task": None}
    owner = f"agent:{slot}"
    try:
        release.admit(q.ops, q.get(task.id), q.controls())  # re-checked just before the lease
        q.lease(task.id, owner)
    except (BudgetRefused, control.Paused) as exc:
        return {"task": task.id, "skipped": str(exc)}
    q.start(task.id, owner)
    failures, note = q.retry_context(task.id)
    jobs = TrackingRunner(svc.executor)
    deps = Deps(svc.store, svc.ops, jobs, agents_config(svc.ctx))
    retry = RetryNote(failures) if failures else None
    use_batch = q.info(task.id).get("batch") or batch
    done = threading.Event()
    watched: dict[str, Any] = {}
    watcher = threading.Thread(
        target=lambda: watched.update(watch_cancel(q, task.id, jobs, done, poll_s)), daemon=True
    )
    watcher.start()
    try:
        result = step(
            svc.ctx, q.get(task.id), retry=retry, note=note, deps=deps, lane=owner, batch=use_batch
        )
    finally:
        done.set()
        watcher.join()
    after = q.get(task.id).state
    out: dict[str, Any] = {
        "task": task.id,
        "run": result.run_id,
        "after_step": after,
        "batch": use_batch,
        **watched,
    }
    if after == "submitted":
        out["verdict"] = q.check(task.id).state
    elif after == "running":
        out["verdict"] = q.abandon(task.id, f"agent step {result.run_id} ended without kta submit")
    return out


def cmd_runner_start(args: argparse.Namespace, make_ctx: ContextFactory) -> int:
    with command(make_ctx, NAME, f"kta runner start --slot {args.slot}") as ctx:
        svc = Services.open(ctx)
        top = svc.ctx.controls.agent_instances["max"]
        if not 1 <= args.slot <= top:
            raise ConfigError(f"slot {args.slot}: budget.agent_instances allows slots 1..{top}")
        q = TaskQueue(ctx, svc.store, svc.ops, svc)
        started = ctx.clock.now()
        holder = f"{NAME}:{UlidFactory(ctx.clock).new()}"
        lock = f"process:{NAME}:{args.slot}"  # one runner per slot (D375)
        if not svc.ops.acquire(lock, holder, 3600):
            raise ConfigError(f"slot {args.slot} already has a runner (lock {lock})")
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
