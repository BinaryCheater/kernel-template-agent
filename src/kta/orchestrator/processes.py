"""Process main loops (docs/design/11 §1). WP6: ``solve``, ``maintain``; WP11 the others.

A process is a long-running foreground program (``kta process start <name>``). One
instance per name runs at a time: it holds the ops lock ``process:<name>`` and renews it
every tick. ``kta process stop <name>`` sets the pointer ``process:<name>:stop``; the
process exits at its next tick (D273). Ticks are separate functions so tests drive them
without waiting.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import UlidFactory
from kta.orchestrator import quotas
from kta.orchestrator.budget import configure_scopes
from kta.orchestrator.pipeline import Services
from kta.orchestrator.proc_calibrate import calibrate_tick
from kta.orchestrator.proc_experiment import experiment_tick
from kta.orchestrator.proc_learn import learn_tick
from kta.orchestrator.proc_summarize import summarize_tick
from kta.orchestrator.runner import Runner

log = logging.getLogger("kta.orchestrator.processes")
ALL = ("solve", "learn", "calibrate", "maintain", "experiment", "summarize")
IMPLEMENTED = ALL
SPAN_RETENTION = timedelta(days=90)


@dataclass
class TickReport:
    advanced: list[str] = field(default_factory=lambda: list[str]())
    checked: list[str] = field(default_factory=lambda: list[str]())
    expired: list[str] = field(default_factory=lambda: list[str]())
    pruned_spans: int = 0


def solve_tick(runner: Runner) -> TickReport:
    """Check submitted tasks of solve loops, then advance every live solve instance once."""
    rep = TickReport()
    ops = runner.svc.ops
    solve_loops = {
        row["id"]
        for row in ops.list_loops("solve")
        if row["data"].get("process", "solve") == "solve"
    }
    for t in ops.list_tasks(states=("submitted",)):
        if t.work_unit in solve_loops:
            runner.queue.check(t.id)
            rep.checked.append(t.id)
    for row in ops.list_loops("solve"):
        if row["data"].get("process", "solve") != "solve":
            continue  # experiment runs are advanced by the experiment process (WP11)
        if runner.advance(str(row["id"])) is not None:
            rep.advanced.append(str(row["id"]))
    return rep


def maintain_tick(runner: Runner) -> TickReport:
    """Return lost leases to the queue, expire silent running tasks, prune old spans."""
    quotas.roll_period(runner.svc.store, runner.svc.ops, runner.ctx.config, runner.ctx.clock.now())
    rep = TickReport(expired=runner.queue.expire())
    rep.pruned_spans = runner.svc.ops.prune_spans(runner.ctx.clock.now() - SPAN_RETENTION)
    return rep


TICKS: dict[str, Callable[[Runner], object]] = {
    "solve": solve_tick,
    "maintain": maintain_tick,
    "learn": learn_tick,
    "calibrate": calibrate_tick,
    "experiment": experiment_tick,
    "summarize": summarize_tick,
}


def stop_requested(svc: Services, name: str, since: datetime | None = None) -> bool:
    """A stop request for ``name`` made at or after ``since`` (any request when None)."""
    raw = svc.ops.get_pointer(f"process:{name}:stop")
    if not raw:
        return False
    return since is None or datetime.fromisoformat(raw) >= since


def request_stop(svc: Services, name: str) -> None:
    """Ask ``name`` to stop. The request is a timestamp, so a process started later
    ignores it and a request made while a process starts is not lost."""
    key = f"process:{name}:stop"
    current = svc.ops.get_pointer(key)
    if not svc.ops.set_pointer(key, current, svc.ctx.clock.now().isoformat()):
        raise ConfigError(f"could not set {key}")


def run(
    svc: Services, name: str, interval_s: float | None = None, max_ticks: int | None = None
) -> int:
    """Run process ``name`` until stopped; returns the number of ticks."""
    if name not in ALL:
        raise ConfigError(f"unknown process {name!r}; one of {', '.join(ALL)}")
    if name not in IMPLEMENTED:
        raise NotSupported(f"process {name} (WP6 implements {', '.join(IMPLEMENTED)}; WP11)")
    started = svc.ctx.clock.now()
    holder = f"{name}:{UlidFactory(svc.ctx.clock).new()}"
    lock = f"process:{name}"
    lease = int(max(60.0, 4 * (interval_s or 10.0)))
    if not svc.ops.acquire(lock, holder, lease):
        raise ConfigError(f"process {name} is already running (lock {lock})")
    configure_scopes(svc.ops, svc.ctx.config)
    quotas.apply_quotas(svc.ops, svc.ctx.config)  # WP11: experiment share when none runs
    runner = Runner(svc, holder)
    pause = threading.Event()
    ticks = 0
    try:
        while max_ticks is None or ticks < max_ticks:
            if stop_requested(svc, name, started):
                break
            svc.ops.acquire(lock, holder, lease)
            rep = TICKS[name](runner)
            ticks += 1
            log.info("%s tick %d: %s", name, ticks, rep)
            pause.wait(interval_s if interval_s is not None else 10.0)
    finally:
        svc.ops.release(lock, holder)
    return ticks
