"""Runner and processes on a real test store: transitions are applied with spans and
records, failures block, restarts continue from the stored state, the inbox fills."""

from __future__ import annotations

import threading

import pytest

from fakes.clock import ManualClock
from knowhelp import seed
from kta.core.context import Context
from kta.orchestrator import processes
from kta.orchestrator.inbox import inbox
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner
from kta.orchestrator.tasks import TaskQueue
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import ctx_with, world


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    seed(ctx, store)
    return Services(ctx, store, ops)


def _run(r: Runner, loop_id: str, n: int) -> list[str]:
    seen = []
    for _ in range(n):
        t = r.advance(loop_id)
        if t is None:
            break
        seen.append(f"{t.src}->{t.to}")
        if t.to is None:
            break
    return seen


def test_solve_loop_reaches_waiting_with_a_t1(svc: Services) -> None:
    w = world(svc)
    r = Runner(svc, "test")
    loop_id = r.start("solve", w["case"], "gemm 256")
    assert _run(r, loop_id, 5) == [
        "new->bounded",
        "bounded->retrieving",
        "retrieving->waiting",
        "waiting->None",
    ]
    row = svc.ops.get_loop(loop_id)
    assert row["state"] == "waiting"
    task = svc.ops.task(row["data"]["waiting_on"])
    assert (task.card_type, task.state, task.work_unit) == ("T1", "queued", loop_id)
    card = svc.store.get(task.card_id).body
    assert w["case"] in card["inputs"] and card["done_checks"][0] == "delivery_note"
    bound = svc.store.get(row["data"]["bounds"]["s0"])
    assert bound.kind == "Bound" and bound.refs["case"] == w["case"]
    spans = [s for s in svc.ops.spans(trace_id=loop_id) if s.kind == "loop_step"]
    assert sorted(s.name for s in spans) == ["bounded", "new", "retrieving", "waiting"]
    assert [s.name for s in spans if s.end_ts is None] == ["waiting"]  # one span per state


def test_restart_continues_from_the_stored_state(svc: Services) -> None:
    w = world(svc)
    loop_id = Runner(svc, "first").start("solve", w["case"], "c")
    _run(Runner(svc, "first"), loop_id, 2)
    assert svc.ops.get_loop(loop_id)["state"] == "retrieving"
    # A fresh process (new Runner, same store) picks up where the first stopped.
    assert _run(Runner(svc, "second"), loop_id, 1) == ["retrieving->waiting"]
    assert Runner(svc, "third").start("solve", w["case"], "c") == loop_id


def test_missing_hw_model_blocks_and_raises_an_inbox_item(svc: Services) -> None:
    w = world(svc, target="other-gpu", hw=False)
    r = Runner(svc, "test")
    loop_id = r.start("solve", w["case"], "c")
    assert _run(r, loop_id, 3) == ["new->bounded"]  # the bounded work raised NotSupported
    row = svc.ops.get_loop(loop_id)
    assert row["state"] == "bounded" and row["data"]["blocked"]["error"] == "NotSupported"
    assert any(i.trigger == "Loop blocked in bounded" for i in inbox(svc))
    r.start("solve", w["case"], "c")  # starting again clears the block
    assert not svc.ops.get_loop(loop_id)["data"]["blocked"]


def test_attention_limit_defers_task_creation(svc: Services, ctx: Context) -> None:
    w = world(svc)
    limited = Services(ctx_with(ctx, attention_limit=0), svc.store, svc.ops)
    r = Runner(limited, "test")
    loop_id = r.start("solve", w["case"], "c")
    _run(r, loop_id, 3)
    row = svc.ops.get_loop(loop_id)
    assert row["state"] == "retrieving" and row["data"]["deferred"]
    assert svc.ops.list_tasks(loop_id) == []


def test_cancelled_task_escalates_the_loop(svc: Services) -> None:
    w = world(svc)
    r = Runner(svc, "test")
    loop_id = r.start("solve", w["case"], "c")
    _run(r, loop_id, 3)
    task = svc.ops.get_loop(loop_id)["data"]["waiting_on"]
    TaskQueue(svc.ctx, svc.store, svc.ops).cancel(task, "not now")
    r.advance(loop_id)
    assert svc.ops.get_loop(loop_id)["data"]["escalated"] is True
    triggers = {i.trigger for i in inbox(svc)}
    assert "Loop needs you (round limit or stuck task)" in triggers


def test_solve_process_ticks_and_stops(svc: Services, clock: ManualClock) -> None:
    w = world(svc)
    Runner(svc, "cli").start("solve", w["case"], "c")
    assert processes.run(svc, "solve", interval_s=0, max_ticks=4) == 4
    assert svc.ops.list_loops("solve")[0]["state"] == "waiting"
    processes.request_stop(svc, "solve")
    clock.advance(1)
    assert processes.run(svc, "solve", interval_s=0, max_ticks=1) == 1  # an older request
    worker = threading.Thread(target=processes.run, args=(svc, "solve", 0.01))
    worker.start()
    processes.request_stop(svc, "solve")  # the clock is fixed: not older than the start
    worker.join(timeout=30)
    assert not worker.is_alive()


def test_maintain_tick_expires_leases(svc: Services, clock: ManualClock) -> None:
    w = world(svc)
    r = Runner(svc, "cli")
    loop_id = r.start("solve", w["case"], "c")
    _run(r, loop_id, 3)
    task = svc.ops.get_loop(loop_id)["data"]["waiting_on"]
    r.queue.lease(task, "agent:1")
    clock.advance(r.settings.lease_s + 1)
    assert processes.maintain_tick(r).expired == [task]
    assert svc.ops.task(task).state == "queued"


def test_unknown_processes_say_so(svc: Services) -> None:
    from kta.core.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown process"):
        processes.run(svc, "teach", max_ticks=1)
