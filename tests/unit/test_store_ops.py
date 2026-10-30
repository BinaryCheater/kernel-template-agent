from __future__ import annotations

import pytest

from fakes.clock import ManualClock
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.schemas import Record
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from storehelp import T0, actor


def _card(store: SqliteStore, task_type: str, goal: str = "g") -> str:
    profile = store.put(
        Record.new(
            "AgentProfile",
            {
                "name": "p",
                "version": 1,
                "provider": "codex",
                "cli_version": "1",
                "auth": "subscription",
                "isolation": "config",
                "model": "m",
                "prefix_template": "a" * 64,
            },
            actor("human"),
            T0,
        )
    )
    body = {"type": task_type, "goal": goal, "profile": profile, "work_unit": "wu"}
    return store.put(Record.new("TaskCard", body, actor("system"), T0))


def test_task_lease_renew_finish(store: SqliteStore, ops: SqliteOps, clock: ManualClock) -> None:
    low = ops.enqueue(_card(store, "T1", "low"), 1, "wu1")
    high = ops.enqueue(_card(store, "T1", "high"), 5, "wu2")
    other = ops.enqueue(_card(store, "T3"), 9, "wu3")
    got = ops.lease("w1", ["T1"], 60)
    assert got is not None and got.id == high and got.attempts == 1 and got.lease_owner == "w1"
    assert ops.renew(high, "w1", 60)
    assert not ops.renew(high, "w2", 60)
    nxt = ops.lease("w2", ["T1"], 60)
    assert nxt is not None and nxt.id == low
    assert ops.lease("w3", ["T1"], 60) is None
    clock.advance(61)
    assert not ops.renew(low, "w2", 60)  # expired
    again = ops.lease("w3", ["T1"], 60)
    assert again is not None and again.id == high and again.attempts == 2
    ops.finish(high, "done")
    with pytest.raises(ConfigError, match="already"):
        ops.finish(high, "failed")
    with pytest.raises(ConfigError, match="state"):
        ops.finish(other, "weird")
    assert ops.task(high).state == "done"
    assert ops.lease("w4", [], 60) is None


def test_enqueue_requires_a_task_card(ops: SqliteOps) -> None:
    with pytest.raises(ConfigError, match="TaskCard"):
        ops.enqueue("0" * 64, 1, "wu")


def test_locks(ops: SqliteOps, clock: ManualClock) -> None:
    assert ops.acquire("gpu:spark:0", "job-a", 30)
    assert not ops.acquire("gpu:spark:0", "job-b", 30)
    assert ops.acquire("gpu:spark:0", "job-a", 30)  # holder extends
    with pytest.raises(PermissionDenied):
        ops.release("gpu:spark:0", "job-b")
    clock.advance(31)
    assert ops.acquire("gpu:spark:0", "job-b", 30)  # expired lease is taken over
    ops.release("gpu:spark:0", "job-b")
    ops.release("gpu:spark:0", "job-b")  # releasing a free lock is a no-op
    assert ops.acquire("gpu:spark:0", "job-c", 30)


def test_budgets(ops: SqliteOps) -> None:
    with pytest.raises(ConfigError, match="no budget"):
        ops.reserve("loop-1", {"gpu_s": 1})
    ops.set_budget("loop-1", {"gpu_s": 100, "steps": 10})
    assert ops.reserve("loop-1", {"gpu_s": 60})
    assert not ops.reserve("loop-1", {"gpu_s": 50})
    ops.settle("loop-1", {"gpu_s": 60}, {"gpu_s": 20, "steps": 1})
    assert ops.reserve("loop-1", {"gpu_s": 80})
    assert ops.budget("loop-1")["used"] == {"gpu_s": 20, "steps": 1}
    with pytest.raises(ConfigError, match="more than was reserved"):
        ops.settle("loop-1", {"gpu_s": 81}, {})


def test_pointer_cas(ops: SqliteOps) -> None:
    assert ops.get_pointer("p") is None
    assert ops.set_pointer("p", None, "a")
    assert not ops.set_pointer("p", None, "b")
    assert not ops.set_pointer("p", "x", "b")
    assert ops.set_pointer("p", "a", "b")
    assert ops.get_pointer("p") == "b"


def test_every_change_is_an_ops_event(store: SqliteStore, ops: SqliteOps) -> None:
    ops.set_pointer("p", None, "a")
    ops.acquire("l", "h", 10)
    ops.heartbeat("spark", "spark-host", True)
    ops.put_loop("loop-1", "solve", "case-1", "running", {"step": 1})
    assert ops.get_loop("loop-1")["data"] == {"step": 1}
    events = ops.events(0)
    assert [e.table_name for e in events] == ["pointers", "locks", "workers", "loops"]
    assert [e.seq for e in ops.events(events[1].seq)] == [events[2].seq, events[3].seq]
