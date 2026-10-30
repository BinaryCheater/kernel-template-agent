"""Task queue and lifecycle on a real test store: done-checks, retries, escalation,
suspension, cancellation, lease expiry and budget reservation (docs/design/10 §3, 11 §3)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from fakes.clock import ManualClock
from knowhelp import seed, source_id, template_body
from kta.core.context import Context
from kta.core.errors import ConfigError
from kta.core.schemas import Actor, Record
from kta.orchestrator import budget
from kta.orchestrator.budget import BudgetRefused
from kta.orchestrator.machines import ANY_STATE, machine
from kta.orchestrator.pipeline import Services, files_of
from kta.orchestrator.tasks import TaskQueue
from kta.orchestrator.transition import TaskRequest
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import world
from solvehelp import GEMM_DIR
from storehelp import T0

NOTE = "# What was done\nx\n# Fixtures and simplifications\nnone\n# Known issues\nnone\n"
HUMAN = Actor(type="human", name="test")
OWNER = "human:test"


@pytest.fixture
def q(ctx: Context, store: SqliteStore, ops: SqliteOps) -> TaskQueue:
    seed(ctx, store)
    return TaskQueue(ctx, store, ops)


def _t2_files(store: SqliteStore, unmatched: int = 1, gaps: int = 1) -> dict[str, bytes]:
    tx = {"level": 3, "pattern_guess": "sync", "description": "per-slot barrier", "where": "x"}
    doc = {
        "chain": {
            "workload": "gemm",
            "items": [{"key": "a", "template": "async_transfer@1", "applies_to": "A tile"}],
        },
        "reconstruction": {
            "matched": 1,
            "total": 1 + unmatched,
            "notes": "test",
            "unmatched_transactions": [tx] * unmatched,
        },
        "gaps": [{"missing": "template", "description": "barrier", "attribution": "space"}] * gaps,
    }
    return {"reconstruction.yaml": yaml.safe_dump(doc).encode()}


def _submit(q: TaskQueue, task: str, files: dict[str, bytes], note: str = NOTE) -> str:
    if q.get(task).state == "queued":
        q.lease(task, OWNER)
    q.start(task, OWNER)
    return q.submit(task, OWNER, files, note, HUMAN)


def _t2(q: TaskQueue, store: SqliteStore) -> str:
    src = source_id(store)
    body = {"name": "ref", "problem": "gemm", "build": "make"}
    ref = store.put(Record.new("Reference", body, HUMAN, T0, refs={"source": src}))
    return q.create(TaskRequest("T2", "decompose", [ref]), "L1", "learn")


def test_t2_accepted_writes_reconstruction_and_gaps(q: TaskQueue, store: SqliteStore) -> None:
    task = _t2(q, store)
    _submit(q, task, _t2_files(store))
    v = q.check(task)
    assert v.state == "accepted" and all(o.passed for o in v.outcomes)
    assert q.get(task).state == "accepted"
    result = store.query("Report", {"title": "task_result"}, limit=1)[0]
    outputs = result.body["data"]["outputs"]
    assert store.get(outputs["reconstruction"]).kind == "Reconstruction"
    assert [store.get(g["id"]).kind for g in outputs["gaps"]] == ["Gap"]
    assert store.get(outputs["reconstruction"]).actor.task == task


def test_rejections_retry_then_escalate(q: TaskQueue, store: SqliteStore) -> None:
    task = _t2(q, store)
    states = []
    for _ in range(3):
        _submit(q, task, _t2_files(store, unmatched=2, gaps=1))  # Gaps do not match
        states.append(q.check(task).state)
    assert states == ["queued", "queued", "escalated"]
    report = store.query("Report", {"title": "done_checks"}, order="-created_at", limit=1)[0]
    assert report.body["data"]["passed"] is False
    q.retry(task, "try again")
    assert q.get(task).state == "queued"


def test_delivery_note_is_required(q: TaskQueue, store: SqliteStore) -> None:
    task = _t2(q, store)
    _submit(q, task, _t2_files(store), note="did it")
    v = q.check(task)
    assert v.state == "queued"
    assert not next(o for o in v.outcomes if o.name == "delivery_note").passed


def test_t3_template_is_proposed_on_acceptance(q: TaskQueue, store: SqliteStore) -> None:
    body = template_body("async_transfer")
    body.update(name="async_transfer_variant", summary="variant for the test")
    doc = {"template": body, "sources": [source_id(store)]}
    task = q.create(TaskRequest("T3", "extract", []), "L1", "learn")
    _submit(
        q,
        task,
        {"template.yaml": yaml.safe_dump(doc).encode()},
        NOTE + "possible duplicate of async_transfer: intended for the test\n",
    )
    v = q.check(task)
    assert v.state == "accepted", [o.to_dict() for o in v.outcomes]
    assert store.query("Template", {"name": "async_transfer_variant"}, limit=1)


def test_checker_crash_suspends_without_counting(q: TaskQueue) -> None:
    task = q.create(TaskRequest("T5", "microbench", []), "L1", "calibrate")
    bench = {
        "quantity": "q",
        "unit": "dram",
        "entry": "k",
        "grid": 1,
        "block": 32,
        "params": [{"name": "Y", "type": "f32*"}],
        "outputs": [["Y", "m"]],
        "shape": {"m": 1},
        "work": "m",
        "value": "s_per_work",
    }
    # A well-formed benchmark whose check cannot run here (no measurement Protocol record).
    _submit(q, task, {"microbench.yaml": yaml.safe_dump(bench).encode(), "kernel.cu": b"k"})
    v = q.check(task)
    assert v.state == "suspended" and q.get(task).attempts == 1
    with pytest.raises(ConfigError):
        q.retry(task, "suspended tasks are rechecked, not retried")


def test_cancel_from_any_live_state(q: TaskQueue, store: SqliteStore) -> None:
    a = _t2(q, store)
    q.cancel(a, "not needed")
    assert q.get(a).state == "cancelled"
    with pytest.raises(ConfigError, match="already"):
        q.cancel(a, "again")
    b = _t2(q, store)
    q.lease(b, OWNER)
    q.cancel(b, "stop")
    assert q.get(b).state == "cancelled"


def test_lease_expiry(q: TaskQueue, store: SqliteStore, clock: ManualClock) -> None:
    leased, running = _t2(q, store), _t2(q, store)
    q.lease(leased, OWNER)
    q.lease(running, OWNER)
    q.start(running, OWNER)
    clock.advance(q.settings.lease_s + 1)
    assert sorted(q.expire()) == sorted([leased, running])
    assert q.get(leased).state == "queued"
    assert q.get(running).state == "queued"  # expired -> rejected -> retry


def test_budget_refusal_keeps_the_task_queued(
    q: TaskQueue, store: SqliteStore, ops: SqliteOps
) -> None:
    ops.set_budget("process:learn", {"agent_min": 1.0})  # less than one task's wall budget
    task = _t2(q, store)
    with pytest.raises(BudgetRefused):
        q.lease(task, OWNER)
    assert q.get(task).state == "queued"
    assert ops.budget("process:learn")["reserved"] == {}


def test_reservation_is_settled_on_finish(q: TaskQueue, store: SqliteStore, ops: SqliteOps) -> None:
    ops.set_budget("global", {"agent_min": 1000.0})
    ops.set_budget("loop:L1", {"agent_min": 500.0})
    task = _t2(q, store)
    q.lease(task, OWNER)
    want = budget.scope_amounts(q.card(q.get(task)).budget)["agent_min"]
    assert ops.budget("global")["reserved"]["agent_min"] == pytest.approx(want)
    assert ops.budget("loop:L1")["reserved"]["agent_min"] == pytest.approx(want)
    q.start(task, OWNER)
    q.submit(task, OWNER, _t2_files(store), NOTE, HUMAN)
    q.check(task)
    assert ops.budget("global")["reserved"]["agent_min"] == pytest.approx(0.0)


def test_refused_lower_scope_releases_the_upper_ones(
    q: TaskQueue, store: SqliteStore, ops: SqliteOps
) -> None:
    ops.set_budget("loop:L1", {"agent_min": 1000.0})
    ops.set_budget("global", {"agent_min": 1.0})
    task = _t2(q, store)
    with pytest.raises(BudgetRefused):
        q.lease(task, OWNER)
    assert ops.budget("loop:L1")["reserved"]["agent_min"] == pytest.approx(0.0)


def test_lifecycle_edges_used_are_in_the_yaml(
    q: TaskQueue, store: SqliteStore, ops: SqliteOps, clock: ManualClock
) -> None:
    """Drive every lifecycle edge; the state sequences in ops_events must use only listed
    edges, and every listed edge must occur."""
    test_rejections_retry_then_escalate(q, store)
    test_t2_accepted_writes_reconstruction_and_gaps(q, store)
    test_checker_crash_suspends_without_counting(q)
    suspended = ops.list_tasks(states=("suspended",))[0].id
    assert q.recheck(suspended, "checker fixed").state == "suspended"  # still cannot run
    test_cancel_from_any_live_state(q, store)
    test_lease_expiry(q, store, clock)
    seen: dict[str, list[str]] = {}
    for ev in ops.events(0):
        if ev.table_name == "tasks" and "state" in ev.change:
            seen.setdefault(ev.row_id, []).append(str(ev.change["state"]))
    used = {("created", "queued")}
    for states in seen.values():
        used |= set(itertools.pairwise(states))
    m = machine("task")
    listed = {(e.src, e.to) for e in m.edges if e.src != ANY_STATE}
    cancels = {(a, b) for a, b in used if b == "cancelled"}
    assert (used - cancels) - listed == set()
    assert listed - used == set()
    assert cancels


def test_services_open_needs_registered_store(ctx: Context) -> None:
    with pytest.raises(ConfigError):
        Services.open(ctx)
    assert isinstance(Path(ctx.data_dir), Path)


def test_t1_family_validates_then_space_check_needs_a_real_generator(
    q: TaskQueue, ctx: Context, store: SqliteStore
) -> None:
    """The WP5 test family validates; its gen.py emits markers only (no launch), so
    verify_space rejects it instead of pretending to build it."""
    w = world(Services(ctx, store, q.ops))
    task = q.create(TaskRequest("T1", "build a family", [w["case"]]), "L1", "solve")
    _submit(q, task, files_of(GEMM_DIR))
    v = q.check(task)
    by_name = {o.name: o for o in v.outcomes}
    assert v.state != "suspended", q.info(task)["suspended"]
    assert by_name["validate_family"].passed, by_name["validate_family"].failures
    assert not by_name["verify_space"].passed
    assert "launch" in " ".join(by_name["verify_space"].failures)
    assert v.state == "queued"
