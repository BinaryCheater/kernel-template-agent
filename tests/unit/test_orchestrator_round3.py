"""Round 3 wiring: experiment pinning (snapshot, knowledge condition, task batch and arm),
the agent runner's queue order, per-task batch and cancel watcher, Bound re-verification,
``kta reference add`` with a Problem and the full ``kta coverage`` text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from fakes.executor import RecordingExecutor
from knowhelp import seed
from kta.cli.agent_runner import TrackingRunner, next_task, step_once
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.jobs import Job
from kta.core.schemas import Actor, Record
from kta.knowledge import Card
from kta.orchestrator.operations import reverify
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner
from kta.orchestrator.solve_work import allowed_cards
from kta.orchestrator.tasks import CURRENT_PROFILE, TaskQueue
from kta.orchestrator.transition import TaskRequest
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import set_controls, world
from storehelp import T0

HUMAN = Actor(type="human", name="test")
NOTE = "# What was done\nx\n# Fixtures and simplifications\nnone\n# Known issues\nnone\n"


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    seed(ctx, store)
    cfg = config_from_mapping({"agents": {"worker": "spark", "batch": "default-batch"}})
    c = Context(ctx.env, cfg, ctx.data_dir, ctx.registry, ctx.clock, ctx.seed, ctx.process)
    s = Services(c, store, ops)
    s._executor = RecordingExecutor()  # type: ignore[assignment]  # the GPU/agent boundary
    return s


def _experiment_loop(svc: Services, case: str, **pinned: Any) -> str:
    return Runner(svc, "exp").start(
        "solve",
        case,
        "run",
        extra={"process": "experiment", "experiment": {"arm": "a", "batch": "exp-b", **pinned}},
        distinct=True,
    )


def _advance(svc: Services, loop: str, n: int) -> dict[str, Any]:
    for _ in range(n):
        Runner(svc, "exp").advance(loop)
    return svc.ops.get_loop(loop)


def test_zero_arm_gets_no_pool_objects_and_tasks_carry_batch_and_arm(svc: Services) -> None:
    w = world(svc)
    loop = _experiment_loop(svc, w["case"], condition={"knowledge": "zero"}, protocol=w["case"])
    row = _advance(svc, loop, 3)
    assert row["state"] == "waiting"
    task = row["data"]["waiting_on"]
    q = TaskQueue(svc.ctx, svc.store, svc.ops, svc)
    info = q.info(task)
    assert info["batch"] == "exp-b" and info["process"] == "experiment"
    assert info["arm"] == {
        "name": "a",
        "knowledge": "zero",
        "condition": {"knowledge": "zero"},
        "snapshot": None,
        "protocol": w["case"],
    }
    card = q.card(q.get(task))
    retrieval = svc.store.get(card.inputs[-1])
    assert retrieval.body["data"]["knowledge"] == "zero"
    assert retrieval.body["data"]["templates"] == []  # no seed Template reaches the zero arm


def test_snapshot_limits_retrieval_to_its_members(svc: Services) -> None:
    w = world(svc)
    cards = [Card(i, "Template", i, [], "proposed", "declared") for i in ("a" * 64, "b" * 64)]
    assert [c.id for c in allowed_cards(cards, None, "pool")] == ["a" * 64, "b" * 64]
    assert [c.id for c in allowed_cards(cards, {"b" * 64}, "pool")] == ["b" * 64]
    assert allowed_cards(cards, None, "text") == [] == allowed_cards(cards, None, "zero")
    snap = svc.store.put(Record.new("PoolSnapshot", {"members": [], "note": "empty"}, HUMAN, T0))
    row = _advance(svc, _experiment_loop(svc, w["case"], snapshot=snap), 3)
    card = svc.store.get(svc.ops.task(row["data"]["waiting_on"]).card_id)
    retrieval = svc.store.get(card.body["inputs"][-1])
    assert retrieval.body["data"]["snapshot"] == 0 and retrieval.body["data"]["templates"] == []


def test_knowledge_aliases_and_text_needs_a_snapshot(svc: Services) -> None:
    from kta.orchestrator.solve_work import knowledge_of

    assert knowledge_of({"knowledge": "from_zero"}) == "zero"
    assert knowledge_of({"knowledge": "template_pool"}) == "pool"
    assert knowledge_of(None) == "pool"
    w = world(svc)
    row = _advance(svc, _experiment_loop(svc, w["case"], condition={"knowledge": "text"}), 3)
    assert "needs a pinned snapshot" in row["data"]["blocked"]["message"]


def test_reverify_request_for_an_unsupported_kind_writes_nothing(svc: Services) -> None:
    from kta.core.errors import NotSupported
    from kta.orchestrator.operations import Operations

    w = world(svc)
    with pytest.raises(NotSupported):
        Operations(svc).request_reverify(w["contract"], "x", HUMAN)
    assert svc.store.query("HumanAction", limit=5) == []


def test_bad_condition_or_snapshot_blocks(svc: Services) -> None:
    w = world(svc)
    bad = _advance(svc, _experiment_loop(svc, w["case"], condition={"knowledge": "all"}), 3)
    assert bad["state"] == "retrieving" and bad["data"]["blocked"]["error"] == "SchemaError"
    wrong = _advance(svc, _experiment_loop(svc, w["case"], snapshot=w["case"]), 3)
    assert "is a Case" in wrong["data"]["blocked"]["message"]


@dataclass
class Result:
    run_id: str = "run1"


def _queue(svc: Services) -> TaskQueue:
    svc.ops.set_pointer(CURRENT_PROFILE, None, "f" * 64)
    set_controls(svc.store, svc.ops, {"release.stage": "S4"})  # agent calls run (WP21)
    return TaskQueue(svc.ctx, svc.store, svc.ops, svc)


def test_runner_order_batch_and_abandon(svc: Services) -> None:
    q = _queue(svc)
    solve = q.create(TaskRequest("T3", "from a loop", []), "L1", "solve", priority=50)
    mine = q.create(TaskRequest("T3", "yours", []), None, "human")
    q.annotate(mine, batch="exp-b")
    assert next_task(q).id == mine  # type: ignore[union-attr]  # your tasks come first
    seen: dict[str, Any] = {}

    def leaves_running(ctx: Any, task: Any, **kw: Any) -> Result:
        seen.update(kw, task=task.id)
        return Result()

    out = step_once(svc, q, 2, None, step=leaves_running, poll_s=0.01)
    assert seen["batch"] == "exp-b" and seen["lane"] == "agent:2" and seen["task"] == mine
    assert out["verdict"] == "queued"  # abandoned: expired -> rejected -> retry
    assert isinstance(seen["deps"].runner, TrackingRunner)

    def submits(ctx: Any, task: Any, **kw: Any) -> Result:
        q.submit(task.id, task.lease_owner, {"x.txt": b"x"}, NOTE, HUMAN)
        return Result()

    out = step_once(svc, q, 1, "cli-batch", step=submits, poll_s=0.01)
    assert out["task"] == mine and out["after_step"] == "submitted"
    assert out["verdict"] == "queued"  # checked: T3 without template.yaml is rejected
    assert q.get(solve).state == "queued"


def test_cancel_during_a_step_cancels_its_jobs(svc: Services) -> None:
    q = _queue(svc)
    task = q.create(TaskRequest("T3", "long", []), None, "human")
    ex = svc.executor
    assert isinstance(ex, RecordingExecutor)

    def cancelled_meanwhile(ctx: Any, t: Any, **kw: Any) -> Result:
        kw["deps"].runner.submit(Job(kind="run", argv=["agent"]))
        q.cancel(t.id, "stop it", HUMAN)
        assert ex.any_cancelled.wait(timeout=30)
        return Result()

    out = step_once(svc, q, 1, None, step=cancelled_meanwhile, poll_s=0.01)
    assert out["cancelled_jobs"] == ["job1"] and ex.cancelled == ["job1"]
    assert q.get(task).state == "cancelled"


def test_cancel_without_executor_support_is_reported() -> None:
    class NoCancel:
        def submit(self, job: Job) -> str:
            return "j"

    tr = TrackingRunner(NoCancel())
    tr.submit(Job(kind="run", argv=["x"]))
    from kta.core.errors import NotSupported

    with pytest.raises(NotSupported):
        tr.cancel_all()


def test_bound_is_recomputed_on_reverify(svc: Services) -> None:
    w = world(svc)
    r = Runner(svc, "t")
    loop = r.start("solve", w["case"], "c")
    r.advance(loop)
    r.advance(loop)  # new -> bounded -> retrieving: Bound written
    bound_id = svc.ops.get_loop(loop)["data"]["bounds"]["s0"]
    [report] = reverify(svc, bound_id, "test")
    data = svc.store.get(report).body["data"]
    assert data["kind"] == "Bound" and data["agrees"] is True


def test_reference_add_links_a_problem_and_coverage_prints_every_key(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    from kta.cli.main import main

    monkeypatch.setenv("KTA_ENV", "test")
    monkeypatch.setenv("KTA_TEST_DATA", str(tmp_path / "data"))
    doc = {
        "source": {"name": "cutlass", "type": "git", "locator": "u", "revision": "v1"},
        "problem": {"name": "gemm_fp16", "reference_impl": "C = A @ B"},
        "reference": {"name": "cutlass_gemm", "problem": "gemm_fp16", "build": "make"},
    }
    path = tmp_path / "ref.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert main(["reference", "add", str(path), "--json"]) == 0
    import json

    out = json.loads(capsys.readouterr().out)
    assert out["problem"]
    assert main(["coverage"]) == 0
    text = capsys.readouterr().out
    assert "reconstructions:" in text and len(text.splitlines()) > 3
