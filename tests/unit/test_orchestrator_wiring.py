"""Integration wiring of WP6's follow-up: human operations, pausing, priorities, retry
notes, abandoned agent steps, re-verification, SASS ``when`` and the sanitizer shape."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from knowhelp import seed
from kta.core.context import Context
from kta.core.errors import NotSupported, PermissionDenied
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Contract
from kta.core.schemas.knowledge import SassExpect
from kta.knowledge import load_family
from kta.orchestrator import control, processes
from kta.orchestrator.operations import Operations, reverify_suspects
from kta.orchestrator.pipeline import CaseView, Services, sanitizer_shape, sass_for
from kta.orchestrator.runner import Runner
from kta.orchestrator.tasks import CURRENT_PROFILE, UNASSIGNED_PROFILE
from kta.orchestrator.transition import TaskRequest
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import world
from solvehelp import GEMM_DIR, chain_templates
from storehelp import T0

ALICE = Actor(type="human", name="alice")


@dataclass(frozen=True)
class Spec:
    type: str
    goal: str
    inputs: list[str] = field(default_factory=lambda: list[str]())
    priority: int = 0


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    seed(ctx, store)
    return Services(ctx, store, ops)


def _actions(svc: Services) -> list[Record]:
    return svc.store.query("HumanAction", order="created_at", limit=100)


def test_human_operations_name_the_operator(svc: Services) -> None:
    ops = Operations(svc)
    made = ops.create_task(Spec("T3", "extract", priority=7), "needed", ALICE)
    task = made["task"]
    assert made["work_unit"].startswith("human:")
    assert svc.ops.task(task).priority == 7
    ops.set_priority(task, 50, "urgent", ALICE)
    assert svc.ops.task(task).priority == 50
    with pytest.raises(PermissionDenied):
        ops.queue.cancel(task, "x", Actor(type="agent", name="worker"))
    ops.cancel_task(task, "not needed", ALICE)
    acts = _actions(svc)
    assert [a.body["action"].split()[1] for a in acts] == ["create", "priority", "cancel"]
    assert {a.actor.name for a in acts} == {"alice"} and all(a.actor.task == task for a in acts)
    assert set(ops.missing()) == {"approve-profile", "env-verify", "write-record"}
    with pytest.raises(NotSupported):
        ops.approve_profile("p", "r", ALICE)


def test_cards_name_the_approved_profile(svc: Services) -> None:
    q = Operations(svc).queue
    first = q.card(q.get(q.create(TaskRequest("T3", "a", []), None, "human")))
    assert first.profile == UNASSIGNED_PROFILE
    svc.ops.set_pointer(CURRENT_PROFILE, None, "f" * 64)
    second = q.card(q.get(q.create(TaskRequest("T3", "b", []), None, "human")))
    assert second.profile == "f" * 64


def test_pause_blocks_leases_loops_and_process_ticks(svc: Services) -> None:
    ops = Operations(svc)
    task = ops.create_task(Spec("T3", "extract"), "", ALICE)["task"]
    ops.pause("global", "maintenance", ALICE)
    with pytest.raises(control.Paused):
        ops.queue.lease(task, "agent:1")
    w = world(svc)
    loop = Runner(svc, "t").start("solve", w["case"], "c")
    assert Runner(svc, "t").advance(loop) is None
    assert processes.run(svc, "solve", interval_s=0, max_ticks=1) == 1
    assert svc.ops.get_loop(loop)["state"] == "new"  # the tick was skipped
    ops.resume("global", "done", ALICE)
    ops.pause(f"loop:{loop}", "look first", ALICE)
    assert Runner(svc, "t").advance(loop) is None
    ops.queue.lease(task, "agent:1")  # other scopes are not paused
    ops.resume(f"loop:{loop}", "go", ALICE)
    assert Runner(svc, "t").advance(loop) is not None
    with pytest.raises(Exception, match="pausable"):
        ops.pause("everything", "x", ALICE)


def test_retry_note_and_last_rejection_reach_the_next_step(svc: Services) -> None:
    q = Operations(svc).queue
    task = q.create(TaskRequest("T3", "extract", []), None, "human")
    for _ in range(3):  # three rejections: escalated
        q.lease(task, "agent:1")
        q.start(task, "agent:1")
        q.submit(task, "agent:1", {"x.txt": b"x"}, "no sections", ALICE)
        q.check(task)
    assert q.get(task).state == "escalated"
    Operations(svc).retry_task(task, "look at the delivery note headings", ALICE)
    failures, note = q.retry_context(task)
    # T3 checks since WP10b: delivery_note, validate_template, template_fills_chain.
    assert {name for name, _ in failures} == {
        "delivery_note",
        "validate_template",
        "template_fills_chain",
    }
    assert note == "look at the delivery note headings"
    assert q.retry_context(task)[1] is None  # the note is handed over once


def test_abandoned_agent_step_is_a_rejection(svc: Services) -> None:
    q = Operations(svc).queue
    task = q.create(TaskRequest("T3", "extract", []), None, "human")
    q.lease(task, "agent:1")
    q.start(task, "agent:1")
    assert q.abandon(task, "no kta submit") == "queued"
    assert q.get(task).attempts == 1


def test_suspect_solution_is_reverified_by_a_new_solve(svc: Services) -> None:
    w = world(svc)
    runner = Runner(svc, "maintain")
    first = runner.start("solve", w["case"], "c")
    svc.ops.put_loop(first, "solve", w["case"], "accepted", svc.ops.get_loop(first)["data"])
    sol = svc.store.put(
        Record.new(
            "Solution",
            {"assignment": {}, "pct_of_bound": 50.0},
            ALICE,
            T0,
            refs={"case": w["case"]},
        )
    )
    svc.store.put(
        Record.new(
            "StatusChange",
            {"subject": sol, "to_status": "suspect", "reason": "too fast"},
            ALICE,
            T0,
        )
    )
    assert reverify_suspects(runner) == [sol]
    loops = [r for r in svc.ops.list_loops("solve") if r["subject"] == w["case"]]
    assert len(loops) == 2 and {r["state"] for r in loops} == {"accepted", "new"}
    assert reverify_suspects(runner) == []  # handled once
    other = w["contract"]
    svc.store.put(
        Record.new(
            "StatusChange", {"subject": other, "to_status": "suspect", "reason": "x"}, ALICE, T0
        )
    )
    assert reverify_suspects(runner) == [] and svc.ops.get_pointer(f"reverify:{other}")


def test_sass_when_is_evaluated_over_item_bindings(svc: Services) -> None:
    """``when`` names Template variables: a constant binding is used as is, a binding
    to a family variable takes the assignment's value (D361)."""
    fp = load_family(GEMM_DIR)
    templates = chain_templates(svc.store, fp.chain)
    item = fp.chain.items[0]
    t = templates[item.template]
    v = t.variables[0].name
    marked = {item.template: [SassExpect(when=f"{v} == 7", must_contain=["HMMA"])]}

    def kept(binding: Any, assignment: dict[str, Any]) -> bool:
        one = item.model_copy(update={"bindings": {**item.bindings, v: binding}})
        chain = fp.chain.model_copy(update={"items": [one]})
        got = sass_for(replace(fp, chain=chain), templates, assignment, marked)
        return any(e.must_contain == ["HMMA"] and e.when == "True" for e in got)

    assert kept(7, {})  # constant binding
    assert not kept(8, {})
    assert kept("fam_v", {"fam_v": 7})  # family variable
    assert not kept("fam_v", {"fam_v": 3})


def test_sanitizer_shape_is_the_first_nontrivial_edge_shape() -> None:
    contract = Contract(
        timing_boundary="k",
        edge_shapes=[
            {"M": 1, "N": 1, "K": 1},
            {"M": 129, "N": 193, "K": 65},
            {"M": 1, "N": 4096, "K": 4096},
        ],
    )
    cv = CaseView("c", None, None, contract, "k", None)  # type: ignore[arg-type]
    assert sanitizer_shape(cv) == {"M": 129, "N": 193, "K": 65}
    empty = CaseView("c", None, None, Contract(timing_boundary="k"), "k", None)  # type: ignore[arg-type]
    assert sanitizer_shape(empty) is None
