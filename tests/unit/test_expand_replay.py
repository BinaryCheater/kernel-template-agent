"""The expand loop end to end on WP10a's seven references (WP10b).

Each reference runs analysis-only from ``frozen`` to ``covered``; its T2 and T3 tasks are
answered by resubmitting WP10a's hand decompositions through the human submission path
(``replayhelp``). No agent output is produced or imitated.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from fakes.clock import ManualClock
from kta.core.context import Context
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner
from kta.orchestrator.transition import TaskRequest
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from replayhelp import failures, in_dependency_order, pool, references, submit, t2_files, t3_files

MAX_STEPS = 12


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    pool(ctx, store)
    return Services(ctx, store, ops)


def _expand(runner: Runner, ref_id: str, name: str, clock: ManualClock) -> dict[str, Any]:
    """Drive one loop to a terminal or held state, answering every task it creates."""
    svc, q = runner.svc, runner.queue
    loop_id = runner.start("expand", ref_id, name)
    row = svc.ops.get_loop(loop_id)
    svc.ops.put_loop(
        loop_id, "expand", ref_id, row["state"], {**row["data"], "analysis_only": "test"}
    )
    counts: Counter[str] = Counter()
    for _ in range(MAX_STEPS):
        clock.advance(1.0)
        t = runner.advance(loop_id)
        row = svc.ops.get_loop(loop_id)
        assert not row["data"].get("blocked"), row["data"]["blocked"]
        if row["state"] == "covered":
            break
        assert t is not None and t.to is not None, t
        tasks = list(row["data"]["waiting_tasks"]) if t.tasks else []
        cards = {task: q.card(q.get(task)) for task in tasks}
        files = {
            task: t2_files(name) if c.type == "T2" else t3_files(svc.store, c.inputs, c.goal)
            for task, c in cards.items()
        }
        for task in in_dependency_order(files):
            counts[cards[task].type] += 1
            v = submit(q, task, files[task], name, cards[task].type)
            assert v.state == "accepted", failures(v)
    return {"row": svc.ops.get_loop(loop_id), "tasks": counts}


def test_wp10a_references_go_from_frozen_to_covered(svc: Services, clock: ManualClock) -> None:
    runner = Runner(svc, "test:learn")
    refs = references(svc.ctx, svc.store)
    results = {name: _expand(runner, rid, name, clock) for name, rid in refs.items()}
    assert {n: r["row"]["state"] for n, r in results.items()} == {n: "covered" for n in refs}
    # One T3 per Template a reference proposed; a re-match T2 after the extraction round.
    t3 = {n: r["tasks"]["T3"] for n, r in results.items()}
    assert t3 == {
        "cutlass_ampere_tf32_gemm": 9,
        "flash_attention_2_fwd": 5,
        "flash_attention_3_fwd": 8,
        "marlin_int4_gemm": 5,
        "flashinfer_batch_decode": 3,
        "vllm_fused_moe": 0,
        "vllm_rms_norm": 0,
    }
    assert {n: r["tasks"]["T2"] for n, r in results.items()} == {n: 2 if t3[n] else 1 for n in refs}
    store = svc.store
    # 7 seed Templates + 30 extracted; WP10a's chains name output_tiling@1 and
    # compute_instruction_selection@1, the seed holds their @2 (WP7), so @1 is extracted too.
    assert len(store.query("Template", limit=1000)) == 38  # + shared_memory_tiling (WP24, D637)
    assert {p.body["name"] for p in store.query("Pattern", limit=1000)} >= {
        "work_assign",
        "issue_order",
        "scan",
    }
    # The last coverage Report describes the whole batch with a value for every item.
    cov = store.query("Report", {"title": "coverage"}, order="-created_at", limit=1)[0]
    assert store.get(cov.body["subject"]).kind == "PoolSnapshot"
    rep = cov.body["data"]["report"]
    assert rep["reconstructions"] == 7
    assert sum(r["unmatched"] for r in rep["references"]) == 55
    assert sum(r["matched"] for r in rep["references"]) == 99
    assert sum(r["total"] for r in rep["references"]) == 135
    assert rep["gaps_by_missing"]["template"] == 37 and sum(rep["gaps_by_missing"].values()) == 55
    assert rep["templates_missing"] == []
    assert rep["direct_test"] == {
        "references": 7,
        "all_steps_matched": 2,
        "nothing_unexplained": 0,
    }
    # templates_missing may be empty: nothing left to extract is a value.
    empty = [k for k, v in rep.items() if v is None or (v in ([], {}) and k != "templates_missing")]
    assert empty == [], f"coverage items without a value: {empty}"
    page = cov.body["data"]
    assert [r["state"] for r in page["references"]] == ["covered"] * 7
    assert len(page["trend"]) == 7 and page["discovery"]["chao1"]["hi"] >= rep["chao1"]


def test_cancelled_extraction_leaves_the_template_open(svc: Services, clock: ManualClock) -> None:
    runner, q = Runner(svc, "test:learn"), Runner(svc, "x").queue
    name = "cutlass_ampere_tf32_gemm"
    ref_id = references(svc.ctx, svc.store)[name]
    loop_id = runner.start("expand", ref_id, name)
    row = svc.ops.get_loop(loop_id)
    svc.ops.put_loop(loop_id, "expand", ref_id, "frozen", {**row["data"], "analysis_only": "test"})
    for _ in range(2):
        runner.advance(loop_id)
    (t2,) = svc.ops.get_loop(loop_id)["data"]["waiting_tasks"]
    v = submit(q, t2, t2_files(name), name, "T2")
    assert v.state == "accepted", failures(v)
    assert runner.advance(loop_id).to == "gap"  # type: ignore[union-attr]
    for task in svc.ops.get_loop(loop_id)["data"]["waiting_tasks"]:
        q.cancel(task, "leave this Template for later")
    t = runner.advance(loop_id)
    assert t is not None and (t.to, t.kind) == ("reconstructed", "normal")
    assert runner.advance(loop_id).to == "covered"  # type: ignore[union-attr]
    cov = svc.store.query("Report", {"title": "coverage"}, limit=1)[0].body["data"]
    assert len(cov["report"]["templates_missing"]) == 9
    assert cov["backlog"][0]["item"].startswith("extract ")


def test_t2_with_inflated_matched_count_is_rejected(svc: Services) -> None:
    runner = Runner(svc, "test:learn")
    ref_id = references(svc.ctx, svc.store)["cutlass_ampere_tf32_gemm"]
    task = runner.queue.create(TaskRequest("T2", "decompose", [ref_id]), "L1", "learn")
    files = t2_files("cutlass_ampere_tf32_gemm")
    files["reconstruction.yaml"] = files["reconstruction.yaml"].replace(
        b"matched: 6", b"matched: 18"
    )
    assert submit(runner.queue, task, files, "cutlass_ampere_tf32_gemm", "T2").state == "queued"
    report = svc.store.query("Report", {"title": "done_checks"}, order="-created_at", limit=1)[0]
    failed = {c["name"] for c in report.body["data"]["checks"] if not c["passed"]}
    assert failed == {"matched_templates_exist"}
