"""SYSDEBUG-2: tiered, library-relative checks, shape scope, step work (D694–D700)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from kta.api.cases import _step_work  # pyright: ignore[reportPrivateUsage]
from kta.checks.full import in_scope
from kta.checks.numeric import library_verdict
from kta.checks.proof.prove import prove_template
from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.ids import canonical_json
from kta.core.schemas.evaluation import Contract
from kta.knowledge import Card
from kta.orchestrator import runner as runner_mod
from kta.orchestrator import solve_work, step
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner
from kta.orchestrator.solve_loop import _shape_text  # pyright: ignore[reportPrivateUsage]
from kta.orchestrator.transition import Loop
from kta.store.base import SpanRow
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import bucket, ctx_with, judging, loop, world
from proofhelp import seeded, template_id
from storehelp import T0

TOL = {"rtol": 0.01, "atol": 0.01, "relative_to": "library", "loose": 4.0, "strict": 1.5}


def _stats(rel_l2: float, max_abs: float, nonfinite: int = 0) -> dict[str, float | int | None]:
    return {
        "rel_l2": rel_l2,
        "max_abs_err": max_abs,
        "mismatches": 480,
        "nonfinite_where_ref_finite": nonfinite,
    }


def test_fp16_rounding_like_the_library_passes_where_fixed_tolerance_failed() -> None:
    # the rehearsal's numbers: 480 element-wise misses, rel L2 2.1e-4, max error 8.06; the
    # library itself makes about the same error on the same inputs
    cand, lib = _stats(2.1e-4, 8.06), _stats(1.9e-4, 7.5)
    assert library_verdict(cand, lib, TOL, "loose")[0]
    ok, detail = library_verdict(cand, lib, TOL, "strict")
    assert ok and detail["factor"] == 1.5 and detail["library_max_abs_err"] == 7.5


def test_a_real_bug_fails_and_strict_is_tighter_than_loose() -> None:
    lib = _stats(1.9e-4, 7.5)
    assert not library_verdict(_stats(0.3, 900.0), lib, TOL, "loose")[0]  # wrong results
    two_x = _stats(3.8e-4, 15.0)  # twice the library's error
    assert library_verdict(two_x, lib, TOL, "loose")[0]
    assert not library_verdict(two_x, lib, TOL, "strict")[0]
    assert not library_verdict(_stats(1e-5, 0.1, nonfinite=3), lib, TOL, "loose")[0]


def test_an_exact_library_does_not_make_every_rounding_a_failure() -> None:
    exact = _stats(0.0, 0.0)
    assert library_verdict(_stats(5e-7, 5e-4), exact, TOL, "strict")[0]  # within the floors
    assert not library_verdict(_stats(1e-3, 0.5), exact, TOL, "strict")[0]


def test_shape_scope_static_range_and_undeclared() -> None:
    own = [{"M": 4096, "N": 4096, "K": 4096}, {"M": 8192, "N": 1024, "K": 4096}]
    unit = {"M": 1, "N": 4096, "K": 4096}
    assert in_scope(own[1], {"kind": "static"}, own)
    assert not in_scope(unit, {"kind": "static"}, own)
    rng = {"kind": "range", "ranges": {"M": [1, 8192], "N": [1024, 8192]}}
    assert in_scope(unit, rng, own) and not in_scope({"M": 4, "N": 512, "K": 64}, rng, own)
    assert in_scope(unit, {}, own)  # undeclared: everything is checked, as before


def test_an_undeclared_scope_keeps_old_contracts_byte_identical() -> None:
    old = {"tolerances": {}, "input_dists": ["normal"], "edge_shapes": [], "timing_boundary": "t"}
    assert canonical_json(Contract.model_validate(old).model_dump(mode="json")) == canonical_json(
        old
    )
    scoped = Contract.model_validate({**old, "shape_scope": {"kind": "static"}})
    assert scoped.model_dump(mode="json")["shape_scope"] == {"kind": "static"}


def test_step_work_counts_jobs_and_their_largest_overlap() -> None:
    t0 = datetime(2026, 11, 2, tzinfo=UTC)

    def job(name: str, start: int, end: int, status: str = "ok") -> SpanRow:
        return SpanRow(
            name, None, "loop", "job_run", "worker", name, t0 + timedelta(seconds=start),
            t0 + timedelta(seconds=end), status, {},
        )  # fmt: skip

    serial = [job(f"compile {i}", 5 * i, 5 * i + 5) for i in range(4)]
    side = [job("check a", 0, 10), job("check b", 2, 9, "fail"), job("check c", 3, 4)]
    work = {w["kind"]: w for w in _step_work(serial + side, t0, t0 + timedelta(seconds=60))}
    assert work["compile"]["count"] == 4 and work["compile"]["max_parallel"] == 1
    assert work["compile"]["busy_s"] == 20.0
    assert work["check"]["max_parallel"] == 3 and work["check"]["failed"] == 1


def test_a_failing_transition_blocks_the_loop_and_logs_it(
    ctx: Context, store: SqliteStore, ops: SqliteOps, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D700/D701: an error while deciding or applying the transition (e.g. creating the
    routed task) blocks the loop and lands in its event log, instead of escaping and
    repeating the state's work on every step."""
    svc = Services(ctx, store, ops)
    w = world(svc)
    r = Runner(svc, "test")
    loop_id = r.start("solve", w["case"], "c")

    def boom(*_: Any, **__: Any) -> Any:
        raise SchemaError("T4 measurements need distinct candidate identities")

    monkeypatch.setattr(runner_mod, "step", boom)
    assert r.advance(loop_id) is None
    blocked = ops.get_loop(loop_id)["data"]["blocked"]
    assert blocked["error"] == "SchemaError" and "distinct candidate" in blocked["message"]
    logged = [
        e.change["blocked"] for e in ops.events_for("loops", loop_id) if e.change.get("blocked")
    ]
    assert logged and logged[-1]["error"] == "SchemaError"


# ---- D702: defer model fixes; D704: multi_buffer_rotation@2 -------------------------------


def _misranked(best: float) -> list[tuple[float, float]]:
    """Pairs (predicted, measured) whose measured order reverses the predicted one."""
    return [(10.0, best + 20.0), (20.0, best + 10.0), (30.0, best)]


def test_a_model_off_round_creates_t4_unless_model_fixes_are_deferred(ctx: Context) -> None:
    lp = judging(bucket(pairs=_misranked(14.0), best_us=14.0))  # worse than the reference
    routed = step(lp, ctx)
    assert (routed.rule, [t.type for t in routed.tasks]) == (2, ["T4"])
    deferred = step(lp, ctx_with(ctx, {"judging.defer_model_fix": True}))
    assert (deferred.rule, [t.type for t in deferred.tasks]) == (4, ["T1"])
    data = deferred.records[-1].body["data"]
    assert data["gap_class"] == "model_fix_deferred" and data["settings"]["defer_model_fix"]
    # D707: the T1 starts with the case header: per-bucket results, family, situation
    header = deferred.tasks[0].header
    assert header is not None and header.situation == "diagnostic"
    assert "s0 " in header.text and "best 14 us, reference 10.5" in header.text
    assert _shape_text({"K": 4096, "M": 8192, "N": 1024}) == "M=8192 N=1024 K=4096"
    assert routed.tasks[0].header is not None and routed.tasks[0].header.situation == "eps"
    # the model metrics are still recorded on the decomposition
    assert deferred.records[-2].body["buckets"][0]["model_off"] is True


def test_deferring_model_fixes_lets_an_otherwise_accepted_round_accept(ctx: Context) -> None:
    lp = judging(bucket(pairs=_misranked(10.4), best_us=10.4))  # beats the reference (10.5)
    assert step(lp, ctx).rule == 2
    assert step(lp, ctx_with(ctx, {"judging.defer_model_fix": True})).rule == 3


def test_rotation_v2_precondition_is_exact_on_the_event_model(
    ctx: Context, store: SqliteStore
) -> None:
    """P <= S - 2 + max(beta, private) + late with late <= P equals the safe set of the
    event model on the declared domains, so the register rotations of CUTLASS and Marlin
    (S 2, P 1, private) and FlashAttention-2's late single-slot refill (S 1, P 1, beta,
    late) are admitted and nothing unsafe is."""
    seeded(ctx, store)
    p = prove_template(store, template_id(store, "multi_buffer_rotation@2"), T0)
    gen = [store.get(r).body for r in p.results]
    table = next(b for b in gen if b.get("obligation") == "generation_correctness")
    assert table["outcome"] == "bounded" and table["data"]["pre_exact"], table.get("reason")
    rows = {(r["S"], r["P"], r["beta"], r["late"], r["private"]): r for r in table["data"]["table"]}
    assert rows[(2, 1, 0, 0, 1)]["safe"] and rows[(1, 1, 1, 1, 0)]["safe"]
    assert not rows[(2, 1, 0, 0, 0)]["safe"] and not rows[(1, 0, 1, 1, 0)]["safe"]


# ---- D706: the next round tries the family an agent delivered -----------------------------


def test_an_accepted_t1_hands_its_family_to_the_next_round(ctx: Context) -> None:
    lp = loop(
        "solve",
        "waiting",
        {"tasks": {"t1": "accepted"}, "delivered_family": "F2"},
        waiting_on="t1",
        waiting_type="T1",
    )
    t = step(lp, ctx)
    assert (t.to, t.data["try_family"]) == ("retrieving", "F2")
    # the retrieving transition consumes it
    after = step(loop("solve", "retrieving", {"families": ["F2", "F1"]}, try_family="F2"), ctx)
    assert after.data == {"families": ["F2", "F1"], "try_family": None}


def test_retrieval_puts_the_delivered_family_before_the_incumbent(
    ctx: Context, store: SqliteStore, ops: SqliteOps, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = Services(ctx, store, ops)
    w = world(svc)
    cards = [Card(i, "Family", f"{i}@1", [], "proposed", "declared", False) for i in "ABC"]
    monkeypatch.setattr(solve_work, "retrieve_families", lambda *_a, **_k: cards)
    base = loop("solve", "retrieving", prefer_family="A")
    lp = Loop(base.id, base.kind, w["case"], base.state, base.data, base.facts)
    assert solve_work.work_retrieving(svc, lp)["families"][0] == "A"  # incumbent
    tried = Loop(lp.id, lp.kind, lp.subject, lp.state, {**lp.data, "try_family": "C"}, {})
    assert solve_work.work_retrieving(svc, tried)["families"][0] == "C"
    gone = Loop(lp.id, lp.kind, lp.subject, lp.state, {**lp.data, "try_family": "Z"}, {})
    out = solve_work.work_retrieving(svc, gone)
    assert out["families"][0] == "A" and "not retrievable" in out["delivered_family_note"]


def test_without_a_family_the_agent_formulates_from_nothing_preselected(
    ctx: Context, store: SqliteStore, ops: SqliteOps, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D717: no family of the case -> a T1 whose only input is the Case (plus the retrieval
    report); no nearest chain, no keep/replace verdicts in the goal."""
    svc = Services(ctx, store, ops)
    w = world(svc)
    monkeypatch.setattr(solve_work, "retrieve_families", lambda *_a, **_k: [])
    base = loop("solve", "retrieving")
    lp = Loop(base.id, base.kind, w["case"], base.state, base.data, base.facts)
    out = solve_work.work_retrieving(svc, lp)
    assert out["families"] == [] and "choose the Templates" in out["t1_goal"]
    assert "chain" not in out["t1_goal"] and out["t1_inputs"][0] == w["case"]
    assert len(out["t1_inputs"]) == 2  # the Case and the retrieval report


def test_a_failed_check_reads_as_its_numbers() -> None:
    """D711: done-check failures name what failed instead of 'path0: None'."""
    from kta.checks.full import failure_reason

    f = {"check": "numeric", "dist": "cancellation", "rel_l2": 2.08e-4, "rel_l2_limit": 1e-3,
         "mismatches": 480, "elements": 16777216, "output": "C"}  # fmt: skip
    text = failure_reason([f, f])
    assert text is not None and text.startswith("check numeric, dist cancellation, rel_l2 0.000208")
    assert "mismatches 480" in text and text.endswith("(+1 more)")
    assert failure_reason([]) is None
