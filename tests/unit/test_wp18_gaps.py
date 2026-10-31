"""WP18: gap decomposition, routing and the loop disciplines (02 §1 P7, §6.1; 12 §3, §4).

The O5 harness is ``test_fault_injection_routes_to_the_right_class``: deterministic faults
planted into one accepted bucket (degraded model, deleted template, capped solver, failed
check) must route to their class.
"""

from __future__ import annotations

from typing import Any

import pytest

from kta.core.context import Context
from kta.core.schemas import Record
from kta.orchestrator import canary, disciplines, gap, step
from kta.orchestrator.improve import summarize
from kta.orchestrator.inbox import inbox
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner
from kta.orchestrator.solve_work import selection_arms
from kta.orchestrator.transition import Loop, Settings, TaskRequest
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import HUMAN, ID, bucket, ctx_with, judging, set_controls, space_loss, world
from solvehelp import TEST_MM, gemm

# Accepted and near SOL: eps 0.2 us, L_rep 0.3 us (band 0.55 us), faster than the reference.
BASE = space_loss(bound_us=10.5, reference_us=11.5)
FAULTS = [
    ("degraded_model", "epsilon", 2, "T4"),
    ("deleted_template", "l_rep", 4, "T1"),
    ("capped_solver", "delta", 5, None),
    ("failed_check", "correctness", 1, "T6"),
]


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    return Services(ctx, store, ops)


def _plan(kind: str) -> dict[str, Any]:
    return next(dict(c["plant"]) for c in gap.routing().canaries if c["kind"] == kind)


def test_the_base_bucket_is_accepted(ctx: Context) -> None:
    assert step(judging(BASE), ctx).rule == 3


@pytest.mark.parametrize(("kind", "cls", "rule", "task"), FAULTS)
def test_fault_injection_routes_to_the_right_class(
    ctx: Context, kind: str, cls: str, rule: int, task: str | None
) -> None:
    planted = canary.plant(BASE, _plan(kind))
    if kind == "failed_check":
        planted["failed_checks"] = [[ID, ID]]
    t = step(judging(planted, references=[]), ctx)
    dec = t.records[-2]
    assert dec.kind == "GapDecomposition" and dec.body["gap_class"] == cls
    assert (t.rule, [r.type for r in t.tasks]) == (rule, [task] if task else [])
    assert canary.classify_inputs(planted, Settings.of(ctx.controls)) == cls


def test_terms_sum_to_the_gap_and_sol_bounds_l_rep() -> None:
    b = space_loss(relaxed_pred_us=9.0, irreducible_us=0.5)
    band = {"use_measurement_ci": True, "rel_of_t": 0.05}
    dec = gap.decompose(b, band)
    total = sum(float(t["us"]) for t in dec["terms"].values())
    assert total == pytest.approx(dec["gap_us"]) and dec["gap_us"] == pytest.approx(6.0)
    assert dec["terms"]["delta_pre"]["us"] == pytest.approx(1.8)
    assert dec["terms"]["l_rep"]["us"] <= dec["l_rep_bound_us"]
    assert dec["band_us"] == pytest.approx(0.55)


def test_unassessed_terms_are_labelled_not_zeroed_silently() -> None:
    dec = gap.decompose(space_loss(), {"use_measurement_ci": True, "rel_of_t": 0.05})
    assert dec["terms"]["delta_pre"]["status"] == "not_assessed"
    assert dec["terms"]["irreducible"]["status"] == "not_assessed"
    assert dec["terms"]["l_rep"]["note"] == "includes delta_pre"


def test_mip_gap_bounds_delta_and_attribution_stops_at_it() -> None:
    order = gap.routing().order
    mip = space_loss(solver={"method": "mip", "model_optimal": False, "mip_gap": 0.2})
    dec = gap.decompose(mip, {"use_measurement_ci": True, "rel_of_t": 0.05})
    assert dec["terms"]["delta"]["status"] == "bounded"
    assert dec["terms"]["delta"]["us"] == pytest.approx(0.2 * 10.8)
    assert gap.classify(dec, order, False, False) == "delta"


def test_routing_table_is_versioned_and_recorded(ctx: Context) -> None:
    table = gap.routing()
    assert table.id.startswith(f"routing@{table.version}:") and len(table.digest) == 12
    t = step(judging(space_loss(), references=[ID]), ctx)
    assert t.records[-2].body["routing"] == table.id
    assert t.records[-1].body["data"]["gap_class"] == "l_rep"


def test_accept_rule_reads_the_registry(ctx: Context) -> None:
    lp = judging(space_loss())  # worse than the reference -> routed, not accepted
    assert step(lp, ctx).rule == 4
    off = {"not_worse_than_strongest_reference": False, "max_unexplained_rel": 0.1,
           "accept_if_fully_irreducible": True}  # fmt: skip
    assert step(lp, ctx_with(ctx, {"escalation.accept": off})).rule == 3


# ---- disciplines ---------------------------------------------------------------------


def _s(ctx: Context, **changes: Any) -> Settings:
    return Settings.of(ctx.controls.with_changes(changes) if changes else ctx.controls)


def test_mechanisms_count_coupled_templates_once() -> None:
    prev = disciplines.basis(["a@1", "b@1"], "F1", "m1")
    now = disciplines.basis(["a@1", "c@1", "d@1"], "F2", "m1")
    names, count = disciplines.mechanisms(prev, now)
    assert names == ["template:+c@1", "template:+d@1", "template:-b@1"] and count == 3
    _, coupled = disciplines.mechanisms(prev, now, [["c", "d"]])
    assert coupled == 2
    assert disciplines.mechanisms(None, now) == (["initial"], 0)
    assert disciplines.mechanisms(now, disciplines.basis(now["templates"], "F3", "m1"))[0] == [
        "family:F3"
    ]


def test_gain_is_credited_only_beyond_the_band_and_attributed(ctx: Context) -> None:
    s = _s(ctx)
    data = {"incumbent": {"t_us": 10.0, "round": 1, "family": "F1"}, "round": 2}
    ok = disciplines.credit(data, 9.0, 0.5, ["template:+c@1"], 1, "F2", s)
    assert ok["credited"] and ok["incumbent"]["family"] == "F2" and ok["null_rounds"] == 0
    noise = disciplines.credit(data, 9.8, 0.5, ["template:+c@1"], 1, "F2", s)
    assert not noise["credited"] and noise["null_rounds"] == 1
    unnamed = disciplines.credit(data, 9.0, 0.5, [], 0, "F1", s)
    assert not unnamed["credited"] and "names no mechanism" in unnamed["note"]
    two = disciplines.credit(data, 9.0, 0.5, ["template:+c@1", "model:m2"], 2, "F2", s)
    assert not two["credited"] and "limit 1" in two["note"]
    looser = _s(ctx, **{"guard.one_mechanism_per_round": 2})
    assert disciplines.credit(data, 9.0, 0.5, ["x", "y"], 2, "F2", looser)["credited"]


def test_a_regressing_round_is_rolled_back_to_the_incumbent(ctx: Context) -> None:
    inc = {"t_us": 8.0, "round": 1, "family": "F1"}
    basis = disciplines.basis(["a@1"], "F1", ID)
    lp = judging(space_loss(), references=[ID], incumbent=inc, basis=basis)
    lp = Loop(lp.id, lp.kind, lp.subject, lp.state, lp.data,
              {**lp.facts, "basis": disciplines.basis(["a@1", "b@1"], "F2", ID)})  # fmt: skip
    t = step(lp, ctx)
    d = t.records[-2].body["discipline"]
    assert d["rolled_back"] and not d["credited"] and d["mechanisms"] == ["template:+b@1"]
    assert t.data["incumbent"] == inc and t.data["prefer_family"] == "F1"
    assert t.data["rolled_back_mechanisms"] == ["template:+b@1"]


def test_model_error_after_calibration_escalates_to_knowledge(ctx: Context) -> None:
    off = bucket(pairs=[(10.0, 20.0), (20.0, 40.0)])  # error 50 %, ranks agree
    assert step(judging(off), ctx).rule == 2
    t = step(judging(off, calibrations=1), ctx)
    assert (t.to, t.rule) == ("closed", 5) and t.data["escalated"] == "knowledge"
    assert t.records[-2].body["route"]["name"] == "model_after_calibration"


def test_selection_controls_are_recorded_per_bucket(ctx: Context) -> None:
    arms = {"model": [(10.0, 11.0), (12.0, 10.5), (20.0, 9.0)], "random": [9.5, 30.0],
            "copied": [10.2], "copied_from": ID}  # fmt: skip
    sel = gap.selection(arms, 2)
    assert sel["model_best_us"] == 9.0 and sel["model_top_e_us"] == 10.5
    assert sel["random_best_us"] == 9.5 and sel["model_beaten_by_random"] is True
    t = step(judging(bucket(arms=arms)), ctx)
    assert t.records[-2].body["buckets"][0]["selection"]["copied_us"] == 10.2


def test_selection_arms_draw_random_feasible_points_and_copy_a_solution(svc: Services) -> None:
    w = world(svc)
    fp, _, f = gemm(svc.ctx, svc.store)
    now = svc.ctx.clock.now()
    fam = svc.store.put(Record.new("Family", fp.family, HUMAN, now))
    mine = svc.store.get(w["case"])
    body = {**mine.body, "shape_dist": [{"shape": {"M": 64, "N": 64, "K": 64}, "weight": 1.0}]}
    other = svc.store.put(Record.new("Case", body, HUMAN, now, refs=dict(mine.refs)))
    svc.store.put(Record.new("Solution", {"bucket": "s0", "assignment": {"bm": 64},
                                          "pct_of_bound": 50.0}, HUMAN, now,
                             refs={"case": other, "family": fam}))  # fmt: skip
    s = Settings.of(svc.ctx.controls)
    b = {"name": "s0", "shape": {"M": 512, "N": 512, "K": 512}, "weight": 1.0}
    lp = Loop("L1", "solve", w["case"], "solving", {})
    arms = selection_arms(svc, lp, s, f, TEST_MM, b, fam, w["case"])
    assert len(arms["random"]) == s.exploration_k == 2
    assert [svc.store.get(c).body["assignment"] for c in arms["copied"]] == [{"bm": 64}]
    again = selection_arms(svc, lp, s, f, TEST_MM, b, fam, w["case"])
    same = [svc.store.get(c).body["assignment"] for c in again["random"]]
    assert same == [svc.store.get(c).body["assignment"] for c in arms["random"]]  # seeded


# ---- duplicate proposals and canaries ------------------------------------------------


def test_an_identical_resubmission_is_rejected_without_running_checks(svc: Services) -> None:
    q = Runner(svc, "t").queue
    task = q.create(TaskRequest("T3", "extract", []), None, "human")
    verdicts = []
    for body in (b"x", b"x"):
        q.lease(task, "agent:1")
        q.start(task, "agent:1")
        q.submit(task, "agent:1", {"x.txt": body}, "same", HUMAN)
        verdicts.append(q.check(task))
    assert len(verdicts[0].outcomes) > 1
    assert [o.name for o in verdicts[1].outcomes] == ["duplicate_proposal"]


def _decomposition(svc: Services, case: str) -> str:
    t = step(judging(BASE), svc.ctx)
    body = dict(t.records[-2].body)
    return svc.store.put(Record.new("GapDecomposition", body, HUMAN, svc.ctx.clock.now(),
                                    refs={"case": case}))  # fmt: skip


def test_canaries_rotate_kinds_and_are_caught(svc: Services) -> None:
    w = world(svc)
    r = Runner(svc, "summarize")
    _decomposition(svc, w["case"])
    kinds = []
    for i in range(len(gap.routing().canaries)):
        rid = summarize(r, f"b{i}", None, svc.ctx.clock.now())
        (cid,) = svc.store.get(rid).refs["canaries"]
        data = svc.store.get(cid).body["data"]
        kinds.append(data["kind"])
        assert data["verdict"] == "caught", data
    assert kinds == [c["kind"] for c in gap.routing().canaries]
    assert not [i for i in inbox(svc) if i.trigger == "canary_missed"]


def test_a_missed_canary_escalates_to_the_user(svc: Services) -> None:
    w = world(svc)
    _decomposition(svc, w["case"])
    # A band so wide that nothing is significant: the planted model fault goes unnoticed.
    blind = {
        "guard.improvement_band": {"use_measurement_ci": True, "rel_of_t": 5.0},
        "judging.model_ranking": {"rank_min": 0.0, "rank_n": 3, "rel_error_max": 10.0},
    }
    set_controls(svc.store, svc.ops, blind)
    rid = summarize(Runner(svc, "summarize"), "b0", None, svc.ctx.clock.now())
    (cid,) = svc.store.get(rid).refs["canaries"]
    assert svc.store.get(cid).body["data"]["verdict"] == "missed"
    (item,) = [i for i in inbox(svc) if i.trigger == "canary_missed"]
    assert item.record == cid


def test_no_decomposition_means_no_base_not_a_pass(svc: Services) -> None:
    rid = summarize(Runner(svc, "summarize"), "b0", None, svc.ctx.clock.now())
    (cid,) = svc.store.get(rid).refs["canaries"]
    assert svc.store.get(cid).body["data"]["verdict"] == "no_base"


def test_summarize_counts_routed_classes_and_escalates_recurring_ones(svc: Services) -> None:
    w = world(svc)
    for i in range(3):
        t = step(judging(space_loss(), references=[ID], round=i + 1), svc.ctx)
        svc.store.put(Record.new("GapDecomposition", dict(t.records[-2].body), HUMAN,
                                 svc.ctx.clock.now(), refs={"case": w["case"]}))  # fmt: skip
    rid = summarize(Runner(svc, "summarize"), "b0", None, svc.ctx.clock.now())
    data = svc.store.get(rid).body["data"]
    assert data["attribution"]["space"] == 3
    (item,) = [i for i in data["items"] if i["class"] == "space"]
    assert "escalate" not in item  # three rounds of one case, not three cases
