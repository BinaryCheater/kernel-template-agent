"""The pure transition functions: every judging rule, every branch, and agreement with
machines.yaml (D96: the YAML is the single definition)."""

from __future__ import annotations

from typing import Any

import pytest

from kta.core.context import Context
from kta.orchestrator import machine, step
from kta.orchestrator.machines import ANY_STATE
from kta.orchestrator.solve_loop import bucket_metrics
from kta.orchestrator.transition import Transition
from orchesthelp import ID, bucket, ctx_with, judging, loop

EdgeKey = tuple[str, str, str, int | None]


def _key(t: Transition) -> EdgeKey:
    assert t.to is not None
    return (t.src, t.to, t.kind, t.rule)


def solve_branches() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(state, facts, data) covering every branch of solve_loop.step."""
    waiting = {"waiting_on": "T", "waiting_tasks": ["T"]}
    return [
        ("new", {"case": ID}, {}),
        ("bounded", {"bounds": {}}, {}),
        ("retrieving", {"families": [ID]}, {}),
        ("retrieving", {"families": [], "t1_goal": "g", "t1_inputs": [ID]}, {}),
        ("solving", {"candidates": {"s0": [ID]}, "empty_buckets": []}, {}),
        ("solving", {"candidates": {"s0": []}, "empty_buckets": ["s0"]}, {}),
        ("building", {"top": {"s0": [ID]}, "builds": {}, "family": ID}, {}),
        ("building", {"top": {"s0": []}, "builds": {}, "family": ID}, {}),
        ("checking", {"checks": {}, "failed": []}, {}),
        ("measuring", {"measurements": {}}, {}),
        ("waiting", {"tasks": {"T": "accepted"}}, {**waiting, "waiting_type": "T1"}),
        ("waiting", {"tasks": {"T": "accepted"}}, {**waiting, "waiting_type": "T4"}),
    ]


def expand_branches() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    done = {"tasks": {"T": "accepted"}}
    wait = {"waiting_tasks": ["T"]}
    return [
        ("frozen", {"source": ID}, {}),
        ("measured", {"reference_point": ID}, {}),
        ("analyzing", {**done, "reconstruction": ID, "gaps": []}, wait),
        (
            "analyzing",
            {**done, "reconstruction": ID, "gaps": [{"id": ID, "missing": "template"}]},
            wait,
        ),
        ("gap", done, {**wait, "resume_at": "analyzing"}),
        ("gap", done, {**wait, "resume_at": "reconstructed"}),
        ("reconstructed", {"error_rel": 0.01, "family": ID}, {}),
        ("reconstructed", {"error_rel": 0.5, "family": ID}, {}),
        ("reproduced", {"report": ID}, {}),
        ("ablated", {"report": ID}, {}),
    ]


def produced(ctx: Context, kind: str) -> set[EdgeKey]:
    branches = solve_branches() if kind == "solve" else expand_branches()
    out = {_key(step(loop(kind, s, f, **d), ctx)) for s, f, d in branches}
    if kind == "solve":
        out |= {_key(step(j, ctx)) for j in rule_loops().values()}
    return out


def rule_loops() -> dict[str, Any]:
    return {
        "rule1": judging(bucket(failed_checks=[[ID, ID]])),
        "rule2": judging(bucket(pairs=[(10.0, 30.0), (20.0, 20.0), (30.0, 10.0)])),
        "rule3": judging(bucket()),
        "rule4_t2": judging(bucket(reference_us=1.0, predicted_us=5.5), references=[ID]),
        "rule4_t1": judging(bucket(reference_us=None, predicted_us=5.5)),
    }


@pytest.mark.parametrize("kind", ["solve", "expand"])
def test_step_produces_exactly_the_yaml_transitions(ctx: Context, kind: str) -> None:
    m = machine(kind)
    listed = {(e.src, e.to, e.kind, e.rule) for e in m.edges if e.src != ANY_STATE}
    got = produced(ctx, kind)
    assert got - listed == set(), "step produced transitions missing from machines.yaml"
    assert listed - got == set(), "machines.yaml lists transitions step never produces"


@pytest.mark.parametrize("kind", ["solve", "expand"])
def test_created_task_types_are_listed_on_the_edge(ctx: Context, kind: str) -> None:
    m = machine(kind)
    branches = solve_branches() if kind == "solve" else expand_branches()
    loops = [loop(kind, s, f, **d) for s, f, d in branches]
    loops += list(rule_loops().values()) if kind == "solve" else []
    for lp in loops:
        t = step(lp, ctx)
        assert t.to is not None
        edge = m.edge(t.src, t.to, t.kind, t.rule)
        assert {r.type for r in t.tasks} <= set(edge.creates)
        assert bool(t.tasks) == (t.kind == "creates_task")


def _judged(ctx: Context, name: str) -> Transition:
    return step(rule_loops()[name], ctx)


def test_judging_rule_1_check_failed_creates_t6(ctx: Context) -> None:
    t = _judged(ctx, "rule1")
    assert (t.to, t.rule, [r.type for r in t.tasks]) == ("waiting", 1, ["T6"])
    assert t.tasks[0].inputs[:2] == [ID, ID] and t.records[-1].body["data"]["rule"] == 1


def test_judging_rule_1_wins_over_rule_2(ctx: Context) -> None:
    lp = judging(bucket(failed_checks=[[ID, ID]], pairs=[(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]))
    assert step(lp, ctx).rule == 1


def test_judging_rule_2_model_off_creates_t4(ctx: Context) -> None:
    t = _judged(ctx, "rule2")
    assert (t.to, t.rule, [r.type for r in t.tasks]) == ("waiting", 2, ["T4"])
    assert t.tasks[0].inputs[0] == ID  # the model version comes first


def test_judging_rule_2_on_error_alone(ctx: Context) -> None:
    lp = judging(bucket(pairs=[(10.0, 20.0), (20.0, 40.0)]))  # ranks agree, error 50%
    assert step(lp, ctx).rule == 2


def test_judging_rule_3_criterion_met_accepts_with_solution(ctx: Context) -> None:
    t = _judged(ctx, "rule3")
    assert (t.to, t.rule, t.tasks) == ("accepted", 3, ())
    kinds = [d.kind for d in t.records]
    assert kinds == ["Solution", "Report"]
    assert t.records[1].links == {"solution": (0,)} and t.records[1].data_key == "judgement"


def test_judging_rule_3_by_explained_gap_without_reference(ctx: Context) -> None:
    lp = judging(bucket(reference_us=None, predicted_us=10.5, best_us=11.0, bound_us=5.0))
    assert step(lp, ctx).rule == 3


def test_judging_rule_4_unexplained_creates_t2_with_a_reference(ctx: Context) -> None:
    t = _judged(ctx, "rule4_t2")
    assert (t.to, t.rule, [r.type for r in t.tasks]) == ("waiting", 4, ["T2"])


def test_judging_rule_4_unexplained_creates_t1_without_reference(ctx: Context) -> None:
    t = _judged(ctx, "rule4_t1")
    assert (t.to, t.rule, [r.type for r in t.tasks]) == ("waiting", 4, ["T1"])


def test_judging_thresholds_come_from_config(ctx: Context) -> None:
    lp = judging(bucket(reference_us=None, predicted_us=8.0))  # explains 50% of the gap
    assert step(lp, ctx).rule == 4
    assert step(lp, ctx_with(ctx, explained_min=0.4)).rule == 3


def test_judgement_report_has_the_api_fields(ctx: Context) -> None:
    report = _judged(ctx, "rule4_t1").records[-1]
    data = report.body["data"]
    assert report.body["title"] == "judgement" and report.refs == {"case": ID}
    assert {"kind", "round", "rule", "units", "explained_share"} <= set(data)
    assert data["units"][0]["bucket"] == "s0"


def test_suspicious_result_is_flagged(ctx: Context) -> None:
    t = step(judging(bucket(best_us=5.1, predicted_us=5.1)), ctx)  # 98% of the bound
    assert t.rule == 3 and t.data["suspicious"] is True


def test_bucket_metrics() -> None:
    m = bucket_metrics(bucket())
    assert m["rho"] == pytest.approx(1.0) and m["n_ranked"] == 3
    assert m["explained_share"] == pytest.approx(5.0 / 6.0)
    assert m["pct_of_bound"] == pytest.approx(100 * 5.0 / 11.0)


@pytest.mark.parametrize(
    ("task_type", "to"),
    [("T1", "retrieving"), ("T2", "retrieving"), ("T6", "retrieving"), ("T4", "solving")],
)
def test_resume_target_and_new_round(ctx: Context, task_type: str, to: str) -> None:
    lp = loop(
        "solve", "waiting", {"tasks": {"T": "accepted"}}, waiting_on="T", waiting_type=task_type
    )
    t = step(lp, ctx)
    assert (t.to, t.kind, t.data["round"]) == (to, "resume", 2)


def test_waiting_holds_until_accepted(ctx: Context) -> None:
    for state in ("queued", "running", "checking", "rejected"):
        t = step(loop("solve", "waiting", {"tasks": {"T": state}}, waiting_on="T"), ctx)
        assert t.to is None and not t.data.get("escalated")


@pytest.mark.parametrize("state", ["escalated", "cancelled", "suspended"])
def test_stuck_task_escalates_the_loop(ctx: Context, state: str) -> None:
    t = step(loop("solve", "waiting", {"tasks": {"T": state}}, waiting_on="T"), ctx)
    assert t.to is None and t.data["escalated"] is True


def test_round_limit_escalates_instead_of_resuming(ctx: Context) -> None:
    lp = loop("solve", "waiting", {"tasks": {"T": "accepted"}}, waiting_on="T", round=5)
    t = step(lp, ctx)
    assert t.to is None and t.data["escalated"] is True
    assert step(lp, ctx_with(ctx, max_rounds=6)).to == "retrieving"


def test_expand_gap_tasks_follow_the_gap_kind(ctx: Context) -> None:
    gaps = [{"id": ID, "missing": "template"}, {"id": ID, "missing": "family"}]
    lp = loop(
        "expand",
        "analyzing",
        {"tasks": {"T": "accepted"}, "reconstruction": ID, "gaps": gaps},
        waiting_tasks=["T"],
    )
    assert [r.type for r in step(lp, ctx).tasks] == ["T3", "T1"]


def test_expand_missed_epsilon_writes_a_gap(ctx: Context) -> None:
    t = step(loop("expand", "reconstructed", {"error_rel": 0.5, "family": ID}), ctx)
    assert t.to == "gap" and t.records[0].kind == "Gap" and t.data["resume_at"] == "reconstructed"


def test_terminal_states_hold(ctx: Context) -> None:
    assert step(loop("solve", "accepted"), ctx).to is None
    assert step(loop("expand", "covered"), ctx).to is None
