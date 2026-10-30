"""API read views over a real SQLite store seeded with the WP9a fixture world.

The world comes from tools/kta_devseed (origin fixture).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from kta.api import cases, opsread, pool, runs, trace
from kta.api.app import create_app, status_for
from kta.api.common import Api, envelope, machines
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.errors import ConfigError, NotSupported, PermissionDenied
from kta.core.registry import Registry
from kta.store import CodeState, open_ops, open_store
from kta.store.sqlite import RecordNotFound
from kta_devseed.knowledge import Seeder, seed_knowledge
from kta_devseed.world import WINDOW_MIN, SeedClock, seed_world

END = datetime(2026, 10, 30, 14, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def api(tmp_path_factory: pytest.TempPathFactory) -> Api:
    clock = SeedClock(END - timedelta(minutes=WINDOW_MIN + 60))
    ctx = Context(
        env="test",
        config=config_from_mapping({"agents": {"slots": 2}}),
        data_dir=Path(tmp_path_factory.mktemp("api")),
        registry=Registry(),
        clock=clock,
    )
    store = open_store(ctx, CodeState("0" * 40, False))
    ops = open_ops(ctx)
    seeder = Seeder(store, clock)
    seed_knowledge(seeder)
    seed_world(seeder, ops, END)
    return Api(store=store, ops=ops, config=ctx.config, env="test")


def test_seed_writes_only_fixture_records(api: Api) -> None:
    assert set(opsread.origin_counts(api.store)) == {"fixture"}


def test_active_maps_ops_states_onto_the_task_lifecycle(api: Api) -> None:
    a = runs.active(api)
    states = sorted((t["type"], t["state"]) for t in a["tasks"])
    # leased with an open agent session -> running; failed after 3 attempts -> escalated.
    assert ("T1", "running") in states and ("T2", "running") in states
    assert ("T3", "escalated") in states and ("T3", "queued") in states
    running = [t for t in a["tasks"] if t["state"] == "running"]
    assert {t["slot"] for t in running} == {"agent:1", "agent:2"}
    assert all(t["used_wall_s"] > 0 for t in running)
    assert a["agent_slots"] == ["agent:1", "agent:2"]


def test_active_jobs_leases_and_workers(api: Api) -> None:
    a = runs.active(api)
    jobs = {(j["name"], j["pool"], j["state"]) for j in a["jobs"]}
    assert ("measure softmax top8", "gpu-exclusive", "running") in jobs
    assert ("queue check gemm@2", "gpu-shared", "queued") in jobs
    assert [x["gpu"] for x in a["leases"]] == ["spark-gpu0:0"]
    spark = next(w for w in a["workers"] if w["name"] == "spark-gpu0")
    exclusive = next(p for p in spark["pools"] if p["pool"] == "gpu-exclusive")
    assert exclusive == {"pool": "gpu-exclusive", "capacity": 1, "running": 1, "queued": 0}
    assert "pools" not in spark["labels"]


def test_inbox_lists_the_escalated_task(api: Api) -> None:
    items = runs.inbox(api)
    assert [i["trigger"] for i in items] == ["Task over retry limit"]


def test_budget_trees_are_per_resource(api: Api) -> None:
    assert [b["key"] for b in runs.budgets(api)] == ["global:agent_min", "global:gpu_min"]
    agent = next(b for b in runs.budgets(api) if b["key"] == "global:agent_min")
    assert agent["used"] == 176 and agent["limit"] == 480
    assert {c["key"] for c in agent["children"]} == {
        "process:solve",
        "process:learn",
        "process:calibrate",
    }


def test_case_list_and_timeline(api: Api) -> None:
    by_name = {c["name"]: c for c in cases.cases(api)}
    assert by_name["vadd_f32"]["state"] == "accepted"
    assert by_name["vadd_f32"]["pct_of_bound"] == pytest.approx(41.9 / 43.6)
    soft = cases.timeline(api, by_name["softmax_bf16"]["id"])
    assert [r["n"] for r in soft["rounds"]] == [1, 2]
    assert soft["rounds"][0]["steps"][-1]["state"] == "waiting"
    assert [t["type"] for t in soft["rounds"][0]["tasks"]] == ["T1"]
    assert soft["rounds"][1]["steps"][-1]["end"] is None
    gemm = cases.timeline(api, by_name["gemm_bf16_256"]["id"])
    assert gemm["rounds"][0]["outcome"]["rule"] == 4
    assert len(gemm["gaps"]) == 1
    assert [r["name"] for r in soft["references"]] == ["triton tutorial", "torch eager"]


def test_lineage_builds_the_full_evidence_chain(api: Api) -> None:
    vadd = next(c for c in cases.cases(api) if c["name"] == "vadd_f32")
    sol = cases.timeline(api, vadd["id"])["best"]["solution"]
    result = cases.lineage(api, sol, 10)["result"]
    assert [link["stage"] for link in result["chain"]] == [
        "Templates",
        "Assignment",
        "Prediction",
        "Build",
        "SASS check",
        "Correctness",
        "Measurement",
        "Gap to bound",
    ]
    assert result["missing"] == []
    assert result["explained_share"] == pytest.approx(0.88)


def test_lineage_of_a_non_solution_has_no_result(api: Api) -> None:
    case = cases.cases(api)[0]["id"]
    assert "result" not in cases.lineage(api, case, 3)


def test_families_follow_snapshot_membership(api: Api) -> None:
    snaps = pool.snapshots(api)
    assert [s["current"] for s in snaps] == [False, False, True]
    first = {f["name"] for f in pool.families(api, snaps[0]["id"])}
    assert first == {"vadd_f32", "gemm_tiled"}
    assert {f["name"] for f in pool.families(api, None)} == {
        "vadd_f32",
        "gemm_tiled",
        "softmax_rowwise",
    }


def test_family_chain_nodes_carry_template_assurance(api: Api) -> None:
    gemm = next(f for f in pool.families(api, None) if f["name"] == "gemm_tiled")
    nodes = pool.chain(api, gemm["id"])["nodes"]
    assert [n["key"] for n in nodes] == ["cta", "warp", "a_ld", "b_ld", "mma", "epi"]
    assert all(n["assurance"] == "tested" and n["template_record"] for n in nodes)


def test_coverage_reads_the_stored_report_or_reports_missing(api: Api) -> None:
    snap = pool.snapshots(api)[-1]["id"]
    cov = pool.coverage(api, snap)
    assert cov["scope"]["total"] == 9 and cov["snapshot"] == snap
    with pytest.raises(NotSupported):
        pool.coverage(api, cases.cases(api)[0]["id"])


def test_trace_window_includes_spans_started_before_it(api: Api) -> None:
    start, end = END - timedelta(minutes=10), END
    spans = trace.window(api, start, end)["spans"]
    assert any(s["start_ts"] < "2026-10-30T14:20" and s["end_ts"] is None for s in spans)
    only_gpu = trace.window(api, start, end, lanes=["gpu"])["spans"]
    assert only_gpu and all(s["lane"].startswith("gpu:") for s in only_gpu)


def test_trace_summary_and_critical_path(api: Api) -> None:
    s = trace.summary(api, END - timedelta(hours=1), END)
    assert 0 < s["gpu_idle"] < 1 and 0 < s["agent_think_share"] < 1
    loop = next(lp for lp in runs.active(api)["loops"] if lp["subject_name"] == "vadd_f32")
    path = trace.critical_path(api, loop["id"])["path"]
    assert path and all(a["start_ts"] <= b["start_ts"] for a, b in pairwise(path))


def test_envelope_has_derived_status(api: Api) -> None:
    meas = cases.cases(api)[0]["bound_record"]
    env = envelope(api, meas)
    assert env["kind"] == "Bound" and env["status"] == "proposed"


def test_errors_map_to_http_status() -> None:
    assert status_for(RecordNotFound("x")) == 404
    assert status_for(NotSupported("x")) == 501
    assert status_for(PermissionDenied("x")) == 403
    assert status_for(ConfigError("x")) == 400


def test_app_serves_every_spec_read_endpoint(api: Api) -> None:
    paths = set(create_app(api).openapi()["paths"])
    spec = {
        "/records/{id}",
        "/records",
        "/search",
        "/lineage/{id}",
        "/cases/{id}/timeline",
        "/families/{id}/chain",
        "/coverage",
        "/model/residuals",
        "/runs/active",
        "/budget",
        "/inbox",
        "/events",
        "/trace",
        "/trace/summary",
        "/trace/critical-path/{trace_id}",
        "/machines",
    }
    assert spec <= paths


def test_machines_come_from_the_orchestrator_file() -> None:
    assert [m["id"] for m in machines()["machines"]] == ["solve", "expand", "task"]


def test_case_index_rows_carry_round_progress(api: Api) -> None:
    by_name = {c["name"]: c for c in cases.cases(api)}
    soft = by_name["softmax_bf16"]["rounds"]
    assert [(r["n"], r["last_state"], r["open"]) for r in soft] == [
        (1, "waiting", False),
        (2, "measuring", True),
    ]
    assert by_name["vadd_f32"]["rounds"][0]["rule"] == 3
    assert by_name["vadd_f32"]["rounds"][0]["pct_of_bound"] == pytest.approx(41.9 / 43.6)
    assert by_name["gemm_bf16_256"]["active_task"] == {"type": "T1", "state": "leased"}


def test_family_index_rows_carry_step_levels(api: Api) -> None:
    gemm = next(f for f in pool.families(api, None) if f["name"] == "gemm_tiled")
    assert [s["key"] for s in gemm["step_detail"]] == ["cta", "warp", "a_ld", "b_ld", "mma", "epi"]
    assert gemm["step_detail"][4]["level_to"] == 5 and gemm["versions"] == 1
