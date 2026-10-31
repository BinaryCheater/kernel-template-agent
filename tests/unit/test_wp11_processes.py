"""WP11: the learn, calibrate, experiment and summarize processes, the budget hierarchy,
the T5 benchmark format and profile v2. No GPU: work that needs one is shown to be
recorded as blocked, never faked."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from fakes.clock import ManualClock
from knowhelp import seed
from kta.agents import _batch_envs  # pyright: ignore[reportPrivateUsage]
from kta.agents.profile import CODEX_V1, CODEX_V2, profile_body, write_profile
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.errors import ConfigError, SchemaError
from kta.core.schemas import Record
from kta.measure.microbench import Microbench, effective_capacity
from kta.orchestrator import improve, processes, quotas
from kta.orchestrator.inbox import inbox
from kta.orchestrator.pipeline import Services
from kta.orchestrator.proc_calibrate import CalSettings, alerts, calibrate_tick, solve_pairs
from kta.orchestrator.proc_common import ProcReport, get_state
from kta.orchestrator.proc_experiment import experiment_tick
from kta.orchestrator.proc_learn import add_to_backlog, learn_tick
from kta.orchestrator.proc_summarize import request, summarize_tick
from kta.orchestrator.runner import Runner
from kta.store.base import Task
from kta.store.snapshot import CURRENT_POOL
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import HUMAN, world


def _ctx(ctx: Context, **sections: Any) -> Context:
    return dataclasses.replace(ctx, config=config_from_mapping(sections))


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    seed(ctx, store)
    return Services(ctx, store, ops)


def _put(svc: Services, kind: str, body: dict[str, Any], **refs: Any) -> str:
    return svc.store.put(Record.new(kind, body, HUMAN, svc.ctx.clock.now(), refs=refs))


# ---- budget hierarchy ----------------------------------------------------------------


def test_experiment_quota_is_redistributed_when_no_batch_runs(ctx: Context) -> None:
    base = quotas.shares(ctx.controls)
    assert base["experiment"] == 0.5 and quotas.effective(base, True) == base
    idle = quotas.effective(base, False)
    assert idle["experiment"] == 0.0
    assert idle["solve"] == pytest.approx(0.5) and idle["learn"] == pytest.approx(0.3)
    assert sum(idle.values()) == pytest.approx(sum(base.values()))
    with pytest.raises(ConfigError):
        ctx.controls.with_changes({"budget.process_quotas": {**base, "solve": 0.9}})


def test_apply_quotas_follows_running_experiments(ctx: Context, ops: SqliteOps) -> None:
    ops.set_budget("global", {"agent_min": 1000.0})
    quotas.apply_quotas(ops, ctx.controls)
    assert ops.budget("process:experiment")["limit"]["agent_min"] == 0.0
    assert ops.budget("process:solve")["limit"]["agent_min"] == pytest.approx(500.0)
    assert quotas.set_active_experiments(ops, ["p1"])
    quotas.apply_quotas(ops, ctx.controls)
    assert ops.budget("process:experiment")["limit"]["agent_min"] == pytest.approx(500.0)
    assert ops.budget("process:solve")["limit"]["agent_min"] == pytest.approx(250.0)
    assert not quotas.set_active_experiments(ops, ["p1"])


def test_budget_period_rolls_over_with_a_report(
    ctx: Context, store: SqliteStore, ops: SqliteOps, clock: ManualClock
) -> None:
    cfg = config_from_mapping({"budget": {"period": "day"}})
    ops.set_budget("global", {"agent_min": 100.0})
    assert quotas.roll_period(store, ops, cfg, clock.now()) is None  # opens the first day
    assert ops.reserve("global", {"agent_min": 30.0})
    ops.settle("global", {"agent_min": 30.0}, {"agent_min": 30.0})
    assert ops.reserve("global", {"agent_min": 5.0})  # a running task keeps its reservation
    clock.advance(86400)
    rid = quotas.roll_period(store, ops, cfg, clock.now())
    assert rid is not None
    assert store.get(rid).body["data"]["scopes"]["global"]["used"] == {"agent_min": 30.0}
    b = ops.budget("global")
    assert b["used"]["agent_min"] == pytest.approx(0.0) and b["reserved"]["agent_min"] == 5.0
    assert quotas.roll_period(store, ops, cfg, clock.now()) is None


def test_queued_tasks_take_turns_by_quota(ctx: Context, ops: SqliteOps) -> None:
    ops.set_budget("global", {"agent_min": 1000.0})
    quotas.apply_quotas(ops, ctx.controls)
    ops.settle("process:solve", {}, {"agent_min": 100.0})  # solve has used 100 / 0.5
    tasks = [
        Task("01A", "c", "T1", "queued", 0, None, None, 0, "a"),
        Task("01B", "c", "T3", "queued", 0, None, None, 0, "b"),
        Task("01C", "c", "T3", "queued", 100, None, None, 0, "c"),
        Task("01D", "c", "T4", "queued", 0, None, None, 0, "d"),
    ]
    proc = {"01A": "solve", "01B": "learn", "01C": "human", "01D": "learn"}
    order = quotas.order_queued(tasks, lambda t: proc[t.id], ops, ctx.controls)
    assert [t.id for t in order] == ["01C", "01B", "01D", "01A"]


# ---- learn -------------------------------------------------------------------------------


def test_learn_writes_a_snapshot_once_and_starts_backlog_loops(svc: Services) -> None:
    r = Runner(svc, "learn")
    rep = learn_tick(r)
    sid = svc.ops.get_pointer(CURRENT_POOL)
    assert rep.done["snapshots"] == [sid] and sid is not None
    members = svc.store.get(sid).body["members"]
    assert {svc.store.get(m).kind for m in members} >= {"Pattern", "Template"}
    assert "snapshots" not in learn_tick(r).done  # nothing accepted since
    w = world(svc)
    src = _put(svc, "Source", {"name": "lib", "type": "git", "locator": "x", "revision": "abc"})
    ref = _put(svc, "Reference", {"name": "ref", "problem": w["problem"], "build": src})
    assert add_to_backlog(svc.store, svc.ops, ref) == [ref]
    assert add_to_backlog(svc.store, svc.ops, ref) == [ref]  # idempotent
    rep = learn_tick(r)
    (loop,) = rep.done["started"]
    assert svc.ops.get_loop(loop)["subject"] == ref and svc.ops.get_pointer("learn:backlog") == "[]"
    with pytest.raises(ConfigError):
        add_to_backlog(svc.store, svc.ops, w["case"])


# ---- calibrate ------------------------------------------------------------------------


MANIFEST = {
    "role": "worker",
    "host": "h",
    "os": "o",
    "arch": "a",
    "python": "p",
    "uv": "u",
    "uv_lock_sha256": "x",
    "commit": "c",
    "dirty": False,
}
WORKER = {"name": "w", "host": "h", "transport": "ssh", "labels": {}, "commit": "c", "devices": []}


def _measured_loop(svc: Services, case: str, family: str, ratio: float, n: int) -> None:
    """A solve loop whose n measured candidates were predicted ``ratio`` x too slow."""
    env, worker = _put(svc, "EnvManifest", MANIFEST), _put(svc, "Worker", WORKER)
    meas: dict[str, str] = {}
    for i in range(n):
        cid = _put(svc, "Candidate", {"assignment": {"i": i}}, case=case)
        _put(
            svc,
            "Prediction",
            {"per_unit": {"dram": 10.0 + i}, "serial": 0.0, "total": 10.0 + i, "model": "m"},
            candidate=cid,
        )
        body = {
            "median_us": (10.0 + i) / ratio,
            "ci_low_us": 1.0,
            "ci_high_us": 1.0,
            "n": 100,
            "protocol": "p",
        }
        meas[cid] = _put(svc, "Measurement", body, case=case, env=env, worker=worker)
    svc.ops.put_loop(
        "L" + family[:8], "solve", case, "judging", {"families": [family], "measurements": meas}
    )


def test_calibrate_records_missing_hardware_and_alerts_on_model_error(svc: Services) -> None:
    w = world(svc)
    family = w["problem"]  # any record with a name: the alert text names the family
    _measured_loop(svc, w["case"], family, ratio=0.6, n=12)  # predicted 40% low
    r = Runner(svc, "calibrate")
    pairs = solve_pairs(r)
    assert len(pairs) == 12 and {p.target for p in pairs} == {"test-gpu"}
    rep = calibrate_tick(r)
    # No Target plugin in the test registry: hardware calibration is blocked and says so.
    assert "calibrate:hw:error" in rep.blocked
    assert get_state(svc.ops, "calibrate:hw:error")["error"].startswith("ConfigError")
    (t4,) = rep.done["T4"]
    card = svc.store.get(svc.ops.task(t4).card_id).body
    assert card["type"] == "T4" and "low by a median 40%" in card["goal"]
    assert len(card["inputs"]) == 1 + 2 * 12
    assert svc.store.get(card["inputs"][1]).kind == "Prediction"
    assert svc.store.get(card["inputs"][2]).kind == "Measurement"
    assert "T4" not in calibrate_tick(r).done  # one open T4 per family
    assert "calibrate:hw:error" in calibrate_tick(r).blocked  # not retried before retry_s


def test_calibrate_is_quiet_when_the_model_is_right(svc: Services) -> None:
    w = world(svc)
    _measured_loop(svc, w["case"], w["problem"], ratio=1.0, n=12)
    r = Runner(svc, "calibrate")
    assert "T4" not in calibrate_tick(r).done
    rep2 = ProcReport("calibrate")
    alerts(r, rep2, solve_pairs(r), CalSettings(alert_rel=0.0, min_pairs=50))
    assert "T4" not in rep2.done  # too few pairs


def test_hw_fact_gaps_get_one_t5(svc: Services) -> None:
    gid = _put(
        svc,
        "Gap",
        {
            "missing": "hw_fact",
            "description": "L2 capacity under streaming",
            "attribution": "model",
        },
    )
    r = Runner(svc, "calibrate")
    (t5,) = calibrate_tick(r).done["T5"]
    card = svc.store.get(svc.ops.task(t5).card_id).body
    assert card["inputs"] == [gid] and card["done_checks"] == [
        "delivery_note",
        "microbench_repeats",
    ]
    assert "T5" not in calibrate_tick(r).done


# ---- T5 format -------------------------------------------------------------------------

DOC = {
    "quantity": "dram_read_bw",
    "unit": "dram",
    "entry": "k",
    "grid": 96,
    "block": 256,
    "params": [{"name": "X", "type": "f32*"}, {"name": "n", "type": "i32"}],
    "inputs": [["X", "n"]],
    "outputs": [["Y", "m"]],
    "shape": {"n": 1024, "m": 1},
    "work": "4 * n",
    "value": "work_per_s",
}


def test_microbench_format_and_value() -> None:
    mb = Microbench.from_doc(DOC, "extern ...")
    assert mb.value_of(1.0) == pytest.approx(4096 / 1e-6)
    assert dataclasses.replace(mb, value="s_per_work").value_of(4096.0) == pytest.approx(1e-6)
    k = mb.kernel()
    assert k.work(k.shape) == {"dram": 4096.0} and k.shape["grid"] == 96
    with pytest.raises(SchemaError):
        Microbench.from_doc({**DOC, "work": "4 * q"}, "")
    with pytest.raises(SchemaError):
        Microbench.from_doc({k: v for k, v in DOC.items() if k != "value"}, "")


def test_l2_effective_capacity_is_where_bandwidth_falls() -> None:
    mib = 1 << 20
    pts = [
        (1 * mib, 400.0),
        (8 * mib, 390.0),
        (16 * mib, 300.0),
        (32 * mib, 120.0),
        (64 * mib, 100.0),
    ]
    assert effective_capacity(pts) == 16 * mib
    with pytest.raises(SchemaError):
        effective_capacity(pts[:2])


# ---- experiment ------------------------------------------------------------------------


def _experiment(svc: Services, store: SqliteStore, w: dict[str, str]) -> str:
    snap = _put(svc, "PoolSnapshot", {"members": [], "note": "empty"})
    profile = write_profile(store, svc.ctx.clock.now(), CODEX_V2)
    params = {
        "arms": ["pool", "text"],
        "runs_per_arm": 1,
        "cases": [w["case"]],
        "snapshot": snap,
        "profile": profile,
        "budget_per_run": {"agent_min": 60.0, "gpu_min": 10.0},
        "conditions": {"text": {"knowledge": "text"}},
    }
    pid = _put(
        svc,
        "Protocol",
        {"name": "first", "version": 1, "type": "experiment", "description": "d", "params": params},
    )
    return pid


def test_experiment_runs_only_approved_protocols_in_their_batch(
    svc: Services, store: SqliteStore, ctx: Context
) -> None:
    ctx.registry.register("store", "test", lambda c: store, fixture=True)
    ctx.registry.register("batch_envs", "agents.batches", _batch_envs)
    svc = Services(_ctx(ctx, impl={"store": "test"}), store, svc.ops)
    w = world(svc)
    pid = _experiment(svc, store, w)
    r = Runner(svc, "experiment")
    assert experiment_tick(r).done == {}  # not approved: nothing happens
    _put(svc, "HumanAction", {"action": "approve_protocol", "reason": "go"}, protocol=pid)
    svc.ops.set_budget("global", {"agent_min": 1000.0, "gpu_min": 100.0})
    rep = experiment_tick(r)
    assert rep.done["environments"] == ["exp-first-v1"] and len(rep.done["started"]) == 2
    assert (ctx.data_dir / "batches" / "exp-first-v1" / "env.lock.json").is_file()
    assert quotas.active_experiments(svc.ops) == [pid]
    assert svc.ops.budget("process:experiment")["limit"]["agent_min"] == pytest.approx(500.0)
    loops = rep.done["started"]
    for loop in loops:
        row = svc.ops.get_loop(loop)
        assert row["subject"] == w["case"] and row["data"]["process"] == "experiment"
        assert svc.ops.budget(f"loop:{loop}")["limit"] == {"agent_min": 60.0, "gpu_min": 10.0}
    assert processes.solve_tick(r).advanced == []  # not the solve process's runs
    for _ in range(4):
        experiment_tick(r)
    tasks = [t for t in svc.ops.list_tasks() if t.work_unit in loops]
    assert tasks and all(r.queue.info(t.id)["process"] == "experiment" for t in tasks)
    assert all(r.queue.info(t.id)["batch"] == "exp-first-v1" for t in tasks)
    for loop in loops:  # a round limit or a human escalates a run: it ends, and is kept
        row = svc.ops.get_loop(loop)
        svc.ops.put_loop(
            loop, "solve", row["subject"], row["state"], {**row["data"], "escalated": True}
        )
    rep = experiment_tick(r)
    assert len(rep.done["runs"]) == 2
    (result,) = rep.done["results"]
    rows = svc.store.get(result).body["data"]["rows"]
    assert sorted(x["arm"] for x in rows) == ["pool", "text"]
    run = svc.store.get(rows[0]["run"])
    assert run.kind == "ExperimentRun" and run.refs["protocol"] == pid
    assert run.body["params"]["batch"] == "exp-first-v1"
    assert quotas.active_experiments(svc.ops) == []
    assert experiment_tick(r).done == {}  # done protocols stay done


def test_experiment_without_batch_envs_is_blocked_and_says_why(
    svc: Services, store: SqliteStore
) -> None:
    pid = _experiment(svc, store, world(svc))
    _put(svc, "HumanAction", {"action": "approve_protocol", "reason": "go"}, protocol=pid)
    rep = experiment_tick(Runner(svc, "experiment"))
    assert "batch_envs" in rep.blocked[f"experiment:{pid}:error"]


# ---- summarize -----------------------------------------------------------------------


def test_summarize_counts_attribution_creates_t3_and_waits_for_you(
    svc: Services, clock: ManualClock
) -> None:
    w = world(svc)
    r = Runner(svc, "summarize")
    assert summarize_tick(r).done == {}  # opens the first window
    clock.advance(60)
    g1 = _put(
        svc, "Gap", {"missing": "template", "description": "no split-K", "attribution": "space"}
    )
    _put(
        svc,
        "Gap",
        {"missing": "check", "description": "racecheck blind", "attribution": "verification"},
    )
    _put(
        svc,
        "Report",
        {
            "title": "judgement",
            "subject": w["case"],
            "summary": "r2",
            "data": {"rule": 2, "explained_share": 0.25},
        },
        case=w["case"],
    )
    assert summarize_tick(r).done == {}  # nothing asked, period not over
    request(svc.ops, "batch-1", clock.now())
    rep = summarize_tick(r)
    (rid,) = rep.done["improvement"]
    data = svc.store.get(rid).body["data"]
    assert data["batch"] == "batch-1"
    assert data["attribution"]["space"] == 1 and data["attribution"]["model"] == 1
    assert data["attribution"]["verification"] == 1 and data["loss"]["model"] == 0.75
    items = {i["class"]: i for i in data["items"]}
    (t3,) = items["space"]["tasks"]
    assert svc.store.get(svc.ops.task(t3).card_id).body["inputs"] == [g1]
    assert r.queue.info(t3)["process"] == "learn"
    assert items["verification"]["kind"] == "system"
    pending = [i for i in inbox(svc) if i.trigger == "improvement_pending"]
    assert [i.record for i in pending] == [rid]
    _put(svc, "HumanAction", {"action": "approve_improvement", "reason": "ok"}, improvement=rid)
    assert not [i for i in inbox(svc) if i.trigger == "improvement_pending"]
    assert summarize_tick(r).done == {}
    clock.advance(int(timedelta(days=1).total_seconds()))
    assert "improvement" in summarize_tick(r).done  # the period closes a batch too
    assert improve.pending_items(svc) == []  # no system item in an empty batch


def test_every_process_runs_a_tick(svc: Services) -> None:
    for name in processes.ALL:
        assert processes.run(svc, name, interval_s=0, max_ticks=1) == 1


# ---- profile v2 ------------------------------------------------------------------------


def test_profile_v2_adds_the_t4_t5_skills_and_keeps_v1(store: SqliteStore) -> None:
    v1, v2 = profile_body(store, CODEX_V1), profile_body(store, CODEX_V2)
    names = lambda refs: [s.partition("@")[0] for s in refs]  # noqa: E731
    assert names(v1.skills["T4"]) == ["use-kta", "find-knowledge"]
    assert names(v2.skills["T4"]) == ["use-kta", "find-knowledge", "diagnose-model"]
    assert names(v2.skills["T5"]) == ["use-kta", "find-knowledge", "write-microbench"]
    assert v1.skills["T1"] == v2.skills["T1"] and (v2.name, v2.version) == ("codex-v2", 2)


def test_runtime_skills_exist_with_front_matter() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "kta" / "agents" / "skills"
    for name in ("diagnose-model", "write-microbench"):
        text = (root / name / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith(f"---\nname: {name}\ndescription: ")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "| `diagnose-model` | T4 | WP11 | written |" in readme
    assert "| `write-microbench` | T5 | WP11 | written |" in readme
