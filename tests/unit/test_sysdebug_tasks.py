"""Adversarial checks of real task/budget/holdout implementations; no model or GPU."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kta.agents import hostfs, sessions, workspace
from kta.agents.profile import CODEX_V1, CODEX_V2, CODEX_V3, profile_body
from kta.core.context import Context
from kta.core.errors import BudgetExceeded, ConfigError, SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.models.calibrate import save_model
from kta.models.unit_model import UnitModel
from kta.orchestrator import model_checks
from kta.orchestrator.pipeline import Services
from kta.orchestrator.tasks import TaskQueue
from kta.orchestrator.transition import TaskRequest
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps

HUMAN = Actor(type="human", name="test")
NOTE = "# Done\nFunctional test.\n# Simplifications\nNone.\n# Known issues\nNone."


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    return Services(ctx, store, ops)


def _queue(svc: Services) -> TaskQueue:
    return TaskQueue(svc.ctx, svc.store, svc.ops, svc)


def environment(svc: Services) -> tuple[str, str]:
    from test_wp11_processes import MANIFEST, WORKER

    env = svc.store.put(Record.new("EnvManifest", MANIFEST, HUMAN, svc.ctx.clock.now()))
    worker = svc.store.put(Record.new("Worker", WORKER, HUMAN, svc.ctx.clock.now()))
    return env, worker


def _run(svc: Services, task: str, session: str, wall_s: float) -> str:
    env, _ = environment(svc)
    blob = svc.store.put_blob(b"{}", "application/json")
    body = dict(
        session=session,
        profile="test",
        cli_version="test",
        model="test",
        session_archive=blob,
        wall_s=wall_s,
        tool_s=0,
        queue_s=0,
        exit="ok",
        context_manifest=blob,
    )
    return svc.store.put(
        Record.new(
            "AgentRun",
            body,
            Actor(type="system", name="test", task=task),
            svc.ctx.clock.now(),
            refs={"env": env},
        )
    )


def test_retry_and_late_cancel_use_are_charged_once(svc: Services) -> None:
    q = _queue(svc)
    svc.ops.set_budget("global", {"agent_min": 1000})
    task = q.create(TaskRequest("T2", "accounting fault injection", []), None, "learn")
    q.lease(task, "agent:1")
    q.start(task, "agent:1")
    _run(svc, task, "first", 120)
    assert q.abandon(task, "did not submit") == "queued"
    assert svc.ops.budget("global")["used"]["agent_min"] == 2
    q.lease(task, "agent:1")
    q.start(task, "agent:1")
    q.cancel(task, "cancel while the run is finishing")
    _run(svc, task, "late", 180)
    # Reconstruct the queue to exercise durable reconciliation after restart.
    for _ in range(2):
        _queue(svc).reconcile_usage(task)
    state = svc.ops.budget("global")
    assert state["used"]["agent_min"] == 5
    assert state["reserved"]["agent_min"] == 0


def test_settlement_receipt_replays_release_atomically(ops: SqliteOps) -> None:
    ops.set_budget("global", {"agent_min": 10})
    assert ops.reserve("global", {"agent_min": 3})
    for _ in range(2):
        ops.settle("global", {"agent_min": 3}, {"agent_min": 2}, receipt="restart")
    assert ops.budget("global")["used"]["agent_min"] == 2
    with pytest.raises(ConfigError, match="different amounts"):
        ops.settle("global", {"agent_min": 3}, {"agent_min": 1}, receipt="restart")


def model_inputs(svc: Services, held_value: float = 1.0) -> tuple[list[str], UnitModel]:
    env, worker = environment(svc)
    old = UnitModel(target="test", eff={"dram": 1.0})
    model = save_model(svc.ctx, svc.store, old, HUMAN)
    ids = [model]
    for i in range(25):
        cid = svc.store.put(
            Record.new("Candidate", {"assignment": {"i": i}}, HUMAN, svc.ctx.clock.now())
        )
        p = svc.store.put(
            Record.new(
                "Prediction",
                dict(per_unit={"dram": 1.0}, serial=0, total=1, model=old.version),
                HUMAN,
                svc.ctx.clock.now(),
                refs={"candidate": cid, "env": env, "worker": worker},
            )
        )
        # A model which overfits every visible point must FAIL the unseen candidates.
        value = held_value if int(cid, 16) % 5 == 0 else 100.0
        m = svc.store.put(
            Record.new(
                "Measurement",
                dict(median_us=value, ci_low_us=value, ci_high_us=value, n=10, protocol="same"),
                HUMAN,
                svc.ctx.clock.now(),
                refs={"candidate": cid, "env": env, "worker": worker},
            )
        )
        ids.extend((p, m))
    return ids, old


def test_holdout_is_stable_unseen_and_rejects_in_sample_overfit(svc: Services) -> None:
    q = _queue(svc)
    ids, old = model_inputs(svc)
    task = q.create(TaskRequest("T4", "holdout canary", ids), None, "calibrate")
    card = svc.store.get(q.get(task).card_id)
    hidden = card.refs["holdout"]
    assert hidden and not set(hidden) & set(card.body["inputs"])
    files = workspace.input_files(svc.store, card.body["inputs"])
    assert all(rid not in files["inputs/index.json"].decode() for rid in hidden)
    # Reordering the next request cannot move a previously visible point to holdout.
    visible2, held2 = model_checks.partition(svc.store, [ids[0], *reversed(ids[1:])])
    assert set(held2) == set(hidden) and set(visible2) == set(card.body["inputs"])
    q.lease(task, "agent:1")
    q.start(task, "agent:1")
    overfit = replace(old, eff={"dram": 0.01})
    q.submit(task, "agent:1", {"model.json": json.dumps(overfit.to_dict()).encode()}, NOTE, HUMAN)
    verdict = q.check(task)
    check = next(o for o in verdict.outcomes if o.name == "model_holdout_improves")
    assert not check.passed and check.data["new_error"] > check.data["old_error"]


def test_prediction_measurement_identity_is_not_positional(svc: Services) -> None:
    ids, _ = model_inputs(svc)
    pairs = model_checks.paired_records(svc.store, ids[1:])
    bad = [record_id(pairs[0][0]), record_id(pairs[1][1])]
    with pytest.raises(SchemaError, match="same candidate"):
        model_checks.paired_records(svc.store, bad)


def test_holdout_improvement_is_accepted_and_written_as_a_new_model(svc: Services) -> None:
    q = _queue(svc)
    ids, old = model_inputs(svc, held_value=2.0)
    task = q.create(TaskRequest("T4", "positive holdout test", ids), None, "calibrate")
    q.lease(task, "agent:1")
    q.start(task, "agent:1")
    improved = replace(old, eff={"dram": 0.5}, parent=ids[0])
    q.submit(task, "agent:1", {"model.json": json.dumps(improved.to_dict()).encode()}, NOTE, HUMAN)
    assert q.check(task).state == "accepted"
    outputs = svc.store.query("Report", {"title": "task_result", "actor.task": task})
    assert svc.store.get(outputs[0].body["data"]["outputs"]["model"]).refs["previous"] == ids[0]


def test_tool_budget_stops_calls_and_resets_on_a_new_attempt(svc: Services) -> None:
    q = _queue(svc)
    task = q.create(TaskRequest("T2", "tool budget test", []), None, "learn")
    q.lease(task, "agent:1")
    q.start(task, "agent:1")
    for _ in range(int(q.card(q.get(task)).budget["steps"])):
        q.consume_step(task)
    with pytest.raises(BudgetExceeded, match="exhausted"):
        q.consume_step(task)
    q.abandon(task, "tool limit")
    q.lease(task, "agent:1")
    q.consume_step(task)
    assert q.info(task)["tool_calls"] == 1


def test_v3_profile_repairs_contracts_without_rewriting_older_profiles(store: SqliteStore) -> None:
    old = [profile_body(store, p) for p in (CODEX_V1, CODEX_V2)]
    new = profile_body(store, CODEX_V3)
    text = store.blob_path(new.step_templates["T1"]).read_text()
    assert "out/family/" not in text and "kta validate-family out" in text
    assert "Template" not in store.blob_path(new.step_templates["T6"]).read_text()
    assert "solve" not in new.allowed_commands["T4"]
    assert [profile_body(store, p) for p in (CODEX_V1, CODEX_V2)] == old


def test_new_task_clears_old_deliverables_but_retry_keeps_draft(tmp_path: Path) -> None:
    job, ws = tmp_path / "job", tmp_path / "ws"
    incoming = job / "bundle/workspace/TASK.md"
    incoming.parent.mkdir(parents=True)
    incoming.write_text("Task `first`")
    hostfs.install_workspace(job, ws, ["TASK.md"])
    (ws / "out/old.yaml").write_text("prior task")
    (ws / "inputs/stale.json").write_text("prior input")
    hostfs.install_workspace(job, ws, ["TASK.md"])
    assert (ws / "out/old.yaml").is_file()  # same-task retry
    incoming.write_text("Task `second`")
    hostfs.install_workspace(job, ws, ["TASK.md"])
    assert not (ws / "out/old.yaml").exists() and not (ws / "inputs/stale.json").exists()


def test_success_breaks_consecutive_failure_history() -> None:
    state = sessions.SessionState(failures=["A", "A"])
    after = sessions.after_run(
        state,
        session="s",
        worker="w",
        run_id="r",
        exit="ok",
        context_ratio=None,
        failure=None,
        started_new=False,
    )
    assert after.failures == []


def test_incomplete_model_task_does_not_stop_or_starve_the_agent_queue(svc: Services) -> None:
    from kta.agents.profile import write_profile
    from kta.cli.agent_runner import next_task
    from kta.orchestrator import release
    from kta.store import controls

    snap = controls.ensure(svc.store, svc.ops, svc.ctx.clock.now())
    controls.propose(
        svc.store,
        svc.ops,
        {"release.stage": "S3"},
        HUMAN,
        "test scheduler",
        svc.ctx.clock.now(),
        snap.version_id,
    )
    profile = write_profile(svc.store, svc.ctx.clock.now(), CODEX_V3)
    svc.ops.set_pointer("current_profile", None, profile)
    q = _queue(svc)
    ids, _ = model_inputs(svc)
    bad = q.create(TaskRequest("T4", "no holdout", ids[:1]), None, "calibrate", priority=100)
    good = q.create(TaskRequest("T3", "ready task", []), None, "calibrate", priority=1)
    selected = next_task(q)
    assert selected is not None and selected.id == good
    reason = release.hold_reason(svc.ops, q.get(bad), q.controls())
    assert reason is not None and "holdout" in reason
    assert q.get(bad).attempts == 0


def test_unattributed_native_cache_change_is_not_an_agent_write() -> None:
    from kta.agents.compliance import from_report

    findings = from_report(
        {
            "watch_changed": ["/home/user/.codex/models_cache.json", "/tmp/other"],
            "exit": "ok",
            "notes_changed": True,
        },
        notes_required=True,
    )
    assert len(findings) == 1 and "/tmp/other" in findings[0]["detail"]
