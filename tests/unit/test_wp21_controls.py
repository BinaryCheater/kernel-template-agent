"""WP21: the controls registry, its versions, the staged release gate and the settings API.

Real code paths throughout: the registry files, the store, the task queue, the agent pool's
``step_once`` and the API app. Only the agent step itself is a stand-in where a test does
not need a CLI (the fake-CLI run of the real agent step is in test_agents_runner.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from apihelp import request, seeded_api
from fakes.executor import RecordingExecutor
from kta.api.app import create_app
from kta.api.common import Api
from kta.cli.agent_runner import next_task, step_once
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.controls import baseline, definitions
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.knobs import Knobs
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import ControlsVersion
from kta.orchestrator import processes, release
from kta.orchestrator.budget import configure_scopes
from kta.orchestrator.pipeline import Services
from kta.orchestrator.tasks import CURRENT_PROFILE, TaskQueue
from kta.orchestrator.transition import Settings, TaskRequest
from kta.store import controls as controls_store
from orchesthelp import HUMAN, set_controls

ROOT = Path(__file__).resolve().parents[2]
DOC12 = ROOT / "docs" / "design" / "12-agent-controls.md"


def _doc_rows() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in DOC12.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith(("| Knob", "| ---")):
            cells = [c.strip() for c in line.strip("|").split(" | ")]
            rows[cells[0]] = cells
    return rows


@pytest.fixture
def svc(ctx: Context, store, ops) -> Services:
    cfg = config_from_mapping({"agents": {"worker": "spark", "batch": "default-batch"}})
    s = Services(replace(ctx, config=cfg), store, ops)
    s._executor = RecordingExecutor()  # type: ignore[assignment]  # the GPU/agent boundary
    return s


# ---- the registry ----------------------------------------------------------------------------


def test_registry_holds_every_row_of_12_with_set_by() -> None:
    defs = definitions()
    rows = _doc_rows()
    by_row = {k.row: k for k in defs.knobs}
    assert set(rows) == set(by_row), set(rows) ^ set(by_row)
    for name, cells in rows.items():
        k = by_row[name]
        assert cells[-1] == (", ".join(k.set_by) or "—"), name
        assert k.prevents and k.where and k.status in ("spec", "default", "set")
    sections = {k.section for k in defs.knobs}
    assert sections == {"12 §1", "12 §2", "12 §3", "12 §4", "12 §7"}


def test_every_knob_has_one_typed_accessor() -> None:
    defs = definitions()
    props = {n for n, v in vars(Knobs).items() if isinstance(v, property)}
    assert props == {k.accessor for k in defs.knobs}
    b = baseline()
    for k in defs.knobs:
        assert getattr(b, k.accessor) is not None, k.id
    assert b.version_id is None and b.definitions_hash == defs.hash


def test_experiments_x1_to_x9_are_defined_with_the_knobs_they_set() -> None:
    defs = definitions()
    assert [e.id for e in defs.experiments] == [f"X{i}" for i in range(1, 10)]
    assert all(e.status == "defined" and e.arms and e.metrics for e in defs.experiments)
    sets = {e.id: set(e.sets) for e in defs.experiments}
    assert sets["X5"] == {"budget.retries_per_task"}
    assert sets["X7"] == {"learning.reproduction_epsilon"}
    assert "budget.task_wall_s" in sets["X9"]
    for k in defs.knobs:
        assert all(k.id in sets[x] for x in k.set_by), k.id
    assert next(e for e in defs.experiments if e.id == "X2").profile_fields == ("reasoning_effort",)


def test_d325_defaults_are_registry_values() -> None:
    b = baseline()
    assert b.reproduction_epsilon == pytest.approx(0.10)
    assert b.learning_scope["version"] == "v1" and b.learning_scope["targets"] == ["gb10"]
    assert b.release_stage == "S0"


def test_settings_are_a_view_of_the_controls() -> None:
    s = Settings.of(baseline())
    assert (s.max_rounds, s.max_retries, s.shortlist, s.top_k) == (5, 2, 64, 8)
    assert s.task_wall_s["T2"] == 5400 and s.card_budget("T4")["wall_s"] == 2700
    assert s.loop_limit() == {"agent_min": 5 * 5400 / 60, "gpu_min": 10 * 600 / 60}
    assert s.epsilon_rel == pytest.approx(0.1) and s.attention_limit == 10


def test_invalid_values_and_invariants_are_refused() -> None:
    b = baseline()
    with pytest.raises(ConfigError, match="below the minimum"):
        b.with_changes({"budget.rounds_per_case_loop": 0})
    with pytest.raises(ConfigError, match="expected an integer"):
        b.with_changes({"budget.retries_per_task": 2.5})
    with pytest.raises(ConfigError, match="keys must be"):
        b.with_changes({"budget.agent_instances": {"start": 2}})
    with pytest.raises(ConfigError, match="sum to"):
        b.with_changes({"budget.process_quotas": {**b.process_quotas, "solve": 0.9}})
    with pytest.raises(ConfigError, match="not one of"):
        b.with_changes({"release.stage": "S9"})
    with pytest.raises(ConfigError, match="no control"):
        b.with_changes({"budget.nonexistent": 1})
    with pytest.raises(PermissionDenied, match="invariant"):
        b.with_changes({"guard.hidden_shapes_and_seeds": False})
    assert b.with_changes({"budget.retries_per_task": 3}).retries_per_task == 3
    assert b.retries_per_task == 2  # snapshots are immutable


def test_moved_configuration_keys_are_refused(ctx: Context, ops) -> None:
    from kta.agents.config import agents_config

    with pytest.raises(ConfigError, match=r"budget\.agent_instances"):
        agents_config(
            Context(
                ctx.env,
                config_from_mapping({"agents": {"worker": "w", "slots": 2}}),
                ctx.data_dir,
                ctx.registry,
            )
        )
    cfg = config_from_mapping({"budget": {"agent_min": 10.0, "quotas": {"solve": 0.5}}})
    with pytest.raises(ConfigError, match=r"budget\.process_quotas"):
        configure_scopes(ops, cfg, baseline())


# ---- versions ---------------------------------------------------------------------------------


def test_a_value_change_is_a_new_version_with_author_and_reason(store, ops, clock) -> None:
    assert controls_store.current(store, ops).version_id is None  # nothing stored yet
    v1 = controls_store.ensure(store, ops, clock.now())
    assert v1.version == 1 and v1.version_id
    rec1 = store.get(v1.version_id)
    assert rec1.actor.type == "system" and "version 1" in rec1.body["reason"]
    assert controls_store.ensure(store, ops, clock.now()) == v1  # written once
    v2 = controls_store.propose(
        store, ops, {"budget.retries_per_task": 3}, HUMAN, "X5 pilot", clock.now(), v1.version_id
    )
    rec2 = store.get(str(v2.version_id))
    assert (v2.version, rec2.refs["previous"], rec2.actor.name) == (2, v1.version_id, "test")
    assert rec2.body["changes"] == [{"knob": "budget.retries_per_task", "old": 2, "new": 3}]
    assert controls_store.current(store, ops) == v2
    assert [rid for rid, _ in controls_store.history(store)] == [v2.version_id, v1.version_id]
    with pytest.raises(ConfigError, match="reload"):  # the author read v1, v2 is current
        controls_store.propose(
            store, ops, {"budget.retries_per_task": 4}, HUMAN, "r", clock.now(), v1.version_id
        )
    with pytest.raises(ConfigError, match="equal"):
        controls_store.propose(
            store, ops, {"budget.retries_per_task": 3}, HUMAN, "r", clock.now(), v2.version_id
        )
    with pytest.raises(PermissionDenied):
        system = Actor(type="system", name="x")
        controls_store.propose(
            store, ops, {"budget.retries_per_task": 4}, system, "r", clock.now(), v2.version_id
        )
    with pytest.raises(ConfigError, match="reason"):
        controls_store.propose(
            store, ops, {"budget.retries_per_task": 4}, HUMAN, " ", clock.now(), v2.version_id
        )


def test_definitions_upgrade_writes_a_system_version(store, ops, clock) -> None:
    """A stored version that lacks a knob the definitions now have (a knob added later,
    e.g. by WP18) is carried over by a system version; user-set values are kept."""
    full = dict(baseline().values)
    full["budget.retries_per_task"] = 4
    del full["release.stage"]
    full["budget.retired_knob"] = 1
    body = ControlsVersion(version=1, definitions_hash="old", values=full, reason="older code")
    old = store.put(Record.new("ControlsVersion", body, HUMAN, clock.now()))
    assert ops.set_pointer(controls_store.POINTER, None, old)
    pending = controls_store.current(store, ops)
    assert pending.version_id is None and pending.release_stage == "S0"
    up = controls_store.ensure(store, ops, clock.now())
    rec = store.get(str(up.version_id))
    assert up.version == 2 and rec.actor.type == "system" and rec.refs["previous"] == old
    assert "added ['release.stage']" in rec.body["reason"]
    assert "dropped ['budget.retired_knob']" in rec.body["reason"]
    assert up.retries_per_task == 4


def test_task_card_and_loop_record_the_controls_version(svc: Services) -> None:
    vid = set_controls(svc.store, svc.ops, {"budget.task_wall_s": {**_walls(), "T1": 1200}})
    q = TaskQueue(svc.ctx, svc.store, svc.ops, svc)
    task = q.get(q.create(TaskRequest("T1", "g", []), None, "human"))
    card = svc.store.get(task.card_id)
    assert card.refs["controls"] == vid and card.body["budget"]["wall_s"] == 1200


def _walls() -> dict[str, float]:
    return baseline().task_wall_s


# ---- staged release ----------------------------------------------------------------------------


@pytest.fixture
def queue(svc: Services) -> TaskQueue:
    svc.ops.set_pointer(CURRENT_PROFILE, None, "f" * 64)
    return TaskQueue(svc.ctx, svc.store, svc.ops, svc)


def _leaves_running(seen: list[str]) -> Callable[..., Any]:
    class Result:
        run_id = "run1"

    def step(ctx: Any, task: Any, **kw: Any) -> Result:
        seen.append(task.id)
        return Result()

    return step


def test_s0_runs_no_agent_calls_and_skips_processes_outside_it(
    queue: TaskQueue, svc: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    tid = queue.create(TaskRequest("T1", "g", []), None, "human")
    assert release.stage_of(queue.controls()).id == "S0"
    assert next_task(queue) is None
    held = release.held_calls(svc.ops, queue.controls(), ("T1",))
    assert held[0]["task"] == tid and "runs no agent calls" in held[0]["held"]
    assert not release.process_allowed(queue.controls(), "learn")
    assert release.process_allowed(queue.controls(), "maintain")
    ticked: list[str] = []
    monkeypatch.setitem(processes.TICKS, "learn", lambda r: ticked.append("learn"))
    monkeypatch.setitem(processes.TICKS, "solve", lambda r: ticked.append("solve"))
    assert processes.run(svc, "learn", interval_s=0.0, max_ticks=1) == 1
    assert processes.run(svc, "solve", interval_s=0.0, max_ticks=1) == 1
    assert ticked == ["solve"]  # learn is outside S0: its tick was skipped


def test_s1_gate_holds_a_fake_adapter_task_until_released(queue: TaskQueue, svc: Services) -> None:
    set_controls(svc.store, svc.ops, {"release.stage": "S1"}, "hand-stepped release")
    tid = queue.create(TaskRequest("T1", "g", []), None, "human")
    other = queue.create(TaskRequest("T1", "h", []), None, "human")
    seen: list[str] = []
    assert step_once(svc, queue, 1, None, step=_leaves_running(seen), poll_s=0.01) == {"task": None}
    held = {h["task"]: h["held"] for h in release.held_calls(svc.ops, queue.controls(), ("T1",))}
    assert "waits for your release" in held[tid]
    action = release.release_call(
        svc.store, svc.ops, queue.get(tid), HUMAN, "read it", svc.ctx.clock.now()
    )
    rec = svc.store.get(action)
    assert rec.body["action"].startswith("release_agent_call") and rec.refs["card"]
    assert json.loads(rec.body["action"].split(" ", 1)[1]) == {"attempt": 1, "task": tid}
    out = step_once(svc, queue, 1, None, step=_leaves_running(seen), poll_s=0.01)
    assert out["task"] == tid and seen == [tid]
    assert out["verdict"] == "queued"  # abandoned without submit: back in the queue
    assert next_task(queue) is None  # the release covered one attempt only
    retry = {h["task"]: h["held"] for h in release.held_calls(svc.ops, queue.controls(), ("T1",))}
    assert "attempt 2 waits" in retry[tid] and "attempt 1 waits" in retry[other]


def test_s1_runs_one_agent_call_at_a_time(queue: TaskQueue, svc: Services) -> None:
    set_controls(svc.store, svc.ops, {"release.stage": "S1"})
    a = queue.create(TaskRequest("T1", "a", []), None, "human")
    b = queue.create(TaskRequest("T1", "b", []), None, "human")
    for t in (a, b):
        release.release_call(svc.store, svc.ops, queue.get(t), HUMAN, "ok", svc.ctx.clock.now())
    queue.lease(a, "agent:1")
    reason = release.hold_reason(svc.ops, queue.get(b), queue.controls())
    assert reason is not None and "one agent call at a time" in reason
    with pytest.raises(release.Held):
        release.admit(svc.ops, queue.get(b), queue.controls())


def test_stage_change_is_a_controls_version(svc: Services) -> None:
    vid = set_controls(svc.store, svc.ops, {"release.stage": "S2"}, "S1 exit criteria met")
    rec = svc.store.get(vid)
    assert rec.body["changes"] == [{"knob": "release.stage", "old": "S0", "new": "S2"}]
    s2 = release.stage_of(controls_store.current(svc.store, svc.ops))
    assert s2.agent_calls == "automatic" and "T2" not in s2.task_types
    assert [s.id for s in release.stages()] == ["S0", "S1", "S2", "S3", "S4"]
    assert all(s.exit for s in release.stages()[:4])


# ---- the settings API -----------------------------------------------------------------------


@pytest.fixture
def api(tmp_path: Path) -> Api:
    token = tmp_path / "operator-token"
    token.write_text("secret-operator-token\n")
    return seeded_api(tmp_path / "d", sections={"api": {"operator_token_file": str(token)}})


def _call(api: Api, method: str, path: str, body: Any = None, token: str | None = None):
    headers = {"X-KTA-Operator-Token": token} if token else {}
    return request(create_app(api), method, path, body, headers)


def test_settings_api_reads_open_and_writes_a_version_with_the_token(api: Api) -> None:
    status, view = _call(api, "GET", "/controls")
    assert status == 200 and len(view["knobs"]) == len(definitions().knobs)
    assert view["version"]["stored"] is False and view["write"]["token_configured"]
    knob = next(k for k in view["knobs"] if k["id"] == "budget.retries_per_task")
    assert knob["value"] == 2 and knob["set_by"] == ["X5"] and knob["editable"]
    body = {"expected": None, "changes": {"budget.retries_per_task": 3}, "reason": "X5 pilot"}
    assert _call(api, "POST", "/controls", body)[0] == 403  # no token
    assert _call(api, "POST", "/controls", body, "wrong")[0] == 403
    # Nothing stored yet: the author read the unstored values (expected null); version 1
    # stores them and version 2 carries the change.
    status, out = _call(api, "POST", "/controls", body, "secret-operator-token")
    assert status == 200 and out["version"] == 2, out
    v1 = api.store.get(out["id"]).refs["previous"]
    rec = api.store.get(out["id"])
    assert rec.actor.type == "human" and rec.body["reason"] == "X5 pilot"
    action = api.store.get(out["action"])
    assert action.body["action"].startswith("set_controls") and action.refs["controls"] == out["id"]
    status, hist = _call(api, "GET", "/controls/versions")
    assert [h["version"] for h in hist] == [2, 1] and hist[0]["author"] == "user"
    stale = {**body, "expected": v1, "changes": {"budget.retries_per_task": 4}}
    assert _call(api, "POST", "/controls", stale, "secret-operator-token")[0] == 400
    locked = {"expected": out["id"], "changes": {"guard.evidence_write_ban": False}, "reason": "r"}
    assert _call(api, "POST", "/controls", locked, "secret-operator-token")[0] == 403
    bad = {"expected": out["id"], "changes": {"budget.lease_s": "long"}, "reason": "r"}
    assert _call(api, "POST", "/controls", bad, "secret-operator-token")[0] == 400
    blank = {"expected": out["id"], "changes": {"budget.lease_s": 900}, "reason": " "}
    assert _call(api, "POST", "/controls", blank, "secret-operator-token")[0] == 400
    machines = _call(api, "GET", "/machines")[1]
    task = next(m for m in machines["machines"] if m["id"] == "task")
    assert task["limits"]["max_retries"] == 3
    assert task["limit_controls"] == {"max_retries": "budget.retries_per_task"}


def test_release_api_shows_stages_and_releases_a_held_call(api: Api) -> None:
    status, view = _call(api, "GET", "/release")
    assert status == 200 and view["stage"]["id"] == "S0"
    assert [s["id"] for s in view["stages"]] == ["S0", "S1", "S2", "S3", "S4"]
    assert all("exit" in s for s in view["stages"])
    queued = [h for h in view["held"]]
    assert queued, "the fixture world has queued agent tasks"
    task = queued[0]["task"]
    assert queued[0]["card_body"]["type"] and queued[0]["pull_log"] is None
    assert "WP19" in queued[0]["pull_log_note"] and "WP19" in queued[0]["plan_note"]
    body = {"task": task, "reason": "read card, pull log and plan"}
    assert _call(api, "POST", "/release", body, "secret-operator-token")[0] == 400  # S0: no gate
    v1 = controls_store.ensure(api.store, api.ops, api.now()).version_id
    change = {"expected": v1, "changes": {"release.stage": "S1"}, "reason": "start S1"}
    assert _call(api, "POST", "/controls", change, "secret-operator-token")[0] == 200
    assert _call(api, "POST", "/release", body)[0] == 403
    status, out = _call(api, "POST", "/release", body, "secret-operator-token")
    assert status == 200 and out["attempt"] >= 1
    after = _call(api, "GET", "/release")[1]
    row = next(h for h in after["held"] if h["task"] == task)
    assert row["released_attempt"] == out["attempt"]
    assert row["held"] is None or "one agent call at a time" in row["held"]


def test_operator_token_file_is_required_for_writes(tmp_path: Path) -> None:
    api = seeded_api(
        tmp_path / "d", sections={"api": {"operator_token_file": str(tmp_path / "none")}}
    )
    body = {"expected": None, "changes": {"budget.lease_s": 900}, "reason": "r"}
    status, out = _call(api, "POST", "/controls", body, "anything")
    assert status == 403 and "kta controls token" in out["detail"]


def test_doc12_rows_name_each_registry_section() -> None:
    text = DOC12.read_text(encoding="utf-8")
    assert re.search(r"^## 7\. Rows added from code and D325", text, re.M)
    assert text.count("| set_by |") == 5


def test_a_task_token_cannot_write_a_controls_version(api: Api) -> None:
    """Design interrogation (C5): an agent's token must not write ControlsVersion records."""
    from dataclasses import replace as dc_replace

    from kta.api import auth
    from kta.api.wiring import Unwired

    class Verified(Unwired):
        def verify_task_token(self, token: str, now: datetime) -> str:
            return "T1"

    body = ControlsVersion(version=9, definitions_hash="x", values={}, reason="agent")
    agent = Actor(type="agent", name="worker", task="T1")
    rec = Record.new("ControlsVersion", body, agent, api.now())
    with pytest.raises(PermissionDenied, match="ControlsVersion"):
        auth.write_record(dc_replace(api, orchestration=Verified()), "tok", rec.to_json_dict())
