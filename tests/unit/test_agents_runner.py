"""run_step (run_task) end to end over the real local Executor and launcher, with a fake CLI.

The CLI is tests/fakes/fake_codex.py.
Only the LLM CLI is replaced. Workspace and CLI-home preparation, the context manifest,
limits, git commits, packing, session parsing, compliance, the AgentRun record and the
session state are the real code paths.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kta.agents import batches, sessions
from kta.agents.cards import RetryNote
from kta.agents.config import agents_config
from kta.agents.profile import ProfileSpec, write_profile
from kta.agents.runner import Deps, parse_session, run_step
from kta.agents.units import close_work_unit
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.errors import PermissionDenied
from kta.core.ids import UlidFactory, record_id
from kta.core.registry import Registry
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import TaskCard
from kta.core.trace import new_tracer, tracing
from kta.execution.executor import JobExecutor
from kta.execution.registry import register, worker_configs
from kta.store import CodeState, SqliteOps, SqliteStore, code_state, open_ops, open_store

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "fake_codex.py"
TEST_PROFILE = ProfileSpec(name="codex-test", isolation="config")


@pytest.fixture
def actx(tmp_path: Path) -> Context:
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text('{"fake": "credential"}')
    (tmp_path / "watched").mkdir()
    config = {
        "workers": {"local": {"transport": "local", "pools": {"cpu": 1, "agent": 1}}},
        "executor": {"queue_timeout_s": 60},
        "agents": {
            "worker": "local",
            "batch": "b1",
            "pool": "agent",
            "token_key_file": str(tmp_path / "keys" / "task-token.key"),
            "watch": [str(tmp_path / "watched")],
            "hosts": {
                "local": {
                    "codex_cmd": [sys.executable, str(FAKE)],
                    "auth_dir": str(auth),
                    "nvidia_smi": "/nonexistent/nvidia-smi",
                }
            },
        },
    }
    return Context(
        env="test", config=config_from_mapping(config), data_dir=tmp_path / "d", registry=Registry()
    )


@pytest.fixture
def parts(actx: Context) -> tuple[SqliteStore, SqliteOps, JobExecutor]:
    store = open_store(actx, CodeState(code_state().commit, False))
    ops = open_ops(actx)
    register(actx, store, ops, worker_configs(actx)["local"], code_state().commit)
    return store, ops, JobExecutor(actx, store, ops, code_state().commit)


def _profile_and_batch(actx: Context, store: SqliteStore) -> str:
    profile = write_profile(store, datetime.now(UTC), TEST_PROFILE)
    batches.build(store, actx.data_dir, "b1", profile, code_state().commit)
    return profile


def _task(store: SqliteStore, ops: SqliteOps, profile: str, wu: str = "wu-1"):
    card = TaskCard(
        type="T1",
        goal="extend the test family to bf16",
        done_checks=["validate_family"],
        budget={"wall_s": 120},
        profile=profile,
        work_unit=wu,
    )
    card_id = store.put(
        Record.new("TaskCard", card, Actor(type="system", name="test"), datetime.now(UTC))
    )
    ops.enqueue(card_id, 1, wu)
    task = ops.lease("runner-test", ["T1"], 600)
    assert task is not None
    return task


def _members(store: SqliteStore, blob: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(store.blob_path(blob).read_bytes()), mode="r:gz") as tar:
        for m in tar.getmembers():
            f = tar.extractfile(m) if m.isfile() else None
            if f is not None:
                out[m.name] = f.read()
    return out


def test_start_resume_violation_and_close(actx: Context, parts, tmp_path: Path) -> None:
    store, ops, ex = parts
    deps = Deps(store, ops, ex, agents_config(actx))
    profile = _profile_and_batch(actx, store)
    assert write_profile(store, datetime.now(UTC), TEST_PROFILE) == profile  # idempotent
    task = _task(store, ops, profile)
    tracer = new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})
    with tracing(tracer):
        first = run_step(actx, task, deps=deps)
    assert first.launch.exit == "ok", first.launch.report
    assert first.decision.startswith("start")
    run = store.get(first.run_id)
    assert run.kind == "AgentRun" and run.refs["profile"] == profile and run.refs["env"]
    assert first.run.tokens_in == 1200 and first.run.session
    assert not [v for v in first.run.violations if v["kind"] != "kta_command_not_allowed"], (
        first.run.violations
    )

    archive = _members(store, first.run.session_archive)
    assert "auth.json" not in archive and "config.toml" in archive
    assert any(n.startswith("sessions/") for n in archive)
    manifest = json.loads(store.blob_path(first.run.context_manifest).read_bytes())
    assert manifest["home"]["files"]["auth.json"] == "credential: provided"
    assert manifest["env"]["KTA_TASK_TOKEN"] == "<provided>"
    assert {"AGENTS.md", "TASK.md", "NOTES.md"} <= set(manifest["workspace"]["files"])
    assert any(k.startswith(".agents/skills/write-family/") for k in manifest["workspace"]["files"])
    env_names = json.loads(_members(store, first.launch.workspace)["out/env.json"])
    assert "KTA_TASK_TOKEN" in env_names and "SSH_AUTH_SOCK" not in env_names

    archive_path = tmp_path / "home.tar.gz"
    archive_path.write_bytes(store.blob_path(first.run.session_archive).read_bytes())
    events = parse_session(archive_path)
    assert any(e.kind == "tool_result" and e.data.get("exec") for e in events)

    retry = RetryNote([("validate_family", "chain item k_split has no code marker")])
    with tracing(tracer):
        second = run_step(actx, task, deps=deps, retry=retry)
    assert second.decision.startswith("resume") and second.run.session == first.run.session
    assert store.get(second.run_id).refs["previous"] == first.run_id

    target = tmp_path / "watched" / "out-of-scope.txt"
    with tracing(tracer):
        third = run_step(actx, task, deps=deps, note=f"INDUCE-WRITE {target}")
    kinds = {v["kind"] for v in third.run.violations}
    assert "write_outside_workspace" in kinds, third.run.violations

    state, _ = sessions.load(ops, "wu-1")
    assert state is not None and len(state.runs) == 3
    report_id = close_work_unit(actx, store, ops, ex, deps.config, "wu-1")
    report = store.get(report_id)
    ws = _members(store, report.blobs["workspace"])
    assert any(n.startswith(".git/") for n in ws) and "NOTES.md" in ws
    data_root = Path(actx.data_dir)
    assert (data_root / "batches" / "b1" / "env.lock.json").exists()
    assert not list(data_root.rglob("work/wu-1")) and not list(data_root.rglob("agent-home/wu-1"))
    state, _ = sessions.load(ops, "wu-1")
    assert state is not None and state.status == "closed"


def test_refuses_an_agents_md_above_the_workspace(actx: Context, parts, tmp_path: Path) -> None:
    store, ops, ex = parts
    deps = Deps(store, ops, ex, agents_config(actx))
    profile = write_profile(store, datetime.now(UTC), TEST_PROFILE)
    task = _task(store, ops, profile, wu="wu-up")
    batches.build(store, actx.data_dir, "b1", profile, code_state().commit)
    (Path(actx.data_dir) / "AGENTS.md").parent.mkdir(parents=True, exist_ok=True)
    (Path(actx.data_dir) / "AGENTS.md").write_text("user-level instructions")
    with tracing(new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})):
        result = run_step(actx, task, deps=deps)
    assert result.launch.exit == "refused" and "AGENTS.md" in result.launch.report["error"]


def test_runs_only_from_a_verified_batch(
    actx: Context, parts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, ops, _ = parts
    profile = _profile_and_batch(actx, store)
    # Nothing may come from the source tree or the user's ~/.codex at run time.
    from kta.agents import profile as profile_mod

    monkeypatch.setattr(profile_mod, "PROMPTS", tmp_path / "no-prompts")
    monkeypatch.setattr(profile_mod, "SKILLS", tmp_path / "no-skills")
    fake_home = tmp_path / "user-home"
    (fake_home / ".codex").mkdir(parents=True)
    (fake_home / ".codex" / "AGENTS.md").write_text("USER-LEVEL-MARKER-7f3a\n")
    monkeypatch.setenv("HOME", str(fake_home))
    # A new executor, so the local worker process inherits the fake HOME.
    deps = Deps(store, ops, JobExecutor(actx, store, ops, code_state().commit), agents_config(actx))
    tracer = new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})
    with tracing(tracer):
        ok = run_step(actx, _task(store, ops, profile, wu="wu-b"), deps=deps)
    assert ok.launch.exit == "ok", ok.launch.report
    manifest = json.loads(store.blob_path(ok.run.context_manifest).read_bytes())
    used = "present, not used (CODEX_HOME points to home)"
    assert manifest["user_level"] == {str(fake_home / ".codex"): used}
    for blob in (ok.run.session_archive, ok.launch.workspace, ok.run.context_manifest):
        assert b"USER-LEVEL-MARKER-7f3a" not in store.blob_path(blob).read_bytes()
    archive = _members(store, ok.run.session_archive)
    assert not any(b"USER-LEVEL-MARKER-7f3a" in data for data in archive.values())

    skill = next((Path(actx.data_dir) / "batches" / "b1" / "skills").glob("use-kta@*/SKILL.md"))
    skill.write_text(skill.read_text() + "\ntampered\n")
    with tracing(tracer), pytest.raises(PermissionDenied, match="changed: skills/use-kta@"):
        run_step(actx, _task(store, ops, profile, wu="wu-c"), deps=deps)


def test_each_arm_shapes_workspace_manifest_and_agent_run(actx: Context, parts) -> None:
    from knowhelp import seed
    from kta.core.schemas.process import PoolSnapshot, Report

    store, ops, ex = parts
    deps = Deps(store, ops, ex, agents_config(actx))
    profile = _profile_and_batch(actx, store)
    seed(actx, store)
    template_id = record_id(store.query("Template", limit=1)[0])
    members = [record_id(r) for kind in ("Pattern", "Template") for r in store.query(kind)]
    actor = Actor(type="system", name="test")
    snap = store.put(
        Record.new(
            "PoolSnapshot", PoolSnapshot(members=members, note="t"), actor, datetime.now(UTC)
        )
    )
    subject = store.put(
        Record.new(
            "Report",
            Report(title="case", subject="case", summary="the task subject"),
            actor,
            datetime.now(UTC),
        )
    )
    manifests: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    tracer = new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})
    for knowledge in ("zero", "text", "pool"):
        card = TaskCard(
            type="T1",
            goal="g",
            done_checks=["validate_family"],
            budget={"wall_s": 120},
            profile=profile,
            work_unit=f"wu-{knowledge}",
            inputs=[template_id, subject],
        )
        card_id = store.put(Record.new("TaskCard", card, actor, datetime.now(UTC)))
        ops.enqueue(card_id, 1, f"wu-{knowledge}")
        task = ops.lease("runner-test", ["T1"], 600)
        assert task is not None
        # The bookkeeping exactly as WP6 writes it (wp/WP6 change 8a9331d), batch included.
        arm = {
            "name": f"arm-{knowledge}",
            "knowledge": knowledge,
            "condition": {"knowledge": knowledge},
            "snapshot": snap,
            "protocol": None,
        }
        assert ops.set_pointer(f"task:{task.id}", None, json.dumps({"arm": arm, "batch": "b1"}))
        with tracing(tracer):
            result = run_step(actx, task, deps=deps)
        assert result.launch.exit == "ok", result.launch.report
        manifests[knowledge] = json.loads(store.blob_path(result.run.context_manifest).read_bytes())
        runs[knowledge] = {"arm": result.run.arm, "refs": store.get(result.run_id).refs}

    def files(k: str) -> set[str]:
        return set(manifests[k]["workspace"]["files"])

    tmpl_input = f"inputs/Template-{template_id[:12]}.json"
    subject_input = f"inputs/Report-{subject[:12]}.json"
    assert all(subject_input in files(k) for k in manifests)  # the task subject is always given
    assert tmpl_input in files("pool") and tmpl_input not in files("zero") | files("text")
    assert any(f.startswith(".agents/skills/find-knowledge/") for f in files("pool"))
    assert not any(
        f.startswith(".agents/skills/find-knowledge/") for f in files("zero") | files("text")
    )
    assert "inputs/knowledge/index.md" in files("text")
    assert not any(f.startswith("inputs/knowledge/") for f in files("zero") | files("pool"))
    for k in manifests:
        assert manifests[k]["arm"] == {
            "name": f"arm-{k}",
            "knowledge": k,
            "snapshot": snap,
            "protocol": None,
            "condition": {"knowledge": k},
        }
        assert runs[k]["arm"] == manifests[k]["arm"] and runs[k]["refs"]["snapshot"] == snap
    # The fixed prefix differs only between conditions (skills listed), not within one.
    prefix = {k: manifests[k]["workspace"]["files"]["AGENTS.md"] for k in manifests}
    assert prefix["zero"] == prefix["text"] != prefix["pool"]


def test_value_change_is_versioned_and_visible_in_the_agent_run(actx: Context, parts) -> None:
    """WP21: a controls change is a new version; the next agent step runs under it and its
    AgentRun names it in refs.controls (the fake CLI replaces only the LLM)."""
    from kta.store import controls as controls_store

    store, ops, ex = parts
    deps = Deps(store, ops, ex, agents_config(actx))
    profile = _profile_and_batch(actx, store)
    tracer = new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})
    with tracing(tracer):
        first = run_step(actx, _task(store, ops, profile, "wu-a"), deps=deps)
    v1 = controls_store.current(store, ops)
    assert v1.version == 1 and store.get(first.run_id).refs["controls"] == v1.version_id
    human = Actor(type="human", name="tester")
    v2 = controls_store.propose(
        store,
        ops,
        {"budget.no_response_timeout_s": 900},
        human,
        "longer idle cut-off for this test",
        datetime.now(UTC),
        v1.version_id,
    )
    with tracing(tracer):
        second = run_step(actx, _task(store, ops, profile, "wu-b"), deps=deps)
    rec = store.get(second.run_id)
    assert rec.refs["controls"] == v2.version_id != v1.version_id
    version = store.get(str(v2.version_id))
    assert version.refs["previous"] == v1.version_id and version.actor.name == "tester"
    assert version.body["changes"] == [
        {"knob": "budget.no_response_timeout_s", "old": 600, "new": 900}
    ]
