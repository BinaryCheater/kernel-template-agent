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

from kta.agents import sessions
from kta.agents.cards import RetryNote
from kta.agents.config import agents_config
from kta.agents.profile import ProfileSpec, write_profile
from kta.agents.runner import Deps, parse_session, run_step
from kta.agents.units import close_work_unit
from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.ids import UlidFactory
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
    profile = write_profile(store, datetime.now(UTC), TEST_PROFILE)
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
    assert not list(data_root.rglob("work/wu-1")) and not list(data_root.rglob("agent-home/wu-1"))
    state, _ = sessions.load(ops, "wu-1")
    assert state is not None and state.status == "closed"


def test_refuses_an_agents_md_above_the_workspace(actx: Context, parts, tmp_path: Path) -> None:
    store, ops, ex = parts
    deps = Deps(store, ops, ex, agents_config(actx))
    profile = write_profile(store, datetime.now(UTC), TEST_PROFILE)
    task = _task(store, ops, profile, wu="wu-up")
    (Path(actx.data_dir) / "AGENTS.md").parent.mkdir(parents=True, exist_ok=True)
    (Path(actx.data_dir) / "AGENTS.md").write_text("user-level instructions")
    with tracing(new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})):
        result = run_step(actx, task, deps=deps)
    assert result.launch.exit == "refused" and "AGENTS.md" in result.launch.report["error"]
