"""Unit tests of the agents module pieces: tokens, session rules, compliance, parsing, packing."""

from __future__ import annotations

import io
import json
import os
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kta.agents import compliance, guard, hostfs, sessions
from kta.agents.adapters import capabilities, codex_session
from kta.agents.adapters.base import StepPlan
from kta.agents.adapters.codex import CodexAdapter, check_flags
from kta.agents.base import AgentEvent
from kta.agents.cards import RetryNote, prefix, task_md
from kta.agents.launch import data_root
from kta.agents.launchspec import LaunchSpec
from kta.agents.profile import CODEX_FLAGS, pack_dir, unpack
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.schemas.process import TaskCard

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


class _Ops:
    def __init__(self, state: str) -> None:
        self.state = state

    def task(self, task_id: str):  # the TaskLookup part of SqliteOps
        from kta.store.base import Task

        return Task(task_id, "c", "T1", self.state, 1, None, None, 0, "wu")


def test_token_roundtrip_expiry_tamper_and_task_state(tmp_path: Path) -> None:
    key = guard.ensure_key(tmp_path / "k" / "task-token.key")
    assert (tmp_path / "k" / "task-token.key").stat().st_mode & 0o077 == 0
    token = guard.issue_token(key, "01TASK", NOW + timedelta(hours=1))
    assert guard.verify_token(key, token, NOW) == "01TASK"
    assert guard.verify_task_token(key, token, NOW, _Ops("leased")) == "01TASK"
    with pytest.raises(PermissionDenied, match="no longer valid"):
        guard.verify_task_token(key, token, NOW, _Ops("done"))
    with pytest.raises(PermissionDenied, match="expired"):
        guard.verify_token(key, token, NOW + timedelta(hours=2))
    with pytest.raises(PermissionDenied, match="signature"):
        guard.verify_token(key, token.replace("01TASK", "01OTHER"), NOW)
    with pytest.raises(PermissionDenied, match="signature"):
        guard.verify_token(os.urandom(32), token, NOW)


def test_token_key_must_be_private(tmp_path: Path) -> None:
    path = tmp_path / "key"
    path.write_bytes(os.urandom(32))
    path.chmod(0o644)
    with pytest.raises(PermissionDenied):
        guard.ensure_key(path)


def test_agent_env_is_whitelisted() -> None:
    env = guard.agent_env({"KTA_TASK_TOKEN": "t", "HOME": "{home}"}, ["{kta_bin}"])
    assert env["PATH"].startswith("{kta_bin}:") and set(env) == {"KTA_TASK_TOKEN", "HOME", "PATH"}
    with pytest.raises(PermissionDenied, match="SSH_AUTH_SOCK"):
        guard.agent_env({"SSH_AUTH_SOCK": "/tmp/x"}, [])


def _state(**kw) -> sessions.SessionState:
    return sessions.SessionState(**{"session": "s1", "worker": "spark", "last_exit": "ok", **kw})


@pytest.mark.parametrize(
    ("state", "resume", "failure", "mode", "reason"),
    [
        (None, True, None, "start", "new work unit"),
        (_state(status="closed"), True, None, "start", "no open session"),
        (_state(), False, None, "start", "cannot resume"),
        (_state(last_exit="idle"), True, None, "start", "abnormally"),
        (_state(context_ratio=0.7), True, None, "start", "context"),
        (_state(failures=["f"]), True, "f", "start", "twice"),
        (_state(failures=["g"]), True, "f", "resume", "continue"),
        (_state(context_ratio=0.5), True, None, "resume", "continue"),
    ],
)
def test_session_rules(state, resume, failure, mode, reason) -> None:
    d = sessions.decide(state, resume, {"reset_context_ratio": 0.6}, failure)
    assert d.mode == mode and reason in d.reason


def test_after_run_tracks_failures_per_session() -> None:
    s = sessions.after_run(
        None,
        session="s",
        worker="w",
        run_id="r1",
        exit="ok",
        context_ratio=0.1,
        failure=None,
        started_new=True,
    )
    s = sessions.after_run(
        s,
        session="s",
        worker="w",
        run_id="r2",
        exit="ok",
        context_ratio=0.2,
        failure="f",
        started_new=False,
    )
    assert s.runs == ["r1", "r2"] and s.failures == ["f"]
    s = sessions.after_run(
        s,
        session="s2",
        worker="w",
        run_id="r3",
        exit="ok",
        context_ratio=0.0,
        failure="f",
        started_new=True,
    )
    assert s.failures == ["f"] and s.session == "s2"
    assert sessions.SessionState.from_json(s.to_json()) == s


def _exec(cmd: str, code: int = 0, out: str = "") -> AgentEvent:
    return AgentEvent(
        "tool_result",
        1.0,
        {
            "call_id": "c",
            "exec": {"command": ["bash", "-lc", cmd], "exit_code": code, "output_tail": out},
        },
    )


def test_compliance_from_events() -> None:
    ws = "/data/work/wu"
    events = [
        _exec("cat x > /data/kta.db", 1, "bash: /data/kta.db: Permission denied"),
        _exec("kta measure b1 && kta submit"),
        _exec("ssh spark nvidia-smi"),
        _exec("echo ok > out/a.txt"),
        AgentEvent(
            "tool_result",
            2.0,
            {"call_id": "p", "patch": {"paths": ["/home/u/.codex/AGENTS.md", f"{ws}/out/f.py"]}},
        ),
    ]
    found = compliance.from_events(events, ws, allowed=["submit", "show"])
    kinds = [v["kind"] for v in found]
    assert kinds.count("denied") == 1
    assert kinds.count("write_outside_workspace") == 2  # the redirect and the patch
    assert any(
        v["detail"].startswith("kta measure")
        for v in found
        if v["kind"] == "kta_command_not_allowed"
    )
    assert kinds.count("forbidden_tool") == 1
    assert compliance.summary(found)["denied"] == 1


def test_compliance_from_report_and_context_changes() -> None:
    report = {
        "exit": "ok",
        "notes_changed": False,
        "workspace_extra": ["scratch"],
        "gpu_guard": [{"pid": 7}],
        "watch_changed": ["/home/u/.codex"],
    }
    kinds = {v["kind"] for v in compliance.from_report(report, notes_required=True)}
    assert kinds == {
        "notes_not_updated",
        "workspace_layout",
        "gpu_outside_kta",
        "write_outside_workspace",
    }
    before = {
        "TASK.md": "a",
        "inputs/x.json": "b",
        "NOTES.md": "c",
        ".agents/skills/s/SKILL.md": "d",
    }
    after = {
        "TASK.md": "a",
        "inputs/x.json": "B",
        "NOTES.md": "C",
        ".agents/skills/s/SKILL.md": "d",
        ".agents/skills/s/extra.md": "e",
    }
    details = {v["detail"] for v in compliance.context_changes(before, after)}
    assert details == {"inputs/x.json", "added .agents/skills/s/extra.md"}


SESSION_0117 = [
    {
        "timestamp": "2026-10-31T10:00:00.000Z",
        "type": "session_meta",
        "payload": {"id": "abc-123", "cli_version": "0.117.0", "cwd": "/w"},
    },
    {
        "timestamp": "2026-10-31T10:00:01.000Z",
        "type": "turn_context",
        "payload": {
            "model": "gpt-5.4",
            "cwd": "/w",
            "sandbox_policy": {"type": "workspace-write"},
            "approval_policy": "never",
        },
    },
    {
        "timestamp": "2026-10-31T10:00:02.000Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "do T1"}],
        },
    },
    {
        "timestamp": "2026-10-31T10:00:03.000Z",
        "type": "response_item",
        "payload": {"type": "reasoning", "summary": [], "encrypted_content": "zzz"},
    },
    {
        "timestamp": "2026-10-31T10:00:04.000Z",
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd": "kta show x"}',
            "call_id": "c1",
        },
    },
    {
        "timestamp": "2026-10-31T10:00:05.000Z",
        "type": "event_msg",
        "payload": {
            "type": "exec_command_end",
            "call_id": "c1",
            "command": ["bash", "-lc", "kta show x"],
            "cwd": "/w",
            "exit_code": 0,
            "aggregated_output": "ok",
            "duration": {"secs": 1, "nanos": 500000000},
        },
    },
    {
        "timestamp": "2026-10-31T10:00:05.100Z",
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    },
    {
        "timestamp": "2026-10-31T10:00:06.000Z",
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": "c2",
            "name": "apply_patch",
            "input": "*** Begin Patch",
        },
    },
    {
        "timestamp": "2026-10-31T10:00:06.500Z",
        "type": "event_msg",
        "payload": {
            "type": "patch_apply_end",
            "call_id": "c2",
            "success": True,
            "changes": {"/w/out/gen.py": {}},
        },
    },
    {
        "timestamp": "2026-10-31T10:00:06.600Z",
        "type": "response_item",
        "payload": {"type": "custom_tool_call_output", "call_id": "c2", "output": "Success"},
    },
    {
        "timestamp": "2026-10-31T10:00:07.000Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 5000,
                    "cached_input_tokens": 4000,
                    "output_tokens": 700,
                },
                "last_token_usage": {"input_tokens": 3000},
                "model_context_window": 10000,
            },
        },
    },
    {
        "timestamp": "2026-10-31T10:00:08.000Z",
        "type": "event_msg",
        "payload": {"type": "something_new"},
    },
    {"timestamp": "2026-10-31T10:00:09.000Z", "type": "brand_new_entry", "payload": {}},
]


def test_parse_codex_0117_session(tmp_path: Path) -> None:
    path = (
        tmp_path / "sessions" / "2026" / "10" / "31" / "rollout-2026-10-31T10-00-00-abc-123.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(e) for e in SESSION_0117) + "\nnot json\n")
    assert codex_session.session_id_of(path) == "abc-123"
    assert codex_session.files_of_session(tmp_path, "abc-123") == [path]
    events = codex_session.parse_session_files([path])
    results = [e for e in events if e.kind == "tool_result"]
    assert len(results) == 2
    assert results[0].data["exec"]["duration_s"] == 1.5 and results[0].data["output"] == "ok"
    assert results[1].data["patch"]["paths"] == ["/w/out/gen.py"]
    assert [e.data["name"] for e in events if e.kind == "tool_call"] == [
        "exec_command",
        "apply_patch",
    ]
    entries = {e.data.get("entry") for e in events if e.kind == "other"}
    assert {
        "event_msg/something_new",
        "brand_new_entry",
        "undecodable",
        "response_item/reasoning",
    } <= entries
    assert all("zzz" not in json.dumps(e.data) for e in events)  # no hidden reasoning content
    usage = codex_session.usage_of(events)
    assert (usage.tokens_in, usage.tokens_out, usage.context_ratio) == (5000, 700, 0.3)
    assert (
        codex_session.session_from_stdout(['{"type":"thread.started","thread_id":"t-9"}']) == "t-9"
    )


def test_codex_argv_and_refused_options() -> None:
    plan = StepPlan(LaunchSpec("codex", "wu", "r", "start"))
    a = CodexAdapter(None, plan, ["codex"], list(CODEX_FLAGS), "gpt-5.4", "medium")  # type: ignore[arg-type]
    assert a.start_argv()[:2] == ["codex", "exec"] and a.start_argv()[-1] == "-"
    assert a.resume_argv("s1")[:3] == ["codex", "exec", "resume"] and a.resume_argv("s1")[-2:] == [
        "s1",
        "-",
    ]
    assert "-m" in a.start_argv() and 'model_reasoning_effort="medium"' in a.start_argv()
    for bad in capabilities.NON_PERSISTENT_OPTIONS:
        with pytest.raises(ConfigError, match="persistence"):
            check_flags([bad])
    with pytest.raises(ConfigError):
        a.resume_argv("")


def test_capability_tables() -> None:
    assert capabilities.CODEX.admitted()
    assert not capabilities.CLAUDE_CODE.admitted()
    assert set(capabilities.ALL) == {"codex", "claude_code", "pi", "deepseek_harness"}
    for caps in capabilities.ALL.values():
        assert set(caps.items) == set(capabilities.REQUIREMENTS)


def test_pack_excludes_credentials_and_links_and_is_deterministic(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "sessions").mkdir(parents=True)
    (home / "sessions" / "s.jsonl").write_text("{}\n")
    (home / "auth.json").write_text("secret")
    (tmp_path / "real").write_text("elsewhere")
    (home / "linked").symlink_to(tmp_path / "real")
    data, flagged = hostfs.pack(home, exclude_top=["auth.json"])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert "auth.json" not in names and "linked" not in names and "sessions/s.jsonl" in names
    assert flagged == [] and hostfs.pack(home, exclude_top=["auth.json"])[0] == data


def test_credentials_link_and_refresh(tmp_path: Path) -> None:
    auth, home = tmp_path / "auth", tmp_path / "home"
    auth.mkdir()
    home.mkdir()
    with pytest.raises(PermissionDenied):
        hostfs.link_credentials(home, auth, ["auth.json"])
    (auth / "auth.json").write_text("v1")
    hostfs.link_credentials(home, auth, ["auth.json"])
    assert (home / "auth.json").is_symlink()
    (home / "auth.json").unlink()
    (home / "auth.json").write_text("v2")  # a CLI that refreshes by rename
    assert hostfs.restore_credentials(home, auth, ["auth.json"]) == ["auth.json"]
    assert (home / "auth.json").is_symlink() and (auth / "auth.json").read_text() == "v2"


def test_upward_agents_md(tmp_path: Path) -> None:
    ws = tmp_path / "a" / "work" / "wu"
    ws.mkdir(parents=True)
    assert hostfs.upward_agents_md(ws) == []
    (tmp_path / "a" / "AGENTS.md").write_text("x")
    assert hostfs.upward_agents_md(ws) == [str((tmp_path / "a" / "AGENTS.md").resolve())]


def test_launcher_only_runs_in_a_worker_job_dir(tmp_path: Path) -> None:
    job = tmp_path / "data" / "worker" / "jobs" / "J1"
    assert data_root(job) == tmp_path / "data"
    with pytest.raises(ConfigError):
        data_root(tmp_path / "elsewhere")


def test_skill_archive_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "s" / "ref").mkdir(parents=True)
    (tmp_path / "s" / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n")
    (tmp_path / "s" / "ref" / "a.md").write_text("a")
    packed = pack_dir(tmp_path / "s")
    assert packed == pack_dir(tmp_path / "s")
    assert unpack(packed) == {
        "SKILL.md": (tmp_path / "s" / "SKILL.md").read_bytes(),
        "ref/a.md": b"a",
    }


def test_prefix_is_byte_identical_and_task_md_is_bounded() -> None:
    tpl = (Path(__file__).resolve().parents[2] / "src/kta/agents/prompts/prefix.md").read_text()
    a = prefix(tpl, {"use-kta": "x", "write-family": "y"})
    assert a == prefix(tpl, {"write-family": "y", "use-kta": "x"}) and "$" not in a
    step = (Path(__file__).resolve().parents[2] / "src/kta/agents/prompts/step-T1.md").read_text()
    card = TaskCard(
        type="T1",
        goal="g",
        done_checks=["validate_family"],
        budget={"wall_s": 60},
        profile="p",
        work_unit="wu",
    )
    md = task_md(
        step,
        task_id="t",
        card_id="c",
        card=card,
        cards=[],
        skills=["use-kta"],
        commands=["show"],
        used={},
        retry=RetryNote([("validate_family", "missing marker")]),
        policy={},
    )
    assert "missing marker" in md and "`kta show`" in md and "$" not in md
    with pytest.raises(ConfigError, match="step_header_max_chars"):
        task_md(
            step,
            task_id="t",
            card_id="c",
            card=card,
            cards=[],
            skills=[],
            commands=[],
            used={},
            retry=None,
            policy={"step_header_max_chars": 100},
        )


def test_arm_of_task_info() -> None:
    from kta.agents.arms import NO_ARM, arm_of

    assert arm_of({}) == NO_ARM
    wp6 = {
        "arm": {"name": "t", "condition": {"knowledge": "text"}, "snapshot": "s1", "protocol": "p1"}
    }
    a = arm_of(wp6)
    assert (a.name, a.knowledge, a.snapshot, a.protocol) == ("t", "text", "s1", "p1")
    assert a.to_dict()["condition"] == {"knowledge": "text"}
    other = arm_of({"arm": {"name": "weak-model", "condition": {"model": "m"}, "snapshot": None}})
    assert other.knowledge == "pool" and other.condition == {"model": "m"}
    z = arm_of({"arm": {"name": "z", "knowledge": "from_zero"}, "snapshot": "s9"})
    assert (z.knowledge, z.snapshot) == ("zero", "s9")
    b = arm_of({"arm": "t", "arm_condition": {"knowledge": "text_knowledge", "snapshot": "s2"}})
    assert (b.name, b.knowledge, b.snapshot) == ("t", "text", "s2")
    for bad in (
        {"arm": {"condition": {"knowledge": "books"}}},
        {"arm": {"condition": {"knowledge": "text"}}},
        {"arm": {"condition": "pool"}},
        {"arm": 3},
    ):
        with pytest.raises(ConfigError):
            arm_of(bad)
