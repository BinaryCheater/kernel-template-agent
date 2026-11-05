"""Experiment launch packaging checks without model calls."""

from pathlib import Path

import pytest

from kta.agents.adapters.base import StepPlan
from kta.agents.launchspec import LaunchSpec
from kta.core.errors import ConfigError
from kta_template_sessions.adapter import RecordedCodexAdapter


def adapter(workspace: Path, state: Path) -> RecordedCodexAdapter:
    # No runner call is made: inspect pure adapter argv/workspace validation only.
    return RecordedCodexAdapter(
        None,
        StepPlan(LaunchSpec(provider="codex", work_unit="test", run="test", mode="start")),
        ["codex"],
        ["--json"],
        "gpt-6.1-sol",
        "high",
        workspace=workspace,
        state=state,
    )


def test_state_cannot_enter_implementation_workspace(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="outside"):
        adapter(tmp_path, tmp_path / "state")


def test_native_persistence_and_resume_identity(tmp_path: Path) -> None:
    cli = adapter(tmp_path / "workspace", tmp_path / "state")
    assert cli.start_argv() == [
        "codex",
        "exec",
        "--json",
        "-m",
        "gpt-6.1-sol",
        "-c",
        'model_reasoning_effort="high"',
        "-",
    ]
    assert cli.resume_argv("native-id")[-2:] == ["native-id", "-"]
    assert "--ephemeral" not in cli.start_argv()
    cli._check_workspace(tmp_path / "workspace")
    with pytest.raises(ConfigError, match="frozen"):
        cli._check_workspace(tmp_path / "other")


def test_dsh_native_step_usage_sum_and_resume_identity(tmp_path: Path) -> None:
    import json

    from kta.agents.adapters.codex_session import usage_of
    from kta_template_sessions.dsh import RecordedDshAdapter, parse_stdout, patch

    rows = [
        {
            "type": "status",
            "phase": "step_end",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 2,
                "cacheReadTokens": 6,
                "cacheWriteTokens": 0,
                "totalTokens": 18,
            },
        },
        {
            "type": "status",
            "phase": "step_end",
            "usage": {
                "inputTokens": 15,
                "outputTokens": 3,
                "cacheReadTokens": 9,
                "cacheWriteTokens": 0,
                "totalTokens": 27,
            },
        },
    ]
    usage = usage_of(parse_stdout("\n".join(map(json.dumps, rows)).encode()))
    assert (usage.tokens_in, usage.tokens_out, usage.cached_in) == (40, 5, 15)
    rows[1]["usage"].pop("cacheReadTokens")
    assert usage_of(parse_stdout("\n".join(map(json.dumps, rows)).encode())).cached_in is None
    cli = RecordedDshAdapter(
        None,
        StepPlan(
            LaunchSpec(provider="deepseek_harness", work_unit="test", run="test", mode="start")
        ),
        ["dsh"],
        [],
        "deepseek-flash",
        "high",
        workspace=tmp_path / "ws",
        state=tmp_path / "state",
    )
    assert cli.resume_argv("session-saved")[-3:] == ["--session-id", "session-saved", "-"]
    model = next(e["config"] for e in patch(tmp_path) if e["id"] == "agent-default-model")
    assert model == {
        "provider": "deepseek-account",
        "model": "deepseek-flash",
        "reasoningEffort": "high",
    }
