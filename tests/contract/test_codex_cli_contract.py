"""Contract between the Codex adapter, the real ``codex`` CLI and tests/fakes/fake_codex.py.

Every option the adapter puts on the command line must be listed by the real CLI's help
for that subcommand and accepted by the fake, and the fake must refuse what the real
adapter refuses. Only ``--help``/``--version`` of the real CLI run here (no model call);
the real CLI is looked up as ``codex`` on PATH or at ``$KTA_CODEX``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kta.agents.adapters.base import StepPlan
from kta.agents.adapters.capabilities import NON_PERSISTENT_OPTIONS
from kta.agents.adapters.codex import CodexAdapter
from kta.agents.launchspec import LaunchSpec
from kta.agents.profile import CODEX_FLAGS

pytestmark = pytest.mark.contract
FAKE = Path(__file__).resolve().parents[1] / "fakes" / "fake_codex.py"
OPTION = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def _adapter() -> CodexAdapter:
    plan = StepPlan(LaunchSpec("codex", "wu", "r", "start"))
    return CodexAdapter(None, plan, ["codex"], list(CODEX_FLAGS), "m", "medium")  # type: ignore[arg-type]


def _options(argv: list[str]) -> set[str]:
    return {a for a in argv if a.startswith("-") and a != "-"}


def _real_cli() -> str | None:
    return os.environ.get("KTA_CODEX") or shutil.which("codex")


def _help(cli: str, *sub: str) -> str:
    proc = subprocess.run([cli, *sub, "--help"], capture_output=True, text=True, timeout=60)
    return proc.stdout + proc.stderr


def test_real_cli_lists_every_adapter_option() -> None:
    cli = _real_cli()
    if cli is None:
        pytest.skip("no real codex CLI on this machine (set KTA_CODEX)")
    a = _adapter()
    for sub, argv in ((("exec",), a.start_argv()), (("exec", "resume"), a.resume_argv("s"))):
        listed = set(OPTION.findall(_help(cli, *sub)))
        missing = _options(argv) - listed
        assert not missing, f"codex {' '.join(sub)} does not list {missing}"


def test_fake_accepts_adapter_argv_and_refuses_non_persistent(tmp_path: Path) -> None:
    a = _adapter()
    env = {**os.environ, "CODEX_HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    (tmp_path / "ws").mkdir()
    start = [sys.executable, str(FAKE), *a.start_argv()[1:]]
    out = subprocess.run(
        start, input="go", capture_output=True, text=True, cwd=tmp_path / "ws", env=env, timeout=60
    )
    assert out.returncode == 0, out.stderr
    sid = next(line for line in out.stdout.splitlines() if "thread.started" in line).split('"')[-2]
    resume = [sys.executable, str(FAKE), *a.resume_argv(sid)[1:]]
    out = subprocess.run(
        resume,
        input="again",
        capture_output=True,
        text=True,
        cwd=tmp_path / "ws",
        env=env,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    bad = [sys.executable, str(FAKE), "exec", NON_PERSISTENT_OPTIONS[0], "-"]
    out = subprocess.run(
        bad, input="x", capture_output=True, text=True, cwd=tmp_path / "ws", env=env, timeout=60
    )
    assert out.returncode == 2


def test_fake_session_has_the_real_rollout_layout(tmp_path: Path) -> None:
    from kta.agents.adapters import codex_session

    env = {**os.environ, "CODEX_HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    (tmp_path / "ws").mkdir()
    subprocess.run(
        [sys.executable, str(FAKE), "exec", "--json", "-"],
        input="go",
        capture_output=True,
        text=True,
        cwd=tmp_path / "ws",
        env=env,
        timeout=60,
        check=True,
    )
    files = codex_session.session_files(tmp_path / "home")
    assert len(files) == 1 and re.match(
        r"rollout-\d{4}-\d\d-\d\dT[\d-]+-[0-9a-f-]{36}\.jsonl", files[0].name
    )
    events = codex_session.parse_session_files(files)
    assert {e.kind for e in events} >= {"text", "tool_call", "tool_result", "usage"}
