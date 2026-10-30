"""A stand-in for the ``codex`` CLI (the LLM boundary), for launcher and runner tests.

Run as ``python fake_codex.py exec [flags] -`` or ``... exec resume [flags] <id> -``. It
accepts only the options the real Codex 0.117 lists (tests/contract/
test_codex_cli_contract.py checks both sides), refuses non-persistent sessions, writes a
session file in the real rollout format under ``$CODEX_HOME/sessions/`` and prints
``--json`` events. It does no model work; it performs these scripted actions in the
workspace (its cwd), each recorded as an ``exec_command_end`` like the real CLI:

- appends a line to NOTES.md and writes out/DELIVERY.md;
- writes out/env.json with the environment variable names it received;
- copies ``$CODEX_HOME/AGENTS.md`` (if any) into the session, as Codex does with the
  user-level instructions, so isolation tests can look for a marker;
- for each prompt line ``INDUCE-WRITE <path>``, tries to write that path (the
  out-of-workspace write that compliance accounting must catch).
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

VERSION = "codex-cli 0.117.0"
EXEC_OPTIONS = {"-c", "--config", "-m", "--model", "--json", "--skip-git-repo-check", "-o"}
VALUE_OPTIONS = {"-c", "--config", "-m", "--model", "-o"}
REFUSED = {"--ephemeral"}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse(args: list[str]) -> tuple[bool, str | None, list[str]]:
    if not args or args[0] != "exec":
        raise SystemExit("fake codex: only `exec` is supported")
    rest = args[1:]
    resume = bool(rest) and rest[0] == "resume"
    rest = rest[1:] if resume else rest
    positional: list[str] = []
    i = 0
    while i < len(rest):
        a = rest[i]
        if a in REFUSED:
            print(f"fake codex: {a} is refused", file=sys.stderr)
            raise SystemExit(2)
        if a.startswith("-") and a != "-":
            if a not in EXEC_OPTIONS:
                print(f"fake codex: unknown option {a}", file=sys.stderr)
                raise SystemExit(2)
            i += 2 if a in VALUE_OPTIONS else 1
            continue
        positional.append(a)
        i += 1
    session = positional[0] if resume and len(positional) == 2 else None
    if resume and session is None:
        raise SystemExit("fake codex: resume needs <id> -")
    return resume, session, positional


def _rollout(home: Path, session: str | None) -> tuple[Path, str]:
    if session is not None:
        for p in sorted(home.glob("sessions/**/rollout-*.jsonl")):
            if session in p.name:
                return p, session
        raise SystemExit(f"fake codex: no session {session}")
    sid = str(uuid.uuid4())
    day = datetime.now(UTC)
    path = (
        home
        / "sessions"
        / day.strftime("%Y/%m/%d")
        / f"rollout-{day:%Y-%m-%dT%H-%M-%S}-{sid}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, sid


def _entry(kind: str, payload: dict[str, object]) -> str:
    return json.dumps({"timestamp": _now(), "type": kind, "payload": payload}) + "\n"


def _exec(lines: list[str], n: int, command: str, code: int, output: str) -> None:
    call = f"call_{n}"
    argv = ["bash", "-lc", command]
    lines.append(
        _entry(
            "response_item",
            {
                "type": "function_call",
                "name": "shell",
                "arguments": json.dumps({"command": argv}),
                "call_id": call,
            },
        )
    )
    lines.append(
        _entry(
            "event_msg",
            {
                "type": "exec_command_end",
                "call_id": call,
                "command": argv,
                "cwd": os.getcwd(),
                "exit_code": code,
                "aggregated_output": output,
                "duration": {"secs": 0, "nanos": 1000},
            },
        )
    )
    lines.append(
        _entry("response_item", {"type": "function_call_output", "call_id": call, "output": output})
    )


def _actions(prompt: str, lines: list[str]) -> None:
    ws = Path.cwd()
    with (ws / "NOTES.md").open("a", encoding="utf-8") as f:
        f.write(f"\n- step at {_now()}\n")
    _exec(lines, 1, "echo step >> NOTES.md", 0, "")
    (ws / "out").mkdir(exist_ok=True)
    (ws / "out" / "DELIVERY.md").write_text(
        "## Done\nfake\n## Simplifications and fixtures\nnone\n## Known issues\nnone\n"
    )
    (ws / "out" / "env.json").write_text(json.dumps(sorted(os.environ)))
    _exec(lines, 2, "kta validate-family out/family", 0, "ok")
    for n, line in enumerate(prompt.splitlines(), start=10):
        if line.startswith("INDUCE-WRITE "):
            target = line.split(" ", 1)[1].strip()
            try:
                Path(target).write_text("induced\n")
                code, out = 0, ""
            except OSError as exc:
                code, out = 1, f"bash: {target}: Permission denied ({exc})"
            _exec(lines, n, f"echo induced > {target}", code, out)


def main(argv: list[str]) -> int:
    if argv == ["--version"]:
        print(VERSION)
        return 0
    resume, session, _ = _parse(argv)
    home = Path(os.environ["CODEX_HOME"])
    prompt = sys.stdin.read()
    path, sid = _rollout(home, session)
    lines: list[str] = []
    if not resume:
        lines.append(
            _entry(
                "session_meta",
                {
                    "id": sid,
                    "timestamp": _now(),
                    "cwd": os.getcwd(),
                    "cli_version": "0.117.0",
                    "originator": "codex_exec",
                },
            )
        )
    lines.append(
        _entry(
            "turn_context",
            {
                "cwd": os.getcwd(),
                "model": "fake",
                "approval_policy": "never",
                "sandbox_policy": {"type": "workspace-write"},
            },
        )
    )
    user_md = home / "AGENTS.md"
    if user_md.exists():
        lines.append(
            _entry(
                "response_item",
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_md.read_text()}],
                },
            )
        )
    lines.append(
        _entry(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        )
    )
    _actions(prompt, lines)
    lines.append(
        _entry(
            "event_msg",
            {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 1200,
                        "cached_input_tokens": 800,
                        "output_tokens": 300,
                    },
                    "last_token_usage": {"input_tokens": 1000},
                    "model_context_window": 10000,
                },
            },
        )
    )
    lines.append(
        _entry(
            "response_item",
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        )
    )
    with path.open("a", encoding="utf-8") as f:
        f.writelines(lines)
    print(json.dumps({"type": "thread.started", "thread_id": sid}))
    print(
        json.dumps(
            {"type": "turn.completed", "usage": {"input_tokens": 1200, "output_tokens": 300}}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
