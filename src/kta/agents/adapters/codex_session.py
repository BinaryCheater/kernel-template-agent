"""Codex session files: locating them and parsing them into AgentEvents.

Codex persists every session as ``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<id>.jsonl``.
Each line is ``{"timestamp", "type", "payload"}``; the entry types used here were read
from Codex 0.117 (spark) and 0.163 (control host) session files (decision D303). Entries
this parser does not describe become ``other`` events carrying their entry name, so a new
Codex version shows up as a growing ``other`` count instead of silently losing events.

This module runs both on the control plane (parsing archives) and inside the launcher on
the agent host (finding the session a run wrote); it imports only the standard library
and ``kta.agents.base``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kta.agents.base import AgentEvent

SESSION_GLOB = "sessions/**/rollout-*.jsonl"
TEXT_MAX_CHARS = 4000  # parsed events are summaries; the session file keeps the full text
CALL_TYPES = ("function_call", "custom_tool_call", "local_shell_call", "web_search_call")
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output", "local_shell_call_output")


@dataclass(frozen=True)
class SessionUsage:
    tokens_in: int | None
    tokens_out: int | None
    cached_in: int | None
    context_tokens: int | None  # input tokens of the latest model call
    context_window: int | None

    @property
    def context_ratio(self) -> float | None:
        if not self.context_tokens or not self.context_window:
            return None
        return self.context_tokens / self.context_window


def _ts(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _clip(text: Any) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return s if len(s) <= TEXT_MAX_CHARS else s[:TEXT_MAX_CHARS] + f"…[{len(s)} chars]"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        str(c.get("text", ""))
        for c in content  # pyright: ignore[reportUnknownVariableType]
        if isinstance(c, dict)
    ]
    return "\n".join(parts)


def read_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Session entries from JSON lines; undecodable lines become ``{"type": "undecodable"}``."""
    out: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            out.append({"type": "undecodable", "payload": {"line": line[:200]}})
            continue
        out.append(value if isinstance(value, dict) else {"type": "undecodable"})
    return out


def read_entries(path: Path) -> list[dict[str, Any]]:
    return read_lines(path.read_text(encoding="utf-8", errors="replace").splitlines())


def session_id_of(path: Path) -> str | None:
    for entry in read_entries(path)[:5]:
        if entry.get("type") == "session_meta":
            payload = entry.get("payload") or {}
            sid = payload.get("id") or payload.get("session_id")
            return str(sid) if sid else None
    return None


def session_files(home: Path) -> list[Path]:
    return sorted(home.glob(SESSION_GLOB))


def files_of_session(home: Path, session: str) -> list[Path]:
    """Every rollout file of one session (resume may append or start a new file)."""
    return [p for p in session_files(home) if session in p.name or session_id_of(p) == session]


def session_from_stdout(lines: Iterable[str]) -> str | None:
    """The thread id from ``codex exec --json`` output (``thread.started``)."""
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            tid = event.get("thread_id")  # pyright: ignore[reportUnknownMemberType]
            return str(tid) if tid else None  # pyright: ignore[reportUnknownArgumentType]
    return None


def _response_item(p: dict[str, Any], at: float) -> AgentEvent:
    kind = p.get("type")
    if kind == "message":
        return AgentEvent(
            "text", at, {"role": p.get("role"), "text": _clip(_content_text(p.get("content")))}
        )
    if kind in CALL_TYPES:
        args = p.get("arguments", p.get("input", p.get("action")))
        return AgentEvent(
            "tool_call",
            at,
            {"name": p.get("name", kind), "call_id": p.get("call_id"), "input": _clip(args)},
        )
    if kind in OUTPUT_TYPES:
        out = p.get("output")
        text = _content_text(out) if isinstance(out, list) else out
        return AgentEvent("tool_result", at, {"call_id": p.get("call_id"), "output": _clip(text)})
    if kind == "reasoning":
        # Only the presence of a reasoning item is kept: hidden reasoning is not required
        # or stored as an observable (docs/design/10 §8).
        return AgentEvent("other", at, {"entry": "response_item/reasoning"})
    return AgentEvent("other", at, {"entry": f"response_item/{kind}"})


def _usage(info: dict[str, Any] | None, at: float) -> AgentEvent:
    info = info or {}
    total = info.get("total_token_usage") or {}
    last = info.get("last_token_usage") or {}
    return AgentEvent(
        "usage",
        at,
        {
            "input_tokens": total.get("input_tokens"),
            "cached_input_tokens": total.get("cached_input_tokens"),
            "output_tokens": total.get("output_tokens"),
            "reasoning_output_tokens": total.get("reasoning_output_tokens"),
            "last_input_tokens": last.get("input_tokens"),
            "context_window": info.get("model_context_window"),
        },
    )


def _exec_detail(p: dict[str, Any]) -> dict[str, Any]:
    duration = p.get("duration") or {}
    secs = float(duration.get("secs", 0)) + float(duration.get("nanos", 0)) / 1e9
    command = p.get("command")
    return {
        "command": command if isinstance(command, list) else [str(command)],
        "cwd": p.get("cwd"),
        "exit_code": p.get("exit_code"),
        "duration_s": secs,
        "output_tail": _clip(str(p.get("aggregated_output") or p.get("stderr") or "")[-2000:]),
    }


def _event_msg(p: dict[str, Any], at: float) -> AgentEvent | tuple[str, dict[str, Any]]:
    kind = p.get("type")
    if kind == "token_count":
        return _usage(p.get("info"), at)
    if kind in ("error", "stream_error"):
        return AgentEvent("error", at, {"message": _clip(p.get("message", ""))})
    if kind == "exec_command_end":
        return (str(p.get("call_id")), {"exec": _exec_detail(p)})
    if kind == "patch_apply_end":
        changes = p.get("changes") or {}
        return (
            str(p.get("call_id")),
            {"patch": {"paths": sorted(changes), "success": p.get("success")}},
        )
    return AgentEvent("other", at, {"entry": f"event_msg/{kind}"})


def parse_entries(entries: list[dict[str, Any]]) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    details: dict[str, dict[str, Any]] = {}
    for entry in entries:
        at = _ts(entry.get("timestamp"))
        kind = entry.get("type")
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        assert isinstance(payload, dict)
        if kind == "response_item":
            events.append(_response_item(payload, at))
        elif kind == "event_msg":
            got = _event_msg(payload, at)
            if isinstance(got, AgentEvent):
                events.append(got)
            else:
                details.setdefault(got[0], {"_at": at}).update(got[1])
        elif kind == "session_meta":
            events.append(
                AgentEvent(
                    "other",
                    at,
                    {
                        "entry": "session_meta",
                        "session": payload.get("id"),
                        "cli_version": payload.get("cli_version"),
                        "cwd": payload.get("cwd"),
                    },
                )
            )
        elif kind == "turn_context":
            events.append(
                AgentEvent(
                    "other",
                    at,
                    {
                        "entry": "turn_context",
                        "model": payload.get("model"),
                        "cwd": payload.get("cwd"),
                        "sandbox": (payload.get("sandbox_policy") or {}).get("type"),
                        "approval": payload.get("approval_policy"),
                    },
                )
            )
        else:
            events.append(AgentEvent("other", at, {"entry": str(kind)}))
    return _attach_details(events, details)


def _attach_details(
    events: list[AgentEvent], details: dict[str, dict[str, Any]]
) -> list[AgentEvent]:
    """Merge exec/patch results into the tool_result of the same call (one event per call)."""
    out: list[AgentEvent] = []
    used: set[str] = set()
    for ev in events:
        cid = ev.data.get("call_id")
        if ev.kind == "tool_result" and isinstance(cid, str) and cid in details:
            extra = {k: v for k, v in details[cid].items() if k != "_at"}
            out.append(AgentEvent(ev.kind, ev.at, {**ev.data, **extra}))
            used.add(cid)
        else:
            out.append(ev)
    # Exec/patch results without a matching output item still count as tool results.
    for cid, d in details.items():
        if cid not in used:
            extra = {k: v for k, v in d.items() if k != "_at"}
            out.append(AgentEvent("tool_result", float(d["_at"]), {"call_id": cid, **extra}))
    return out


def parse_session_files(paths: list[Path]) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for path in paths:
        events += parse_entries(read_entries(path))
    return sorted(events, key=lambda e: e.at)


def usage_of(events: list[AgentEvent]) -> SessionUsage:
    usage = [e for e in events if e.kind == "usage"]
    if not usage:
        return SessionUsage(None, None, None, None, None)
    last = usage[-1].data
    return SessionUsage(
        last.get("input_tokens"),
        last.get("output_tokens"),
        last.get("cached_input_tokens"),
        last.get("last_input_tokens"),
        last.get("context_window"),
    )
