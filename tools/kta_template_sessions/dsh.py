"""Experiment-local DSH adapter for the installed headless persistence interface."""

from __future__ import annotations

import json
from pathlib import Path

from kta.agents.base import AgentEvent, Launch
from kta.core.errors import NotSupported

from .adapter import RecordedCodexAdapter


class RecordedDshAdapter(RecordedCodexAdapter):
    name = "deepseek_harness"

    def start_argv(self) -> list[str]:
        return [
            *self.cli,
            "--profile",
            "headless",
            "--patch",
            str(self.state / "dsh.patch.json"),
            "--json",
            "-",
        ]

    def resume_argv(self, session: str) -> list[str]:
        return [*self.start_argv()[:-1], "--session-id", session, "-"]

    def events_of(self, launch: Launch) -> list[AgentEvent]:
        return parse_stdout(self.runner.read_blob(launch.stdout))

    def parse_session(self, path: Path) -> list[AgentEvent]:
        raise NotSupported(
            "DSH native archive is retained losslessly; use stdout events for accounting"
        )


def parse_stdout(raw: bytes) -> list[AgentEvent]:
    """Headless usage is per step, including failed attempts, rather than cumulative."""
    events: list[AgentEvent] = []
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    uncached_input = 0
    cache_write = 0
    input_complete = True
    seen = False
    cache_complete = True
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("{"):
            events.append(AgentEvent("other", 0.0, {"diagnostic": line}))
            continue
        row = json.loads(line)
        kind = row.get("type", "other")
        if row.get("phase") == "step_end" and "usage" in row:
            usage = row["usage"]
            uncached_input += usage["inputTokens"]
            cache_write += usage.get("cacheWriteTokens", 0)
            if "totalTokens" in usage:
                totals["input_tokens"] += usage["totalTokens"] - usage["outputTokens"]
            elif "cacheReadTokens" in usage and "cacheWriteTokens" in usage:
                totals["input_tokens"] += (
                    usage["inputTokens"] + usage["cacheReadTokens"] + usage["cacheWriteTokens"]
                )
            else:
                input_complete = False
            totals["output_tokens"] += usage["outputTokens"]
            totals["cached_input_tokens"] += usage.get("cacheReadTokens", 0)
            cache_complete &= "cacheReadTokens" in usage
            seen = True
        events.append(
            AgentEvent(kind if kind in ("text", "tool_call", "tool_result") else "other", 0.0, row)
        )
    if seen:
        events.append(
            AgentEvent(
                "usage",
                0.0,
                {
                    **totals,
                    "input_tokens": totals["input_tokens"] if input_complete else None,
                    "uncached_input_tokens": uncached_input,
                    "cache_write_input_tokens": cache_write,
                    "cached_input_tokens": totals["cached_input_tokens"]
                    if cache_complete
                    else None,
                },
            )
        )
    return events


def patch(state: Path) -> list[dict]:
    """Verified installed plugin ids; preserve native credential/provider configuration."""
    return [
        {"id": "authorization", "disabled": False},
        {"id": "credentials", "disabled": False},
        {"id": "deepseek-account", "disabled": False, "config": {"desktopPlatform": "darwin"}},
        {
            "id": "agent-default-model",
            "config": {
                "provider": "deepseek-account",
                "model": "deepseek-flash",
                "reasoningEffort": "high",
            },
        },
        {"id": "session-persistence-jsonl", "config": {"root": str(state / "home/sessions")}},
        {"id": "session-title-llm", "disabled": True},
    ]
