"""Claude Code (D65: interface only in the first phase)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from kta.agents.base import AgentEvent, Launch, Limits
from kta.core.errors import NotSupported

REASON = "claude_code adapter is not implemented in the first phase (spec/workplan.md WP8)"


class ClaudeCodeAdapter:
    name = "claude_code"
    supports_resume = False

    def start(
        self, prompt: str, workspace: Path, env: dict[str, str], limits: Limits
    ) -> tuple[str, Iterator[AgentEvent]]:
        raise NotSupported(REASON)

    def resume(
        self, session: str, message: str, workspace: Path, env: dict[str, str], limits: Limits
    ) -> Iterator[AgentEvent]:
        raise NotSupported(REASON)

    def parse_session(self, path: Path) -> list[AgentEvent]:
        raise NotSupported(REASON)

    def last_launch(self) -> Launch | None:
        raise NotSupported(REASON)
