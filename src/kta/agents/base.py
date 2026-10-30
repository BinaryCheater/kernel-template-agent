"""Agent adapter interface (spec/modules/agents.md "接口").

An adapter turns one agent step into a CLI invocation and reads back what the CLI
recorded. The spec's ``start``/``resume`` return the session handle and the events; here
both also return the :class:`Launch` that ran them, because the CLI does not run in this
process: it runs as an Executor job on the agent host (decision D300), and the runner
needs the job's archive, manifest and exit reason to write the AgentRun.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

EventKind = Literal["text", "tool_call", "tool_result", "usage", "error", "exit", "other"]
EVENT_KINDS: tuple[EventKind, ...] = (
    "text",
    "tool_call",
    "tool_result",
    "usage",
    "error",
    "exit",
    "other",
)


@dataclass(frozen=True)
class AgentEvent:
    """One event of a session; ``at`` is seconds since the epoch (UTC).

    ``other`` is not in the spec's list: it counts session entries the parser does not
    describe, so coverage statistics see them instead of losing them (AGENTS.md
    principle 1).
    """

    kind: EventKind
    at: float
    data: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


@dataclass(frozen=True)
class Limits:
    wall_s: int
    idle_s: int = 600
    tokens: int | None = None  # only enforced when the adapter sees usage live


@dataclass(frozen=True)
class Launch:
    """What one CLI invocation on the agent host produced (blob IDs are store sha256)."""

    session: str
    exit: str  # ok | nonzero | wall | idle | gpu_guard | start_failed | infra
    exit_code: int | None
    wall_s: float
    queue_s: float
    archive: str  # blob: agent-home session files + CLI output, no credentials
    manifest: str  # blob: context manifest (JSON)
    stdout: str  # blob: the CLI's stdout (JSONL events)
    workspace: str  # blob: tar of the workspace after the step (out/, NOTES.md, ...)
    report: dict[str, Any]  # launcher report: watch results, git commit, guard findings
    env_manifest: str  # EnvManifest record of the agent host
    worker: str  # Worker record of the agent host


class AgentAdapter(Protocol):
    name: str
    supports_resume: bool

    def start(
        self, prompt: str, workspace: Path, env: dict[str, str], limits: Limits
    ) -> tuple[str, Iterator[AgentEvent]]: ...

    def resume(
        self, session: str, message: str, workspace: Path, env: dict[str, str], limits: Limits
    ) -> Iterator[AgentEvent]: ...

    def parse_session(self, path: Path) -> list[AgentEvent]: ...

    def last_launch(self) -> Launch | None:
        """The Launch behind the most recent start/resume, for the AgentRun."""
        ...
