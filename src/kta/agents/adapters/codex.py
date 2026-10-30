"""Codex adapter: ``codex exec`` / ``codex exec resume`` with an isolated CODEX_HOME.

Parameters checked against Codex 0.117 (spark) help text (decision D303):

- ``codex exec [flags] -`` reads the prompt from stdin; ``codex exec resume [flags] <id> -``
  continues a persisted session. ``resume`` has no ``-C``/``-s``, so the working root is
  the process cwd (set by the launcher) and the sandbox is a ``-c`` override, which both
  subcommands accept.
- ``--json`` prints events as JSON lines; ``thread.started`` carries the session id.
- ``CODEX_HOME`` (set by the launcher to the work unit's agent-home) replaces ``~/.codex``:
  config, AGENTS.md, skills, credentials and the session files all come from there.
- Session persistence stays on: the non-persistent option is refused (conventions §7.1).
"""

from __future__ import annotations

from pathlib import Path

from kta.agents.adapters import codex_session
from kta.agents.adapters.base import CliAdapterBase, StepPlan, archive_members
from kta.agents.adapters.capabilities import NON_PERSISTENT_OPTIONS
from kta.agents.base import AgentEvent
from kta.core.errors import ConfigError
from kta.core.jobs import JobRunner

CREDENTIAL_FILES = ("auth.json",)  # linked from the protected directory, never archived


def check_flags(flags: list[str]) -> None:
    bad = [f for f in flags if f in NON_PERSISTENT_OPTIONS]
    if bad:
        raise ConfigError(f"codex flags {bad} disable session persistence (conventions §7.1)")


class CodexAdapter(CliAdapterBase):
    name = "codex"
    supports_resume = True

    def __init__(
        self,
        runner: JobRunner,
        plan: StepPlan,
        cli: list[str],
        flags: list[str],
        model: str,
        effort: str | None,
    ) -> None:
        super().__init__(runner, plan)
        check_flags(flags)
        if not cli:
            raise ConfigError("codex adapter needs the CLI command ([agents] codex_cmd)")
        self.cli = list(cli)
        self.flags = list(flags)
        self.model = model
        self.effort = effort

    def _common(self) -> list[str]:
        out = [*self.flags, "-m", self.model]
        if self.effort:
            out += ["-c", f'model_reasoning_effort="{self.effort}"']
        return out

    def start_argv(self) -> list[str]:
        return [*self.cli, "exec", *self._common(), "-"]

    def resume_argv(self, session: str) -> list[str]:
        if not session:
            raise ConfigError("codex resume needs a session id")
        return [*self.cli, "exec", "resume", *self._common(), session, "-"]

    def parse_bytes(self, files: dict[str, bytes]) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        for name in sorted(files):
            lines = files[name].decode("utf-8", "replace").splitlines()
            entries = codex_session.read_lines(lines)
            events += codex_session.parse_entries(entries)
        return sorted(events, key=lambda e: e.at)

    def parse_session(self, path: Path) -> list[AgentEvent]:
        """Events of a session file, a CODEX_HOME directory or a packed archive."""
        return parse_session(path)


def parse_session(path: Path) -> list[AgentEvent]:
    if path.is_dir():
        return codex_session.parse_session_files(codex_session.session_files(path))
    if path.name.endswith(".tar.gz"):
        files = archive_members(path.read_bytes())
        events: list[AgentEvent] = []
        for name in sorted(files):
            events += codex_session.parse_entries(
                codex_session.read_lines(files[name].decode("utf-8", "replace").splitlines())
            )
        return sorted(events, key=lambda e: e.at)
    return codex_session.parse_session_files([path])
