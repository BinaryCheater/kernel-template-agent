"""The existing agents adapter with an experiment-only Executor launcher."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from kta.agents.adapters.base import EXIT_MARGIN_S, collect
from kta.agents.adapters.codex import CodexAdapter
from kta.agents.base import Launch, Limits
from kta.agents.launchspec import PROMPT_FILE, SPEC_FILE, LaunchSpec
from kta.core.errors import ConfigError
from kta.core.jobs import Job


class RecordedCodexAdapter(CodexAdapter):
    """Keep native argv/parser; change only workspace installation for this experiment."""

    def __init__(self, *args, workspace: Path, state: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.workspace = workspace.resolve()
        self.state = state.resolve()
        if self.workspace == self.state or self.workspace in self.state.parents:
            raise ConfigError("session state must be outside the implementation workspace")

    def _check_workspace(self, workspace: Path) -> None:
        if workspace.resolve() != self.workspace:
            raise ConfigError("workspace differs from the frozen experiment package")

    def _run(self, spec: LaunchSpec, prompt: str, limits: Limits, env: dict[str, str]) -> Launch:
        spec = replace(spec, wall_s=limits.wall_s, idle_s=limits.idle_s, env={**spec.env, **env})
        payload = {
            "spec": json.loads(spec.to_json()),
            "workspace": str(self.workspace),
            "state": str(self.state),
        }
        inputs = {
            SPEC_FILE: self.runner.put_blob(json.dumps(payload).encode(), "application/json"),
            PROMPT_FILE: self.runner.put_blob(prompt.encode(), "text/markdown"),
            "session-launch.py": self.runner.put_blob(
                Path(__file__).with_name("launch.py").read_bytes(), "text/x-python"
            ),
        }
        inputs["export-dsh.cjs"] = self.runner.put_blob(
            Path(__file__).with_name("export-dsh.cjs").read_bytes(), "text/javascript"
        )
        result = self.runner.run(
            Job(
                kind="shell",
                argv=["{python}", "{job}/session-launch.py", "{job}"],
                requires=dict(self.plan.requires),
                inputs=inputs,
                timeout_s=limits.wall_s + EXIT_MARGIN_S,
            )
        )
        for name in ("stderr.txt", "stdout.jsonl", "report.json"):
            if name in result.outputs:
                target = self.state / spec.run
                target.mkdir(parents=True, exist_ok=True)
                (target / name).write_bytes(self.runner.read_blob(result.outputs[name]))
        self._last = collect(self.runner, result, spec.session)
        return self._last
