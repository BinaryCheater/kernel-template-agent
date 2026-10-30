"""Shared part of CLI adapters: one agent step = one launcher job on the agent host (D300).

An adapter only decides the CLI argv and how to read its session files; submitting the
job, collecting the outputs by hash and turning them into a :class:`Launch` is the same
for every CLI and lives here.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from kta.agents.base import AgentEvent, Launch, Limits
from kta.agents.launchspec import (
    OUT_ARCHIVE,
    OUT_MANIFEST,
    OUT_REPORT,
    OUT_STDOUT,
    OUT_WORKSPACE,
    PROMPT_FILE,
    SPEC_FILE,
    LaunchSpec,
)
from kta.core.errors import ExecutionError, NotSupported
from kta.core.jobs import Job, JobResult, JobRunner

LAUNCHER = ["{python}", "-m", "kta.agents.launch", "{job}"]
EXIT_MARGIN_S = 300  # the launcher's own epilogue (git commit, packing) after the CLI ends


@dataclass(frozen=True)
class StepPlan:
    """What the runner fixed for this step; the adapter adds argv and session handling."""

    spec: LaunchSpec  # mode/argv/session filled in by the adapter
    inputs: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # job name -> blob
    requires: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    hard_cap_s: int = 0  # Executor timeout: 3x the wall budget (docs/design/11 §3.3)


def launch_job(runner: JobRunner, plan: StepPlan, spec: LaunchSpec, prompt: str) -> JobResult:
    spec_blob = runner.put_blob(spec.to_json(), "application/json")
    prompt_blob = runner.put_blob(prompt.encode("utf-8"), "text/markdown")
    job = Job(
        kind="shell",
        argv=list(LAUNCHER),
        requires=dict(plan.requires),
        inputs={**plan.inputs, SPEC_FILE: spec_blob, PROMPT_FILE: prompt_blob},
        timeout_s=max(plan.hard_cap_s, spec.wall_s + EXIT_MARGIN_S),
    )
    return runner.run(job)


def collect(runner: JobRunner, result: JobResult, fallback_session: str | None) -> Launch:
    """Turn the launcher job's outputs into a Launch; a launcher crash is an infra error."""
    if not result.ok or OUT_REPORT not in result.outputs:
        tail = runner.read_blob(result.stderr_blob)[-2000:].decode("utf-8", "replace")
        raise ExecutionError(
            "infra" if result.error_kind == "none" else result.error_kind,
            f"agent launcher job {result.job_id} failed ({result.error_kind}): {tail}",
        )
    report: dict[str, Any] = json.loads(runner.read_blob(result.outputs[OUT_REPORT]))
    missing = [
        n for n in (OUT_ARCHIVE, OUT_MANIFEST, OUT_STDOUT, OUT_WORKSPACE) if n not in result.outputs
    ]
    if missing and report.get("exit") not in ("refused", "start_failed"):
        raise ExecutionError("infra", f"agent launcher job {result.job_id} lacks {missing}")
    return Launch(
        session=str(report.get("session") or fallback_session or ""),
        exit=str(report.get("exit")),
        exit_code=report.get("exit_code"),
        wall_s=float(report.get("wall_s", 0.0)),
        queue_s=result.queue_s,
        archive=result.outputs.get(OUT_ARCHIVE, ""),
        manifest=result.outputs.get(OUT_MANIFEST, ""),
        stdout=result.outputs.get(OUT_STDOUT, ""),
        workspace=result.outputs.get(OUT_WORKSPACE, ""),
        report=report,
        env_manifest=result.env_manifest,
        worker=result.worker,
    )


def archive_members(
    data: bytes, suffix: str = ".jsonl", under: str = "sessions/"
) -> dict[str, bytes]:
    """Files of a tar.gz archive below ``under`` with ``suffix``, by member name."""
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith(under) and member.name.endswith(suffix):
                handle = tar.extractfile(member)
                if handle is not None:
                    out[member.name] = handle.read()
    return out


class CliAdapterBase:
    """start/resume through the launcher; subclasses provide argv and session parsing."""

    name = "cli"
    supports_resume = False

    def __init__(self, runner: JobRunner, plan: StepPlan) -> None:
        self.runner = runner
        self.plan = plan
        self._last: Launch | None = None

    # ---- subclass hooks ---------------------------------------------------------
    def start_argv(self) -> list[str]:
        raise NotSupported(f"{self.name}: start")

    def resume_argv(self, session: str) -> list[str]:
        raise NotSupported(f"{self.name}: resume")

    def parse_bytes(self, files: dict[str, bytes]) -> list[AgentEvent]:
        raise NotSupported(f"{self.name}: session parsing")

    def parse_session(self, path: Path) -> list[AgentEvent]:
        raise NotSupported(f"{self.name}: session parsing")

    # ---- interface ---------------------------------------------------------------
    def last_launch(self) -> Launch | None:
        return self._last

    def _run(self, spec: LaunchSpec, prompt: str, limits: Limits, env: dict[str, str]) -> Launch:
        spec = replace(spec, wall_s=limits.wall_s, idle_s=limits.idle_s, env={**spec.env, **env})
        result = launch_job(self.runner, self.plan, spec, prompt)
        self._last = collect(self.runner, result, spec.session)
        return self._last

    def events_of(self, launch: Launch) -> list[AgentEvent]:
        if not launch.archive:
            return []
        files = archive_members(self.runner.read_blob(launch.archive))
        if launch.session:
            files = {n: b for n, b in files.items() if launch.session in n} or files
        return self.parse_bytes(files)

    def start(
        self, prompt: str, workspace: Path, env: dict[str, str], limits: Limits
    ) -> tuple[str, Iterator[AgentEvent]]:
        self._check_workspace(workspace)
        spec = replace(self.plan.spec, mode="start", argv=self.start_argv(), session=None)
        launch = self._run(spec, prompt, limits, env)
        return launch.session, iter(self.events_of(launch))

    def resume(
        self, session: str, message: str, workspace: Path, env: dict[str, str], limits: Limits
    ) -> Iterator[AgentEvent]:
        if not self.supports_resume:
            raise NotSupported(f"{self.name} cannot resume sessions")
        self._check_workspace(workspace)
        spec = replace(
            self.plan.spec, mode="resume", argv=self.resume_argv(session), session=session
        )
        launch = self._run(spec, message, limits, env)
        return iter(self.events_of(launch))

    def _check_workspace(self, workspace: Path) -> None:
        # The workspace lives on the agent host; the runner names it relative to that
        # host's data directory, and the launcher refuses anything else.
        expected = Path("work") / self.plan.spec.work_unit
        if workspace != expected:
            raise NotSupported(f"workspace must be {expected} on the agent host, got {workspace}")
