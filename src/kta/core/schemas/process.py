"""Process and audit records (spec/schemas.md §4.9), plus the kinds that docs/design/20
§2.3 lists without fields in spec/schemas.md: Source, Protocol, ExperimentRun, Report
(decision D143)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from kta.core.schemas.common import Assurance, Body, EvidenceStatus
from kta.core.schemas.plan import TaskHeader


class StatusChange(Body):
    subject: str
    to_status: EvidenceStatus | None = None
    to_assurance: Assurance | None = None
    reason: str

    @model_validator(mode="after")
    def _changes_something(self) -> StatusChange:
        if self.to_status is None and self.to_assurance is None:
            raise ValueError("a StatusChange sets to_status, to_assurance, or both")
        return self


class TaskCard(Body):
    type: Literal["T1", "T2", "T3", "T4", "T5", "T6"]
    goal: str
    inputs: list[str] = Field(default_factory=lambda: list[str]())
    done_checks: list[str] = Field(default_factory=lambda: list[str]())
    budget: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    profile: str  # AgentProfile record ID
    work_unit: str
    header: TaskHeader | None = None  # case-loop tasks: the one-screen header (WP19)


class AgentRun(Body):
    session: str
    profile: str
    cli_version: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    session_archive: str  # blob
    wall_s: float
    tool_s: float
    queue_s: float
    exit: str
    context_manifest: str  # blob
    violations: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())
    # experiment arm: name, knowledge condition (zero | text | pool), snapshot (D430)
    arm: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())


class AgentProfile(Body):
    name: str
    version: int
    provider: Literal["codex", "claude_code", "pi", "deepseek_harness"]
    cli_version: str
    auth: Literal["subscription", "api"]
    api_base: str | None = None
    api_key_name: str | None = None  # the key's name in secrets.env, never the key
    isolation: Literal["config", "config+user", "container"]
    model: str
    reasoning_effort: str | None = None
    cli_flags: list[str] = Field(default_factory=lambda: list[str]())
    home_files: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    prefix_template: str
    step_templates: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    message_templates: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    skills: dict[str, list[str]] = Field(default_factory=lambda: dict[str, list[str]]())
    # "<skill>@<sha256>" -> blob of the deterministic skill tar (D322)
    skill_bundles: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    allowed_commands: dict[str, list[str]] = Field(default_factory=lambda: dict[str, list[str]]())
    context_policy: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    loop_policy: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())


class Submission(Body):
    files: list[str] = Field(default_factory=lambda: list[str]())
    note: str


class HumanAction(Body):
    action: str
    reason: str


class PoolSnapshot(Body):
    members: list[str] = Field(default_factory=lambda: list[str]())
    note: str


# ---- Kinds from docs/design/20 §2.3 without fields in spec/schemas.md (D143) ------


class Source(Body):
    """A frozen external source: repository at a commit, an archive by hash, a paper."""

    name: str
    type: Literal["git", "archive", "paper", "doc", "other"]
    locator: str  # URL or path where it was obtained
    revision: str  # commit, sha256 of the archive, DOI or edition
    license: str | None = None
    note: str | None = None


class Protocol(Body):
    """A measurement or experiment protocol approved by a human."""

    name: str
    version: int
    type: Literal["measurement", "experiment", "other"]
    description: str
    params: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())


class ExperimentRun(Body):
    name: str
    condition: str
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())
    note: str


class Report(Body):
    """A tool-generated report; ``data`` holds the numbers, ``summary`` one paragraph."""

    title: str
    subject: str
    summary: str
    data: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())


class ControlsVersion(Body):
    """One version of the controls registry's values (WP21, D481).

    ``values`` holds every knob of the definitions ``definitions_hash`` names; ``changes``
    lists what differs from the previous version (``refs.previous``). The actor is the
    author (a human for settings-page writes, the system for version 1 and for definition
    upgrades); ``reason`` is required. Runs record the version they used in ``refs.controls``.
    """

    version: int = Field(ge=1)
    definitions_hash: str
    values: dict[str, Any]
    changes: list[dict[str, Any]] = Field(default_factory=lambda: list[dict[str, Any]]())
    reason: str = Field(min_length=1)
