"""The launch spec: what the runner asks the launcher on the agent host to do.

The runner (control plane) and the launcher (an Executor job on the agent host, D300)
share no file system, so everything the launcher needs travels as one JSON file plus
the bundle files, all as job inputs by hash. Paths inside the spec are relative to the
agent host's data directory or use the placeholders below; the launcher resolves them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Placeholders the launcher substitutes in argv and env values.
HOME = "{home}"  # the work unit's isolated CLI home: <data>/agent-home/<work unit>
WORKSPACE = "{workspace}"  # <data>/work/<work unit>
KTA_BIN = "{kta_bin}"  # bin directory of the deployed kta on the agent host
SPEC_FILE = "launch.json"
PROMPT_FILE = "prompt.md"
BUNDLE_WORKSPACE = "bundle/workspace/"  # job-input prefix of files installed into the workspace
BUNDLE_BATCH = "batch/"  # job-input prefix of the batch template files and env.lock.json
# Top-level workspace entries allowed by spec/modules/agents.md ("工作区文件").
WORKSPACE_ENTRIES: tuple[str, ...] = (
    "AGENTS.md",
    "TASK.md",
    "NOTES.md",
    "inputs",
    "out",
    ".agents",
    ".git",
)
# Directories the runner replaces wholesale on every step (the agent may not keep edits).
REPLACED_DIRS: tuple[str, ...] = (".agents/skills",)

Mode = Literal["start", "resume", "close", "remove"]


@dataclass(frozen=True)
class AuthSpec:
    provider: str
    dir: str  # protected credential directory on the agent host, e.g. ~/.config/kta/auth/codex
    files: list[str] = field(default_factory=lambda: list[str]())  # linked into the CLI home


@dataclass(frozen=True)
class LaunchSpec:
    provider: str
    work_unit: str
    run: str  # ULID of this run
    mode: Mode
    argv: list[str] = field(default_factory=lambda: list[str]())
    env: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    session: str | None = None
    wall_s: int = 3600
    idle_s: int = 600
    workspace_files: list[str] = field(default_factory=lambda: list[str]())
    batch: str = ""  # batch root <data>/batches/<batch>/ on the agent host (D322)
    batch_files: list[str] = field(default_factory=lambda: list[str]())  # template + lock
    auth: AuthSpec | None = None
    run_as: str | None = None  # system user of config+user isolation (D09); None = job user
    secret_env: list[str] = field(default_factory=lambda: list[str]())  # never in the manifest
    notes_file: str = "NOTES.md"
    gpu_guard: bool = True
    nvidia_smi: str = "nvidia-smi"
    watch: list[str] = field(default_factory=lambda: list[str]())  # host paths, depth 1
    arm: dict[str, Any] = field(
        default_factory=lambda: dict[str, Any]()
    )  # knowledge condition (D430)
    expect_version: str | None = None  # the profile's CLI version; another version is refused
    # WP22 (D584): provider, policy and, when required, url (no credentials), source,
    # check_hosts, no_proxy, credentials, secrets_file. The launcher runs the preflight.
    proxy: dict[str, Any] | None = None

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, indent=1).encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaunchSpec:
        auth = data.get("auth")
        return cls(**{**data, "auth": AuthSpec(**auth) if auth else None})

    @classmethod
    def from_json(cls, raw: bytes) -> LaunchSpec:
        return cls.from_dict(json.loads(raw.decode("utf-8")))


# Names of the files the launcher writes under the job's out/ directory.
OUT_REPORT = "report.json"
OUT_MANIFEST = "manifest.json"
OUT_ARCHIVE = "home.tar.gz"
OUT_WORKSPACE = "workspace.tar.gz"
OUT_STDOUT = "stdout.jsonl"
OUT_STDERR = "stderr.txt"
