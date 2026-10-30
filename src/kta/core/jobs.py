"""Job and JobResult: what the Executor runs and returns (spec/modules/target-exec.md).

They live in core rather than in ``kta.execution`` because ``kta.targets`` (layer 2) must
build jobs for its probes and cannot import execution (layer 3); ``kta.execution.base``
re-exports them (decision D180).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

JobKind = Literal["compile", "run", "measure", "profile", "check", "microbench", "shell"]
ErrorKind = Literal["none", "timeout", "oom", "gpu_hang", "infra", "nonzero"]
JOB_KINDS: tuple[JobKind, ...] = (
    "compile",
    "run",
    "measure",
    "profile",
    "check",
    "microbench",
    "shell",
)
ERROR_KINDS: tuple[ErrorKind, ...] = ("none", "timeout", "oom", "gpu_hang", "infra", "nonzero")

# Pool a job runs in when ``requires`` names none (docs/design/11 §4.1, decision D183).
DEFAULT_POOL: dict[str, str] = {
    "compile": "cpu",
    "shell": "cpu",
    "run": "gpu-shared",
    "check": "gpu-shared",
    "measure": "gpu-exclusive",
    "profile": "gpu-exclusive",
    "microbench": "gpu-exclusive",
}
GPU_POOLS = frozenset({"gpu-shared", "gpu-exclusive"})


@dataclass(frozen=True)
class Job:
    """A unit of remote work.

    ``argv`` may use the placeholders ``{python}`` (the worker's interpreter for this
    commit), ``{cuda_home}`` and ``{job}`` (the job directory); the worker substitutes
    them (decision D185). ``inputs`` maps a file name in the job directory to a blob
    sha256; files the job writes under ``out/`` come back as ``outputs``.
    """

    kind: JobKind
    argv: list[str]
    requires: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    env: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    inputs: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    timeout_s: int = 600
    mem_limit_mb: int | None = None
    container: str | None = None  # image reference pinned by digest; None = worker host

    @property
    def pool(self) -> str:
        return self.requires.get("pool", DEFAULT_POOL[self.kind])

    @property
    def uses_gpu(self) -> bool:
        return self.pool in GPU_POOLS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(**data)


@dataclass(frozen=True)
class JobResult:
    ok: bool
    exit_code: int
    wall_s: float
    queue_s: float
    worker: str  # Worker record ID
    device: str | None
    env_manifest: str  # EnvManifest record ID
    stdout_blob: str
    stderr_blob: str
    outputs: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    error_kind: ErrorKind = "none"
    job_id: str = ""
    worker_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobRunner(Protocol):
    """The part of the Executor that targets and backends need (structural type)."""

    def submit(self, job: Job) -> str: ...

    def wait(self, job_id: str) -> JobResult: ...

    def run(self, job: Job) -> JobResult: ...

    def read_blob(self, sha256: str) -> bytes: ...

    def put_blob(self, data: bytes, media: str) -> str: ...
