"""Executor protocol, Job, JobResult, transports (spec/modules/target-exec.md)."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from kta.core.errors import ConfigError, ExecutionError
from kta.core.jobs import Job, JobResult, JobRunner
from kta.execution.blobs import frame, parse_frames
from kta.execution.workerprobe import WorkerSpec

TransportName = Literal["local", "ssh", "pull"]
TRANSPORTS: tuple[TransportName, ...] = ("local", "ssh", "pull")
DEFAULT_CALL_TIMEOUT_S = 300.0
RUN_MARGIN_S = 120.0  # the worker kills a job at its timeout; this covers the epilogue


class Executor(Protocol):
    def submit(self, job: Job) -> str: ...

    def wait(self, job_id: str) -> JobResult: ...

    def run(self, job: Job) -> JobResult: ...

    def quiet_lease(
        self, worker: str, device: str, holder: str, max_s: int = 600
    ) -> AbstractContextManager[None]: ...

    def read_blob(self, sha256: str) -> bytes: ...

    def put_blob(self, data: bytes, media: str) -> str: ...


class Transport(Protocol):
    """Starts ``python -m kta.execution.worker <args>`` on one worker."""

    @property
    def name(self) -> TransportName: ...

    def call(self, args: list[str], stdin: bytes, timeout_s: float) -> bytes:
        """The command's stdout; raises ExecutionError (infra) or NotSupported."""
        ...


@dataclass(frozen=True)
class WorkerConfig:
    """One ``[workers.<name>]`` section of the configuration."""

    name: str
    transport: TransportName
    ssh: str | None = None
    root: str | None = None
    path: list[str] = field(default_factory=lambda: list[str]())
    env: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    cuda_home: str = "/usr/local/cuda"
    image: str | None = None
    pools: dict[str, int] = field(default_factory=lambda: {"cpu": 1})

    @property
    def spec(self) -> WorkerSpec:
        return WorkerSpec(self.name, dict(self.pools), self.image, self.cuda_home)

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> WorkerConfig:
        transport = data.get("transport")
        if transport not in TRANSPORTS:
            raise ConfigError(f"[workers.{name}] transport must be one of {TRANSPORTS}")
        known = {f for f in cls.__dataclass_fields__ if f != "name"}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"[workers.{name}] unknown keys {sorted(unknown)}")
        pools = {str(k): int(v) for k, v in dict(data.get("pools", {"cpu": 1})).items()}
        return cls(name=name, **{**data, "pools": pools})


class WorkerClient:
    """Typed calls to the worker commands over any transport."""

    def __init__(self, config: WorkerConfig, transport: Transport) -> None:
        self.config = config
        self.transport = transport

    def _json(self, args: list[str], payload: Any = None, timeout_s: float | None = None) -> Any:
        stdin = b"" if payload is None else json.dumps(payload).encode("utf-8")
        out = self.transport.call(args, stdin, timeout_s or DEFAULT_CALL_TIMEOUT_S)
        try:
            return json.loads(out.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutionError("infra", f"{self.config.name}: bad worker reply: {exc}") from exc

    def info(self) -> dict[str, Any]:
        return self._json(["info", json.dumps(self.config.spec.to_dict())])

    def missing(self, shas: list[str]) -> list[str]:
        return list(self._json(["has"], shas))

    def put(self, blobs: dict[str, bytes]) -> None:
        stdin = frame(sorted(blobs.items()))
        out = self.transport.call(["put"], stdin, DEFAULT_CALL_TIMEOUT_S)
        stored = set(json.loads(out.decode("utf-8")))
        if stored != set(blobs):
            raise ExecutionError("infra", f"{self.config.name}: worker stored {len(stored)} blobs")

    def get(self, shas: list[str]) -> dict[str, bytes]:
        if not shas:
            return {}
        got = parse_frames(self.transport.call(["get", *shas], b"", DEFAULT_CALL_TIMEOUT_S))
        if set(got) != set(shas):
            raise ExecutionError("infra", f"{self.config.name}: missing blobs in transfer")
        return got

    def run(self, job_id: str, job: Job, device: str | None) -> dict[str, Any]:
        payload = {
            "job_id": job_id,
            "job": job.to_dict(),
            "spec": self.config.spec.to_dict(),
            "device": device,
        }
        return self._json(["run"], payload, job.timeout_s + RUN_MARGIN_S)


__all__ = [
    "Executor",
    "Job",
    "JobResult",
    "JobRunner",
    "Transport",
    "TransportName",
    "WorkerClient",
    "WorkerConfig",
]
