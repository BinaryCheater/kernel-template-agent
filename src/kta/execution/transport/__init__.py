"""Transports from the control plane to workers: local, ssh, pull (conventions §6)."""

from __future__ import annotations

from pathlib import Path

from kta.core.errors import ConfigError
from kta.execution.base import Transport, WorkerConfig
from kta.execution.transport.local import LocalTransport
from kta.execution.transport.pull import PullTransport
from kta.execution.transport.ssh import SshTransport


def make_transport(
    config: WorkerConfig, env: str, wp: str | None, data_root: Path, commit: str
) -> Transport:
    """The transport for one worker.

    ``data_root`` is the control plane's ``KTA_DATA`` (the local worker shares it) and
    holds the ssh control sockets; ``commit`` selects the worker's deployment.
    """
    if config.transport == "local":
        return LocalTransport(env, wp, data_root)
    if config.transport == "ssh":
        return SshTransport(config, env, wp, commit, data_root / "ssh")
    if config.transport == "pull":
        return PullTransport(config)
    raise ConfigError(f"unknown transport {config.transport!r}")


__all__ = ["LocalTransport", "PullTransport", "SshTransport", "make_transport"]
