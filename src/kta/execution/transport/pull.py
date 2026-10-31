"""``pull`` transport: the worker daemon fetches jobs from the control-plane API.

Phase 1 does not implement it (conventions §6); the protocol is in ``pull.md``.
"""

from __future__ import annotations

import subprocess
from typing import Literal

from kta.core.errors import NotSupported
from kta.execution.base import WorkerConfig


class PullTransport:
    @property
    def name(self) -> Literal["pull"]:
        return "pull"

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config

    def call(self, args: list[str], stdin: bytes, timeout_s: float) -> bytes:
        raise NotSupported(
            f"worker {self.config.name}: the pull transport is specified in "
            "kta/execution/transport/pull.md but not implemented in phase 1"
        )

    def popen(self, args: list[str]) -> subprocess.Popen[bytes]:
        raise NotSupported(f"worker {self.config.name}: the pull transport (phase 1)")
