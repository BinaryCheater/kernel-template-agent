"""``local`` transport: the worker runs on the control-plane machine, same code tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from kta.core.errors import ExecutionError, NotSupported

WORKER_MODULE = "kta.execution.worker"
EXIT_NOT_SUPPORTED = 3


def check_reply(name: str, proc: subprocess.CompletedProcess[bytes]) -> bytes:
    """stdout of a finished worker command, or the matching error."""
    detail = proc.stderr.decode("utf-8", "replace").strip()[-2000:]
    if proc.returncode == EXIT_NOT_SUPPORTED:
        raise NotSupported(f"{name}: {detail}")
    if proc.returncode != 0:
        raise ExecutionError("infra", f"{name}: worker command exit {proc.returncode}: {detail}")
    return proc.stdout


class LocalTransport:
    @property
    def name(self) -> Literal["local"]:
        return "local"

    def __init__(self, env: str, wp: str | None, data_root: Path) -> None:
        self.environ = {k: v for k, v in os.environ.items() if not k.startswith("KTA_")}
        self.environ.update({"KTA_ENV": env, "KTA_DATA": str(data_root)})
        if env == "test":
            self.environ["KTA_TEST_DATA"] = str(data_root)
        if wp:
            self.environ["KTA_WP"] = wp

    def call(self, args: list[str], stdin: bytes, timeout_s: float) -> bytes:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", WORKER_MODULE, *args],
                input=stdin,
                capture_output=True,
                env=self.environ,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError("infra", f"local worker did not answer in {timeout_s}s") from exc
        return check_reply("local", proc)

    def popen(self, args: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environ,
        )
