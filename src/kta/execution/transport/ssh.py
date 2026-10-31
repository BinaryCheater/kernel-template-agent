"""``ssh`` transport: the control plane runs worker commands over the user's ssh alias.

The command runs in ``<root>/src/<commit>``, the deployment ``scripts/worker-setup``
made for this commit; a worker without it refuses work (decision D181). Connections
are multiplexed through a control socket under the control plane's data directory
(decision D196) so each command costs one round trip, not one handshake.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from kta.core.errors import ConfigError, ExecutionError
from kta.execution.base import WorkerConfig
from kta.execution.transport.local import WORKER_MODULE, check_reply

EXIT_NOT_DEPLOYED = 97
SSH_EXIT_FAILED = 255
MAX_SOCKET_DIR = 60
# sshd refuses bursts of unauthenticated connections (MaxStartups, default 10); calls per
# worker are capped, and connections refused before the command started are retried.
# ``cancel`` skips the cap: the calls holding it may be the long ``run`` calls it stops
# (one extra session stays under sshd's MaxSessions of 10 per multiplexed connection).
MAX_CONCURRENT = 8
UNCAPPED = frozenset({"cancel"})
CONNECT_RETRIES = 4
NOT_CONNECTED = (
    "kex_exchange_identification",
    "Connection reset by",
    "Connection refused",
    "Connection closed by",
    "ssh_exchange_identification",
)


def remote_path(path: str) -> str:
    """A path for a remote shell: ``~`` expands on the remote side, the rest is quoted."""
    if path == "~" or path.startswith("~/"):
        return '"$HOME"' + shlex.quote(path[1:]) if len(path) > 1 else '"$HOME"'
    return shlex.quote(path)


class SshTransport:
    @property
    def name(self) -> Literal["ssh"]:
        return "ssh"

    def __init__(
        self, config: WorkerConfig, env: str, wp: str | None, commit: str, sockets: Path
    ) -> None:
        if not config.ssh or not config.root:
            raise ConfigError(f"[workers.{config.name}] ssh transport needs ssh and root")
        self.config = config
        self.env = env
        self.wp = wp
        self.commit = commit
        self.sockets = sockets
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)

    def script(self, args: list[str]) -> str:
        root = remote_path(self.config.root or "")
        deploy = f"{root}/src/{shlex.quote(self.commit)}"
        exports = [f"KTA_ENV={shlex.quote(self.env)}", f"KTA_DATA={root}/data"]
        if self.wp:
            exports.append(f"KTA_WP={shlex.quote(self.wp)}")
        if self.env == "test":
            exports.append(f"KTA_TEST_DATA={root}/data/test-worker")
        exports += [f"{k}={shlex.quote(v)}" for k, v in sorted(self.config.env.items())]
        path = ":".join(remote_path(p) for p in self.config.path)
        if path:
            exports.append(f'PATH={path}:"$PATH"')
        return (
            f"if [ ! -x {deploy}/.venv/bin/python ]; then "
            f"echo 'commit {self.commit} is not deployed on {self.config.name}' >&2; "
            f"exit {EXIT_NOT_DEPLOYED}; fi; "
            f"cd {deploy} && export {' '.join(exports)} && "
            f"exec .venv/bin/python -m {WORKER_MODULE} {shlex.join(args)}"
        )

    def argv(self, args: list[str]) -> list[str]:
        opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        # Unix socket paths are limited to ~104 bytes and %C expands to 40 characters;
        # with a longer directory (pytest temp dirs) each call opens its own connection.
        if len(str(self.sockets)) <= MAX_SOCKET_DIR:
            self.sockets.mkdir(mode=0o700, parents=True, exist_ok=True)
            opts += ["-o", "ControlMaster=auto", "-o", f"ControlPath={self.sockets}/%C"]
            opts += ["-o", "ControlPersist=120"]
        return ["ssh", *opts, self.config.ssh or "", self.script(args)]

    def _once(
        self, args: list[str], stdin: bytes, timeout_s: float
    ) -> subprocess.CompletedProcess[bytes]:
        slot = nullcontext() if args[:1] and args[0] in UNCAPPED else self.slots
        try:
            with slot:
                return subprocess.run(
                    self.argv(args),
                    input=stdin,
                    capture_output=True,
                    timeout=timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                "infra", f"{self.config.name}: no answer over ssh in {timeout_s}s"
            ) from exc

    def popen(self, args: list[str]) -> subprocess.Popen[bytes]:
        # The remote command ends when ssh closes its stdin (the hold protocol relies on it).
        return subprocess.Popen(
            self.argv(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def call(self, args: list[str], stdin: bytes, timeout_s: float) -> bytes:
        proc = self._once(args, stdin, timeout_s)
        detail = proc.stderr.decode("utf-8", "replace").strip()[-500:]
        for attempt in range(CONNECT_RETRIES):
            # Retry only when the connection was never established: the command did not run.
            if proc.returncode != SSH_EXIT_FAILED or not any(m in detail for m in NOT_CONNECTED):
                break
            time.sleep(0.5 * (attempt + 1))  # kta:allowed backoff before reconnecting
            proc = self._once(args, stdin, timeout_s)
            detail = proc.stderr.decode("utf-8", "replace").strip()[-500:]
        if proc.returncode == EXIT_NOT_DEPLOYED:
            raise ExecutionError("infra", proc.stderr.decode("utf-8", "replace").strip())
        if proc.returncode == SSH_EXIT_FAILED:
            raise ExecutionError("infra", f"{self.config.name}: ssh failed: {detail}")
        return check_reply(self.config.name, proc)
