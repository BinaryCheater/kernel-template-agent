"""Host-level readers–writer lock of a compute host (WP23, D603).

On a unified-memory host (GB10) CPU work and GPU work share one memory system, so a
measurement is only clean when nothing else heavy runs on the whole machine. Every heavy
activity on that host takes this lock **shared** (Executor cpu and gpu-shared jobs, the
control plane's in-process MIP solves and L2 simulations when the control plane runs
there); measurement, profiling and microbenchmarks take it **exclusive**.

The lock is two ``flock`` files in one directory on the host, so it holds across
processes and across control planes (dev on the Mac over ssh, long-run on the host):

- ``intent.lock`` is the turnstile. A writer holds it exclusive from the moment it asks
  until it releases; a reader passes it shared only while it takes ``rw.lock``. So a
  waiting writer stops new readers (writer preference) and is never starved.
- ``rw.lock`` is the lock itself: shared for readers, exclusive for the writer.

Waits are bounded (``wait_s``); a writer's hold is bounded too (``max_hold_s``): past it
the holder's process drops both locks, logs it and leaves ``forced.jsonl`` behind. The
holder of the exclusive side is written to ``holder.json`` for the Pipeline view; the
file is informative, the flocks are the lock.

``python -m kta.core.hostlock hold <dir> <shared|exclusive> --wait-s W --max-s M`` holds
the lock for another process: it prints ``{"state": "waiting"}`` (exclusive, once the
turnstile is taken), ``{"state": "acquired", "wait_s": ...}``, and releases when its stdin
closes or ``M`` seconds pass (``{"state": "released", "forced": bool}``). The Executor
uses it, through its transport, for measurement windows (``JobExecutor.quiet_lease``).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from kta.core.errors import ConfigError, ExecutionError

log = logging.getLogger("kta.core.hostlock")
Mode = Literal["shared", "exclusive"]
POLL_S = 0.05
RW, INTENT, HOLDER, FORCED = "rw.lock", "intent.lock", "holder.json", "forced.jsonl"
HOST_SECTION = "host"


class HostLockTimeout(ExecutionError):
    """The lock was not granted within the wait bound (kind ``infra``: the host is busy)."""

    def __init__(self, message: str) -> None:
        super().__init__("infra", message)


@dataclass(frozen=True)
class Grant:
    mode: Mode
    holder: str
    wait_s: float
    directory: str


def _try(fd: IO[bytes], op: int) -> bool:
    try:
        fcntl.flock(fd, op | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _until(fd: IO[bytes], op: int, deadline: float, what: str) -> None:
    while not _try(fd, op):
        if time.monotonic() >= deadline:
            raise HostLockTimeout(what)
        time.sleep(POLL_S)  # kta:allowed flock has no timed wait; poll until the bound


class HostLock:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _open(self, name: str) -> IO[bytes]:
        self.directory.mkdir(parents=True, exist_ok=True)
        return (self.directory / name).open("a+b")

    @contextmanager
    def shared(self, holder: str, wait_s: float) -> Generator[Grant]:
        """Readers queue behind a waiting writer: the turnstile is taken shared first."""
        start = time.monotonic()
        deadline = start + wait_s
        what = f"host lock {self.directory}: shared for {holder} not granted in {wait_s}s"
        intent, rw = self._open(INTENT), self._open(RW)
        try:
            _until(intent, fcntl.LOCK_SH, deadline, what)
            try:
                _until(rw, fcntl.LOCK_SH, deadline, what)
            finally:
                fcntl.flock(intent, fcntl.LOCK_UN)
            yield Grant("shared", holder, time.monotonic() - start, str(self.directory))
        finally:
            rw.close()  # closing drops the flock
            intent.close()

    @contextmanager
    def exclusive(
        self,
        holder: str,
        wait_s: float,
        max_hold_s: float,
        on_waiting: Callable[[], None] | None = None,
    ) -> Generator[Grant]:
        start = time.monotonic()
        deadline = start + wait_s
        what = f"host lock {self.directory}: exclusive for {holder} not granted in {wait_s}s"
        intent, rw = self._open(INTENT), self._open(RW)
        timer: threading.Timer | None = None
        try:
            _until(intent, fcntl.LOCK_EX, deadline, what)
            if on_waiting is not None:
                on_waiting()  # new readers are blocked from here on
            _until(rw, fcntl.LOCK_EX, deadline, what)
            grant = Grant("exclusive", holder, time.monotonic() - start, str(self.directory))
            self._write_holder(holder, max_hold_s)
            timer = threading.Timer(max_hold_s, self._force, args=(rw, intent, holder, max_hold_s))
            timer.daemon = True
            timer.start()
            yield grant
        finally:
            if timer is not None:
                timer.cancel()
            (self.directory / HOLDER).unlink(missing_ok=True)
            rw.close()
            intent.close()

    def _write_holder(self, holder: str, max_hold_s: float) -> None:
        data = {
            "holder": holder,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "since_epoch_s": time.time(),
            "max_hold_s": max_hold_s,
        }
        (self.directory / HOLDER).write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    def _force(self, rw: IO[bytes], intent: IO[bytes], holder: str, max_hold_s: float) -> None:
        fcntl.flock(rw, fcntl.LOCK_UN)
        fcntl.flock(intent, fcntl.LOCK_UN)
        log.warning("host lock %s: %s force-released after %ss", self.directory, holder, max_hold_s)
        line = {"holder": holder, "max_hold_s": max_hold_s, "at_epoch_s": time.time()}
        with (self.directory / FORCED).open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")

    def holder(self) -> dict[str, Any] | None:
        """The exclusive holder as last written (informative), or None."""
        try:
            return json.loads((self.directory / HOLDER).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None


# ---- the control plane's own heavy work ---------------------------------------------------


@dataclass(frozen=True)
class HostSettings:
    """``[host]`` of the mode config: set only when the control plane runs on the compute
    host (long-run mode). Then its in-process solves and simulations take the lock shared."""

    name: str
    lock_dir: Path | None
    wait_s: float

    @classmethod
    def of(cls, section: Mapping[str, Any]) -> HostSettings:
        lock = section.get("lock_dir")
        if lock is not None and not isinstance(lock, str):
            raise ConfigError("[host] lock_dir must be a path string")
        return cls(
            name=str(section.get("name", "")),
            lock_dir=Path(lock).expanduser() if lock else None,
            wait_s=float(section.get("shared_wait_s", 3600)),
        )


@contextmanager
def heavy(host: Mapping[str, Any], what: str) -> Generator[Grant | None]:
    """Hold the host lock shared around in-process heavy work (no-op off the host)."""
    settings = HostSettings.of(host)
    if settings.lock_dir is None:
        yield None
        return
    with HostLock(settings.lock_dir).shared(
        f"control-plane:{what}:{os.getpid()}", settings.wait_s
    ) as g:
        yield g


# ---- `hold`: the lock held for another process ---------------------------------------------


def _say(state: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(state, sort_keys=True) + "\n")  # kta:allowed hold protocol
    sys.stdout.flush()


def hold(directory: Path, mode: Mode, holder: str, wait_s: float, max_s: float) -> int:
    lock = HostLock(directory)
    released = threading.Event()

    def wait_for_stdin() -> None:
        sys.stdin.buffer.read()  # returns at EOF: the requester let go (or died)
        released.set()

    try:
        if mode == "exclusive":
            ctx = lock.exclusive(holder, wait_s, max_s, lambda: _say({"state": "waiting"}))
        else:
            ctx = lock.shared(holder, wait_s)
        with ctx as grant:
            _say({"state": "acquired", "wait_s": round(grant.wait_s, 6), "mode": mode})
            threading.Thread(target=wait_for_stdin, daemon=True).start()
            forced = not released.wait(max_s)
        _say({"state": "released", "forced": forced})
    except HostLockTimeout as exc:
        _say({"state": "timeout", "error": str(exc)})
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kta.core.hostlock")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hold")
    h.add_argument("directory")
    h.add_argument("mode", choices=["shared", "exclusive"])
    h.add_argument("--holder", default=f"hold:{os.getpid()}")
    h.add_argument("--wait-s", type=float, default=600.0)
    h.add_argument("--max-s", type=float, default=600.0)
    args = p.parse_args(argv)
    return hold(Path(args.directory), args.mode, args.holder, args.wait_s, args.max_s)


if __name__ == "__main__":
    sys.exit(main())
