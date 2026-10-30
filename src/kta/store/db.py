"""SQLite connection setup, migrations and timestamp format shared by Store and Ops."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from kta.core.errors import ConfigError

BUSY_TIMEOUT_MS = 30_000
_MIGRATION_RE = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def ts(value: datetime) -> str:
    """Fixed-width UTC timestamp, so text comparison in SQL orders correctly."""
    if value.tzinfo is None:
        raise ConfigError(f"naive datetime {value!r}; use timezone-aware UTC")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def _migrations() -> list[tuple[int, str]]:
    folder = resources.files("kta.store").joinpath("migrations")
    found: list[tuple[int, str]] = []
    for entry in folder.iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            found.append((int(m.group(1)), entry.read_text(encoding="utf-8")))
    return sorted(found)


def _user_version(con: sqlite3.Connection) -> int:
    return int(con.execute("PRAGMA user_version").fetchone()[0])


def migrate(con: sqlite3.Connection) -> int:
    """Apply pending migrations; returns the schema version.

    Each migration runs as one script that ends by setting ``user_version``. When two
    processes migrate at once, the loser's script fails on existing tables; it then
    re-reads the version instead of failing.
    """
    for number, sql in _migrations():
        if _user_version(con) >= number:
            continue
        try:
            con.executescript(f"BEGIN IMMEDIATE;\n{sql}\nPRAGMA user_version = {number};\nCOMMIT;")
        except sqlite3.OperationalError:
            if con.in_transaction:
                con.execute("ROLLBACK")
            if _user_version(con) < number:
                raise
    return _user_version(con)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Autocommit mode: transactions are opened explicitly with BEGIN IMMEDIATE.
    con = sqlite3.connect(
        path, isolation_level=None, check_same_thread=False, timeout=BUSY_TIMEOUT_MS / 1000
    )
    con.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    migrate(con)
    # Recommended for long-lived connections: gather planner statistics when missing, so
    # expression indexes such as (kind, $.body.name) are chosen over the ORDER BY index.
    con.execute("PRAGMA optimize = 0x10002")
    return con


class Database:
    """One connection with a lock; every write goes through ``write()``.

    sqlite3 connections are not safe for concurrent use from several threads, so reads
    take the same lock. Other processes use their own Database on the same file and are
    serialized by SQLite's file lock (BEGIN IMMEDIATE + busy timeout).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.con = connect(path)
        self.lock = threading.RLock()

    @contextmanager
    def write(self) -> Generator[sqlite3.Connection]:
        with self.lock:
            self.con.execute("BEGIN IMMEDIATE")
            try:
                yield self.con
            except BaseException:
                self.con.execute("ROLLBACK")
                raise
            self.con.execute("COMMIT")

    @contextmanager
    def read(self) -> Generator[sqlite3.Connection]:
        with self.lock:
            yield self.con

    def optimize(self) -> None:
        """Refresh planner statistics after bulk writes."""
        with self.lock:
            self.con.execute("PRAGMA optimize")

    def close(self) -> None:
        with self.lock:
            self.con.close()
