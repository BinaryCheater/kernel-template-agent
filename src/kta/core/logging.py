"""JSON-lines logging with contextual IDs and secret redaction (spec/conventions.md §4).

Logs are for debugging only; nothing in them counts as evidence.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kta.core.context import Context
from kta.core.secrets import redact

ROOT_LOGGER = "kta"
CONTEXT_FIELDS = ("loop_id", "task_id", "job_id", "session")

_context_vars: dict[str, ContextVar[str | None]] = {
    name: ContextVar(f"kta_log_{name}", default=None) for name in CONTEXT_FIELDS
}


@contextmanager
def bind(**ids: str) -> Generator[None]:
    """Attach loop/task/job/session IDs to every log line emitted inside the block."""
    unknown = set(ids) - set(CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"unknown log context fields: {sorted(unknown)}")
    tokens: list[tuple[str, Token[str | None]]] = [
        (name, _context_vars[name].set(value)) for name, value in ids.items()
    ]
    try:
        yield
    finally:
        for name, token in reversed(tokens):
            _context_vars[name].reset(token)


class JsonFormatter(logging.Formatter):
    def __init__(self, env: str, process: str) -> None:
        super().__init__()
        self._env = env
        self._process = process

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
            "env": self._env,
            "process": self._process,
        }
        for name in CONTEXT_FIELDS:
            value = _context_vars[name].get()
            if value is not None:
                line[name] = value
        if record.exc_info:
            line["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(line, ensure_ascii=False)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, UTC).isoformat()


def log_path(ctx: Context) -> Path:
    day = ctx.clock.now().date().isoformat()
    return ctx.data_dir / "logs" / ctx.process / f"{day}.jsonl"


def setup(ctx: Context) -> Path:
    """Configure the ``kta`` logger tree for this process; returns the log file path."""
    path = log_path(ctx)
    os.makedirs(path.parent, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter(ctx.env, ctx.process))
    root = logging.getLogger(ROOT_LOGGER)
    for old in list(root.handlers):
        root.removeHandler(old)
        old.close()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if ctx.env == "dev" else logging.INFO)
    root.propagate = False
    return path
