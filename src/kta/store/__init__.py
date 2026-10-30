"""kta.store: evidence records, blobs, operations tables (see README.md).

Implementations are registered on the Context's registry (decision D77):
``resolve("store", ctx)`` and ``resolve("ops", ctx)`` open ``<data_dir>/kta.db``.
"""

from __future__ import annotations

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.envprobe import git_state
from kta.core.ids import UlidFactory
from kta.core.registry import Registry
from kta.store.db import Database
from kta.store.sqlite import CodeState, SqliteStore
from kta.store.sqlite_ops import SqliteOps

DB_NAME = "kta.db"
OBJECTS_DIR = "objects"


def code_state() -> CodeState:
    commit, dirty = git_state(repo_root())
    return CodeState(commit, dirty)


def open_store(ctx: Context, code: CodeState | None = None) -> SqliteStore:
    db = Database(ctx.data_dir / DB_NAME)
    return SqliteStore(db, ctx.data_dir / OBJECTS_DIR, ctx.env, code or code_state(), ctx.clock)


def open_ops(ctx: Context) -> SqliteOps:
    return SqliteOps(Database(ctx.data_dir / DB_NAME), ctx.clock, UlidFactory(ctx.clock))


def register_implementations(registry: Registry) -> None:
    registry.register("store", "sqlite", open_store)
    registry.register("ops", "sqlite", open_ops)


__all__ = [
    "CodeState",
    "SqliteOps",
    "SqliteStore",
    "code_state",
    "open_ops",
    "open_store",
    "register_implementations",
]
