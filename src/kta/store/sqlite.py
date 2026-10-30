"""SQLite implementation of Store (spec/modules/core-store.md).

Writes: one Database (connection + lock) per process; the API service holds the
production writer (api-web.md). ``put`` validates the envelope and body, the
environment rules, refs and blobs, and the status-promotion rules, then writes the
record, its edges and its full-text row in one transaction.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kta.core.context import Clock
from kta.core.errors import PermissionDenied, SchemaError
from kta.core.ids import canonical_json, record_id
from kta.core.schemas import Assurance, EvidenceStatus, Record, kind_spec
from kta.core.schemas.process import StatusChange
from kta.store import querybuild, status, textindex
from kta.store.base import Direction, Graph, Hit
from kta.store.blobs import BlobFiles
from kta.store.db import Database, ts

# Origins each environment accepts (spec/conventions.md §2).
ALLOWED_ORIGINS: dict[str, frozenset[str]] = {
    "test": frozenset({"real", "fixture", "simulated", "historical"}),
    "dev": frozenset({"real", "fixture"}),
    "main": frozenset({"real", "historical"}),
}
SNIPPET_TOKENS = 12
BULK_OPTIMIZE = 1000  # refresh planner statistics after writes this large


class RecordNotFound(SchemaError):
    pass


@dataclass(frozen=True)
class CodeState:
    """The commit and dirty flag of the code this process runs (decision D140)."""

    commit: str
    dirty: bool


class SqliteStore:
    def __init__(
        self, db: Database, objects: Path, env: str, code: CodeState, clock: Clock
    ) -> None:
        if env not in ALLOWED_ORIGINS:
            raise SchemaError(f"unknown env {env!r}")
        self.db = db
        self.clock = clock
        self.blobs = BlobFiles(objects)
        self.env = env
        self.code = code

    # ---- writes -----------------------------------------------------------------

    def put(self, record: Record) -> str:
        return self.put_many([record])[0]

    def put_many(self, records: Iterable[Record]) -> list[str]:
        """Write records in one transaction (later records may reference earlier ones)."""
        ids: list[str] = []
        with self.db.write() as con:
            for record in records:
                ids.append(self._put_one(con, self._stamp(record)))
        if len(ids) >= BULK_OPTIMIZE:
            self.db.optimize()
        return ids

    def _stamp(self, record: Record) -> Record:
        if record.origin not in ALLOWED_ORIGINS[self.env]:
            raise PermissionDenied(f"env {self.env} does not accept origin {record.origin!r}")
        actor = record.actor
        if actor.commit is None:
            actor = actor.model_copy(update={"commit": self.code.commit, "dirty": self.code.dirty})
            record = record.model_copy(update={"actor": actor})
        if self.env == "main" and actor.dirty:
            raise PermissionDenied("env main does not accept records written by dirty code")
        return record

    def _put_one(self, con: sqlite3.Connection, record: Record) -> str:
        data = record.to_json_dict()
        rid = record_id(data)
        if con.execute("SELECT 1 FROM records WHERE id = ?", (rid,)).fetchone():
            return rid
        self._check_refs(con, record)
        self._check_blobs(con, record)
        edges = record.ref_ids()
        if record.kind == "StatusChange":
            change = StatusChange.model_validate(record.body)
            if not self._exists(con, change.subject):
                raise SchemaError(f"StatusChange subject {change.subject} is not in the store")
            current = self._status(con, change.subject)
            status.check_change(current, change, record.actor.type)
            edges.append(("subject", change.subject))
        con.execute(
            "INSERT INTO records (id, kind, schema, created_at, actor_type, actor_name, task, "
            "origin, json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                record.kind,
                record.schema_,
                ts(record.created_at),
                record.actor.type,
                record.actor.name,
                record.actor.task,
                record.origin,
                canonical_json(data).decode("utf-8"),
            ),
        )
        con.executemany(
            "INSERT OR IGNORE INTO edges (src, rel, dst) VALUES (?, ?, ?)",
            [(rid, rel, dst) for rel, dst in edges],
        )
        text = textindex.extract(record.kind, record.body)
        if text is not None:
            con.execute(
                "INSERT INTO records_fts (id, kind, title, text) VALUES (?, ?, ?, ?)",
                (rid, record.kind, text[0], text[1]),
            )
        return rid

    def _check_refs(self, con: sqlite3.Connection, record: Record) -> None:
        pairs = record.ref_ids()
        if not pairs:
            return
        wanted = sorted({dst for _, dst in pairs})
        rows = con.execute(
            f"SELECT id, kind FROM records WHERE id IN ({', '.join('?' * len(wanted))})", wanted
        ).fetchall()
        kinds: dict[str, str] = {row[0]: row[1] for row in rows}
        missing = [d for d in wanted if d not in kinds]
        if missing:
            raise SchemaError(f"{record.kind}: refs not in the store: {missing}")
        expected = kind_spec(record.kind).ref_kinds
        for rel, dst in pairs:
            if rel in expected and kinds[dst] != expected[rel]:
                raise SchemaError(
                    f"{record.kind}: refs.{rel} must point to {expected[rel]}, not {kinds[dst]}"
                )

    def _check_blobs(self, con: sqlite3.Connection, record: Record) -> None:
        for name, sha in record.blobs.items():
            if not con.execute("SELECT 1 FROM blobs WHERE sha256 = ?", (sha,)).fetchone():
                raise SchemaError(f"{record.kind}: blobs.{name} {sha[:12]} is not stored")

    def put_blob(self, data: bytes | Path, media: str) -> str:
        sha, size = self.blobs.write(data)
        with self.db.write() as con:
            con.execute(
                "INSERT OR IGNORE INTO blobs (sha256, size, media, first_seen) VALUES (?, ?, ?, ?)",
                (sha, size, media, ts(self.clock.now())),
            )
        return sha

    # ---- reads ------------------------------------------------------------------

    def blob_path(self, sha256: str) -> Path:
        with self.db.read() as con:
            row = con.execute("SELECT 1 FROM blobs WHERE sha256 = ?", (sha256,)).fetchone()
        path = self.blobs.path(sha256)
        if row is None or not path.exists():
            raise RecordNotFound(f"blob {sha256} is not stored")
        return path

    def blob_media(self, sha256: str) -> str:
        with self.db.read() as con:
            row = con.execute("SELECT media FROM blobs WHERE sha256 = ?", (sha256,)).fetchone()
        if row is None:
            raise RecordNotFound(f"blob {sha256} is not stored")
        return str(row[0])

    def get(self, id: str) -> Record:
        with self.db.read() as con:
            row = con.execute("SELECT json FROM records WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise RecordNotFound(f"record {id} is not in the store")
        return Record.model_validate(json.loads(row[0]))

    def get_json(self, id: str) -> dict[str, Any]:
        """The stored JSON exactly as hashed (for transfer and ID verification)."""
        with self.db.read() as con:
            row = con.execute("SELECT json FROM records WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise RecordNotFound(f"record {id} is not in the store")
        return json.loads(row[0])

    def exists(self, id: str) -> bool:
        with self.db.read() as con:
            return self._exists(con, id)

    @staticmethod
    def _exists(con: sqlite3.Connection, id: str) -> bool:
        return con.execute("SELECT 1 FROM records WHERE id = ?", (id,)).fetchone() is not None

    def kind_of(self, id: str) -> str:
        with self.db.read() as con:
            row = con.execute("SELECT kind FROM records WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise RecordNotFound(f"record {id} is not in the store")
        return str(row[0])

    def query(
        self,
        kind: str,
        where: Mapping[str, Any] | None = None,
        order: str | None = None,
        limit: int = 100,
    ) -> list[Record]:
        kind_spec(kind)
        sql, params = querybuild.build(kind, where, order, limit)
        with self.db.read() as con:
            rows = con.execute(sql, params).fetchall()
        return [Record.model_validate(json.loads(r[0])) for r in rows]

    def search(self, text: str, kinds: list[str] | None = None, limit: int = 10) -> list[Hit]:
        match = textindex.match_query(text)
        if match is None:
            return []
        sql = (
            f"SELECT id, kind, title, snippet(records_fts, -1, '[', ']', '…', {SNIPPET_TOKENS}), "
            "rank FROM records_fts WHERE records_fts MATCH ?"
        )
        params: list[Any] = [match]
        if kinds:
            sql += f" AND kind IN ({', '.join('?' * len(kinds))})"
            params += kinds
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self.db.read() as con:
            rows = con.execute(sql, params).fetchall()
        return [Hit(r[0], r[1], r[2], r[3], -float(r[4])) for r in rows]

    def neighbors(self, id: str, rel: str | None = None, direction: Direction = "out") -> list[str]:
        mine, other = ("src", "dst") if direction == "out" else ("dst", "src")
        sql = f"SELECT {other} FROM edges WHERE {mine} = ?"
        params: list[Any] = [id]
        if rel is not None:
            sql += " AND rel = ?"
            params.append(rel)
        with self.db.read() as con:
            return sorted({r[0] for r in con.execute(sql + f" ORDER BY {other}", params)})

    def lineage(self, id: str, depth: int = 10) -> Graph:
        """Everything ``id`` depends on through refs, up to ``depth`` hops."""
        nodes = {id: self.kind_of(id)}
        edges: list[tuple[str, str, str]] = []
        frontier: deque[tuple[str, int]] = deque([(id, 0)])
        with self.db.read() as con:
            while frontier:
                node, level = frontier.popleft()
                if level >= depth:
                    continue
                rows = con.execute(
                    "SELECT e.rel, e.dst, r.kind FROM edges e JOIN records r ON r.id = e.dst "
                    "WHERE e.src = ? ORDER BY e.rel, e.dst",
                    (node,),
                ).fetchall()
                for rel, dst, kind in rows:
                    edges.append((node, rel, dst))
                    if dst not in nodes:
                        nodes[dst] = kind
                        frontier.append((dst, level + 1))
        return Graph(root=id, nodes=nodes, edges=edges)

    def status(self, id: str) -> tuple[EvidenceStatus, Assurance | None]:
        with self.db.read() as con:
            if not self._exists(con, id):
                raise RecordNotFound(f"record {id} is not in the store")
            return self._status(con, id)

    @staticmethod
    def _status(con: sqlite3.Connection, id: str) -> tuple[EvidenceStatus, Assurance | None]:
        rows = con.execute(
            "SELECT json_extract(r.json, '$.body') FROM edges e JOIN records r ON r.id = e.src "
            "WHERE e.dst = ? AND e.rel = 'subject' AND r.kind = 'StatusChange' "
            "ORDER BY r.created_at, r.rowid",
            (id,),
        ).fetchall()
        return status.fold(StatusChange.model_validate(json.loads(r[0])) for r in rows)

    def count(self, kind: str | None = None) -> int:
        with self.db.read() as con:
            if kind is None:
                return int(con.execute("SELECT count(*) FROM records").fetchone()[0])
            sql = "SELECT count(*) FROM records WHERE kind = ?"
            return int(con.execute(sql, (kind,)).fetchone()[0])
