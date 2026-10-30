"""A read-only view of a Store plus records not yet written.

Seed import validates a whole batch (Templates whose relations point at each other,
Chains over Templates of the same batch) before writing anything, so a rejected batch
leaves the append-only store untouched (decision D204).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from kta.core.errors import NotSupported, PermissionDenied
from kta.core.ids import record_id
from kta.core.schemas import Assurance, EvidenceStatus, Record
from kta.store.base import Direction, Graph, Hit, Store


def _get_path(body: Mapping[str, Any], path: str) -> Any:
    node: Any = body
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]  # pyright: ignore[reportUnknownVariableType]
    return node  # pyright: ignore[reportUnknownVariableType]


class PendingView:
    """Implements the read side of ``Store``; pending records are ``proposed``."""

    def __init__(self, base: Store, pending: Iterable[Record]) -> None:
        self.base = base
        self.pending = {record_id(r): r for r in pending}

    def put(self, record: Record) -> str:
        raise PermissionDenied("PendingView is read-only")

    def put_many(self, records: Iterable[Record]) -> list[str]:
        raise PermissionDenied("PendingView is read-only")

    def put_blob(self, data: bytes | Path, media: str) -> str:
        raise PermissionDenied("PendingView is read-only")

    def blob_path(self, sha256: str) -> Path:
        return self.base.blob_path(sha256)

    def get(self, id: str) -> Record:
        return self.pending[id] if id in self.pending else self.base.get(id)

    def exists(self, id: str) -> bool:
        return id in self.pending or self.base.exists(id)

    def query(
        self,
        kind: str,
        where: Mapping[str, Any] | None = None,
        order: str | None = None,
        limit: int = 100,
    ) -> list[Record]:
        if order is not None or any(isinstance(v, Mapping) for v in (where or {}).values()):
            raise NotSupported("PendingView.query supports equality filters only")
        found = self.base.query(kind, where=where, limit=limit)
        for rec in self.pending.values():
            if rec.kind == kind and all(
                _get_path(rec.body, k) == v for k, v in (where or {}).items()
            ):
                found.append(rec)
        return found[:limit]

    def search(self, text: str, kinds: list[str] | None = None, limit: int = 10) -> list[Hit]:
        raise NotSupported("PendingView does not search pending records")

    def neighbors(self, id: str, rel: str | None = None, direction: Direction = "out") -> list[str]:
        out = set(self.base.neighbors(id, rel, direction)) if self.base.exists(id) else set()
        for rid, rec in self.pending.items():
            for key, dst in rec.ref_ids():
                if rel is not None and key != rel:
                    continue
                if direction == "out" and rid == id:
                    out.add(dst)
                if direction == "in" and dst == id:
                    out.add(rid)
        return sorted(out)

    def lineage(self, id: str, depth: int = 10) -> Graph:
        raise NotSupported("PendingView does not compute lineage")

    def status(self, id: str) -> tuple[EvidenceStatus, Assurance | None]:
        return ("proposed", None) if id in self.pending else self.base.status(id)


def as_store(view: PendingView) -> Store:
    return view
