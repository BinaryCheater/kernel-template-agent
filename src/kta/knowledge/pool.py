"""Read access to the knowledge pool in the Store (records by name, status, references).

Knowledge objects are records; a Template is named ``<name>@<version>`` (spec/schemas.md
§4.3). Everything here reads through the ``Store`` protocol only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kta.core.errors import SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Record
from kta.core.schemas.knowledge import Template
from kta.store.assurance import effective
from kta.store.base import Store
from kta.store.querybuild import MAX_LIMIT
from kta.store.sqlite import SqliteStore

# Store.query needs a limit; the knowledge pool is thousands of records (docs/design/23).
ALL = MAX_LIMIT
# Objects in these states are kept for history but never offered for use (21 §1.1).
UNUSABLE = frozenset({"rejected", "deprecated", "stale"})


def records(
    store: Store, kind: str, where: Mapping[str, Any] | None = None, order: str | None = None
) -> list[Record]:
    """All matching records; ``order`` ties keep insertion order (D147)."""
    return store.query(kind, where=where, order=order, limit=ALL)


def with_ids(recs: list[Record]) -> list[tuple[str, Record]]:
    return [(record_id(r), r) for r in recs]


def template_ref(name: str, version: int) -> str:
    return f"{name}@{version}"


def parse_template_ref(ref: str) -> tuple[str, int]:
    name, sep, version = ref.partition("@")
    if not sep or not name or not version.isdigit():
        raise SchemaError(f"template reference {ref!r} is not <name>@<version>")
    return name, int(version)


def find_template(store: Store, ref: str) -> tuple[str, Template] | None:
    """The Template record for ``name@version``; the newest when several bodies exist."""
    name, version = parse_template_ref(ref)
    found = records(store, "Template", {"name": name, "version": version})
    if not found:
        return None
    rec = found[-1]
    return record_id(rec), Template.model_validate(rec.body)


def templates(store: Store) -> list[tuple[str, Template]]:
    return [
        (rid, Template.model_validate(r.body)) for rid, r in with_ids(records(store, "Template"))
    ]


def names(store: Store, kind: str) -> set[str]:
    return {str(r.body["name"]) for r in records(store, kind)}


def find_by_name(store: Store, kind: str, name: str) -> tuple[str, Record] | None:
    found = records(store, kind, {"name": name})
    return (record_id(found[-1]), found[-1]) if found else None


def status_of(store: Store, rid: str) -> tuple[str, str]:
    """(status, assurance); no assurance StatusChange means ``declared`` (D203). On the
    SQLite store bounded / proved come only from live proof records (WP24, D634)."""
    status, assurance = store.status(rid)
    if isinstance(store, SqliteStore):
        return status, effective(store, rid)
    return status, assurance or "declared"


def usable(store: Store, rid: str) -> bool:
    return status_of(store, rid)[0] not in UNUSABLE


def chain_template_refs(chain_body: Mapping[str, Any]) -> list[str]:
    return [str(item["template"]) for item in chain_body.get("items", [])]
