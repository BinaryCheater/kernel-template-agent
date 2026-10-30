"""Move records and blobs between stores as a verified bundle (decision D153).

A bundle is JSON lines: a header, then blobs (base64), then records in dependency
order. Records travel verbatim: the ID is the hash of the content, so the importing
store recomputes every ID and blob sha256 and refuses any mismatch. The actor keeps
the commit and dirty flag of the code that produced the record on the worker.

This is how records produced on a worker reach the control-plane store before the
Executor (WP2) returns job results itself.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from kta.core.errors import SchemaError
from kta.core.ids import record_id, sha256_hex
from kta.core.schemas import Record
from kta.store.sqlite import SqliteStore

FORMAT = "kta-bundle"
VERSION = 1


def closure(store: SqliteStore, ids: list[str]) -> list[str]:
    """``ids`` and everything they reference, dependencies first."""
    ordered: list[str] = []
    done: set[str] = set()

    def visit(rid: str, stack: frozenset[str]) -> None:
        if rid in done:
            return
        if rid in stack:
            raise SchemaError(f"reference cycle through {rid}")
        for dep in store.neighbors(rid, direction="out"):
            visit(dep, stack | {rid})
        done.add(rid)
        ordered.append(rid)

    for rid in ids:
        visit(rid, frozenset())
    return ordered


def export_bundle(store: SqliteStore, ids: list[str], out: Path) -> dict[str, Any]:
    records = [store.get_json(rid) for rid in closure(store, ids)]
    blob_ids = sorted({sha for r in records for sha in r.get("blobs", {}).values()})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", "format": FORMAT, "version": VERSION}) + "\n")
        for sha in blob_ids:
            data = store.blob_path(sha).read_bytes()
            line = {
                "type": "blob",
                "sha256": sha,
                "media": store.blob_media(sha),
                "data_b64": base64.b64encode(data).decode("ascii"),
            }
            f.write(json.dumps(line) + "\n")
        for rec in records:
            line = {"type": "record", "id": record_id(rec), "record": rec}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return {"records": [record_id(r) for r in records], "blobs": blob_ids, "path": str(out)}


def import_bundle(store: SqliteStore, path: Path) -> dict[str, Any]:
    """Verify and write every blob and record of a bundle; returns their IDs."""
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines or lines[0].get("format") != FORMAT or lines[0].get("version") != VERSION:
        raise SchemaError(f"{path} is not a {FORMAT} v{VERSION} file")
    blobs: list[str] = []
    records: list[Record] = []
    expected: list[str] = []
    for line in lines[1:]:
        if line["type"] == "blob":
            data = base64.b64decode(line["data_b64"])
            if sha256_hex(data) != line["sha256"]:
                raise SchemaError(f"blob {line['sha256'][:12]} content does not match its hash")
            blobs.append(store.put_blob(data, line["media"]))
        elif line["type"] == "record":
            rec = Record.model_validate(line["record"])
            # Check both the raw and the re-validated form, before anything is written.
            if record_id(line["record"]) != line["id"] or record_id(rec) != line["id"]:
                raise SchemaError(f"record {line['id'][:12]} content does not match its ID")
            if rec.actor.commit is None:
                raise SchemaError(f"record {line['id'][:12]} has no actor.commit")
            records.append(rec)
            expected.append(line["id"])
        else:
            raise SchemaError(f"unknown bundle line type {line['type']!r}")
    written = store.put_many(records)
    if written != expected:
        raise SchemaError("imported record IDs differ from the bundle (normalization changed)")
    return {"records": written, "blobs": blobs}
