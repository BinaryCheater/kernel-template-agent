"""Read the index and its four layers from the repository or a frozen package."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.json"
BUNDLE = HERE / "bundle.json.gz"
LAYERS = {"search": 1, "card": 1, "guide": 2, "template": 3, "proof": 3, "realisation": 4}


class Library:
    """Layer text comes from a bundle when one exists, so a package needs no library copy."""

    def __init__(self, exclude_source: str | None = None) -> None:
        if not INDEX.exists():
            from .build import write_index  # A frozen package always ships its index.

            write_index()
        index = json.loads(INDEX.read_text())
        self.lexicon = index["lexicon"]
        self.fingerprint = index["library_fingerprint"]
        entries = index["entries"]
        if exclude_source:
            entries = [e for e in entries if exclude_source not in lineage(e)]
        self.entries = entries
        self.by_identity = {e["identity"]: e for e in entries}
        self.bundle = json.loads(gzip.open(BUNDLE).read()) if BUNDLE.exists() else None
        self.root = HERE.parents[1]

    def entry(self, identity: str) -> dict:
        if identity not in self.by_identity:
            raise SystemExit(f"unknown identity: {identity}; identities look like name@version")
        return self.by_identity[identity]

    def text(self, identity: str, layer: str) -> str:
        entry = self.entry(identity)
        if self.bundle is not None:
            return self.bundle[identity][layer]
        return (self.root / entry["paths"][layer]).read_text()

    def realisations(self, identity: str) -> dict[str, str]:
        entry = self.entry(identity)
        if self.bundle is not None:
            return self.bundle[identity]["realisation"]
        return {p: (self.root / p).read_text() for p in entry["paths"]["realisation"]}


def lineage(entry: dict) -> set[str]:
    origin = entry["origin"]
    return {origin["study"], *origin["lineage"]}


class ReadLog:
    """Append-only JSON lines; one row per read, per layer. Tools never rewrite it."""

    def __init__(self, path: Path, session: str) -> None:
        self.path = path
        self.session = session

    def write(self, event: str, *, text: str = "", **fields: object) -> None:
        count = 0
        if self.path.exists():
            with self.path.open("rb") as handle:
                count = sum(1 for _ in handle)
        body = text.encode()
        row = {
            "seq": count,
            "ts": datetime.now(UTC).isoformat(),
            "session": self.session,
            "event": event,
            "layer": LAYERS.get(event),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest() if body else None,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def default_log() -> Path:
    return Path(os.environ.get("KTA_ACCESS_LOG", "access-log.jsonl"))


def default_session() -> str:
    return os.environ.get("KTA_ACCESS_SESSION", "unnamed")
