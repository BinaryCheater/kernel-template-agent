"""What a fresh session of a work unit with history is told (D716), written from records.

A resumed session keeps its own context; a fresh one (context limit, abnormal exit, same
failure twice, force retry) would otherwise rediscover the case. The handoff lists the
case's family lineage with each version's delivery note, from the store, not from any
agent's memory.
"""

from __future__ import annotations

import re

from kta.core.schemas.process import TaskCard
from kta.store.base import Store

NOTE_CHARS = 600
MAX_VERSIONS = 12


def case_of(store: Store, card: TaskCard) -> str | None:
    """The card's Case: the header names it for case-loop tasks, else the first input."""
    if card.header is not None and card.header.case:
        return card.header.case
    return next((i for i in card.inputs if store.exists(i) and store.get(i).kind == "Case"), None)


def _done(store: Store, blobs: dict[str, str]) -> str:
    sha = blobs.get("files/DELIVERY.md")
    if not sha:
        return ""
    text = store.blob_path(sha).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^## Done\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    body = " ".join((m.group(1) if m else text).split())
    return body[:NOTE_CHARS] + ("…" if len(body) > NOTE_CHARS else "")


def handoff_text(store: Store, card: TaskCard) -> str:
    """One line per family version delivered for the card's case (empty when none)."""
    case = case_of(store, card)
    if case is None:
        return ""
    rows: list[tuple[str, str]] = []
    for rec in store.query("Family", limit=500):
        if rec.refs.get("case") != case:
            continue
        name = f"{rec.body['name']}@{rec.body['version']}"
        note = _done(store, rec.blobs)
        line = f"- {name} ({rec.created_at:%Y-%m-%d %H:%M} UTC)" + (f": {note}" if note else "")
        rows.append((rec.created_at.isoformat(), line))
    return "\n".join(line for _, line in sorted(rows)[-MAX_VERSIONS:])
