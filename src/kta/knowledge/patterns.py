"""Pattern table: load from the Store, match a described transaction (docs/design/22 §2.2).

The table is data: no Pattern name is hard-coded except ``other``, the fallback the
schema itself defines (spec/schemas.md §4.1).
"""

from __future__ import annotations

import re

from kta.core.schemas.knowledge import Pattern
from kta.knowledge import pool
from kta.knowledge.types import TxDescription
from kta.store.base import Store

OTHER = "other"
MATCH_THRESHOLD = 0.3
_WORD = re.compile(r"[a-z0-9]+")
# Words that say nothing about the kind of transaction.
_STOP = frozenset(
    "a an and the of to in into on for with by from per each one is are be as at or it its "
    "this that then than over under".split()
)


def load_patterns(store: Store) -> dict[str, Pattern]:
    """Every Pattern in the store that is not rejected, deprecated or stale, by name."""
    out: dict[str, Pattern] = {}
    for rid, rec in pool.with_ids(pool.records(store, "Pattern")):
        if pool.usable(store, rid):
            out[str(rec.body["name"])] = Pattern.model_validate(rec.body)
    return out


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _score(tx: TxDescription, p: Pattern, tx_words: set[str]) -> float:
    score = 0.0
    if tx.category is not None and tx.category == p.category:
        score += 0.5
    if tx.level is not None:
        score += {0: 0.2, 1: 0.1}.get(abs(tx.level - p.level), 0.0)
    if tx.params:
        known = {s.name for s in p.params}
        score += 0.2 * len(known & set(tx.params)) / len(tx.params)
    if p.name in tx_words or p.name.replace("_", " ") in tx.text.lower():
        score += 0.3
    vocab = _words(" ".join([p.name, p.description, *p.examples, *(s.name for s in p.params)]))
    if tx_words:
        score += 0.3 * len(tx_words & vocab) / len(tx_words)
    return min(score, 1.0)


def match_pattern(tx: TxDescription, patterns: dict[str, Pattern]) -> list[tuple[str, float]]:
    """Candidate Patterns with a match score in [0, 1], best first; ``[("other", 0)]`` if none.

    The score is a deterministic heuristic (decision D208): schema category, level
    distance, shared parameter names, the Pattern's name in the text and word overlap
    with its description and examples. It ranks candidates for a human or agent to
    confirm; it is not a classification proof.
    """
    tx_words = _words(tx.text)
    scored = [
        (name, round(_score(tx, p, tx_words), 4)) for name, p in patterns.items() if name != OTHER
    ]
    hits = [(n, s) for n, s in scored if s >= MATCH_THRESHOLD]
    if not hits:
        return [(OTHER, 0.0)]
    return sorted(hits, key=lambda h: (-h[1], h[0]))
