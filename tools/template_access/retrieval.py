"""Deterministic fielded BM25 over index entries. Standard library only."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOP = frozenset(
    """a an and are as at be been but by can do does each for from has have if in into is it its
    may more most no not of on one only or other same so such than that the their then there these
    this to two use used uses using via was were when where which while will with without""".split()
)

K1, B = 1.2, 0.75

# Flat is the plain card text the earlier experiment offered. The access weights were
# fixed once, before any gold set existed; they are not fitted (see results note).
ACCESS_WEIGHTS = {
    "name": 3.0,
    "card": 2.0,
    "pattern": 1.5,
    "transformation": 1.0,
    "resource": 1.0,
    "symptom": 1.0,
}
FLAT_WEIGHTS = {"flat": 1.0}


def stem(word: str) -> str:
    """Conflate inflections only; a heavier stemmer would merge unrelated hardware words."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    for suffix in ("ization", "isation", "ation", "ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[: -len(suffix)]
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOP and len(w) > 1]


def document_fields(entry: dict) -> dict[str, str]:
    """Split one entry into searchable fields. 'flat' is exactly the catalog card."""
    card = entry["card"]
    card_text = " ".join((card["when"], card["choice"], card["boundary"]))
    return {
        "flat": " ".join((entry["identity"], entry["title"], entry["category"], card_text)),
        "name": entry["title"],
        "card": card_text,
        "pattern": " ".join(
            [entry["category"], entry["pattern"]["refines"], *entry["pattern"]["produces"]]
        ),
        "transformation": " ".join(entry["transformation"].values()),
        "resource": " ".join([*entry["facets"]["resource"], entry["resource_text"]]),
        "symptom": " ".join([*entry["facets"]["symptom"], entry["summary"], entry["guide_when"]]),
    }


class Ranker:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries
        self.tf: dict[str, list[Counter]] = defaultdict(list)
        self.length: dict[str, list[int]] = defaultdict(list)
        for entry in entries:
            for field, text in document_fields(entry).items():
                bag = Counter(tokens(text))
                self.tf[field].append(bag)
                self.length[field].append(sum(bag.values()))
        self.df = {
            field: Counter(term for bag in bags for term in bag) for field, bags in self.tf.items()
        }

    def _field(self, field: str, terms: Counter) -> list[float]:
        n = len(self.entries)
        lengths = self.length[field]
        mean = (sum(lengths) / n) if n else 1.0
        scores = [0.0] * n
        for term, query_count in terms.items():
            df = self.df[field].get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, bag in enumerate(self.tf[field]):
                tf = bag.get(term, 0)
                if tf:
                    norm = tf + K1 * (1 - B + B * lengths[i] / (mean or 1.0))
                    scores[i] += query_count * idf * tf * (K1 + 1) / norm
        return scores

    def rank(self, query: str, weights: dict[str, float], allowed: set[int] | None = None) -> list:
        """Return (entry, score) by descending score; ties break on identity for repeatability."""
        terms = Counter(tokens(query))
        total = [0.0] * len(self.entries)
        for field, weight in weights.items():
            for i, value in enumerate(self._field(field, terms)):
                total[i] += weight * value
        rows = [
            (self.entries[i], s)
            for i, s in enumerate(total)
            if s > 0 and (allowed is None or i in allowed)
        ]
        return sorted(rows, key=lambda row: (-row[1], row[0]["identity"]))
