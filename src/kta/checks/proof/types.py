"""In-memory results of the checker before they become ProofResult records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kta.core.schemas.proof import Query

# The ordering used to fold outcomes into one level: lower is weaker.
RANK = {"counterexample": 0, "unresolved": 1, "bounded": 2, "proved": 3}


@dataclass
class Draft:
    condition: str  # K1 | K2 | K3 | composition | candidate | wp
    obligation: str
    outcome: str  # proved | bounded | counterexample | unresolved
    domain: str
    queries: list[Query] = field(default_factory=lambda: list[Query]())
    seconds: float = 0.0
    counterexample: dict[str, Any] | None = None
    replay: dict[str, Any] | None = None
    reason: str = ""
    data: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    # premise groups of the Template this result rests on (paper §4.4: named groups)
    groups: tuple[str, ...] = ()

    def add(self, q: Query) -> None:
        self.queries.append(q)
        self.seconds += q.seconds


def weakest(outcomes: list[str]) -> str:
    return min(outcomes, key=lambda o: RANK[o]) if outcomes else "unresolved"


def cap(outcome: str, ceiling: str) -> str:
    """``proved`` lowered to ``ceiling`` (bounded) when the method or domain says so."""
    return ceiling if outcome == "proved" and RANK[ceiling] < RANK["proved"] else outcome
