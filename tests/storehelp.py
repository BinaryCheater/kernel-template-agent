"""Record builders shared by store tests (plain helpers, not test doubles)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kta.core.schemas import Actor, Record, RefValue
from kta.core.schemas.common import ActorType, Origin

TEST_COMMIT = "c0ffee" + "0" * 34
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def actor(kind: ActorType = "tool", name: str = "test") -> Actor:
    return Actor(type=kind, name=name)


def pattern_body(name: str, description: str = "sum over k with rescaling") -> dict[str, Any]:
    return {
        "name": name,
        "level": 1,
        "category": "reduce",
        "description": description,
        "examples": ["softmax denominator", "split-k partial sums"],
    }


def pattern(
    name: str,
    description: str = "sum over k with rescaling",
    origin: Origin = "real",
    who: Actor | None = None,
) -> Record:
    return Record.new("Pattern", pattern_body(name, description), who or actor(), T0, origin=origin)


def gap(description: str, refs: dict[str, RefValue] | None = None, n: int = 0) -> Record:
    body = {
        "missing": "template",
        "description": description,
        "attribution": "space",
        "level": 1 + n % 7,  # levels 1-7 (D441)
    }
    return Record.new("Gap", body, actor(), T0, refs=refs or {})


def status_change(
    subject: str,
    who: ActorType,
    to_status: str | None = None,
    to_assurance: str | None = None,
) -> Record:
    body: dict[str, Any] = {"subject": subject, "reason": "test"}
    if to_status:
        body["to_status"] = to_status
    if to_assurance:
        body["to_assurance"] = to_assurance
    return Record.new("StatusChange", body, actor(who), T0)
