"""Status and assurance derivation and the promotion rules (spec/schemas.md §4.9).

- -> proposed: any actor while the object is proposed; a human from any status.
- proposed / stale / a lower promotion -> reproduced, calibrated, holdout_validated: checker.
- any -> rejected / deprecated / suspect: human.
- any -> stale: system.
- a promotion level -> a lower one: system (demotion by counter-evidence, D448).
- assurance upward (declared < tested < bounded < proved): checker.

Interpretation of the spec table (decision D149): checkers may also re-promote stale
objects (re-verification) and move forward between promotion levels; assurance never
moves down. With no StatusChange the status is ``proposed`` and the assurance ``None``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import get_args

from kta.core.errors import PermissionDenied
from kta.core.schemas import Assurance, EvidenceStatus
from kta.core.schemas.common import ActorType
from kta.core.schemas.process import StatusChange

PROMOTION_RANK: dict[str, int] = {
    "proposed": 0,
    "reproduced": 1,
    "calibrated": 2,
    "holdout_validated": 3,
}
HUMAN_ONLY = {"rejected", "deprecated", "suspect"}
ASSURANCE_RANK: dict[str, int] = {a: i for i, a in enumerate(get_args(Assurance))}
DEFAULT_STATUS: EvidenceStatus = "proposed"


def fold(changes: Iterable[StatusChange]) -> tuple[EvidenceStatus, Assurance | None]:
    """Current (status, assurance) from changes in write order (all already validated)."""
    status: EvidenceStatus = DEFAULT_STATUS
    assurance: Assurance | None = None
    for change in changes:
        if change.to_status is not None:
            status = change.to_status
        if change.to_assurance is not None:
            assurance = change.to_assurance
    return status, assurance


def _status_allowed(current: EvidenceStatus, target: EvidenceStatus, actor: ActorType) -> bool:
    if target == "proposed":
        demotion = actor == "system" and PROMOTION_RANK.get(current, 0) > 0
        return actor == "human" or current == "proposed" or demotion
    if target in PROMOTION_RANK:
        if actor == "system":
            # Demotion (D448): only down the ladder, never up and never out of stale.
            return current in PROMOTION_RANK and PROMOTION_RANK[target] < PROMOTION_RANK[current]
        if actor != "checker":
            return False
        if current == "stale":
            return True
        return current in PROMOTION_RANK and PROMOTION_RANK[current] < PROMOTION_RANK[target]
    if target in HUMAN_ONLY:
        return actor == "human"
    if target == "stale":
        return actor == "system"
    return False


def check_change(
    current: tuple[EvidenceStatus, Assurance | None], change: StatusChange, actor: ActorType
) -> None:
    """Raise PermissionDenied when ``actor`` may not make ``change`` from ``current``."""
    status, assurance = current
    if change.to_status is not None and not _status_allowed(status, change.to_status, actor):
        raise PermissionDenied(
            f"{actor} may not change status {status} -> {change.to_status} "
            f"(subject {change.subject[:12]})"
        )
    if change.to_assurance is not None:
        have = ASSURANCE_RANK[assurance or "declared"]
        want = ASSURANCE_RANK[change.to_assurance]
        if actor != "checker" or want <= have:
            raise PermissionDenied(
                f"{actor} may not change assurance {assurance or 'declared'} -> "
                f"{change.to_assurance} (only checkers, only upward)"
            )
