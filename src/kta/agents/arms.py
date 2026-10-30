"""Experiment arms: the knowledge condition of a step (docs/design/50 §1.2, 20 §5; D430).

The orchestrator writes the arm into the task bookkeeping (ops pointer ``task:<id>``,
``info["arm"]``, WP6). The agents module only reads it and shapes the step:

| knowledge | input records of pool kinds | ``inputs/knowledge/`` | find-knowledge | ``kta query`` |
| --- | --- | --- | --- | --- |
| ``zero`` | left out | none | left out | not allowed |
| ``text`` | left out | the snapshot rendered as Markdown | left out | not allowed |
| ``pool`` | as on the card | none | given | allowed |

Records of the task's own subject (Case, Problem, Contract, ...) are inputs in every arm;
"pool kinds" are the knowledge kinds that have cards (``knowledge.retrieve.CARD_KINDS``).
The condition is recorded in the context manifest and in ``AgentRun.arm``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kta.core.errors import ConfigError
from kta.knowledge.render import render_text_knowledge
from kta.knowledge.retrieve import CARD_KINDS
from kta.store.base import Ops, Store

KNOWLEDGE = ("zero", "text", "pool")
ALIASES = {"from_zero": "zero", "text_knowledge": "text", "template_pool": "pool"}
WITHOUT_POOL_SKILLS = frozenset({"find-knowledge"})
WITHOUT_POOL_COMMANDS = frozenset({"query"})
TEXT_DIR = "inputs/knowledge/"


@dataclass(frozen=True)
class Arm:
    name: str | None
    knowledge: str
    snapshot: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def uses_pool(self) -> bool:
        return self.knowledge == "pool"


NO_ARM = Arm(None, "pool", None)


def _knowledge(value: Any) -> str:
    k = ALIASES.get(str(value), str(value))
    if k not in KNOWLEDGE:
        raise ConfigError(f"arm knowledge condition {value!r} is not one of {KNOWLEDGE}")
    return k


def arm_of(info: Mapping[str, Any]) -> Arm:
    """The arm from task bookkeeping; no ``arm`` key means a normal (pool) run."""
    raw = info.get("arm")
    if raw is None:
        return NO_ARM
    if isinstance(raw, str):
        cond = info.get("arm_condition") or {}
        if not isinstance(cond, Mapping):
            raise ConfigError("task info arm_condition must be a mapping")
        name, knowledge, snapshot = raw, cond.get("knowledge"), cond.get("snapshot")
    elif isinstance(raw, Mapping):
        name, knowledge, snapshot = raw.get("name"), raw.get("knowledge"), raw.get("snapshot")
    else:
        raise ConfigError(f"task info arm must be a name or a mapping, got {type(raw).__name__}")
    if knowledge is None:
        raise ConfigError(f"arm {name!r} names no knowledge condition (zero, text, pool)")
    snapshot = snapshot or info.get("snapshot")
    arm = Arm(
        str(name) if name else None, _knowledge(knowledge), str(snapshot) if snapshot else None
    )
    if arm.knowledge == "text" and arm.snapshot is None:
        raise ConfigError(f"arm {arm.name!r}: text knowledge needs a pinned snapshot")
    return arm


def task_info(ops: Ops, task_id: str) -> dict[str, Any]:
    """The orchestrator's bookkeeping of a task (ops pointer ``task:<id>``, WP6)."""
    raw = ops.get_pointer(f"task:{task_id}")
    return json.loads(raw) if raw else {}


def skills(arm: Arm, names: list[str]) -> list[str]:
    return names if arm.uses_pool else [n for n in names if n not in WITHOUT_POOL_SKILLS]


def commands(arm: Arm, allowed: list[str]) -> list[str]:
    return allowed if arm.uses_pool else [c for c in allowed if c not in WITHOUT_POOL_COMMANDS]


def keeps_input(arm: Arm, kind: str) -> bool:
    return arm.uses_pool or kind not in CARD_KINDS


def text_book(store: Store, arm: Arm) -> dict[str, bytes]:
    """The snapshot rendered as Markdown (the deterministic renderer of WP4) for ``text``."""
    if arm.knowledge != "text":
        return {}
    with tempfile.TemporaryDirectory(prefix="kta-text-knowledge-") as scratch:
        render_text_knowledge(store, arm.snapshot, Path(scratch))
        root = Path(scratch)
        return {
            TEXT_DIR + p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
