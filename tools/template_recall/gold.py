"""Load tasks and gold moves, and decide what each move can match in a library view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/runbooks/learndemo-1/template-recall"
TASKS = ("attention_causal_fp16", "int4_weight_gemm", "paged_batch_decode", "grouped_fp8_gemm")


@dataclass(frozen=True)
class Move:
    id: str
    role: str
    need: str
    source_rule: str
    accept: frozenset[str]
    equivalent: frozenset[str]

    def targets(self, scoring: str, view: set[str]) -> set[str]:
        """Identities that count as this move in a view; empty means the view lacks it."""
        wanted = self.accept if scoring == "strict" else self.accept | self.equivalent
        return set(wanted) & view


@dataclass(frozen=True)
class Task:
    id: str
    source_study: str
    statement: str
    moves: tuple[Move, ...]


def load_task(task_id: str) -> Task:
    gold = yaml.safe_load((BASE / "gold" / f"{task_id}.yaml").read_text())
    moves = tuple(
        Move(
            id=m["id"],
            role=m["role"],
            need=m["need"],
            source_rule=m["source_rule"],
            accept=frozenset(m["accept"]),
            equivalent=frozenset(m["equivalent"]),
        )
        for m in gold["moves"]
    )
    statement = (BASE / "tasks" / f"{task_id}.md").read_text()
    return Task(task_id, gold["source_study"], statement, moves)


def load_tasks() -> list[Task]:
    return [load_task(t) for t in TASKS]
