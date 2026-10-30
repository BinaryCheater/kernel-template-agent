"""Structural checks on src/kta/orchestrator/machines.yaml (D96).

WP6 adds the consistency test between `step` and this definition; this file only keeps the
definition self-consistent so the frontend can draw it.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest
import yaml

import kta.orchestrator

MACHINES = Path(kta.orchestrator.__file__).parent / "machines.yaml"
STATE_KINDS = {"step", "wait", "terminal"}
TRANSITION_KINDS = {"normal", "creates_task", "resume", "retry", "human", "any"}


def _load() -> dict[str, Any]:
    return yaml.safe_load(MACHINES.read_text(encoding="utf-8"))


def _machines() -> list[dict[str, Any]]:
    return _load()["machines"]


def test_expected_machines_present() -> None:
    assert [m["id"] for m in _machines()] == ["solve", "expand", "task"]


@pytest.mark.parametrize("machine", _machines(), ids=lambda m: m["id"])
def test_machine_is_well_formed(machine: dict[str, Any]) -> None:
    task_ids = {t["id"] for t in _load()["task_types"]}
    states = {s["id"]: s for s in machine["states"]}
    assert len(states) == len(machine["states"]), "duplicate state id"
    assert all(s["kind"] in STATE_KINDS for s in states.values())
    assert all(s.get("does") for s in states.values())
    initial = [s["id"] for s in states.values() if s.get("initial")]
    assert len(initial) == 1

    edges: dict[str, set[str]] = {sid: set() for sid in states}
    for t in machine["transitions"]:
        assert t["kind"] in TRANSITION_KINDS
        assert len(t.get("label", "")) <= 16, t
        assert t["to"] in states, t
        assert t["from"] == "*" or t["from"] in states, t
        assert (t["kind"] == "creates_task") == bool(t.get("creates")), t
        assert set(t.get("creates", [])) <= task_ids, t
        if t["from"] != "*":
            assert states[t["from"]]["kind"] != "terminal", t
            edges[t["from"]].add(t["to"])

    # Every state is reachable from the initial state ("*" edges reach their target from any).
    any_targets = {t["to"] for t in machine["transitions"] if t["from"] == "*"}
    seen = {initial[0]} | any_targets
    queue = deque(seen)
    while queue:
        for nxt in edges[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    assert seen == set(states)


def test_judging_rules_are_ordered_one_to_four() -> None:
    solve = next(m for m in _machines() if m["id"] == "solve")
    rules = [t["rule"] for t in solve["transitions"] if t["from"] == "judging"]
    assert rules == [1, 2, 3, 4]
