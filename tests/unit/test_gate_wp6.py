"""The WP6 gate check: transition tests and one test per judging rule (D275)."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from kta.core.config import repo_root
from kta_gate.specific import WP6_MACHINES, WP6_TESTS, wp6

RULE_TESTS = {f"test_judging_rule_{n}_x" for n in (1, 2, 3, 4)}
PASSED = {f"tests.unit.test_orchestrator_step::{t}" for t in (*WP6_TESTS, *RULE_TESTS)}


def _root(tmp: Path) -> Path:
    (tmp / WP6_MACHINES).parent.mkdir(parents=True)
    shutil.copy(repo_root() / WP6_MACHINES, tmp / WP6_MACHINES)
    return tmp


def test_gate_wp6_passes_with_all_tests(tmp_path: Path) -> None:
    results = wp6(_root(tmp_path), PASSED)
    assert [r.status for r in results] == ["PASS", "PASS"], [r.render() for r in results]


def test_gate_wp6_fails_when_a_rule_is_untested(tmp_path: Path) -> None:
    passed = {t for t in PASSED if "rule_2" not in t}
    check = wp6(_root(tmp_path), passed)[1]
    assert check.status == "FAIL" and any("rule 2" in f for f in check.findings)


def test_gate_wp6_fails_when_a_new_rule_appears(tmp_path: Path) -> None:
    root = _root(tmp_path)
    doc = yaml.safe_load((root / WP6_MACHINES).read_text(encoding="utf-8"))
    solve = next(m for m in doc["machines"] if m["id"] == "solve")
    solve["transitions"].append({"from": "judging", "to": "accepted", "kind": "normal", "rule": 5})
    (root / WP6_MACHINES).write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert wp6(root, PASSED)[1].status == "FAIL"


def test_gate_wp6_fails_without_transition_tests(tmp_path: Path) -> None:
    assert wp6(_root(tmp_path), RULE_TESTS)[0].status == "FAIL"
