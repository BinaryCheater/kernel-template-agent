"""The WP9b gate: named API tests, operation coverage, code-stage WARN, full-stage records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from kta_gate.result import CheckResult
from kta_gate.specific import stage
from kta_gate.specific_wp9b import WP9B_TESTS, real_operations, wp9b

ROOT = Path(__file__).resolve().parents[2]


def _built(_: Path) -> CheckResult:
    return CheckResult("web build", "PASS", "not run in this unit test")


def _passed() -> set[str]:
    return {f"tests/unit/x.py::{t}" for t in WP9B_TESTS} | {
        "t.py::test_reject_and_deprecate_need_a_reason[a-b]"
    }


def test_code_stage_passes_with_tests_and_warns_on_real_runs() -> None:
    with stage("code"):
        results = {r.name: r.status for r in wp9b(ROOT, _passed(), build=_built)}
    assert results["WP9b API tests"] == "PASS"
    assert results["operations tested"] == "PASS"
    assert results["no sample data in web/"] == "PASS"
    assert results["WP9b real operations"] == "WARN"


def test_a_missing_api_test_fails() -> None:
    passed = {t for t in _passed() if "budget" not in t}
    with stage("code"):
        results = {r.name: r for r in wp9b(ROOT, passed, build=_built)}
    assert results["WP9b API tests"].status == "FAIL"
    assert "test_budget_needs_reason_and_merges_limits" in results["WP9b API tests"].render()


def _db(path: Path, rows: list[tuple[str, str, dict[str, object]]]) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE records (id TEXT, kind TEXT, origin TEXT, json TEXT)")
    for rid, origin, rec in rows:
        con.execute(
            "INSERT INTO records VALUES (?, 'HumanAction', ?, ?)", (rid, origin, json.dumps(rec))
        )
    return con


def _action(action: str, dirty: bool = False) -> dict[str, object]:
    return {
        "actor": {"type": "human", "name": "user", "commit": "c1", "dirty": dirty},
        "body": {"action": action, "reason": "r"},
    }


def test_real_operations_need_all_three(tmp_path: Path) -> None:
    con = _db(
        tmp_path / "a.db",
        [
            ("a", "real", _action('reject {"kind":"Solution"}')),
            ("b", "real", _action("task cancel 01X")),
            ("c", "fixture", _action('create_task {"type":"T1"}')),
            ("d", "real", _action('create_task {"type":"T1"}', dirty=True)),
        ],
    )
    findings = real_operations(con, {"a", "b", "c", "d"}, lambda c: c == "c1")
    assert findings == ["no listed, real, clean HumanAction to start a task"]
    con.execute(
        "INSERT INTO records VALUES ('e', 'HumanAction', 'real', ?)",
        (json.dumps(_action("create_task")),),
    )
    assert real_operations(con, {"a", "b", "e"}, lambda c: c == "c1") == []
    assert real_operations(con, {"a", "b", "e"}, lambda c: False) != []


def test_full_stage_fails_without_the_dev_store(tmp_path: Path) -> None:
    results = {r.name: r.status for r in wp9b(ROOT, _passed(), data_root=tmp_path, build=_built)}
    assert results["WP9b real operations"] == "FAIL"
