"""The WP10b gate checks: named tests, expand transitions, done-checks, and the full-stage
evidence of a real expand loop."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import yaml

from kta.core.config import repo_root
from kta_gate.specific import stage
from kta_gate.specific_wp10b import MACHINES, WP10B_TESTS, _covered_reports, wp10b

PASSED = {f"tests.unit.test_x::{t}" for t in WP10B_TESTS}


def _root(tmp: Path, references: int = 7) -> Path:
    (tmp / MACHINES).parent.mkdir(parents=True)
    shutil.copy(repo_root() / MACHINES, tmp / MACHINES)
    for i in range(references):
        (tmp / "knowledge" / "seed" / "references" / f"r{i}").mkdir(parents=True)
    return tmp


def test_code_stage_passes_and_leaves_the_real_loop_pending(tmp_path: Path) -> None:
    with stage("code"):
        results = wp10b(_root(tmp_path), PASSED)
    assert [r.status for r in results] == ["PASS", "PASS", "PASS", "WARN"], [
        r.render() for r in results
    ]


def test_a_missing_test_fails(tmp_path: Path) -> None:
    with stage("code"):
        check = wp10b(_root(tmp_path), PASSED - {next(iter(sorted(PASSED)))})[0]
    assert check.status == "FAIL"


def test_an_expand_edge_creating_t4_fails(tmp_path: Path) -> None:
    root = _root(tmp_path)
    doc = yaml.safe_load((root / MACHINES).read_text(encoding="utf-8"))
    expand = next(m for m in doc["machines"] if m["id"] == "expand")
    expand["transitions"][1]["creates"] = ["T4"]
    (root / MACHINES).write_text(yaml.safe_dump(doc), encoding="utf-8")
    with stage("code"):
        assert wp10b(root, PASSED)[1].status == "FAIL"


def _con(body: dict[str, object], commit: str = "abc") -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE records (id TEXT, origin TEXT, json TEXT)")
    rec = {"kind": "Report", "body": body, "actor": {"commit": commit, "dirty": False}}
    con.execute("INSERT INTO records VALUES ('R1', 'real', ?)", (json.dumps(rec),))
    return con


def test_full_stage_finds_a_listed_coverage_report_with_a_covered_loop() -> None:
    body = {"title": "coverage", "data": {"references": [{"name": "x", "state": "covered"}]}}
    assert _covered_reports(_con(body), {"R1"}, lambda c: c == "abc") == []
    assert _covered_reports(_con(body), set(), lambda c: True) != []  # not listed
    assert _covered_reports(_con(body, "zzz"), {"R1"}, lambda c: c == "abc") != []  # not clean
    blocked = {"title": "coverage", "data": {"references": [{"name": "x", "state": "frozen"}]}}
    assert _covered_reports(_con(blocked), {"R1"}, lambda c: True) != []


def test_full_stage_needs_three_new_references(tmp_path: Path) -> None:
    results = wp10b(_root(tmp_path), PASSED, data_root=tmp_path / "none")
    assert results[3].status == "FAIL" and "need at least 10" in results[3].findings[0]
