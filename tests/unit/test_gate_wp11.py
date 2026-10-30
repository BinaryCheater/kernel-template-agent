"""The WP11 gate checks: named tests, processes, skills, afternoon records and spans."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import timedelta
from pathlib import Path

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.core.trace import SpanStart
from kta.store import CodeState, open_ops, open_store
from kta_gate import specific
from kta_gate.specific_wp11 import (
    LANES,
    WP11_TESTS,
    check_processes,
    check_skills,
    check_tests,
    record_findings,
    wp11,
)
from storehelp import T0

MANIFEST = {
    "role": "worker",
    "host": "spark",
    "os": "linux",
    "arch": "arm64",
    "python": "3.12.13",
    "uv": "0.11.2",
    "uv_lock_sha256": "x",
    "commit": "c" * 40,
    "dirty": False,
}


def _world(
    ctx: Context, tmp: Path, *, lanes: tuple[str, ...] = LANES, apart: bool = False
) -> tuple[Path, set[str]]:
    data = tmp / "dev" / "WP11"
    c = dataclasses.replace(ctx, data_dir=data)
    store, ops = open_store(c, CodeState("c" * 40, False)), open_ops(c)
    sys = Actor(type="system", name="kta.orchestrator (calibrate)")
    put = lambda kind, body, **refs: store.put(Record.new(kind, body, sys, T0, refs=refs))  # noqa: E731
    env = put("EnvManifest", MANIFEST)
    fact = {"unit": "l2", "quantity": "l2_effective_bytes", "value": 1.0, "method": "measured"}
    listed = {
        put("HwFact", fact, measurements=[env]),
        put("ExperimentRun", {"name": "e", "condition": "pool", "note": "n"}),
        put("TaskCard", {"type": "T4", "goal": "g", "profile": "p", "work_unit": "w"}),
        put("Report", {"title": "improvement", "subject": "b", "summary": "s"}),
    }
    for i, lane in enumerate(lanes):
        start = T0 + timedelta(hours=10 * i if apart else i)
        ops.start_span(SpanStart(f"S{i}", None, "t", "process_action", lane, "x", start, {}))
    return data, listed


def test_wp11_records_and_afternoon_pass(ctx: Context, tmp_path: Path) -> None:
    data, listed = _world(ctx, tmp_path)
    con = sqlite3.connect(data / "kta.db")
    assert record_findings(con, listed) == []
    assert record_findings(con, set())[0].startswith("no listed, real, clean microbenchmark")


def test_wp11_afternoon_needs_overlapping_lanes(ctx: Context, tmp_path: Path) -> None:
    data, listed = _world(ctx, tmp_path / "a", apart=True)
    found = record_findings(sqlite3.connect(data / "kta.db"), listed)
    assert found and "window" in found[-1]
    data, listed = _world(ctx, tmp_path / "b", lanes=LANES[:3])
    assert "window" in record_findings(sqlite3.connect(data / "kta.db"), listed)[-1]


def test_wp11_static_checks(tmp_path: Path) -> None:
    root = repo_root()
    assert check_processes().status == "PASS"
    assert check_skills(root).status == "PASS"
    assert check_tests(set()).status == "FAIL"
    (tmp_path / "src" / "kta" / "agents" / "skills").mkdir(parents=True)
    (tmp_path / "src" / "kta" / "agents" / "skills" / "README.md").write_text("x")
    assert check_skills(tmp_path).status == "FAIL"


def test_wp11_code_stage_leaves_the_afternoon_pending() -> None:
    passed = {f"tests/unit/x.py::{t}" for t in WP11_TESTS}
    with specific.stage("code"):
        results = wp11(repo_root(), passed)
    assert [r.status for r in results] == ["PASS", "PASS", "PASS", "WARN"]
