from __future__ import annotations

import dataclasses
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.store import CodeState, open_store
from kta_gate import specific
from kta_gate.specific_wp7 import (
    CASE,
    FAULTS,
    WP7_UNIT_TESTS,
    case_findings,
    family_findings,
    record_findings,
    wp7,
)
from storehelp import T0

HEAD = subprocess.run(
    ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
ACTOR = Actor(type="tool", name="test")
PASSED = {f"tests.unit.x::{t}" for t in WP7_UNIT_TESTS}
MANIFEST = {
    "role": "worker",
    "host": "spark",
    "os": "Ubuntu",
    "arch": "aarch64",
    "python": "3.12.13",
    "uv": "0.11.2",
    "uv_lock_sha256": "x",
    "commit": HEAD,
    "dirty": False,
}


def _report(subject: str, data: dict[str, Any], title: str = "r") -> dict[str, Any]:
    return {"title": title, "subject": subject, "summary": "s", "data": data}


def _store(ctx: Context, tmp: Path, dirty: bool = False, caught: bool = True) -> set[str]:
    store = open_store(
        dataclasses.replace(ctx, data_dir=tmp / "data" / "dev" / "WP7"), CodeState(HEAD, dirty)
    )

    def put(kind: str, body: dict[str, Any], refs: dict[str, Any] | None = None) -> str:
        return store.put(Record.new(kind, body, ACTOR, T0, refs=refs or {}))

    env = put("EnvManifest", MANIFEST)
    worker = put(
        "Worker",
        {
            "name": "spark",
            "host": "spark",
            "transport": "ssh",
            "labels": {"gpu": "sm_121"},
            "devices": ["0"],
            "commit": HEAD,
        },
    )
    checks = {
        f: put(
            "Check", {"passed": False, "failures": [{"check": "x"}]}, {"worker": worker, "env": env}
        )
        for f in FAULTS
    }
    faults = {f: {"check": checks[f], "caught": caught} for f in FAULTS}
    baseline = put("Report", _report("baseline", {"call": "torch.matmul"}))
    ids = {
        worker,
        baseline,
        *checks.values(),
        put("Report", _report("case", {}, title="judgement")),
        put("Report", _report("wp7_faults", faults)),
        put("Report", _report("replay", {"consistent": True})),
        put("Report", _report("wp7_reuse", {"bf16": {"as_expected": True}})),
        put("HwModel", {"target": "gb10", "version": 1, "peaks": {"tensor": 1.0}, "limits": {}}),
        put(
            "Measurement",
            {
                "median_us": 1.0,
                "ci_low_us": 0.9,
                "ci_high_us": 1.1,
                "n": 100,
                "clocks": {},
                "temps": {},
                "protocol": "p",
            },
            {"baseline": baseline, "worker": worker, "env": env},
        ),
    }
    return ids


def _findings(tmp: Path, listed: set[str]) -> list[str]:
    con = sqlite3.connect(tmp / "data" / "dev" / "WP7" / "kta.db")
    try:
        return record_findings(repo_root(), con, listed)
    finally:
        con.close()


def test_static_family_and_case_pass(tmp_path: Path) -> None:
    assert family_findings(repo_root(), tmp_path) == []
    assert case_findings(repo_root()) == []


def test_case_check_rejects_a_weakened_contract(tmp_path: Path) -> None:
    doc = yaml.safe_load((repo_root() / CASE).read_text(encoding="utf-8"))
    doc["contract"]["tolerances"]["f16"]["rtol"] = 0.05
    doc["case"]["shape_dist"] = doc["case"]["shape_dist"][:1]
    (tmp_path / CASE).parent.mkdir(parents=True)
    (tmp_path / CASE).write_text(yaml.safe_dump(doc), encoding="utf-8")
    found = case_findings(tmp_path)
    assert any("tolerances" in f for f in found) and any("non-square" in f for f in found)


def test_records_accepted_when_listed_real_and_clean(ctx: Context, tmp_path: Path) -> None:
    ids = _store(ctx, tmp_path)
    assert _findings(tmp_path, ids) == []
    assert any("judgement" in f for f in _findings(tmp_path, set()))


def test_records_rejected_when_a_fault_is_not_caught(ctx: Context, tmp_path: Path) -> None:
    ids = _store(ctx, tmp_path, caught=False)
    assert {f"fault {f} not caught" for f in FAULTS} <= set(_findings(tmp_path, ids))


def test_records_rejected_when_dirty(ctx: Context, tmp_path: Path) -> None:
    ids = _store(ctx, tmp_path, dirty=True)
    assert len(_findings(tmp_path, ids)) >= 5


def test_wp7_full_and_code_stage(
    ctx: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _store(ctx, tmp_path)
    (tmp_path / "deliveries").mkdir()
    claim = [{"claim": "x", "record_ids": sorted(ids)}]
    (tmp_path / "deliveries" / "WP7.evidence.json").write_text(json.dumps(claim), "utf-8")
    for rel in ("knowledge", "tools"):
        (tmp_path / rel).symlink_to(repo_root() / rel)
    monkeypatch.delenv("KTA_DATA", raising=False)
    full = wp7(tmp_path, PASSED, data_root=tmp_path / "data", git_root=repo_root())
    assert [r.status for r in full] == ["PASS", "PASS", "PASS", "PASS"], full
    with specific.stage("code"):
        code = wp7(tmp_path, PASSED)
    assert code[-1].status == "WARN" and "pending real run" in code[-1].detail
    missing = wp7(tmp_path, set(), data_root=tmp_path / "data", git_root=repo_root())
    assert missing[0].status == "FAIL"
