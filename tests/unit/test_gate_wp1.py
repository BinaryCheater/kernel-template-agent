from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas import Record
from kta.store import CodeState, open_store
from kta.store.bench import ACTOR
from kta_gate.specific import WP1_ACCEPTANCE, wp1
from storehelp import T0

HEAD = subprocess.run(
    ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
PASSED = {f"tests.acceptance.WP1.test_core_store_acceptance::{t}" for t in WP1_ACCEPTANCE}
MANIFEST = {
    "host": "spark",
    "os": "Ubuntu",
    "arch": "aarch64",
    "python": "3.12.13",
    "uv": "0.11.2",
    "uv_lock_sha256": "x",
    "commit": HEAD,
    "dirty": False,
}


def _setup(
    ctx: Context,
    tmp: Path,
    p95_ms: float = 3.0,
    role: str = "worker",
    dirty: bool = False,
    listed: bool = True,
) -> str:
    data_dir = tmp / "data" / "dev" / "WP1"
    store = open_store(dataclasses.replace(ctx, data_dir=data_dir), CodeState(HEAD, dirty))
    env = store.put(Record.new("EnvManifest", {**MANIFEST, "role": role}, ACTOR, T0))
    samples = {"query_name_eq": [1.0] * 19 + [p95_ms], "search_1word": [2.0] * 20}
    blob = store.put_blob(json.dumps(samples).encode(), "application/json")
    body = {
        "title": "store benchmark",
        "subject": "store-benchmark",
        "summary": "s",
        "data": {"records_in_store": 100_000},
    }
    rid = store.put(
        Record.new("Report", body, ACTOR, T0, refs={"env": env}, blobs={"latencies_ms": blob})
    )
    (tmp / "deliveries").mkdir(exist_ok=True)
    claim = {"record_ids": [rid] if listed else []}
    (tmp / "deliveries" / "WP1.evidence.json").write_text(json.dumps([claim]))
    return rid


def _run(tmp: Path, passed: set[str] = PASSED) -> list[tuple[str, str, list[str]]]:
    results = wp1(tmp, passed, data_root=tmp / "data", git_root=repo_root())
    return [(r.name, r.status, r.findings) for r in results]


def test_wp1_gate_accepts_a_valid_benchmark_record(ctx: Context, tmp_path: Path) -> None:
    _setup(ctx, tmp_path)
    assert [s for _, s, _ in _run(tmp_path)] == ["PASS", "PASS"]


def test_wp1_gate_recomputes_p95_from_raw_samples(ctx: Context, tmp_path: Path) -> None:
    _setup(ctx, tmp_path, p95_ms=80.0)
    (_, _, _), (_, status, findings) = _run(tmp_path)
    assert status == "FAIL" and any("query_name_eq: p95 80.000" in f for f in findings)


def test_wp1_gate_rejects_dirty_control_or_unlisted(ctx: Context, tmp_path: Path) -> None:
    for kwargs, needle in (
        ({"dirty": True}, "dirty"),
        ({"role": "control"}, "worker EnvManifest"),
        ({"listed": False}, "not in WP1.evidence.json"),
    ):
        tmp = tmp_path / needle.replace(" ", "_")
        tmp.mkdir()
        _setup(ctx, tmp, **kwargs)  # type: ignore[arg-type]
        (_, _, _), (_, status, findings) = _run(tmp)
        assert status == "FAIL" and any(needle in f for f in findings), (kwargs, findings)


def test_wp1_gate_requires_acceptance_tests(ctx: Context, tmp_path: Path) -> None:
    _setup(ctx, tmp_path)
    (_, status, findings), _ = _run(tmp_path, passed=set())
    assert status == "FAIL" and len(findings) == len(WP1_ACCEPTANCE)


def test_wp1_gate_without_store(tmp_path: Path) -> None:
    (_, _, _), (_, status, _) = _run(tmp_path)
    assert status == "FAIL"
