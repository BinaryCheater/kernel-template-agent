from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.store import CodeState, open_store
from kta_gate.specific_wp3 import WP3_ACCEPTANCE, wp3
from storehelp import T0

HEAD = subprocess.run(
    ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
GPU_PASSED = {f"tests.acceptance.WP3.test_measure_check_acceptance::{t}" for t in WP3_ACCEPTANCE}
ACTOR = Actor(type="tool", name="test")
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
VARIANT_OK = {"as_expected": True}


def _setup(
    ctx: Context,
    tmp: Path,
    dirty: bool = False,
    variants_ok: bool = True,
    groups: int = 30,
    unlisted: str = "",
) -> None:
    store = open_store(
        dataclasses.replace(ctx, data_dir=tmp / "data" / "dev" / "WP3"), CodeState(HEAD, dirty)
    )
    ids: dict[str, str] = {}

    def put(
        name: str, kind: str, body: dict[str, object], refs: Mapping[str, object] | None = None
    ) -> str:
        rid = store.put(Record.new(kind, body, ACTOR, T0, refs=dict(refs or {})))  # type: ignore[arg-type]
        ids[name] = rid
        return rid

    env = put("env", "EnvManifest", MANIFEST)
    worker = put(
        "worker",
        "Worker",
        {
            "name": "spark",
            "host": "spark",
            "transport": "ssh",
            "labels": {"gpu": "sm_121"},
            "devices": ["0"],
            "commit": HEAD,
        },
        {"env": env},
    )
    job = {"env": env, "worker": worker}
    proto = {"name": "measure", "type": "measurement", "description": "d", "params": {}}
    v1 = put("v1", "Protocol", {**proto, "version": 1})
    m = put(
        "m",
        "Measurement",
        {"median_us": 1.0, "ci_low_us": 0.9, "ci_high_us": 1.1, "n": 100, "protocol": v1},
        {**job, "protocol": v1},
    )
    noise = put(
        "noise",
        "Report",
        {"title": "n", "subject": "noise", "summary": "s", "data": {"groups": groups}},
        {"env": env, "measurements": [m]},
    )
    v2 = put(
        "v2",
        "Protocol",
        {**proto, "version": 2, "params": {"noise_floor_rel": 0.004}},
        {"base": v1, "noise_reports": [noise]},
    )
    put("check", "Check", {"passed": True, "failures": []}, job)
    data = {
        v: (VARIANT_OK if variants_ok or v != "cublas" else {"as_expected": False})
        for v in ("correct", "oob_write", "missing_row", "cublas")
    }
    put("selftest", "Report", {"title": "c", "subject": "checks", "summary": "s", "data": data})
    for sig in (True, False):
        put(
            f"cmp{sig}",
            "Report",
            {
                "title": "c",
                "subject": "compare",
                "summary": "s",
                "data": {"significant": sig, "protocol": v2},
            },
        )
    (tmp / "deliveries").mkdir(exist_ok=True)
    claim = {"record_ids": [rid for name, rid in ids.items() if name != unlisted]}
    (tmp / "deliveries" / "WP3.evidence.json").write_text(json.dumps([claim]))


def _run(tmp: Path, gpu_passed: set[str] = GPU_PASSED) -> list[tuple[str, str, list[str]]]:
    results = wp3(tmp, set(), data_root=tmp / "data", git_root=repo_root(), gpu_passed=gpu_passed)
    return [(r.name, r.status, r.findings) for r in results]


def test_wp3_gate_accepts_complete_real_records(ctx: Context, tmp_path: Path) -> None:
    _setup(ctx, tmp_path)
    assert [(n, s, f) for n, s, f in _run(tmp_path)] == [
        ("WP3 acceptance tests", "PASS", []),
        ("WP3 records", "PASS", []),
    ]


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"dirty": True}, "no listed, real, clean WP3 Measurement"),
        ({"variants_ok": False}, "all four variants"),
        ({"groups": 5}, "fewer than 30 groups"),
        ({"unlisted": "v2"}, "no listed Protocol measure-v2"),
        ({"unlisted": "cmpFalse"}, "non-significant compare"),
        ({"unlisted": "noise"}, "is not listed"),
    ],
)
def test_wp3_gate_rejects_incomplete_records(
    ctx: Context, tmp_path: Path, kwargs: dict[str, object], needle: str
) -> None:
    _setup(ctx, tmp_path, **kwargs)  # type: ignore[arg-type]
    (_, _, _), (_, status, findings) = _run(tmp_path)
    assert status == "FAIL" and any(needle in f for f in findings), findings


def test_wp3_gate_needs_every_gpu_acceptance_test(ctx: Context, tmp_path: Path) -> None:
    _setup(ctx, tmp_path)
    (_, status, findings), _ = _run(tmp_path, set(list(GPU_PASSED)[:-1]))
    assert status == "FAIL" and len(findings) == 1
