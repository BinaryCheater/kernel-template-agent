from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.store import CodeState, open_store
from kta_gate.specific import WP2_ACCEPTANCE, wp2
from storehelp import T0

HEAD = subprocess.run(
    ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
GPU_PASSED = {f"tests.acceptance.WP2.test_target_exec_acceptance::{t}" for t in WP2_ACCEPTANCE}
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


def _setup(
    ctx: Context,
    tmp: Path,
    dirty: bool = False,
    listed: bool = True,
    backends: tuple[str, ...] = ("cuda", "tilelang"),
) -> None:
    store = open_store(
        dataclasses.replace(ctx, data_dir=tmp / "data" / "dev" / "WP2"), CodeState(HEAD, dirty)
    )
    ids: list[str] = []

    def put(
        kind: str,
        body: dict[str, object],
        refs: dict[str, str] | None = None,
        blobs: dict[str, str] | None = None,
    ) -> str:
        rid = store.put(Record.new(kind, body, ACTOR, T0, refs=dict(refs or {}), blobs=blobs))
        ids.append(rid)
        return rid

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
        {"env": env},
    )
    job = {"env": env, "worker": worker}
    put(
        "Target",
        {
            "name": "gb10",
            "gpu": "GB10",
            "sm": "sm_121",
            "driver": "580",
            "cuda": "13.0",
            "ptxas": "13.0",
            "image": "none",
        },
        job,
    )
    codegen = store.put_blob(b"{}", "application/json")
    for backend in backends:
        blobs = {"codegen": codegen} if backend == "tilelang" else {}
        build = put("Build", {"ok": True, "log_summary": backend}, job, blobs)
        put("Check", {"passed": True, "failures": []}, {**job, "build": build})
    (tmp / "deliveries").mkdir(exist_ok=True)
    claim = {"record_ids": ids if listed else []}
    (tmp / "deliveries" / "WP2.evidence.json").write_text(json.dumps([claim]))


def _run(tmp: Path, gpu_passed: set[str] = GPU_PASSED) -> list[tuple[str, str, list[str]]]:
    results = wp2(tmp, set(), data_root=tmp / "data", git_root=repo_root(), gpu_passed=gpu_passed)
    return [(r.name, r.status, r.findings) for r in results]


def test_wp2_gate_accepts_complete_real_records(ctx: Context, tmp_path: Path) -> None:
    _setup(ctx, tmp_path)
    assert [(n, s) for n, s, _ in _run(tmp_path)] == [
        ("WP2 acceptance tests", "PASS"),
        ("WP2 records", "PASS"),
    ]


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"dirty": True}, "no listed, real, clean WP2 Target"),
        ({"listed": False}, "no listed, real, clean WP2 Build"),
        ({"backends": ("cuda",)}, "tilelang Build"),
    ],
)
def test_wp2_gate_rejects_incomplete_records(
    ctx: Context, tmp_path: Path, kwargs: dict[str, object], needle: str
) -> None:
    _setup(ctx, tmp_path, **kwargs)  # type: ignore[arg-type]
    (_, _, _), (_, status, findings) = _run(tmp_path)
    assert status == "FAIL" and any(needle in f for f in findings), findings


def test_wp2_gate_needs_every_gpu_acceptance_test(ctx: Context, tmp_path: Path) -> None:
    _setup(ctx, tmp_path)
    (_, status, findings), _ = _run(tmp_path, set(list(GPU_PASSED)[:-1]))
    assert status == "FAIL" and len(findings) == 1
