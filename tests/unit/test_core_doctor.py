from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from kta.core.config import config_from_mapping, repo_root
from kta.core.context import Context
from kta.core.doctor import run_doctor
from kta.core.envmanifest import collect_manifest
from kta.core.locks import python_mismatches, uv_lock_mismatches
from kta.core.registry import Registry


def fixture_clock(ctx: Context) -> object:
    return ctx.clock


def test_manifest_fields_match_schema(tmp_path: Path) -> None:
    m = collect_manifest("control", repo_root(), tmp_path / "no-cuda").to_dict()
    expected = {
        "role",
        "host",
        "os",
        "arch",
        "python",
        "uv",
        "uv_lock_sha256",
        "node",
        "npm",
        "npm_lock_sha256",
        "cuda",
        "driver",
        "nvcc",
        "ptxas",
        "backends",
        "image_digest",
        "commit",
        "dirty",
        "lock_mismatches",
    }
    assert set(m) == expected
    assert m["python"].startswith("3.12.") and m["cuda"] is None and m["nvcc"] is None
    assert len(m["uv_lock_sha256"]) == 64


def test_repo_environment_matches_locks() -> None:
    assert python_mismatches(repo_root()) == []
    assert uv_lock_mismatches(repo_root()) == []


def test_lock_mismatch_detected(tmp_path: Path) -> None:
    (tmp_path / ".python-version").write_text("3.11.0\n")
    assert python_mismatches(tmp_path)
    assert uv_lock_mismatches(tmp_path) == ["uv.lock is missing"]


def test_doctor_main_rejects_lock_mismatch(ctx: Context, tmp_path: Path) -> None:
    (tmp_path / ".python-version").write_text("3.11.0\n")
    report = run_doctor(dataclasses.replace(ctx, env="main"), root=tmp_path)
    assert not report.ok and any("python" in p for p in report.problems)
    dev = run_doctor(dataclasses.replace(ctx, env="dev"), root=tmp_path)
    assert dev.ok and any("python" in w for w in dev.warnings)


def test_doctor_reports_fixture_selection(ctx: Context) -> None:
    reg = Registry()
    reg.register("clock", "manual", fixture_clock, fixture=True)
    cfg = config_from_mapping({"impl": {"clock": "manual"}})
    in_test = run_doctor(dataclasses.replace(ctx, registry=reg, config=cfg))
    assert in_test.ok and in_test.interfaces[0].fixture
    in_dev = run_doctor(dataclasses.replace(ctx, env="dev", registry=reg, config=cfg))
    assert not in_dev.ok


def test_kta_doctor_cli_json(tmp_path: Path) -> None:
    env = {**os.environ, "KTA_ENV": "dev", "KTA_DATA": str(tmp_path), "KTA_WP": "WP0"}
    kta = Path(sys.executable).parent / "kta"
    proc = subprocess.run([kta, "doctor", "--json"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["env"] == "dev" and out["data_dir"] == str(tmp_path / "dev" / "WP0")
    missing = subprocess.run([kta, "doctor"], env={**env, "KTA_ENV": ""}, capture_output=True)
    assert missing.returncode == 2


def test_parse_version_prefers_most_specific() -> None:
    from kta.core.envmanifest import parse_version

    nvcc = (
        "nvcc: NVIDIA (R) Cuda compiler driver\n"
        "Cuda compilation tools, release 13.0, V13.0.88\n"
        "Build cuda_13.0.r13.0/compiler.36424714_0"
    )
    assert parse_version(nvcc) == "13.0.88"
    assert parse_version("uv 0.11.2 (aarch64-unknown-linux-gnu)") == "0.11.2"
    assert parse_version("v26.12.1") == "26.12.1"
    assert parse_version("no version") is None
