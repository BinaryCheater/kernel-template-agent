"""The WP8 gate checks: named tests, runtime skills, forbidden options, agent-run records."""

from __future__ import annotations

import dataclasses
import gzip
import io
import json
import sqlite3
import tarfile
from pathlib import Path

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import AgentRun
from kta.store import CodeState, open_store
from kta_gate import specific
from kta_gate.specific_wp8 import WP8_TESTS, check_options, check_skills, run_findings, wp8
from storehelp import T0

MANIFEST = {
    "role": "control",
    "host": "mac",
    "os": "macOS",
    "arch": "arm64",
    "python": "3.12.13",
    "uv": "0.11.2",
    "uv_lock_sha256": "x",
    "commit": "c" * 40,
    "dirty": False,
}


def _tgz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue())


def _runs(
    ctx: Context,
    tmp: Path,
    *,
    credential: bool = False,
    resumed: bool = True,
    violation: bool = True,
) -> set[str]:
    data = tmp / "dev" / "WP8"
    store = open_store(dataclasses.replace(ctx, data_dir=data), CodeState("c" * 40, False))
    actor = Actor(type="system", name="test")
    env = store.put(Record.new("EnvManifest", MANIFEST, actor, T0))
    files = {"sessions/2026/10/31/rollout-x.jsonl": b"{}\n"}
    if credential:
        files["auth.json"] = b"{}"
    archive = store.put_blob(_tgz(files), "application/gzip")
    manifest = store.put_blob(json.dumps({"schema": "x"}).encode(), "application/json")

    def run(session: str, violations: list[dict[str, object]], previous: str | None) -> str:
        body = AgentRun(
            session=session,
            profile="p",
            cli_version="0.117.0",
            model="m",
            session_archive=archive,
            wall_s=1.0,
            tool_s=0.1,
            queue_s=0.0,
            exit="ok",
            context_manifest=manifest,
            violations=violations,
        )
        refs: dict[str, object] = {"env": env}
        if previous:
            refs["previous"] = previous
        return store.put(Record.new("AgentRun", body, actor, T0, refs=refs))  # type: ignore[arg-type]

    first = run("s1", [], None)
    induced: list[dict[str, object]] = (
        [{"kind": "write_outside_workspace", "detail": "x"}] if violation else []
    )
    second = run("s1" if resumed else "s2", induced, first)
    return {first, second}


def _findings(tmp: Path, listed: set[str]) -> list[str]:
    data = tmp / "dev" / "WP8"
    con = sqlite3.connect(f"file:{data / 'kta.db'}?mode=ro", uri=True)
    try:
        return run_findings(con, data, listed)
    finally:
        con.close()


def test_wp8_runs_pass_with_resume_archives_and_violation(ctx: Context, tmp_path: Path) -> None:
    listed = _runs(ctx, tmp_path)
    assert _findings(tmp_path, listed) == []


def test_wp8_runs_fail_without_resume_or_with_credentials(ctx: Context, tmp_path: Path) -> None:
    listed = _runs(ctx, tmp_path / "a", resumed=False, violation=False)
    found = _findings(tmp_path / "a", listed)
    assert any("resumed" in f for f in found) and any("induced" in f for f in found)
    listed = _runs(ctx, tmp_path / "b", credential=True)
    assert any("auth.json" in f for f in _findings(tmp_path / "b", listed))
    assert _findings(tmp_path / "b", set()) == ["no listed, real, clean AgentRun"]


def test_wp8_static_checks(tmp_path: Path) -> None:
    root = repo_root()
    assert check_skills(root).status == "PASS"
    assert check_options(root).status == "PASS"
    bad = tmp_path / "src" / "kta" / "agents"
    bad.mkdir(parents=True)
    (bad / "x.py").write_text('ARGV = ["codex", "exec", "--ephemeral"]\n')
    assert check_options(tmp_path).status == "FAIL"
    assert check_skills(tmp_path).status == "FAIL"


def test_wp8_code_stage_leaves_runs_pending() -> None:
    passed = {f"tests/unit/x.py::{t}" for t in WP8_TESTS}
    with specific.stage("code"):
        results = wp8(repo_root(), passed)
    assert [r.status for r in results] == ["PASS", "PASS", "PASS", "WARN"]
