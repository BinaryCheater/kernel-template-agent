"""Batch environments and profile export (D322): determinism, refusal, verification."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kta.agents import batches
from kta.agents.profile import CODEX_V1, profile_body, write_profile
from kta.core.context import Context
from kta.core.errors import ConfigError, PermissionDenied, SchemaError
from kta.core.schemas import Actor, Record
from kta.store import CodeState, SqliteStore, open_store
from storehelp import TEST_COMMIT


def _tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_same_profile_and_commit_give_byte_identical_batches(
    ctx: Context, store: SqliteStore, tmp_path: Path
) -> None:
    pid = write_profile(store, datetime.now(UTC))
    a = batches.build(store, tmp_path / "a", "b1", pid, TEST_COMMIT)
    # A second store on another "machine", with the same profile record content.
    other = open_store(
        dataclasses.replace(ctx, data_dir=tmp_path / "other"), CodeState(TEST_COMMIT, False)
    )
    rec = store.get(pid)
    for sha in {
        *rec.body["home_files"].values(),
        rec.body["prefix_template"],
        *rec.body["step_templates"].values(),
        *rec.body["message_templates"].values(),
        *rec.body["skill_bundles"].values(),
    }:
        other.put_blob(store.blob_path(sha).read_bytes(), "application/octet-stream")
    assert other.put(rec) == pid
    b = batches.build(other, tmp_path / "b", "b1", pid, TEST_COMMIT)
    assert _tree(a) == _tree(b)
    lock = json.loads((a / "env.lock.json").read_text())
    assert (
        lock["profile"] == pid
        and lock["commit"] == TEST_COMMIT
        and lock["cli_version"] == CODEX_V1.cli_version
    )
    assert {
        "agent-home-template/config.toml",
        "prompts/prefix.md",
        "prompts/step-T1.md",
        "prompts/msg-rejection.md",
    } <= set(lock["files"])
    assert any(f.startswith("skills/write-family@") for f in lock["files"])
    assert batches.verify(store, a).ok


def test_existing_batch_names_and_bad_names_are_refused(store: SqliteStore, tmp_path: Path) -> None:
    pid = write_profile(store, datetime.now(UTC))
    batches.build(store, tmp_path, "b1", pid, TEST_COMMIT)
    with pytest.raises(ConfigError, match="never overwritten"):
        batches.build(store, tmp_path, "b1", pid, TEST_COMMIT)
    for bad in ("../x", "a/b", ".hidden"):
        with pytest.raises(ConfigError):
            batches.build(store, tmp_path, bad, pid, TEST_COMMIT)


def test_verify_reports_tampered_added_and_missing_files(
    store: SqliteStore, tmp_path: Path
) -> None:
    pid = write_profile(store, datetime.now(UTC))
    root = batches.build(store, tmp_path, "b1", pid, TEST_COMMIT)
    skill = next((root / "skills").glob("use-kta@*/SKILL.md"))
    skill.write_text("tampered")
    (root / "prompts" / "extra.md").write_text("x")
    (root / "prompts" / "step-T3.md").unlink()
    (root / "agent-home" / "wu").mkdir(parents=True)  # per work unit dirs are not template
    (root / "agent-home" / "wu" / "auth.json").write_text("{}")
    result = batches.verify(store, root)
    assert not result.ok
    assert any(p.startswith("changed: skills/use-kta@") for p in result.problems)
    assert "not in the lock: prompts/extra.md" in result.problems
    assert "missing prompts/step-T3.md" in result.problems
    assert not any("agent-home/wu" in p for p in result.problems)
    with pytest.raises(PermissionDenied):
        batches.require_verified(store, root)
    (root / "env.lock.json").write_text((root / "env.lock.json").read_text().replace(pid, "0" * 64))
    with pytest.raises(SchemaError):  # the lock now names a profile that does not exist
        batches.verify(store, root)


def test_credentials_never_enter_a_template(store: SqliteStore) -> None:
    body = profile_body(store)
    body.home_files["auth.json"] = store.put_blob(b"{}", "application/json")
    with pytest.raises(PermissionDenied, match="credential"):
        batches.template_files(store, body)


def test_export_profile_writes_every_blob(store: SqliteStore, tmp_path: Path) -> None:
    pid = write_profile(store, datetime.now(UTC))
    files = batches.export_profile(store, pid, tmp_path / "out")
    assert "profile.json" in files and "prompts/prefix.md" in files
    bundles = [f for f in files if f.startswith("bundles/")]
    body = store.get(pid).body
    assert len(bundles) == len(body["skill_bundles"]) > 0
    assert json.loads((tmp_path / "out" / "profile.json").read_text())["id"] == pid
    with pytest.raises(ConfigError, match="not empty"):
        batches.export_profile(store, pid, tmp_path / "out")


def test_skill_bundles_cover_every_skill_reference(store: SqliteStore) -> None:
    body = profile_body(store)
    refs = {r for names in body.skills.values() for r in names}
    assert refs == set(body.skill_bundles)
    assert all(ref.endswith("@" + sha) for ref, sha in body.skill_bundles.items())
    rec = Record.new("AgentProfile", body, Actor(type="system", name="t"), datetime.now(UTC))
    assert rec.body["skill_bundles"]
