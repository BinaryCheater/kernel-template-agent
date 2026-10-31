"""Batch environments and profile export (spec/modules/agents.md "Skills, backup and
regenerable batch environments", D321, D322).

A batch root ``<data>/batches/<batch>/`` is generated only from records: one AgentProfile
and the code commit that generated it. Its template parts are

    env.lock.json            batch, profile, commit, CLI version, every template file -> sha256
    agent-home-template/     exactly profile.home_files (never a credential)
    skills/<name>@<hash>/    unpacked skill_bundles
    prompts/                 prefix.md, step-<T>.md, msg-<name>.md

and per work unit ``agent-home/<wu>/`` and ``work/<wu>/`` are added at run time (on the
agent host, by the launcher). ``verify`` regenerates the template into a temporary
directory, compares it with the lock and with what is on disk; any difference refuses the
start. The same profile id and code commit always give byte-identical template parts.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kta.agents.adapters.codex import CREDENTIAL_FILES
from kta.agents.hostfs import safe_rel, sha256_file
from kta.agents.profile import unpack
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.ids import sha256_hex
from kta.core.schemas.process import AgentProfile
from kta.store.base import Store

LOCK = "env.lock.json"
LOCK_SCHEMA = "kta-batch-lock/1"
HOME_TEMPLATE = "agent-home-template"
TEMPLATE_DIRS = (HOME_TEMPLATE, "skills", "prompts")
RUNTIME_DIRS = ("agent-home", "work")  # per work unit; not part of the template


@dataclass(frozen=True)
class Verification:
    ok: bool
    problems: list[str] = field(default_factory=lambda: list[str]())


def batches_root(data_dir: Path) -> Path:
    return data_dir / "batches"


def check_name(batch: str) -> None:
    rel = safe_rel(batch)
    if len(rel.parts) != 1 or batch.startswith("."):
        raise ConfigError(f"batch name {batch!r} must be one plain path component")


def profile_batch(profile: AgentProfile) -> str:
    """The batch root named after a profile version (WP22, D595): ``<name>-v<version>``."""
    return f"{profile.name}-v{profile.version}"


def template_files(store: Store, profile: AgentProfile) -> dict[str, bytes]:
    """Every template file of a batch, by path relative to the batch root (records only)."""

    def blob(sha: str) -> bytes:
        return store.blob_path(sha).read_bytes()

    out: dict[str, bytes] = {}
    for name, sha in profile.home_files.items():
        if safe_rel(name).name in CREDENTIAL_FILES:
            raise PermissionDenied(f"profile home file {name} is a credential; never templated")
        out[f"{HOME_TEMPLATE}/{name}"] = blob(sha)
    refs = sorted({r for names in profile.skills.values() for r in names})
    for ref in refs:
        sha = profile.skill_bundles.get(ref)
        if sha is None:
            raise ConfigError(f"profile has no skill bundle for {ref} (skill_bundles, D322)")
        data = blob(sha)
        if sha256_hex(data) != ref.partition("@")[2]:
            raise ConfigError(f"skill bundle of {ref} does not hash to its id")
        for rel, content in unpack(data).items():
            out[f"skills/{ref}/{safe_rel(rel)}"] = content
    out["prompts/prefix.md"] = blob(profile.prefix_template)
    for t, sha in profile.step_templates.items():
        out[f"prompts/step-{t}.md"] = blob(sha)
    for name, sha in profile.message_templates.items():
        out[f"prompts/msg-{name}.md"] = blob(sha)
    return out


def lock_for(
    batch: str, profile_id: str, profile: AgentProfile, commit: str, files: dict[str, bytes]
) -> bytes:
    lock = {
        "schema": LOCK_SCHEMA,
        "batch": batch,
        "profile": profile_id,
        "commit": commit,
        "cli_version": profile.cli_version,
        "files": {rel: sha256_hex(data) for rel, data in sorted(files.items())},
    }
    return json.dumps(lock, sort_keys=True, indent=1).encode("utf-8") + b"\n"


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in sorted(files.items()):
        dest = root / safe_rel(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o644)


def build(store: Store, data_dir: Path, batch: str, profile_id: str, commit: str) -> Path:
    """Create ``<data>/batches/<batch>/``; an existing batch name is refused."""
    check_name(batch)
    root = batches_root(data_dir) / batch
    if root.exists():
        raise ConfigError(f"batch {batch} exists; batches are never overwritten")
    profile = load_profile(store, profile_id)
    files = template_files(store, profile)
    tmp = root.with_name(f".{batch}.building")
    if tmp.exists():
        shutil.rmtree(tmp)  # an interrupted build of this same new name
    write_tree(tmp, files)
    (tmp / LOCK).write_bytes(lock_for(batch, profile_id, profile, commit, files))
    tmp.rename(root)
    return root


def read_lock(root: Path) -> dict[str, Any]:
    path = root / LOCK
    if not path.exists():
        raise ConfigError(f"{root} is not a batch root (no {LOCK})")
    return json.loads(path.read_text(encoding="utf-8"))


def on_disk(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in TEMPLATE_DIRS:
        for path in sorted((root / d).rglob("*")):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                out[path.relative_to(root).as_posix()] = (
                    "symlink" if path.is_symlink() else sha256_file(path)
                )
    return out


def verify(store: Store, root: Path) -> Verification:
    """Regenerate the template from the lock's profile and compare (D322)."""
    lock = read_lock(root)
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="kta-batch-verify-") as scratch:
        profile = load_profile(store, str(lock["profile"]))
        files = template_files(store, profile)
        write_tree(Path(scratch), files)
        regenerated = lock_for(
            str(lock["batch"]), str(lock["profile"]), profile, str(lock["commit"]), files
        )
    if regenerated != (root / LOCK).read_bytes():
        problems.append("env.lock.json differs from the lock regenerated from the profile")
    expected: dict[str, str] = dict(lock["files"])
    actual = on_disk(root)
    for rel in sorted(set(expected) | set(actual)):
        if rel not in actual:
            problems.append(f"missing {rel}")
        elif rel not in expected:
            problems.append(f"not in the lock: {rel}")
        elif actual[rel] != expected[rel]:
            problems.append(f"changed: {rel}")
    return Verification(not problems, problems)


def require_verified(store: Store, root: Path) -> dict[str, Any]:
    result = verify(store, root)
    if not result.ok:
        shown = "; ".join(result.problems[:5])
        raise PermissionDenied(
            f"batch {root.name} fails verification ({len(result.problems)}): {shown}"
        )
    return read_lock(root)


def load_profile(store: Store, profile_id: str) -> AgentProfile:
    rec = store.get(profile_id)
    if rec.kind != "AgentProfile":
        raise ConfigError(f"{profile_id} is not an AgentProfile")
    return AgentProfile.model_validate(rec.body)


def export_profile(store: Store, profile_id: str, out: Path) -> list[str]:
    """Write a profile and all its blobs as files: ``profile.json`` plus the template tree."""
    if out.exists() and any(out.iterdir()):
        raise ConfigError(f"{out} is not empty")
    profile = load_profile(store, profile_id)
    files = template_files(store, profile)
    for ref, sha in profile.skill_bundles.items():
        files[f"bundles/{ref}.tar.gz"] = store.blob_path(sha).read_bytes()
    body = {"id": profile_id, "body": profile.model_dump(mode="json")}
    files["profile.json"] = json.dumps(body, sort_keys=True, indent=1).encode("utf-8") + b"\n"
    write_tree(out, files)
    return sorted(files)


@dataclass(frozen=True)
class BatchRoot:
    """Read access to a verified batch root: the only source of home, skills and prompts."""

    root: Path
    lock: dict[str, Any]

    @classmethod
    def open(cls, store: Store, data_dir: Path, batch: str) -> BatchRoot:
        check_name(batch)
        root = batches_root(data_dir) / batch
        return cls(root, require_verified(store, root))

    def text(self, rel: str) -> str:
        return (self.root / "prompts" / rel).read_text(encoding="utf-8")

    def skill_files(self, ref: str) -> dict[str, bytes]:
        base = self.root / "skills" / ref
        if not base.is_dir():
            raise ConfigError(f"batch {self.root.name} has no skill {ref}")
        return {
            p.relative_to(base).as_posix(): p.read_bytes()
            for p in sorted(base.rglob("*"))
            if p.is_file()
        }

    def template(self) -> dict[str, bytes]:
        """All template files with the lock, for the agent host's copy of the batch root."""
        files = {rel: (self.root / rel).read_bytes() for rel in self.lock["files"]}
        files[LOCK] = (self.root / LOCK).read_bytes()
        return files
