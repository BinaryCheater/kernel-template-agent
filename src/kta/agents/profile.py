"""AgentProfile ``codex-v1``: the versioned settings that alone control a worker.

The record's blobs are the prompt templates, the isolated CLI config and one packed
archive per skill; a skill is named ``<name>@<sha256 of its archive>``, the archive is
stored as that blob, so a run can be rebuilt from the record even after the source tree
changed. ``write_profile`` is idempotent: the same name and version with the same
content returns the existing record; a different content under the same version is
refused (bump the version). Switching the pointer ``current_profile`` needs a human
(``approve_profile``), as spec/modules/agents.md requires.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from kta.core.errors import ConfigError, PermissionDenied
from kta.core.ids import record_id, sha256_hex
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import AgentProfile, HumanAction
from kta.store.base import Ops, Store

PACKAGE = Path(__file__).resolve().parent
PROMPTS = PACKAGE / "prompts"
SKILLS = PACKAGE / "skills"
CURRENT_POINTER = "current_profile"
TASK_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")
MESSAGES = ("rejection", "budget_warning", "human_note", "new_session")
# kta subcommands an agent may call (docs/design/10 §5, spec/modules/orchestrator-cli.md).
AGENT_COMMANDS = [
    "show", "query", "why", "src", "instantiate", "build", "sass", "check", "run",
    "measure", "profile", "bound", "solve", "validate-template", "validate-family",
    "note", "submit",
]  # fmt: skip
SKILLS_BY_TYPE: dict[str, list[str]] = {
    "T1": ["use-kta", "find-knowledge", "write-family"],
    "T2": ["use-kta", "find-knowledge", "decompose-kernel", "write-template"],
    "T3": ["use-kta", "find-knowledge", "write-template"],
    "T4": ["use-kta", "find-knowledge"],
    "T5": ["use-kta", "find-knowledge"],
    "T6": ["use-kta", "find-knowledge", "write-family"],
}
# codex-v2 (WP11, D407): the T4 and T5 skills join; everything else is as in codex-v1.
SKILLS_BY_TYPE_V2: dict[str, list[str]] = {
    **SKILLS_BY_TYPE,
    "T4": ["use-kta", "find-knowledge", "diagnose-model"],
    "T5": ["use-kta", "find-knowledge", "write-microbench"],
}


@dataclass(frozen=True)
class ProfileSpec:
    """The choices of one profile version that are not files (decisions D308, D309)."""

    name: str = "codex-v1"
    version: int = 1
    cli_version: str = "0.117.0"
    model: str = "gpt-5.4"
    reasoning_effort: str | None = "medium"
    isolation: str = "config+user"
    skills_by_type: dict[str, list[str]] = field(default_factory=lambda: dict(SKILLS_BY_TYPE))


CODEX_V1 = ProfileSpec()
CODEX_V2 = ProfileSpec(name="codex-v2", version=2, skills_by_type=SKILLS_BY_TYPE_V2)
CODEX_FLAGS = [
    "--json",
    "-c", 'sandbox_mode="workspace-write"',
    "-c", 'approval_policy="never"',
    "-c", 'web_search="disabled"',
]  # fmt: skip
CONTEXT_POLICY: dict[str, Any] = {
    "cards_k": 5,
    "card_max_lines": 5,
    "tool_output_max_chars": 4000,
    "step_header_max_chars": 6000,
    "notes_required": True,
}
LOOP_POLICY: dict[str, Any] = {
    "max_retries": 2,
    "reset_context_ratio": 0.6,
    "reset_on_repeat_failure": True,
    "idle_s": 600,
    "steps_per_work_unit": 10,
    "wall_s": {"T1": 3600, "T2": 5400, "T3": 1800, "T4": 3600, "T5": 3600, "T6": 2700},
}


def pack_dir(root: Path) -> bytes:
    """Deterministic tar.gz of a directory (sorted names, zeroed times and owners)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            data = path.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), mtime=0)


def unpack(data: bytes) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            handle = tar.extractfile(m) if m.isfile() else None
            if handle is not None:
                out[m.name] = handle.read()
    return out


def skill_description(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    raise ConfigError("skill without a description line in its front matter")


def skill_ref(store: Store, name: str) -> str:
    path = SKILLS / name / "SKILL.md"
    if not path.exists():
        raise ConfigError(f"runtime skill {name} has no {path}")
    sha = store.put_blob(pack_dir(SKILLS / name), "application/gzip")
    return f"{name}@{sha}"


def profile_body(store: Store, spec: ProfileSpec = CODEX_V1) -> AgentProfile:
    def blob(name: str, media: str = "text/markdown") -> str:
        return store.put_blob((PROMPTS / name).read_bytes(), media)

    by_type = spec.skills_by_type
    refs = {n: skill_ref(store, n) for n in sorted({s for v in by_type.values() for s in v})}
    messages = {m: blob(f"msg-{m}.md") for m in MESSAGES}
    messages["notes_seed"] = blob("notes-seed.md")
    return AgentProfile(
        name=spec.name,
        version=spec.version,
        provider="codex",
        cli_version=spec.cli_version,
        auth="subscription",
        isolation=spec.isolation,  # type: ignore[arg-type]
        model=spec.model,
        reasoning_effort=spec.reasoning_effort,
        cli_flags=list(CODEX_FLAGS),
        home_files={"config.toml": blob("codex-config.toml", "application/toml")},
        prefix_template=blob("prefix.md"),
        step_templates={t: blob(f"step-{t}.md") for t in TASK_TYPES},
        message_templates=messages,
        skills={t: [refs[s] for s in names] for t, names in by_type.items()},
        skill_bundles={ref: ref.partition("@")[2] for ref in refs.values()},
        allowed_commands={t: list(AGENT_COMMANDS) for t in TASK_TYPES},
        context_policy=dict(CONTEXT_POLICY),
        loop_policy=dict(LOOP_POLICY),
    )


def write_profile(store: Store, at: datetime, spec: ProfileSpec = CODEX_V1) -> str:
    body = profile_body(store, spec)
    dumped = body.model_dump(mode="json")
    for rec in store.query("AgentProfile", where={"name": spec.name, "version": spec.version}):
        if rec.body == dumped:
            return record_id(rec)
        raise ConfigError(f"AgentProfile {spec.name} v{spec.version} exists with other content")
    rec = Record.new("AgentProfile", body, Actor(type="system", name="kta.agents.profile"), at)
    return store.put(rec)


def approve_profile(
    store: Store, ops: Ops, profile_id: str, human: str, reason: str, at: datetime
) -> str:
    """Point ``current_profile`` at a profile; only a human may do this (HumanAction)."""
    if not human:
        raise PermissionDenied("switching the current profile needs a named human")
    if store.get(profile_id).kind != "AgentProfile":
        raise ConfigError(f"{profile_id} is not an AgentProfile")
    action = Record.new(
        "HumanAction",
        HumanAction(action="approve_profile", reason=reason),
        Actor(type="human", name=human),
        at,
        refs={"profile": profile_id},
    )
    action_id = store.put(action)
    current = ops.get_pointer(CURRENT_POINTER)
    if not ops.set_pointer(CURRENT_POINTER, current, profile_id):
        raise ConfigError("current_profile changed concurrently; approve again")
    return action_id


def content_hash(data: bytes) -> str:
    return sha256_hex(data)
