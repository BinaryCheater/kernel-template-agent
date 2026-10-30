"""Step bundle: the files the launcher installs into the worker workspace and CLI home.

Built on the control plane from the AgentProfile's blobs and the task's input records;
the launcher on the agent host copies them in (``kta.agents.hostfs``). Nothing else
enters the workspace (spec/modules/agents.md "工作区文件").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kta.agents import cards
from kta.agents.batches import BatchRoot
from kta.agents.profile import skill_description, unpack
from kta.core.errors import ConfigError, SchemaError
from kta.core.schemas.process import AgentProfile, TaskCard
from kta.knowledge.retrieve import CARD_KINDS, card
from kta.store.base import Store

INPUT_BLOB_MAX_BYTES = 32 * 1024 * 1024
TAR_BLOBS = ("package", "source", "sources")  # blobs that hold a directory tree


@dataclass
class Bundle:
    workspace: dict[str, bytes] = field(default_factory=lambda: dict[str, bytes]())
    batch: dict[str, bytes] = field(default_factory=lambda: dict[str, bytes]())  # template + lock


def skills_of(
    batch: BatchRoot, profile: AgentProfile, task_type: str
) -> dict[str, dict[str, bytes]]:
    """Skill files by name, from the batch root's ``skills/<name>@<hash>/``."""
    out: dict[str, dict[str, bytes]] = {}
    for ref in profile.skills.get(task_type, []):
        name, _, sha = ref.partition("@")
        if not sha:
            raise ConfigError(f"profile skill {ref!r} lacks its archive hash")
        out[name] = batch.skill_files(ref)
    return out


def descriptions(batch: BatchRoot, profile: AgentProfile) -> dict[str, str]:
    """One line per skill of every task type: the prefix lists them all, so it is
    byte-identical across sessions of the profile."""
    seen: dict[str, str] = {}
    for task_type in sorted(profile.skills):
        for name, files in skills_of(batch, profile, task_type).items():
            seen[name] = skill_description(files["SKILL.md"].decode("utf-8"))
    return seen


def input_cards(store: Store, ids: list[str]) -> list[cards.InputCard]:
    out: list[cards.InputCard] = []
    for rid in ids:
        rec = store.get(rid)
        if rec.kind in CARD_KINDS:
            c = card(store, rid)
            out.append(cards.InputCard(rid, rec.kind, c.name, list(c.lines)))
        else:
            title = str(
                rec.body.get("name") or rec.body.get("goal") or rec.body.get("description") or ""
            )
            out.append(cards.InputCard(rid, rec.kind, title[:120], []))
    return out


def input_files(store: Store, ids: list[str]) -> dict[str, bytes]:
    """``inputs/<kind>-<id12>.json`` per input record, plus its blobs (trees unpacked)."""
    files: dict[str, bytes] = {}
    for rid in ids:
        rec = store.get(rid)
        stem = f"inputs/{rec.kind}-{rid[:12]}"
        files[f"{stem}.json"] = json.dumps(
            {"id": rid, "kind": rec.kind, "refs": rec.refs, "body": rec.body},
            indent=1,
            sort_keys=True,
        ).encode("utf-8")
        for name, sha in rec.blobs.items():
            path = store.blob_path(sha)
            if path.stat().st_size > INPUT_BLOB_MAX_BYTES:
                raise SchemaError(f"input blob {name} of {rid[:12]} is too large for the workspace")
            data = path.read_bytes()
            if name in TAR_BLOBS and data[:2] == b"\x1f\x8b":
                for member, content in unpack(data).items():
                    files[f"{stem}/{name}/{member}"] = content
            else:
                files[f"{stem}/{name}"] = data
    return files


def build(
    store: Store,
    profile: AgentProfile,
    batch: BatchRoot,
    *,
    task_id: str,
    card_id: str,
    card_body: TaskCard,
    used: dict[str, float],
    retry: cards.RetryNote | None,
) -> Bundle:
    """Workspace files; templates and skills come only from the verified batch root (D322)."""
    t = card_body.type
    if t not in profile.step_templates:
        raise ConfigError(f"profile {profile.name} has no step template for {t}")
    skills = skills_of(batch, profile, t)
    bundle = Bundle()
    ws = bundle.workspace
    ws["AGENTS.md"] = cards.prefix(batch.text("prefix.md"), descriptions(batch, profile)).encode()
    ws["TASK.md"] = cards.task_md(
        batch.text(f"step-{t}.md"),
        task_id=task_id,
        card_id=card_id,
        card=card_body,
        cards=input_cards(store, card_body.inputs),
        skills=sorted(skills),
        commands=profile.allowed_commands.get(t, []),
        used=used,
        retry=retry,
        policy=profile.context_policy,
    ).encode()
    if "notes_seed" in profile.message_templates:
        ws["NOTES.md"] = batch.text("msg-notes_seed.md").encode()
    for name, files in skills.items():
        for rel, content in files.items():
            ws[f".agents/skills/{name}/{rel}"] = content
    ws.update(input_files(store, card_body.inputs))
    bundle.batch = batch.template()
    return bundle
