"""Draft AgentProfile versions edited from the workflow page (WP22 Part D, D593–D596).

A profile record is immutable and runs reference it, so an edit never changes a profile:
``write_draft`` writes a new AgentProfile version of the same name (``refs.base`` = the
profile it was edited from) with the changed templates, skills or CLI settings. It does
not move ``current_profile``; ``approve_profile`` does, with a human and a reason, and
``batch_for`` builds the new version's batch root so new tasks can run it. Past runs keep
the profile version they recorded.

Placeholders are checked before anything is written: every template is rendered through
the same ``kta.agents.cards`` functions the runner uses, with sample values, so a
``$name`` no renderer provides is an error, and a provided placeholder the template no
longer uses is a warning.
"""

from __future__ import annotations

import difflib
import gzip
import io
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

from kta.agents import batches, cards
from kta.agents.profile import CURRENT_POINTER, unpack
from kta.core.errors import ConfigError
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import AgentProfile, TaskCard
from kta.store.base import Ops, Store

# Placeholders each renderer of kta.agents.cards provides (the runner's own sets).
PLACEHOLDERS: dict[str, frozenset[str]] = {
    "prefix": frozenset({"skills_table"}),
    "step": frozenset(
        {
            "task_id",
            "work_unit",
            "card_id",
            "goal",
            "done_checks",
            "inputs",
            "budget",
            "skills",
            "commands",
            "previous",
        }
    ),
    "rejection": frozenset({"task_id", "failures"}),
    "new_session": frozenset({"work_unit", "extra"}),
    "human_note": frozenset({"note"}),
    "budget_warning": frozenset({"task_id", "resource", "used", "remaining"}),
}
UNRENDERED = ("notes_seed",)  # copied verbatim (NOTES.md seed); home files likewise
EDITABLE_FIELDS = ("model", "reasoning_effort", "provider", "cli_version")
SAMPLE_POLICY = {"cards_k": 5, "card_max_lines": 5, "step_header_max_chars": 10**9}


def template_keys(p: AgentProfile) -> dict[str, str]:
    """Editable template key -> blob: ``prefix``, ``step:Tn``, ``message:<m>``, ``home:<f>``."""
    out = {"prefix": p.prefix_template}
    out.update({f"step:{t}": b for t, b in p.step_templates.items()})
    out.update({f"message:{m}": b for m, b in p.message_templates.items()})
    out.update({f"home:{f}": b for f, b in p.home_files.items()})
    return out


def renderer_of(key: str) -> str | None:
    kind, _, name = key.partition(":")
    if kind == "prefix":
        return "prefix"
    if kind == "step":
        return "step"
    if kind == "message" and name in PLACEHOLDERS:
        return name
    return None  # notes_seed, home files: not rendered


def sample_card(task_type: str, profile_id: str) -> TaskCard:
    return TaskCard(
        type=task_type,  # type: ignore[arg-type]
        goal=f"(sample) a {task_type} goal",
        inputs=[],
        done_checks=["delivery_note"],
        budget={"wall_s": 3600},
        profile=profile_id,
        work_unit="sample-unit",
    )


def check_template(key: str, text: str) -> tuple[list[str], list[str]]:
    """(errors, warnings) of one template, by rendering it as the runner would."""
    which = renderer_of(key)
    if which is None:
        return [], []
    tpl = Template(text)
    if not tpl.is_valid():
        return [f"{key}: a `$` that is not a placeholder (write `$$` for a literal $)"], []
    used = set(tpl.get_identifiers())
    allowed = PLACEHOLDERS[which]
    errors = [f"{key}: ${n} is not provided (allowed: {', '.join(sorted(allowed))})"
              for n in sorted(used - allowed)]  # fmt: skip
    warnings = [f"{key}: ${n} is provided but not used" for n in sorted(allowed - used)]
    return errors, warnings


def render_task(
    store: Store, text: str, card: TaskCard, card_id: str, skills: list[str], commands: list[str]
) -> str:
    """TASK.md as the runner writes it (cards.task_md), for a real or sample card."""
    from kta.agents.workspace import input_cards

    shown = input_cards(store, [i for i in card.inputs if store.exists(i)])
    return cards.task_md(
        text, task_id="(preview)", card_id=card_id, card=card, cards=shown, skills=skills,
        commands=commands, used={}, retry=None, policy=SAMPLE_POLICY,
    )  # fmt: skip


@dataclass
class Changes:
    templates: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # key -> text
    skills: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # name -> SKILL.md
    fields: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    @classmethod
    def of(cls, data: dict[str, Any]) -> Changes:
        unknown = set(data) - {"templates", "skills", "fields"}
        if unknown:
            raise ConfigError(f"draft changes have no section {sorted(unknown)}")
        bad = set(data.get("fields") or {}) - set(EDITABLE_FIELDS)
        if bad:
            raise ConfigError(f"profile fields {sorted(bad)} are not editable here")
        return cls(
            dict(data.get("templates") or {}),
            dict(data.get("skills") or {}),
            dict(data.get("fields") or {}),
        )


def text_of(store: Store, blob: str) -> str:
    return store.blob_path(blob).read_text("utf-8")


def skill_text(store: Store, p: AgentProfile, name: str) -> tuple[str, str] | None:
    """(ref, SKILL.md text) of the profile's bundle for skill ``name``."""
    for ref, sha in p.skill_bundles.items():
        if ref.partition("@")[0] == name:
            files = unpack(store.blob_path(sha).read_bytes())
            return ref, files["SKILL.md"].decode("utf-8")
    return None


def pack_files(files: dict[str, bytes]) -> bytes:
    """Deterministic tar.gz of ``{relative path: content}`` (as profile.pack_dir)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    return gzip.compress(buf.getvalue(), mtime=0)


def check(store: Store, base: AgentProfile, ch: Changes) -> tuple[list[str], list[str]]:
    keys = template_keys(base)
    errors: list[str] = [f"no template {k}" for k in ch.templates if k not in keys]
    warnings: list[str] = []
    for key, blob in keys.items():
        e, w = check_template(key, ch.templates.get(key, text_of(store, blob)))
        errors += e
        warnings += w
    for name, text in ch.skills.items():
        if skill_text(store, base, name) is None:
            errors.append(f"skill {name} is not in the profile")
        elif not text.lstrip().startswith("---") or "description:" not in text:
            errors.append(f"skill {name}: SKILL.md needs front matter with a description")
    if "provider" in ch.fields and ch.fields["provider"] != "codex":
        warnings.append("only the codex adapter is implemented; another provider cannot run")
    return errors, warnings


def apply(store: Store, base: AgentProfile, ch: Changes) -> dict[str, Any]:
    body = base.model_dump(mode="json")
    for key, text in ch.templates.items():
        kind, _, name = key.partition(":")
        blob = store.put_blob(text.encode("utf-8"), "text/markdown")
        if kind == "prefix":
            body["prefix_template"] = blob
        else:
            section = {"step": "step_templates", "message": "message_templates"}.get(kind)
            body[section or "home_files"][name] = blob
    for name, text in ch.skills.items():
        found = skill_text(store, base, name)
        assert found is not None  # check() refused unknown skills
        old_ref = found[0]
        files = unpack(store.blob_path(base.skill_bundles[old_ref]).read_bytes())
        files["SKILL.md"] = text.encode("utf-8")
        sha = store.put_blob(pack_files(files), "application/gzip")
        new_ref = f"{name}@{sha}"
        body["skill_bundles"] = {
            (new_ref if r == old_ref else r): (sha if r == old_ref else s)
            for r, s in body["skill_bundles"].items()
        }
        body["skills"] = {
            t: [new_ref if r == old_ref else r for r in refs] for t, refs in body["skills"].items()
        }
    body.update(ch.fields)
    return body


def next_version(store: Store, name: str) -> int:
    found = store.query("AgentProfile", where={"name": name}, limit=10000)
    return max((int(r.body["version"]) for r in found), default=0) + 1


def write_draft(
    store: Store, base_id: str, ch: Changes, author: str, at: datetime
) -> tuple[str, list[str]]:
    """A new profile version from ``base_id`` plus ``ch``; (record ID, warnings).

    Refused (ConfigError, nothing written) when a placeholder is not provided, a skill or
    template is unknown, or nothing changes.
    """
    base = AgentProfile.model_validate(store.get(base_id).body)
    errors, warnings = check(store, base, ch)
    if errors:
        raise ConfigError("; ".join(errors))
    body = apply(store, base, ch)
    if body == base.model_dump(mode="json"):
        raise ConfigError("the draft changes nothing")
    body["version"] = next_version(store, base.name)
    new = AgentProfile.model_validate(body)
    rec = Record.new("AgentProfile", new, Actor(type="human", name=author), at, {"base": base_id})
    return store.put(rec), warnings


def diff(store: Store, a_id: str, b_id: str) -> list[dict[str, Any]]:
    """Field and text differences from profile ``a_id`` to ``b_id``."""
    a = AgentProfile.model_validate(store.get(a_id).body)
    b = AgentProfile.model_validate(store.get(b_id).body)
    out: list[dict[str, Any]] = []
    for f in EDITABLE_FIELDS:
        if getattr(a, f) != getattr(b, f):
            out.append({"what": f, "old": getattr(a, f), "new": getattr(b, f)})
    ka, kb = template_keys(a), template_keys(b)
    for key in sorted(ka.keys() | kb.keys()):
        if ka.get(key) != kb.get(key):
            old = text_of(store, ka[key]).splitlines() if key in ka else []
            new = text_of(store, kb[key]).splitlines() if key in kb else []
            lines = difflib.unified_diff(old, new, f"{key} (old)", f"{key} (new)", lineterm="")
            out.append({"what": key, "diff": "\n".join(lines)})
    names = {r.partition("@")[0] for r in (*a.skill_bundles, *b.skill_bundles)}
    for name in sorted(names):
        sa, sb = skill_text(store, a, name), skill_text(store, b, name)
        if (sa and sa[0]) != (sb and sb[0]):
            old = sa[1].splitlines() if sa else []
            new = sb[1].splitlines() if sb else []
            lines = difflib.unified_diff(old, new, f"{name} (old)", f"{name} (new)", lineterm="")
            out.append({"what": f"skill:{name}", "diff": "\n".join(lines)})
    return out


def batch_for(store: Store, data_dir: Path, profile_id: str, commit: str) -> str:
    """The batch root of a profile version, ``<name>-v<version>``; built when absent."""
    p = AgentProfile.model_validate(store.get(profile_id).body)
    name = batches.profile_batch(p)
    root = batches.batches_root(data_dir) / name
    if not root.exists():
        batches.build(store, data_dir, name, profile_id, commit)
    elif batches.read_lock(root)["profile"] != profile_id:
        raise ConfigError(f"batch {name} exists for another profile record; rename the profile")
    return name


def current(ops: Ops) -> str | None:
    return ops.get_pointer(CURRENT_POINTER)


__all__ = ["Changes", "batch_for", "check", "diff", "write_draft"]
