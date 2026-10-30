"""GET /agents: runtime skills, AgentProfile versions, the profile pointer, batch environments.

- Runtime skills are the directories under ``kta/agents/skills/`` (spec/modules/agents.md
  "Runtime skills"), with their front matter, plus the rows of the catalogue table in
  ``skills/README.md`` when it exists (D321). Construction skills are not listed here.
- Profiles are AgentProfile records; ``current_profile`` is the ops pointer, its history
  comes from the ops event log and the approve_profile HumanActions.
- Batch environments are ``<data dir>/batches/<batch>/env.lock.json`` (D322); whether a
  batch still matches its lock is WP8's ``kta agents env verify``, delegated (wiring.py).
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

from kta.api import operate, opsread
from kta.api.common import Api, iso
from kta.core.errors import ConfigError
from kta.core.ids import record_id

CURRENT_PROFILE = "current_profile"
# Profile fields compared between consecutive versions of one profile name.
DIFF_FIELDS = (
    "provider", "cli_version", "auth", "isolation", "model", "reasoning_effort", "cli_flags",
    "home_files", "prefix_template", "step_templates", "message_templates", "skills",
    "allowed_commands", "context_policy", "loop_policy", "api_base",
)  # fmt: skip


def _front_matter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and not line.startswith((" ", "\t")):
            out[key.strip()] = value.strip()
    return out


def markdown_table(text: str) -> list[dict[str, str]]:
    """Rows of the first pipe table in ``text``, keyed by its header cells."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            rows.append([c.strip() for c in s.strip("|").split("|")])
        elif rows:
            break
    if len(rows) < 2:
        return []
    header = [h.strip("`* ").lower() for h in rows[0]]
    return [dict(zip(header, r, strict=False)) for r in rows[2:]]


def _skills_root() -> Traversable:
    return resources.files("kta.agents").joinpath("skills")


def skills(used: dict[str, list[str]]) -> dict[str, Any]:
    """The runtime skill catalogue; ``used`` maps skill name -> task types (current profile)."""
    root = _skills_root()
    readme = root.joinpath("README.md")
    catalogue = markdown_table(readme.read_text("utf-8")) if readme.is_file() else []
    by_name = {r.get("skill", "").strip("`"): r for r in catalogue}
    out: list[dict[str, Any]] = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        skill_md = d.joinpath("SKILL.md")
        if not skill_md.is_file():
            continue
        text = skill_md.read_text("utf-8")
        fm = _front_matter(text)
        out.append(
            {
                "name": d.name,
                "description": fm.get("description", ""),
                "lines": len(text.splitlines()),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "files": sorted(p.name for p in d.iterdir() if p.is_file()),
                "catalogue": by_name.get(d.name),
                "used_by": used.get(d.name, []),
            }
        )
    listed = {s["name"] for s in out}
    planned = [r for name, r in by_name.items() if name and name not in listed]
    return {"has_catalogue": bool(catalogue), "skills": out, "planned": planned}


def _profile(api: Api, rid: str, current: str | None) -> dict[str, Any]:
    rec = api.store.get_json(rid)
    b = rec["body"]
    return {
        "id": rid,
        "created_at": rec["created_at"],
        "origin": rec["origin"],
        "status": api.store.status(rid)[0],
        "current": rid == current,
        **{k: b.get(k) for k in ("name", "version", "provider", "cli_version", "auth")},
        **{k: b.get(k) for k in ("isolation", "model", "reasoning_effort", "cli_flags")},
        "skills": b.get("skills", {}),
        "skill_bundles": b.get("skill_bundles", {}),
        "allowed_commands": {t: len(v) for t, v in b.get("allowed_commands", {}).items()},
        "context_policy": b.get("context_policy", {}),
        "loop_policy": b.get("loop_policy", {}),
        "templates": {
            "prefix": b.get("prefix_template"),
            **{f"step {t}": v for t, v in b.get("step_templates", {}).items()},
            **{f"message {t}": v for t, v in b.get("message_templates", {}).items()},
            **{f"home {t}": v for t, v in b.get("home_files", {}).items()},
        },
    }


def _diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return [f for f in DIFF_FIELDS if old.get(f) != new.get(f)]


def profiles(api: Api) -> dict[str, Any]:
    current = api.ops.get_pointer(CURRENT_PROFILE)
    recs = api.store.query("AgentProfile", order="created_at", limit=1000)
    rows: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] = {}
    for rec in recs:
        rid = record_id(rec.to_json_dict())
        row = _profile(api, rid, current)
        before = previous.get(row["name"])
        row["changed"] = _diff(before, rec.body) if before is not None else None
        previous[row["name"]] = rec.body
        rows.append(row)
    history = [
        {"at": iso(at), "from": ch.get("from"), "to": ch.get("to")}
        for at, ch in opsread.pointer_history(api.ops, CURRENT_PROFILE)
    ]
    approvals = [a for a in _actions(api) if a["verb"] == "approve_profile"]
    return {"current": current, "profiles": rows, "history": history, "approvals": approvals}


def _actions(api: Api) -> list[dict[str, Any]]:
    out = []
    for rec in api.store.query("HumanAction", order="-created_at", limit=500):
        out.append(
            {
                "id": record_id(rec.to_json_dict()),
                "at": iso(rec.created_at),
                "by": rec.actor.name,
                "action": rec.body["action"],
                "reason": rec.body["reason"],
                "refs": rec.refs,
                **operate.parse_action(rec.body["action"]),
            }
        )
    return out


def batches(api: Api) -> list[dict[str, Any]]:
    if api.data_dir is None:
        return []
    root = api.data_dir / "batches"
    if not root.is_dir():
        return []
    gap = api.orchestration.missing().get("env-verify")
    out: list[dict[str, Any]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        lock_path = d / "env.lock.json"
        row: dict[str, Any] = {"batch": d.name, "path": str(d), "lock": None}
        if lock_path.is_file():
            try:
                lock = json.loads(lock_path.read_text("utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{lock_path} is not JSON: {exc}") from exc
            files = lock.get("files", {})
            row["lock"] = {k: v for k, v in lock.items() if k != "files"}
            row["files"] = len(files) if isinstance(files, dict) else None
            row["lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        row["work_units"] = (
            sorted(p.name for p in (d / "work").iterdir() if p.is_dir())
            if (d / "work").is_dir()
            else []
        )
        row["verify"] = {"state": "not_supported", "needs": gap} if gap else {"state": "unknown"}
        out.append(row)
    return out


def agents(api: Api) -> dict[str, Any]:
    prof = profiles(api)
    current = next((p for p in prof["profiles"] if p["current"]), None)
    used: dict[str, list[str]] = {}
    for ttype, refs in (current or {}).get("skills", {}).items():
        for ref in refs:
            used.setdefault(str(ref).split("@", 1)[0], []).append(ttype)
    return {
        "skills": skills(used),
        **prof,
        "batches": batches(api),
        "batches_root": str(api.data_dir / "batches") if api.data_dir else None,
    }
