"""The agent workflow page (WP22 Part D, D593–D597): what configures each step of a task.

``GET /workflow`` derives the pipeline of every task type from the code and data that
run it, so the page cannot drift from them:

- who creates the task: ``creates_task`` transitions of ``machines.yaml`` and the
  ``TaskRequest("Tn", ...)`` calls in ``kta/orchestrator`` (scanned from the source);
- prompt assembly: the profile's prefix, step and message templates and skills (blobs);
- session: the ``Decision`` reasons of ``kta.agents.sessions.decide`` (read from its AST)
  and the session knobs in force;
- checks: ``DONE_CHECKS`` with each check function's source location; the task lifecycle
  of ``machines.yaml``;
- knobs from the controls registry, the provider's proxy policy (system configuration).

Edits never change a profile. ``POST /workflow/preview`` checks placeholders and renders
TASK.md; ``POST /workflow/drafts`` writes a new profile version (operator token);
``POST /workflow/drafts/{id}/approve`` moves ``current_profile`` and builds its batch.
"""

from __future__ import annotations

import ast
import inspect
import re
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict, StringConstraints

from kta.agents import drafts, sessions
from kta.agents.profile import CURRENT_POINTER
from kta.api import operate
from kta.api.common import Api, controls, machines
from kta.api.controls import TOKEN_HEADER, check_operator
from kta.core.controls import definitions
from kta.core.errors import ConfigError
from kta.core.ids import record_id
from kta.core.schemas.process import AgentProfile, TaskCard
from kta.core.sysconfig import load as load_system
from kta.orchestrator import done_checks
from kta.store import code_state

Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# Which controls knobs act at which node of the pipeline (ids are checked against the
# registry by the tests; the values always come from the registry).
NODE_KNOBS: dict[str, tuple[str, ...]] = {
    "card": ("budget.task_wall_s", "budget.gpu_s", "budget.tool_call_steps"),
    "prompt": (
        "session.step_header_size", "session.tool_output_truncation",
        "session.task_header_max_lines", "session.access_discipline_version",
        "session.notes_required_sections",
    ),
    "session": ("session.resume_scope", "session.fresh_session_threshold",
                "session.same_failure_rule"),
    "run": ("budget.hard_cap_factor", "budget.no_response_timeout_s", "budget.lease_s",
            "budget.agent_instances"),
    "retry": ("budget.retries_per_task", "budget.user_attention_cap"),
}  # fmt: skip
TASK_REQUEST = re.compile(r'TaskRequest\(\s*"(T\d)"')


def _src(obj: Any) -> str:
    path = Path(inspect.getsourcefile(obj) or "?")
    rel = path.as_posix().split("/src/", 1)[-1]
    return f"{rel}:{inspect.getsourcelines(obj)[1]}"


def _text(node: ast.expr) -> str:
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.IfExp):  # "a" if cond else "b": both readings
        return f"{_text(node.body)} / {_text(node.orelse)}"
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(v.value) if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"  # type: ignore[attr-defined]
            for v in node.values
        )
    return ast.unparse(node)


def session_rules() -> list[dict[str, str]]:
    """The (mode, reason) of each ``Decision`` in ``sessions.decide``, in source order."""
    tree = ast.parse(inspect.getsource(sessions.decide))
    out: list[tuple[int, dict[str, str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Decision":
            if len(node.args) == 2:
                rule = {"mode": _text(node.args[0]), "reason": _text(node.args[1])}
                out.append((node.lineno, rule))
    return [r for _, r in sorted(out, key=lambda x: x[0])]


def process_creators() -> dict[str, list[dict[str, str]]]:
    """``TaskRequest("Tn", ...)`` call sites in the orchestrator package."""
    out: dict[str, list[dict[str, str]]] = {}
    root = resources.files("kta.orchestrator")
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".py"):
            continue
        for no, line in enumerate(entry.read_text("utf-8").splitlines(), 1):
            for m in TASK_REQUEST.finditer(line):
                out.setdefault(m.group(1), []).append(
                    {"kind": "process", "where": f"kta/orchestrator/{entry.name}:{no}"}
                )
    return out


def machine_creators() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for m in machines()["machines"]:
        for e in m.get("transitions", []):
            if e.get("kind") != "creates_task":
                continue
            for t in e.get("creates", []):
                out.setdefault(t, []).append(
                    {"kind": "loop", "machine": m["id"], "from": e["from"], "to": e["to"],
                     "label": e.get("label"), "when": e.get("when"), "rule": e.get("rule")}
                )  # fmt: skip
    return out


def checks_of(task_type: str) -> list[dict[str, str]]:
    return [
        {"name": n, "source": _src(done_checks.CHECKS[n]) if n in done_checks.CHECKS else "?"}
        for n in done_checks.DONE_CHECKS.get(task_type, ())
    ]


def knobs(api: Api) -> dict[str, Any]:
    c = controls(api)
    defs = {k.id: k for k in definitions().knobs}
    return {
        "version": c.version_id,
        "nodes": {
            node: [
                {"id": i, "row": defs[i].row, "value": c.value(i), "editable": defs[i].editable,
                 "type": defs[i].type, "unit": defs[i].unit, "prevents": defs[i].prevents}
                for i in ids
            ]
            for node, ids in NODE_KNOBS.items()
        },
    }  # fmt: skip


def _profile_id(api: Api, profile: str | None) -> str:
    if profile:
        return profile
    cur = api.ops.get_pointer(CURRENT_POINTER)
    if cur:
        return cur
    found = api.store.query("AgentProfile", order="-created_at", limit=1)
    if not found:
        raise ConfigError("no AgentProfile stored yet (kta agents profile write)")
    return record_id(found[0])


def versions(api: Api, name: str, current: str | None) -> list[dict[str, Any]]:
    rows = api.store.query("AgentProfile", where={"name": name}, limit=1000)
    return sorted(
        (
            {"id": record_id(r), "version": r.body["version"], "current": record_id(r) == current,
             "base": r.refs.get("base"), "author": r.actor.name, "author_type": r.actor.type,
             "origin": r.origin}
            for r in rows
        ),
        key=lambda v: -int(v["version"]),
    )  # fmt: skip


def templates(api: Api, p: AgentProfile) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, blob in drafts.template_keys(p).items():
        which = drafts.renderer_of(key)
        out[key] = {
            "blob": blob,
            "text": drafts.text_of(api.store, blob),
            "placeholders": sorted(drafts.PLACEHOLDERS[which]) if which else None,
        }
    return out


def skills_view(api: Api, p: AgentProfile) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ref in sorted(p.skill_bundles):
        name = ref.partition("@")[0]
        found = drafts.skill_text(api.store, p, name)
        out[name] = {
            "ref": ref,
            "text": found[1] if found else "",
            "used_by": sorted(t for t, refs in p.skills.items() if ref in refs),
        }
    return out


def workflow_view(api: Api, profile: str | None) -> dict[str, Any]:
    pid = _profile_id(api, profile)
    rec = api.store.get(pid)
    p = AgentProfile.model_validate(rec.body)
    cur = api.ops.get_pointer(CURRENT_POINTER)
    by_loop, by_proc = machine_creators(), process_creators()
    data = machines()
    task_machine = next(m for m in data["machines"] if m["id"] == "task")
    return {
        "profile": {
            "id": pid, "name": p.name, "version": p.version, "provider": p.provider,
            "model": p.model, "reasoning_effort": p.reasoning_effort, "cli_version": p.cli_version,
            "isolation": p.isolation, "auth": p.auth, "current": pid == cur,
            "base": rec.refs.get("base"), "origin": rec.origin,
        },
        "versions": versions(api, p.name, cur),
        "task_types": [
            {
                **tt,
                "creators": by_loop.get(tt["id"], []) + by_proc.get(tt["id"], []),
                "step_template": f"step:{tt['id']}",
                "skills": [r.partition("@")[0] for r in p.skills.get(tt["id"], [])],
                "commands": p.allowed_commands.get(tt["id"], []),
                "done_checks": checks_of(tt["id"]),
            }
            for tt in data["task_types"]
        ],
        "lifecycle": {"states": task_machine["states"], "transitions": task_machine["transitions"],
                      "limits": task_machine.get("limits", {})},
        "session": {"rules": session_rules(), "source": _src(sessions.decide)},
        "templates": templates(api, p),
        "skills": skills_view(api, p),
        "knobs": knobs(api),
        "proxy": {"provider": p.provider,
                  **load_system(api.data_dir).provider(p.provider).model_dump(mode="json")},
        "invariants": [
            {"what": "done checks per task type (DONE_CHECKS; cards store the names)",
             "source": "kta/orchestrator/done_checks.py"},
            {"what": "agents cannot change done checks, profiles, templates or skills",
             "source": "spec/modules/agents.md (受控配置)"},
            {"what": "AGENTS.md above the workspace refuses the start",
             "source": "kta/agents/launch.py step"},
        ],
        "write": {"header": TOKEN_HEADER},
    }  # fmt: skip


class PreviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: str
    changes: dict[str, Any] = {}
    task_type: str = "T1"
    card: str | None = None  # a real TaskCard ID; else a sample card


class DraftBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: str
    changes: dict[str, Any]
    reason: Reason


class ApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: Reason


def preview(api: Api, b: PreviewBody) -> dict[str, Any]:
    base = AgentProfile.model_validate(api.store.get(b.base).body)
    ch = drafts.Changes.of(b.changes)
    errors, warnings = drafts.check(api.store, base, ch)
    if b.card:
        card = TaskCard.model_validate(api.store.get(b.card).body)
    else:
        card = drafts.sample_card(b.task_type, b.base)
    key = f"step:{card.type}"
    out: dict[str, Any] = {"errors": errors, "warnings": warnings, "card": b.card or "sample"}
    text = ch.templates.get(key) or drafts.text_of(api.store, base.step_templates[card.type])
    if not any(e.startswith(key + ":") for e in errors):
        skills = [r.partition("@")[0] for r in base.skills.get(card.type, [])]
        cmds = base.allowed_commands.get(card.type, [])
        out["task_md"] = drafts.render_task(api.store, text, card, b.card or "sample", skills, cmds)
    return out


def write_draft(api: Api, b: DraftBody) -> dict[str, Any]:
    human = operate.operator(api)
    pid, warnings = drafts.write_draft(
        api.store, b.base, drafts.Changes.of(b.changes), human.name, api.now()
    )
    action = operate.write_action(api, "draft_profile", b.reason, {"profile": pid, "base": b.base})
    return {"id": pid, "warnings": warnings, "diff": drafts.diff(api.store, b.base, pid),
            "action": action}  # fmt: skip


def approve(api: Api, pid: str, b: ApproveBody) -> dict[str, Any]:
    out = api.orchestration.approve_profile(pid, b.reason, operate.operator(api))
    batch = drafts.batch_for(api.store, api.data_dir, pid, code_state().commit)
    return {**out, "batch": batch}


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/workflow")
    def get_workflow(profile: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return workflow_view(api, profile)

    @r.get("/workflow/diff")
    def get_diff(a: str, b: str) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return drafts.diff(api.store, a, b)

    @r.post("/workflow/preview")
    def post_preview(b: PreviewBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return preview(api, b)

    @r.post("/workflow/drafts")
    def post_draft(  # pyright: ignore[reportUnusedFunction]
        b: DraftBody, token: str | None = Header(None, alias=TOKEN_HEADER)
    ) -> dict[str, Any]:
        check_operator(api, token)
        return write_draft(api, b)

    @r.post("/workflow/drafts/{pid}/approve")
    def post_approve(  # pyright: ignore[reportUnusedFunction]
        pid: str, b: ApproveBody, token: str | None = Header(None, alias=TOKEN_HEADER)
    ) -> dict[str, Any]:
        check_operator(api, token)
        return approve(api, pid, b)

    return r
