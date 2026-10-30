"""Compliance accounting: violations of one run, from its session events and launcher report.

Written into ``AgentRun.violations`` (spec/modules/agents.md "行为合规监测"). Every
violation is ``{"kind", "detail", "at", ...}``; kinds are an open table (AGENTS.md
principle 1), the ones produced here are listed in KINDS. Detection reads what the run
left behind: tool results (denials, exit codes), the commands it ran, the files its
patches touched, and the launcher's watch, guard and workspace checks.
"""

from __future__ import annotations

import re
import shlex
from collections import Counter
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

from kta.agents.base import AgentEvent

KINDS = (
    "denied",  # the sandbox or the OS refused an operation
    "kta_command_not_allowed",  # a kta subcommand outside allowed_commands for the task type
    "forbidden_tool",  # direct remote, GPU or network tools instead of kta
    "write_outside_workspace",  # patch/redirect outside the workspace, or a watched path changed
    "gpu_outside_kta",  # the GPU guard found a compute process under the agent
    "notes_not_updated",
    "workspace_layout",  # top-level workspace entries other than the allowed ones
    "context_file_modified",  # AGENTS.md, TASK.md, inputs/ or skills changed by the agent
    "secret_in_session",
)
DENIAL = re.compile(
    r"permission denied|operation not permitted|read-only file system"
    r"|sandbox(?:ing)? (?:denied|blocked)|landlock|seatbelt|EACCES|EPERM",
    re.IGNORECASE,
)
FORBIDDEN_TOOLS = frozenset(
    {"ssh", "scp", "rsync", "nvidia-smi", "nvcc", "ncu", "nsys", "curl", "wget", "sudo"}
)
REDIRECT = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*(/[^\s;|&]+)")
PROTECTED_CONTEXT = ("AGENTS.md", "TASK.md")
PROTECTED_DIRS = ("inputs/", ".agents/skills/")


def _commands(ev: AgentEvent) -> list[list[str]]:
    """Shell words of each simple command in an exec result (``bash -lc "..."`` unwrapped)."""
    exec_ = ev.data.get("exec") or {}
    argv = [str(a) for a in exec_.get("command") or []]
    if (
        len(argv) >= 3
        and PurePosixPath(argv[0]).name in ("bash", "sh", "zsh")
        and argv[1] in ("-lc", "-c")
    ):
        script = argv[2]
    elif argv:
        return [argv]
    else:
        return []
    out: list[list[str]] = []
    for part in re.split(r"&&|\|\||;|\||\n", script):
        try:
            words = shlex.split(part)
        except ValueError:
            words = part.split()
        if words:
            out.append(words)
    return out


def _script(ev: AgentEvent) -> str:
    exec_ = ev.data.get("exec") or {}
    return " ".join(str(a) for a in exec_.get("command") or [])


def _outside(path: str, workspace: str) -> bool:
    p = PurePosixPath(path)
    if not p.is_absolute():
        return ".." in p.parts
    ws = PurePosixPath(workspace)
    return p != ws and ws not in p.parents and not str(p).startswith("/dev/null")


def from_events(
    events: Iterable[AgentEvent], workspace: str, allowed: list[str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.kind != "tool_result":
            continue
        exec_ = ev.data.get("exec") or {}
        text = f"{ev.data.get('output', '')}\n{exec_.get('output_tail', '')}"
        code = exec_.get("exit_code")
        if code not in (None, 0) and DENIAL.search(text):
            out.append({"kind": "denied", "detail": _script(ev)[:300] or text[:300], "at": ev.at})
        for words in _commands(ev):
            tool = PurePosixPath(words[0]).name
            if (
                tool == "kta"
                and len(words) > 1
                and words[1] not in allowed
                and not words[1].startswith("-")
            ):
                out.append(
                    {
                        "kind": "kta_command_not_allowed",
                        "detail": " ".join(words)[:300],
                        "at": ev.at,
                    }
                )
            elif tool in FORBIDDEN_TOOLS:
                out.append({"kind": "forbidden_tool", "detail": " ".join(words)[:300], "at": ev.at})
        for target in REDIRECT.findall(_script(ev)):
            if _outside(target, workspace):
                out.append(
                    {
                        "kind": "write_outside_workspace",
                        "detail": f"redirect to {target}",
                        "at": ev.at,
                    }
                )
        for path in (ev.data.get("patch") or {}).get("paths") or []:
            if _outside(str(path), workspace):
                out.append(
                    {"kind": "write_outside_workspace", "detail": f"patch {path}", "at": ev.at}
                )
    return out


def from_report(report: dict[str, Any], notes_required: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in report.get("watch_changed") or []:
        out.append(
            {
                "kind": "write_outside_workspace",
                "detail": f"watched path changed: {path}",
                "at": None,
            }
        )
    for g in report.get("gpu_guard") or []:
        out.append({"kind": "gpu_outside_kta", "detail": str(g), "at": None})
    if (
        notes_required
        and report.get("exit") in ("ok", "nonzero")
        and not report.get("notes_changed")
    ):
        out.append(
            {"kind": "notes_not_updated", "detail": "NOTES.md unchanged by the step", "at": None}
        )
    for name in report.get("workspace_extra") or []:
        out.append(
            {"kind": "workspace_layout", "detail": f"unexpected top-level entry {name}", "at": None}
        )
    for member in report.get("secret_members") or []:
        out.append(
            {"kind": "secret_in_session", "detail": f"secret pattern in {member}", "at": None}
        )
    return out


def context_changes(manifest_files: dict[str, str], after: dict[str, str]) -> list[dict[str, Any]]:
    """Files the agent must not change (prefix, task, inputs, skills) compared with the manifest."""
    out: list[dict[str, Any]] = []
    for rel, sha in manifest_files.items():
        if rel in PROTECTED_CONTEXT or rel.startswith(PROTECTED_DIRS):
            if after.get(rel) != sha:
                out.append({"kind": "context_file_modified", "detail": rel, "at": None})
    for rel in after:
        if rel.startswith(PROTECTED_DIRS) and rel not in manifest_files:
            out.append({"kind": "context_file_modified", "detail": f"added {rel}", "at": None})
    return out


def summary(violations: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(v["kind"]) for v in violations))
