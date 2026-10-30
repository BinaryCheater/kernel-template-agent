"""Shared loaders for the WP13 analyses of the Formulaic F2 (v7) Attention batch.

The inputs live in the read-only research repository. Nothing here writes to it.
Every derived value keeps the raw file paths it came from so that the delivery
note can cite a source file for each number.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESEARCH_ROOT = Path("/Users/dingyi/Projects/agentic-codesign")
F2_ROOT = RESEARCH_ROOT / "kernel-lab/results/formulaic-reuse-f2-20261027-01"
V7_DIR = "formal-v7"
EXPORT_DIR = "formal-v7/export-attention-20261027-01"
TASKS = ("attn-small", "attn-medium", "attn-long")
ARMS = ("reconstruct", "text", "template")
REPS = (0, 1, 2)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: Path | str, root: Path = RESEARCH_ROOT) -> str:
    """Path relative to the research repo, the form used in citations."""
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def trajectory_path(f2_root: Path, task: str, arm: str, rep: int) -> Path:
    return f2_root / V7_DIR / f"trajectory-{task}-{arm}-rep{rep}-v7.json"


def iter_trajectories(f2_root: Path):
    for task in TASKS:
        for arm in ARMS:
            for rep in REPS:
                path = trajectory_path(f2_root, task, arm, rep)
                yield task, arm, rep, path, load_json(path)


def export_records(f2_root: Path) -> tuple[Path, dict[tuple[str, str], dict]]:
    path = f2_root / EXPORT_DIR / "results.json"
    data = load_json(path)
    return path, {(r["task"], r["run_id"]): r for r in data["records"]}


def is_v7_stage(name: str) -> bool:
    return "-v7-" in name


def feedback_dir_for(evaluation: str) -> Path:
    p = Path(evaluation)
    return p.parent / p.name.replace("evaluate-", "feedback-", 1)


def solve_receipt_for(solver: str) -> Path:
    p = Path(solver)
    return p.parent / f"{p.name}.receipt.json"


# --- statistics -------------------------------------------------------------


def quantile(values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile (same as numpy's default 'linear')."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def describe(values: list[float]) -> dict[str, float | int | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "sum": sum(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p10": quantile(vals, 0.10),
        "p90": quantile(vals, 0.90),
        "min": min(vals),
        "max": max(vals),
    }


def kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


# --- author session parsing -------------------------------------------------

_ZSH = re.compile(r"^/bin/zsh -c (['\"])(.*)\1$", re.DOTALL)
_WRITE_HINTS = (
    "write_text(",
    "write_bytes(",
    ".write(",
    "json.dump(",
    "cp ",
    "mv ",
    "tee ",
    "apply_patch",
)
_READ_HEADS = ("cat", "sed", "head", "tail", "ls", "rg", "grep", "find", "wc", "pwd")
_STDERR_TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z\s+(\w+)\s+(\S+):")
_RETRY_MARKERS = ("retrying sampling request", "falling back to HTTP")
RETRY_CHAIN_GAP_S = 60.0


def _inner(command: str) -> str:
    m = _ZSH.match(command)
    return m.group(2) if m else command


def classify_command(command: str) -> str:
    """Heuristic action class of one shell command an author ran.

    read  : prints existing files or listings into the context;
    write : creates or rewrites workspace files (code, formulation, notes);
    check : runs Python that computes or validates (audits, CPU checks, parsing
            feedback numbers) without writing files;
    other : anything else.
    The classes describe what entered or left the context, not time spent:
    the F2 sessions carry no per-command timestamps.
    """
    inner = _inner(command).strip()
    if not inner:
        return "other"
    if any(h in inner for h in _WRITE_HINTS) or re.search(r"(^|[^2])>\s*\S", inner):
        return "write"
    head = re.split(r"[\s;&|]+", inner, maxsplit=1)[0]
    head = os.path.basename(head)
    if head.startswith("python"):
        return "check"
    if head in _READ_HEADS or head == "zsh":
        return "read"
    return "other"


def read_targets(command: str) -> set[str]:
    """Which input group a read command touches (knowledge pack, feedback...)."""
    inner = _inner(command)
    groups = set()
    for group, pat in (
        ("knowledge", r"\bknowledge/"),
        ("feedback", r"\bfeedback\b"),
        ("history", r"\bhistory/"),
        ("reference", r"\breference/"),
        ("skill", r"\.agents/skills"),
        ("task_inputs", r"\b(task\.json|hardware\.json|eager\.py|request\.md)\b"),
        (
            "own_artifacts",
            r"\b(program\.py|formulation\.json|hypothesis\.md|audit\.py)\b",
        ),
    ):
        if re.search(pat, inner):
            groups.add(group)
    return groups


@dataclass
class SessionStats:
    commands: dict[str, int] = field(default_factory=dict)
    output_bytes: dict[str, int] = field(default_factory=dict)
    read_bytes_by_target: dict[str, int] = field(default_factory=dict)
    failed_commands: int = 0
    file_change_items: int = 0
    agent_messages: int = 0
    stream_errors: int = 0
    reads_before_first_write: int | None = None


def parse_stdout(path: Path) -> SessionStats:
    stats = SessionStats()
    reads_so_far = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            item = event.get("item") or {}
            if etype == "error":
                stats.stream_errors += 1
            if etype != "item.completed":
                continue
            itype = item.get("type")
            if itype == "agent_message":
                stats.agent_messages += 1
            elif itype == "file_change":
                stats.file_change_items += 1
                stats.commands["write"] = stats.commands.get("write", 0) + 1
                if stats.reads_before_first_write is None:
                    stats.reads_before_first_write = reads_so_far
            elif itype == "command_execution":
                cmd = item.get("command", "")
                cls = classify_command(cmd)
                if cls == "read":
                    reads_so_far += 1
                elif cls == "write" and stats.reads_before_first_write is None:
                    stats.reads_before_first_write = reads_so_far
                nbytes = len(item.get("aggregated_output") or "")
                stats.commands[cls] = stats.commands.get(cls, 0) + 1
                stats.output_bytes[cls] = stats.output_bytes.get(cls, 0) + nbytes
                if item.get("exit_code") not in (0, None):
                    stats.failed_commands += 1
                if cls == "read":
                    for tgt in read_targets(cmd) or {"unclassified"}:
                        stats.read_bytes_by_target[tgt] = (
                            stats.read_bytes_by_target.get(tgt, 0) + nbytes
                        )
    return stats


def retry_chains(stderr: Path) -> list[tuple[datetime, datetime]]:
    """Provider reconnect chains, from stderr timestamps (UTC).

    A chain is a run of retry warnings / HTTP fallbacks each at most
    RETRY_CHAIN_GAP_S after the previous one.
    """
    if not stderr.exists():
        return []
    marks: list[datetime] = []
    with stderr.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _STDERR_TS.match(line)
            if m and any(k in line for k in _RETRY_MARKERS):
                ts = datetime.fromisoformat(m.group(1)).replace(tzinfo=UTC)
                marks.append(ts)
    chains: list[tuple[datetime, datetime]] = []
    for ts in sorted(marks):
        if chains and (ts - chains[-1][1]).total_seconds() <= RETRY_CHAIN_GAP_S:
            chains[-1] = (chains[-1][0], ts)
        else:
            chains.append((ts, ts))
    return chains


def retry_spans_s(stderr: Path) -> float:
    """Sum of chain spans: a lower bound on time lost to provider reconnects,
    since the request that timed out began before the first warning."""
    return sum((b - a).total_seconds() for a, b in retry_chains(stderr))


def startup_stall_s(stderr: Path, session_start_s: float) -> float:
    """Upper bound on the connection stall at session start: from the session
    start to the end of the first chain, if that chain begins within
    RETRY_CHAIN_GAP_S of the start; otherwise 0."""
    chains = retry_chains(stderr)
    if not chains:
        return 0.0
    first, last = chains[0]
    if first.timestamp() - session_start_s > RETRY_CHAIN_GAP_S:
        return 0.0
    return last.timestamp() - session_start_s


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=False)
        fh.write("\n")
