"""Score recorded sessions against gold: surfaced, opened and selected, per move."""

from __future__ import annotations

import json
from pathlib import Path

from template_access.store import Library

from .gold import Task, load_task
from .transcript import access_observations, audit, flat_observations, tool_events


def move_stage(targets: set[str], seen: dict) -> str:
    if not targets:
        return "absent_from_view"
    if targets & set(seen["selected"]):
        return "selected"
    if targets & set(seen["opened"]):
        return "opened_not_selected"
    return "surfaced_not_opened" if targets & set(seen["surfaced"]) else "not_surfaced"


def flat_selection(workspace: Path, view: set[str]) -> tuple[list[str], list[str]]:
    path = workspace / "SELECTION.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    picks = [r["identity"] for r in rows]
    needs = workspace / "NEEDS.md"
    return [p for p in picks if p in view], [needs.read_text()] if needs.exists() else []


def score_run(run: Path, task: Task, arm: str, view_name: str) -> dict:
    session = json.loads((run / "state/session.json").read_text())
    workspace = run / "workspace"
    library = Library(task.source_study if view_name == "loso" else None)
    view = set(library.by_identity)
    events = tool_events(Path(session["native_transcripts"][0]))
    if arm == "access":
        seen = access_observations(workspace / "access-log.jsonl", view)
    else:
        seen = flat_observations(events, view)
        seen["selected"], seen["needs"] = flat_selection(workspace, view)
        # Selecting a rule proves it reached the session, whatever the parser saw.
        seen["surfaced"] = sorted(set(seen["surfaced"]) | set(seen["selected"]))
    gold_targets = {t for move in task.moves for t in move.targets("lenient", view)}
    moves = []
    for move in task.moves:
        row = {"move": move.id, "role": move.role}
        for scoring in ("strict", "lenient"):
            row[scoring] = move_stage(move.targets(scoring, view), seen)
        moves.append(row)
    return {
        "run": run.name,
        "task": task.id,
        "arm": arm,
        "view": view_name,
        "moves": moves,
        "counts": {k: len(seen[k]) for k in ("surfaced", "opened", "selected")},
        "layer_counts": seen.get("layer_counts"),
        "selected": seen["selected"],
        "selected_outside_gold": sorted(set(seen["selected"]) - gold_targets),
        "needs": seen["needs"],
        "audit": audit(events, arm, workspace),
        "session": session["session"],
        "agent_run": session["agent_run"],
        "model": session["model"],
        "effort": session["effort"],
        "exit": session["exit"],
        "wall_s": session["wall_s"],
        "usage": session["usage"],
    }


def totals(rows: list[dict]) -> dict:
    out: dict = {}
    for row in rows:
        group = out.setdefault(f"{row['arm']}/{row['view']}", {"strict": {}, "lenient": {}})
        for move in row["moves"]:
            for scoring in ("strict", "lenient"):
                stage = move[scoring]
                group[scoring][stage] = group[scoring].get(stage, 0) + 1
    return out


def score_all(runs: Path) -> dict:
    rows = []
    for meta in sorted(runs.glob("*/run.json")):
        info = json.loads(meta.read_text())
        rows.append(score_run(meta.parent, load_task(info["task"]), info["arm"], info["view"]))
    return {"runs": rows, "totals": totals(rows)}
