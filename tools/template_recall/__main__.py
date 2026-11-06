"""Run the recall evaluation: deterministic baseline, session packages, session scoring."""

from __future__ import annotations

import argparse
import json
import subprocess  # kta:allowed git identity for an offline report
from pathlib import Path

from template_access.store import Library

from .deterministic import evaluate_task, summarise
from .gold import ROOT, TASKS, load_task, load_tasks
from .package import build_package
from .score import score_all
from .sessions import run_matrix


def cmd_deterministic(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    full: dict = {"git_commit": commit, "results": []}
    brief: dict = {"git_commit": commit, "note": "No LLM. Retrieval from task text.", "rows": []}
    for task in tasks:
        for view, exclude in (("full", None), ("loso", task.source_study)):
            library = Library(exclude)
            result = evaluate_task(task, view, library)
            full["library_fingerprint"] = library.fingerprint
            full["results"].append(result)
            brief["rows"].append(
                {
                    "task": task.id,
                    "view": view,
                    "view_size": result["view_size"],
                    **summarise(result),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(full, indent=1))
    args.summary.write_text(json.dumps(brief, indent=1) + "\n")
    print(f"{len(full['results'])} task views written to {args.out}")
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    print(build_package(load_task(args.task), args.arm, args.view, args.workspace))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    report = score_all(args.runs)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["totals"], indent=1))
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    print(json.dumps(run_matrix(args.runs, args.repeats, args.tasks, args.arms, args.parallel)))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("deterministic")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--summary", type=Path, required=True)
    c.set_defaults(run=cmd_deterministic)
    c = sub.add_parser("package")
    c.add_argument("--task", required=True)
    c.add_argument("--arm", choices=("access", "flat"), required=True)
    c.add_argument("--view", choices=("full", "loso"), required=True)
    c.add_argument("--workspace", type=Path, required=True)
    c.set_defaults(run=cmd_package)
    c = sub.add_parser("score")
    c.add_argument("--runs", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(run=cmd_score)
    c = sub.add_parser("sessions")
    c.add_argument("--runs", type=Path, required=True)
    c.add_argument("--repeats", type=int, default=2)
    c.add_argument("--tasks", nargs="*", default=list(TASKS))
    c.add_argument("--arms", nargs="*", default=["access", "flat"])
    c.add_argument("--parallel", type=int, default=2)
    c.set_defaults(run=cmd_sessions)
    args = p.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
