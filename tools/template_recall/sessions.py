"""Launch recorded sessions through the existing persistent session launcher."""

from __future__ import annotations

import json
import os
import subprocess  # kta:allowed launches the recorded-session CLI, which uses the Executor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .gold import ROOT, load_task
from .package import build_package


def run_cell(runs: Path, task_id: str, arm: str, repeat: int) -> dict:
    """One fresh session per cell. A finished cell is kept, never rerun over."""
    run = runs / f"{task_id}-{arm}-loso-r{repeat}"
    if (run / "state/session.json").exists():
        return {"run": run.name, "status": "kept"}
    if run.exists():
        return {"run": run.name, "status": "failed earlier; inspect, then use a new --runs"}
    run.mkdir(parents=True)
    prompt = build_package(load_task(task_id), arm, "loso", run / "workspace")
    (run / "run.json").write_text(
        json.dumps({"task": task_id, "arm": arm, "view": "loso", "repeat": repeat})
    )
    done = subprocess.run(
        [
            str(ROOT / "scripts/template-session"),
            "start",
            "--workspace",
            str(run / "workspace"),
            "--state",
            str(run / "state"),
            "--prompt",
            str(prompt),
            "--wall-s",
            "1500",
        ],
        env={
            **os.environ,
            "KTA_ENV": "dev",
            "KTA_WP": "template-recall-e2",
            "KTA_MODE": "dev-mac",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    (run / "launcher.out").write_text(done.stdout)
    (run / "launcher.err").write_text(done.stderr)
    return {"run": run.name, "status": "ok" if done.returncode == 0 else f"rc={done.returncode}"}


def run_matrix(runs: Path, repeats: int, tasks: list[str], arms: list[str], parallel: int) -> list:
    cells = [(t, a, r) for r in range(1, repeats + 1) for t in tasks for a in arms]
    with ThreadPoolExecutor(parallel) as pool:
        return list(pool.map(lambda c: run_cell(runs, *c), cells))
