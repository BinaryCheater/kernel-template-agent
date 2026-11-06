"""python -m tools.template_chain chain-check WORKSPACE JSON | round RUN TASK N | session ...

`session` runs a cell: authoring, round 1, the repair turn, round 2.
"""

import fcntl
import json
import os
import subprocess  # kta:allowed launches the recorded-session CLI, as tools/template_recall does
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.template_chain import package, rounds, workers  # noqa: E402
from tools.template_chain.tasks import TASKS  # noqa: E402

REPAIR = """{report}

The file above is also saved as report.md. You may revise form.py and family.py once, using
what the measurements show: missing mechanisms in the amounts, or a better structure. Keep
checking with ./chain check. Append to DESIGN.md what you changed and why. Then stop; a second
search runs on what you leave here. The checks you have left still apply.
"""


def chain_check(workspace: Path, x: str) -> int:
    state_path = workspace / "state.json"
    with state_path.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.load(stream)
        if state["checks"] >= package.MAX_CHECKS:
            print("check budget used")
            return 1
        state["checks"] += 1
        stream.seek(0)
        stream.truncate()
        json.dump(state, stream)
    task = TASKS[state["task"]]
    assignment = json.loads(x)
    job = {"id": "check", "kind": "candidate", "x": assignment, "shape": task["dev_shape"]}
    got = workers.compile_jobs(state["task"], workspace, [job], timeout=900)["check"]
    log = workspace / "checks.jsonl"
    with log.open("a") as stream:
        stream.write(json.dumps({"x": assignment, **got}) + "\n")
    print(json.dumps({k: v for k, v in got.items() if k != "samples_us"}, indent=1))
    print(f"checks used {state['checks']} of {package.MAX_CHECKS}")
    return 0 if got.get("correct") else 1


def session(mode: str, state: Path, workspace: Path, prompt: Path) -> dict:
    done = subprocess.run(
        [
            str(ROOT / "scripts/template-session"),
            mode,
            "--workspace",
            str(workspace),
            "--state",
            str(state),
            "--prompt",
            str(prompt),
            "--wall-s",
            str(package.WALL_S),
        ],
        env={
            **os.environ,
            "KTA_DATA": "/Users/dingyi/kta-data",
            "KTA_ENV": "dev",
            "KTA_WP": "template-chain-e3b",
            "KTA_MODE": "dev-mac",
        },
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    (state.parent / f"launcher-{mode}.out").write_text(done.stdout + done.stderr)
    return {"rc": done.returncode}


def cell(runs: Path, task: str, arm: str, repeat: int) -> dict:
    run = runs / f"{task}-{arm}-r{repeat}"
    log: dict = {"run": run.name}
    done = run / "cell.json"
    if done.exists():
        return json.loads(done.read_text())
    run.mkdir(parents=True, exist_ok=True)
    ws = run / "workspace"
    if not ws.exists():
        prompt = package.build(ws, task, arm)
        (run / "run.json").write_text(json.dumps({"task": task, "arm": arm, "repeat": repeat}))
        log["author"] = session("start", run / "state", ws, prompt)
    for n in (1, 2):
        if not (ws / "form.py").exists() or not (ws / "family.py").exists():
            log["status"] = "no family written"
            break
        try:
            r = rounds.run_round(run, task, n)
        except Exception as e:  # noqa: BLE001
            log[f"round{n}"] = {"status": f"harness or form failure: {type(e).__name__}: {e}"}
            r = {"status": "failed"}
        else:
            log[f"round{n}"] = {
                k: r.get(k) for k in ("status", "best_us", "comparator_us", "median_abs_error")
            }
        if r.get("status") != "ok" or n == 2:
            if r.get("status") != "ok" and n == 1:
                (run / "failure.json").write_text(json.dumps(r, indent=1, default=str)[:20000])
            break
        text = rounds.report(run, n)
        (ws / "report.md").write_text(text)
        prompt = run / "repair.md"
        prompt.write_text(REPAIR.format(report=text))
        log["repair"] = session("resume", run / "state", ws, prompt)
    done.write_text(json.dumps(log, indent=1))
    return log


def main() -> int:
    command = sys.argv[1]
    if command == "chain-check":
        return chain_check(Path(sys.argv[2]), sys.argv[-1])
    if command == "round":
        out = rounds.run_round(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=1)[:4000])
        return 0
    if command == "cells":
        runs = Path(sys.argv[2])
        tasks, arms = sys.argv[3].split(","), sys.argv[4].split(",")
        repeats, parallel = int(sys.argv[5]), int(sys.argv[6])
        cells = [(t, a, r) for r in range(1, repeats + 1) for t in tasks for a in arms]
        with ThreadPoolExecutor(parallel) as pool:
            print(json.dumps(list(pool.map(lambda c: cell(runs, *c), cells)), indent=1))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
