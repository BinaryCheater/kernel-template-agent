"""The full table, once per round, on spark (LOOP §8): deploy the committed HEAD, carry the
extracted upstream inputs and the replay cache over, run there, bring table, cache and run record
back. Refused during a GPU measurement and from a dirty tree."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from . import upstream
from .measure import HOST, REMOTE_ROOT

ROOT = upstream.ROOT
LOG = ROOT / "perfmodel" / "table-runs.jsonl"
STATE = ROOT / "docs" / "design" / "perfmodel" / "STATE.md"
JOBS = 18  # of spark's 20 cores


def current_round() -> int:
    """The round in progress: one past the last report in STATE.md."""
    done = re.findall(r"^Round (\d+):", STATE.read_text(), re.M)
    return max(map(int, done), default=0) + 1


def log(record: dict) -> None:
    rec = {
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "round": current_round(),
        **record,
    }
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def runs(round_no: int) -> list[dict]:
    lines = LOG.read_text().splitlines() if LOG.exists() else []
    return [r for r in map(json.loads, lines) if r["round"] == round_no]


def refuse_second() -> None:
    n = current_round()
    if any(r["mode"] == "full" for r in runs(n)):
        raise SystemExit(f"refused: round {n} has had its full table (perfmodel/table-runs.jsonl)")


def _ssh(cmd: str) -> str:
    print("+ ssh", HOST, cmd, flush=True)
    return subprocess.run(["ssh", HOST, cmd], check=True, text=True, stdout=subprocess.PIPE).stdout


def _rsync(src: str, dst: str) -> None:
    print("+ rsync", src, dst, flush=True)
    subprocess.run(["rsync", "-a", src, dst], check=True)


def on_spark(table_dir: Path) -> None:
    refuse_second()
    head, dirty = upstream.kta_commit()
    if dirty:
        raise SystemExit("refused: uncommitted changes; a table belongs to a commit")
    busy = _ssh(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader; "
        "pgrep -f [p]erfmodel/remote/run.py || true"
    ).strip()
    if busy:
        raise SystemExit(f"refused: spark's GPU is in use: {busy}")
    subprocess.run([str(ROOT / "scripts" / "remote-run"), "sync", HOST], check=True)
    tree = f"{REMOTE_ROOT}/src/{head}"
    _ssh(f"mkdir -p {tree}/data/perfmodel/cache")
    data = ROOT / "data" / "perfmodel"
    for up in data.glob("upstream-*"):
        _rsync(f"{up}", f"{HOST}:{tree}/data/perfmodel/")
    _rsync(f"{data}/cache/", f"{HOST}:{tree}/data/perfmodel/cache/")
    _ssh(
        f"cd {tree} && PATH=$HOME/.local/bin:$PATH PYTHONPATH=tools nice -n 19 "
        f"uv run --frozen --quiet python -m kta_perfmodel --full-here --jobs {JOBS} "
        f"> table.log 2>&1 || {{ tail -30 table.log; exit 1; }}; tail -3 table.log"
    )
    _rsync(f"{HOST}:{tree}/data/perfmodel/cache/", f"{data}/cache/")
    for name in ("error-table.json", "error-table.md"):
        _rsync(f"{HOST}:{tree}/docs/design/perfmodel/{name}", f"{table_dir}/{name}")
    last = _ssh(f"tail -1 {tree}/perfmodel/table-runs.jsonl")
    with LOG.open("a") as f:
        f.write(last)
    print(f"full table of {head[:12]} on {HOST}: {last}", end="")
