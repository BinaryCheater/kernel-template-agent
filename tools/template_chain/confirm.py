"""Final confirmation and the cell table. No model is involved.

python -m tools.template_chain.confirm RUNS   three fresh-seed windows per cell, comparator around each
python -m tools.template_chain.confirm RUNS --table
"""

import json
import statistics
import subprocess  # kta:allowed fetches the measured family source from the timing host
import sys
from pathlib import Path

from tools.template_chain import workers
from tools.template_chain.tasks import TASKS

WINDOWS = 3


def best_row(run: Path) -> tuple[dict, str, int] | None:
    best = None
    for n in (1, 2):
        rows = run / f"round{n}/rows.json"
        if not rows.exists():
            continue
        remote = json.loads((run / f"round{n}/batch.json").read_text())["remote"]
        for r in json.loads(rows.read_text()):
            if r.get("us") and (best is None or r["us"] < best[0]["us"]):
                best = (r, remote, n)
    return best


def confirm(run: Path) -> dict:
    out = run / "confirm.json"
    if out.exists():
        return json.loads(out.read_text())
    meta = json.loads((run / "run.json").read_text())
    task = TASKS[meta["task"]]
    found = best_row(run)
    if not found:
        return {"run": run.name, "status": "nothing measured"}
    row, remote, n = found
    family = run / "confirm-family"
    if not family.exists():
        host, path = remote.split(":")
        subprocess.run(["scp", "-qr", f"{host}:{path}/family", str(family)], check=True)
    windows = []
    for w in range(WINDOWS):
        job = {
            "id": "best",
            "kind": "candidate",
            "x": row["x"],
            "shape": task["shape"],
            "seed": 1000 + w,
            "timing": True,
        }
        batch = workers.measure(task, meta["task"], family, task["shape"], [job])
        got = batch["rows"].get("best", {})
        windows.append(
            {
                "valid": batch["valid"],
                "candidate_us": got.get("median_us"),
                "correct": bool(got.get("correct") and got.get("replay_correct")),
                "comparator_us": [
                    (batch.get(k) or {}).get("median_us") for k in ("before", "after")
                ],
                "attempt": batch["attempt"],
                "remote": batch["remote"],
                "error": got.get("error"),
            }
        )
    ratios = [
        w["candidate_us"] / statistics.mean(w["comparator_us"])
        for w in windows
        if w["valid"] and w["correct"] and w["candidate_us"]
    ]
    result = {
        "run": run.name,
        "x": row["x"],
        "round": n,
        "search_us": row["us"],
        "windows": windows,
        "ratio_median": statistics.median(ratios) if ratios else None,
        "valid_windows": len(ratios),
    }
    out.write_text(json.dumps(result, indent=1))
    return result


def usage(run: Path) -> dict:
    tokens = {"input": 0, "cached_input": 0, "output": 0}
    wall = 0.0
    for path in run.glob("state/*/result.json"):
        t = json.loads(path.read_text())
        wall += t.get("wall_s") or 0
        for k in tokens:
            tokens[k] += (t.get("per_turn_usage") or {}).get(k, 0)
    return {**tokens, "wall_s": wall}


def table(runs: Path) -> list[dict]:
    out = []
    for run in sorted(p for p in runs.iterdir() if (p / "run.json").exists()):
        meta = json.loads((run / "run.json").read_text())
        rows = {}
        summaries = {}
        for n in (1, 2):
            if (run / f"round{n}/summary.json").exists():
                rows[n] = json.loads((run / f"round{n}/rows.json").read_text())
                summaries[n] = json.loads((run / f"round{n}/summary.json").read_text())
        timed = {n: [r for r in v if r.get("us")] for n, v in rows.items()}
        anchor = next(
            (r["us"] for r in rows.get(1, []) if r["role"] == "anchor" and r.get("us")), None
        )
        randoms = [r["us"] for r in rows.get(1, []) if r["role"] == "random" and r.get("us")]
        checks = json.loads((run / "workspace/state.json").read_text())["checks"]
        conf = (
            json.loads((run / "confirm.json").read_text())
            if (run / "confirm.json").exists()
            else {}
        )
        out.append(
            {
                **meta,
                "domain": [summaries[n]["domain"] for n in summaries],
                "anchor_us": anchor,
                "random_us": randoms,
                "best_us": {n: min((r["us"] for r in v), default=None) for n, v in timed.items()},
                "top_pick_us": {
                    n: next(
                        (
                            r["us"]
                            for r in sorted(v, key=lambda r: r["post_us"])
                            if r["role"] == "mip"
                        ),
                        None,
                    )
                    for n, v in timed.items()
                },
                "form_error": {n: s.get("median_abs_error") for n, s in summaries.items()},
                "gpu_rows": sum(len(v) for v in rows.values()),
                "compiled": sum(len(s["candidates"]) for s in summaries.values()),
                "compile_rejected": sum(
                    1 for s in summaries.values() for r in s["candidates"] if not r["correct"]
                ),
                "mip_error": [s.get("mip_minus_enumeration_us") for s in summaries.values()],
                "checks": checks,
                "usage": usage(run),
                "confirm_ratio": conf.get("ratio_median"),
                "confirm_windows": conf.get("valid_windows"),
                "confirm_x": conf.get("x"),
            }
        )
    return out


def main() -> None:
    runs = Path(sys.argv[1])
    if "--table" in sys.argv:
        print(json.dumps(table(runs), indent=1))
        return
    for run in sorted(p for p in runs.iterdir() if (p / "run.json").exists()):
        r = confirm(run)
        print(r["run"], r.get("ratio_median"), r.get("valid_windows"), flush=True)


if __name__ == "__main__":
    main()
