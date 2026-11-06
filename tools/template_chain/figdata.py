"""Flat tables behind the E3 figures, from retained records only. No model and no GPU.

python -m tools.template_chain.figdata RUNS OUT_DIR
"""

import json
import statistics
import sys
from pathlib import Path

from tools.template_chain import confirm


def comparator(batch: dict) -> float | None:
    got = [(batch.get(k) or {}).get("median_us") for k in ("before", "after")]
    got = [g for g in got if g]
    return statistics.mean(got) if got else None


def main() -> None:
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows, cells, funnel, windows = [], [], [], []
    for run in sorted(p for p in runs.iterdir() if (p / "run.json").exists()):
        meta = json.loads((run / "run.json").read_text())
        cell = f"{meta['arm']} {meta['repeat']}"
        base = {"task": meta["task"], "arm": meta["arm"], "repeat": meta["repeat"], "cell": cell}
        for n in (1, 2):
            if not (run / f"round{n}/rows.json").exists():
                continue
            batch = json.loads((run / f"round{n}/batch.json").read_text())
            pre = json.loads((run / f"round{n}/predictions.json").read_text())
            cmp_us = comparator(batch)
            measured = json.loads((run / f"round{n}/rows.json").read_text())
            for i, r in enumerate(measured):
                if not r.get("us"):
                    continue
                rows.append(
                    {
                        **base,
                        "id": f"{meta['task']}/{cell}/{n}/{i}",
                        "round": n,
                        "role": r["role"],
                        "config": json.dumps(r["x"]),
                        "predicted_us": r["post_us"],
                        "measured_us": r["us"],
                        "iqr_us": r.get("iqr_us"),
                        "comparator_us": cmp_us,
                        "ratio": r["us"] / cmp_us,
                    }
                )
            funnel.append(
                {
                    **base,
                    "id": f"{meta['task']}/{cell}/{n}",
                    "round": n,
                    "domain": pre["domain"],
                    "compiled": len(pre["candidates"]),
                    "rejected": sum(not r["correct"] for r in pre["candidates"]),
                    "measured": len(measured),
                }
            )
        usage = confirm.usage(run)
        cells.append(
            {
                **base,
                "id": f"{meta['task']}/{cell}",
                "input_tokens": usage["input"],
                "cached_tokens": usage["cached_input"],
                "output_tokens": usage["output"],
                "model_minutes": usage["wall_s"] / 60,
                "checks": json.loads((run / "workspace/state.json").read_text())["checks"],
            }
        )
        conf = run / "confirm.json"
        if conf.exists():
            c = json.loads(conf.read_text())
            for w, win in enumerate(c.get("windows", [])):
                comp = [x for x in win["comparator_us"] if x]
                windows.append(
                    {
                        **base,
                        "id": f"{meta['task']}/{cell}/{w + 1}",
                        "window": w + 1,
                        "config": json.dumps(c["x"]),
                        "search_round": c["round"],
                        "candidate_us": win["candidate_us"],
                        "comparator_us": statistics.mean(comp) if comp else None,
                        "ratio": win["candidate_us"] / statistics.mean(comp)
                        if comp and win["candidate_us"]
                        else None,
                        "valid": bool(win["valid"] and win["correct"]),
                    }
                )
    for name, data in (("rows", rows), ("cells", cells), ("funnel", funnel), ("windows", windows)):
        (out / f"{name}.json").write_text(json.dumps(data))
        print(name, len(data))


if __name__ == "__main__":
    main()
