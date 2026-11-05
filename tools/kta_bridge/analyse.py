"""Search results of one batch: each role's best against the measured best, by family and cache.

python -m tools.kta_bridge.analyse BATCH_DIR SELECTION.json
"""

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def spearman(a: list[float], b: list[float]) -> float:
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((p - ma) * (q - mb) for p, q in zip(ra, rb, strict=True))
    den = sum((p - ma) ** 2 for p in ra) * sum((q - mb) ** 2 for q in rb)
    return num / den**0.5 if den else 0.0


def main() -> None:
    measured = json.loads((Path(sys.argv[1]) / "measured.json").read_text())
    results = {**measured["results"], **measured.get("invalid_results", {})}
    rows = json.loads(Path(sys.argv[2]).read_text())
    groups = defaultdict(list)
    for r in rows:
        got = results.get(r["name"] + " kernel")
        if got:
            groups[r["family"], r["cache"]].append({**r, "us": got["median"], "iqr": got["iqr"]})
    out = {}
    for key, rs in sorted(groups.items()):
        best = min(r["us"] for r in rs)
        randoms = [r["us"] for r in rs if "random" in r["roles"]]
        short = sorted((r for r in rs if "random" not in r["roles"]), key=lambda r: r["pre_us"])
        rng = random.Random(1)
        k = len(short)
        draws = sorted(min(rng.sample(randoms, min(k, len(randoms)))) for _ in range(2000))
        out[" ".join(key)] = {
            "kernels": len(rs),
            "measured_best_us": best,
            "measured_best": next(r["x"] for r in rs if r["us"] == best),
            "shortlist": k,
            "first_pick_us": short[0]["us"],
            "shortlist_best_us": min(r["us"] for r in short),
            "random_median_us": statistics.median(randoms),
            "random_best_us": min(randoms),
            f"random{k}_best_median_us": draws[len(draws) // 2],
            "spearman_pred_all": round(
                spearman([r["pre_us"] for r in rs], [r["us"] for r in rs]), 3
            ),
            "error_median_pct": round(
                100 * statistics.median(abs(r["pre_us"] / r["us"] - 1) for r in rs), 1
            ),
        }
    sys.stdout.write(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
