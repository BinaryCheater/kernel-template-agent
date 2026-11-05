"""Decision scores of a predictor over design groups, the same quantities as 28 §1.

A group is one `set`: the variants of one family at one shape and cache state.
"""

import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable


def groups(rows: Iterable[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("set") and not str(r["set"]).startswith("whatif"):
            out[r["set"]].append(r)
    return {k: v for k, v in out.items() if len(v) >= 3}


def score(rows: list[dict], predict: Callable[[dict], float | None], gap: float = 0.03) -> dict:
    """Pairs right among pairs `gap` apart, best in top 3, first-pick loss, error."""
    right = total = top3 = n_groups = 0
    losses: list[float] = []
    errors: list[float] = []
    within: list[float] = []
    for rs in groups(rows).values():
        have = [(predict(r), r["measured_us"]) for r in rs]
        have = [(p, m) for p, m in have if p is not None]
        if len(have) < 3:
            continue
        n_groups += 1
        for i, (p, m) in enumerate(have):
            for q, n in have[i + 1 :]:
                if abs(m - n) / min(m, n) >= gap:
                    total += 1
                    right += (p - q) * (m - n) > 0
        ranked = sorted(have)
        best = min(m for _, m in have)
        top3 += best in [m for _, m in ranked[:3]]
        losses.append(ranked[0][1] / best - 1)
        errors += [abs(p - m) / m for p, m in have]
        level = statistics.median(p / m for p, m in have)
        within += [abs(p / level - m) / m for p, m in have]
    return {
        "groups": n_groups,
        "pairs": f"{right}/{total}",
        "pairs_pct": round(100 * right / total, 1) if total else None,
        "top3": f"{top3}/{n_groups}",
        "loss_median_pct": round(100 * statistics.median(losses), 1) if losses else None,
        "loss_worst_pct": round(100 * max(losses), 1) if losses else None,
        "loss_gt3": sum(x > 0.03 for x in losses),
        "error_median_pct": round(100 * statistics.median(errors), 1) if errors else None,
        "within_median_pct": round(100 * statistics.median(within), 1) if within else None,
    }
