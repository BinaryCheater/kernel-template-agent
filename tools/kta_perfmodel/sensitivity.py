"""LOOP §3b step 2: the parameters a wrong pair's rows read, each ±30 %: can a value flip the pair
(a value or a rule is wrong) or can none (structure is missing). A diagnostic on seen rows: nothing
here sets a parameter (LOOP §4.2).

    python -m kta_perfmodel.sensitivity PAIRS [--params a,b] [--step 0.3] [--jobs 6]

PAIRS: `rowA|rowB,rowC|rowD`, row ids without ` kernel`. Without --params: the parameters on the
critical paths of the rows at 1 % or more of a path, from one traced replay each. Counted in the
round's limit below the core (LOOP §8); at most 600 CPU s a round.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from typing import Any

from . import full, table, upstream

LIMIT_S = 600.0


def _rows(ids: list[str]) -> list[dict[str, Any]]:
    cal, _ = upstream.calibration_dir(upstream.UPSTREAM, upstream.UPSTREAM_REV)
    want = {f"{i} kernel" for i in ids}
    return [r for r in table.latest(iter(table.specs(cal, ("batches",)))) if r["id"] in want]


def _replay(ids: list[str], over: dict[str, Any] | None, jobs: int, trace: bool) -> dict[str, Any]:
    cal, sha = upstream.calibration_dir(upstream.UPSTREAM, upstream.UPSTREAM_REV)
    return {
        r["id"].removesuffix(" kernel"): r
        for r in table.run(cal, sha, _rows(ids), jobs, overrides=over, trace=trace)
    }


def on_paths(rows: dict[str, Any]) -> list[str]:
    """Parameters on the rows' critical paths, largest share first."""
    share: Counter = Counter()
    for r in rows.values():
        for a in r.get("attribution") or []:
            p = a.get("param")
            if p:
                share[p] += a["share"]
    return [p for p, s in share.most_common() if s >= 0.01]


def value_of(r: dict[str, Any], p: str) -> float | None:
    v = (r.get("values") or {}).get(p)
    v = v[0] if isinstance(v, list) and len(v) == 2 and isinstance(v[1], str) else v
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) and v else None


def report(pairs: list[tuple[str, str]], params: list[str], step: float, jobs: int) -> str:
    ids = sorted({i for p in pairs for i in p})
    with full.scratch("sensitivity", len(ids)):
        base = _replay(ids, None, jobs, True)
        params = params or on_paths(base)
        some = next(iter(base.values()))
        seen = {}
        for p in params:
            v = value_of(some, p)
            if v is None:
                continue
            seen[p] = v
        out = [f"Sensitivity ±{100 * step:.0f} %: {len(pairs)} pairs, {len(seen)} parameters"]
        flips: dict[str, list[str]] = {}
        moved: dict[str, float] = {}
        for p, v in seen.items():
            for sign in (1 - step, 1 + step):
                got = _replay(ids, {p: v * sign}, jobs, False)
                for a, b in pairs:
                    was = base[a]["predicted"] - base[b]["predicted"]
                    now = got[a]["predicted"] - got[b]["predicted"]
                    meas = base[a]["measured"] - base[b]["measured"]
                    if was * meas <= 0 < now * meas:
                        flips.setdefault(f"{a} / {b}", []).append(f"{p} ×{sign:.1f}")
                    d = abs(got[a]["predicted"] / base[a]["predicted"] - 1)
                    moved[p] = max(moved.get(p, 0.0), d)
        out.append("Largest move of a row's time: ")
        out[-1] += ", ".join(
            f"{p} {100 * d:.0f} %" for p, d in sorted(moved.items(), key=lambda x: -x[1])
        )
        for a, b in pairs:
            k = f"{a} / {b}"
            was = (
                base[a]["predicted"] / base[a]["measured"]
                - base[b]["predicted"] / base[b]["measured"]
            )
            what = ", ".join(flips.get(k, [])) or "none flips it: structure"
            out.append(f"{k}: errors apart {100 * was:+.1f} points; {what}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="sensitivity")
    ap.add_argument("pairs")
    ap.add_argument("--params", default="")
    ap.add_argument("--step", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    pairs = [tuple(re.split(r"\|", p)) for p in a.pairs.split(",")]
    params = [p for p in a.params.split(",") if p]
    print(report(pairs, params, a.step, a.jobs))  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
