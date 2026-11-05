"""``perfmodel/core.txt``: the fixed list `scripts/perfmodel-table --core` replays (LOOP §8, D874).
About 800 rows, about 2500 CPU s, and it stays that size as batches are added: no rule reads the
newest batch, and blind rows join only when a family's representative is replaced by decision.

The rule, in the order it is applied; no rule looks at how rows moved or at whether they fail:

1. Design groups: the shapes of `SHAPES`, warm and rotating, every variant.
2. What-ifs: every what-if pair whose base is in the core or in a what-if group of `SHAPES`,
   at most `WHATIF_AXIS` pairs per family, axis and cache state, cheapest first.
3. Everything else (a probe, a sweep, the upstream sets, the own kernels): per group, per mechanism,
   `ceil(sqrt(n))` rows: first the cheapest row for each value a row's name sweeps (a greedy
   cover of the name's tokens), then rows evenly spread over the measured span. A row above
   `DEAR_S` CPU s is taken only if its group has nothing cheaper.
4. Launch modes: `kernel` always; `b2b` and `graph` only for design and what-if rows under 30 µs.

    python -m kta_perfmodel.core            # rewrite the list from the full table of round 15
    python -m kta_perfmodel.core --backtest # how often a change in a full table shows in the core
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict

from . import table, upstream

FILE = upstream.ROOT / "perfmodel" / "core.txt"
TABLE = "docs/design/perfmodel/error-table.json"
FULL_TABLE_REV = "1b9cb9c"  # the full table the list was drawn from: round 15's
# (round whose full table it is, commit that holds it)
FULL_TABLES = ((12, "239f138"), (13, "154837e"), (14, "583d2d8"), (15, "1b9cb9c"))
SHAPES = (
    "hgemm2048",
    "hgemm 3072x1536x2048",
    "hgemm 2688x1920x1728",
    "hgemm 1024x1024x1024",
    "fa 2304x6x144",
    "fa_forward",
    "gemv 3072x6144",
    "gemv_w4a16",
    "decode 36x2304x144",
    "decode_gqa",
    "rmsnorm 1920x1792",
    "rmsnorm 1280x2560",
    "softmax 1920x1792",
    "softmax 1280x2560",
    "nvfp4gemm",
    "fp8gemm2048",
    "w4gemm",
    "moe_gemm",
)
WHATIF_AXIS = 2
DEAR_S = 6.0
SHORT_US = table.SHORT_US
MODE = re.compile(r" (kernel|b2b|graph)$")
HGEMM_SHAPE = re.compile(r"^testset (hgemm)( ?\d+.*)$")


def base_group(group: str) -> str:
    """The group a row belongs to, without launch mode or cache state."""
    g = re.sub(r" (b2b|graph)$", "", group)
    g = re.sub(r" (warm|rotating)$", "", g)
    return g.removeprefix("testset ")


def family(group: str) -> str:
    """The kernel class or mechanism: what a change to the model reaches whatever the shape."""
    g = base_group(group)
    m = re.match(r"(whatif )?(hgemm|fa|gemv|decode|rmsnorm|softmax)\b", g)
    return m[2] if m else g


def unit_of(rid: str) -> str:
    return MODE.sub("", rid)


def _tokens(unit: str) -> set[str]:
    return {t for t in re.split(r"[^A-Za-z0-9]+|(?<=\D)(?=\d)", unit) if t}


def sample(units: dict[str, dict]) -> list[str]:
    """Rule 3 over one group: `units` is unit -> {cpu_s, measured}."""
    k = math.ceil(math.sqrt(len(units)))
    cheap = {u: v for u, v in units.items() if v["cpu_s"] <= DEAR_S} or {
        min(units, key=lambda u: units[u]["cpu_s"]): 0
    }
    pool = {u: units[u] for u in cheap}
    picked: list[str] = []
    left = set().union(*(_tokens(u) for u in pool))
    while left and len(picked) < k:
        u = max(pool, key=lambda u: (len(_tokens(u) & left), -pool[u]["cpu_s"]))
        if not _tokens(u) & left:
            break
        picked.append(u)
        left -= _tokens(u)
    by_time = sorted(pool, key=lambda u: pool[u]["measured"] or 0.0)
    for i in range(len(by_time)):
        if len(picked) >= k:
            break
        u = by_time[i * len(by_time) // max(1, k - len(picked)) % len(by_time)]
        if u not in picked:
            picked.append(u)
    for u in by_time:
        if len(picked) >= min(k, len(pool)):
            break
        if u not in picked:
            picked.append(u)
    return picked


def _whatifs() -> dict[str, dict]:
    out = {}
    for lj in table.BATCHES.glob("*/build/testset/*/launch.json"):
        w = json.loads(lj.read_text()).get("whatif")
        if w:
            out[lj.parent.name] = w
    return out


def choose(rows: list[dict]) -> dict[str, str]:
    """Row id -> the rule that took it."""
    live = [r for r in rows if not r["excluded"]]
    units: dict[str, dict] = {}
    for r in live:
        u = unit_of(r["id"])
        d = units.setdefault(u, {"cpu_s": 0.0, "measured": None, "group": base_group(r["set"])})
        d["cpu_s"] = max(d["cpu_s"], r.get("cpu_s") or 0.0)
        if r["id"].endswith(" kernel") or d["measured"] is None:
            d["measured"] = r["measured"]
        d["raw_group"] = r["set"]
    why: dict[str, str] = {}
    for u, d in units.items():
        if d["group"] in SHAPES:
            why[u] = f"design group {d['group']}"
    wi = _whatifs()
    pairs: dict[tuple[str, str, bool], list[str]] = defaultdict(list)
    for u, w in wi.items():
        if u not in units or w["base"] not in units:
            continue
        rotating = "rot" in u.split("_")[-1] or u.endswith("_rotating")
        pairs[(family(units[u]["raw_group"]), w["axis"], rotating)].append(u)
    for key, us in pairs.items():
        us.sort(key=lambda u: units[u]["cpu_s"] + units[wi[u]["base"]]["cpu_s"])
        for u in us[:WHATIF_AXIS]:
            why[u] = f"what-if {key[0]} {key[1]}"
            why.setdefault(wi[u]["base"], f"what-if base {key[0]} {key[1]}")
    rest: dict[str, dict[str, dict]] = defaultdict(dict)
    for u, d in units.items():
        if u not in why and not d["group"].startswith(("whatif", "hgemm", "fa ", "gemv ")):
            if not d["group"].startswith(("decode ", "rmsnorm ", "softmax ")):
                rest[d["group"]][u] = d
    for g, us in rest.items():
        for u in sample(us):
            why[u] = f"sample {g}"
    picked: dict[str, str] = {}
    for r in live:
        u = unit_of(r["id"])
        if u not in why:
            continue
        short = (units[u]["measured"] or 1e9) < SHORT_US
        sampled = why[u].startswith("sample")
        if r["id"].endswith((" b2b", " graph")) and (sampled or not short):
            continue
        picked[r["id"]] = why[u]
    return picked


def read() -> set[str]:
    if not FILE.exists():
        raise SystemExit(f"{FILE} missing: python -m kta_perfmodel.core")
    return {s for line in FILE.read_text().splitlines() if (s := line.split("#")[0].strip())}


def _doc(rev: str) -> dict:
    show = ["git", "-C", str(upstream.ROOT), "show", f"{rev}:{TABLE}"]
    return json.loads(subprocess.run(show, check=True, capture_output=True, text=True).stdout)


def cost(rows: list[dict], ids: set[str]) -> float:
    """CPU s the way the run log counts it: one replay per launch mode kept."""
    seen: dict[str, tuple[float, int]] = {}
    for r in rows:
        if r["id"] in ids:
            c, n = seen.get(unit_of(r["id"]), (0.0, 0))
            seen[unit_of(r["id"])] = (max(c, r.get("cpu_s") or 0.0), n + 1)
    return sum(c * n for c, n in seen.values())


def write() -> None:
    rows = _doc(FULL_TABLE_REV)["rows"]
    picked = choose(rows)
    lines = [
        "# rows of `scripts/perfmodel-table --core`: fixed; rule in tools/kta_perfmodel/core.py",
        f"# drawn from the full table at {FULL_TABLE_REV}; python -m kta_perfmodel.core rewrites",
        *(f"{rid}  # {w}" for rid, w in sorted(picked.items())),
    ]
    FILE.write_text("\n".join(lines) + "\n")
    print(f"{len(picked)} rows, about {cost(rows, set(picked)):.0f} CPU s, in {FILE}")


def _moved(a: dict, b: dict, gap: float) -> dict[str, float]:
    out = {}
    for rid, r in b.items():
        p, q = r.get("predicted"), a.get(rid, {}).get("predicted")
        if p and q and abs(p / q - 1) > gap:
            out[rid] = p / q - 1
    return out


def _pairs(rows: list[dict], keep: set[str] | None) -> tuple[int, int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ok = r["id"].endswith(" kernel") and r.get("predicted") and r["measured"] == r["measured"]
        if ok and not r["excluded"] and table._design(r["set"]) and (not keep or r["id"] in keep):
            groups[r["set"]].append(r)
    right = total = 0
    for rs in groups.values():
        g = table.goal(rs)
        right += g["pairs_right"]
        total += g["pairs"]
    return right, total


def backtest(gap: float = 0.03, shown: float = 0.01) -> dict:
    """Between two consecutive full tables: every family with a row moved past `gap`: did a core
    row of that family move past `shown`? And did a row that worsened past `gap` have a core row of
    its family worsened too? The list's rule reads no movement, so all of this is out of sample."""
    core = read()
    docs = [(n, {r["id"]: r for r in _doc(rev)["rows"]}) for n, rev in FULL_TABLES]
    out = []
    for (n0, a), (n1, b) in itertools.pairwise(docs):
        moved, small = _moved(a, b, gap), _moved(a, b, shown)
        fams: dict[str, list[str]] = defaultdict(list)
        for rid in moved:
            fams[family(b[rid]["set"])].append(rid)
        seen = {f for f in fams if any(r in core and family(b[r]["set"]) == f for r in small)}
        held = {f for f in fams if any(family(b[r]["set"]) == f for r in core if r in b)}
        groups = {b[r]["set"] for r in moved}
        gseen = {b[r]["set"] for r in small if r in core}
        worse = {
            r for r in moved if abs(b[r].get("error") or 0) > abs(a[r].get("error") or 0) + gap
        }
        wfam = {family(b[r]["set"]) for r in worse}
        wseen = {family(b[r]["set"]) for r in worse if r in core}
        full0, full1 = _pairs(list(a.values()), None), _pairs(list(b.values()), None)
        both = set(a) & set(b)
        core0 = _pairs([a[r] for r in both], core)
        core1 = _pairs([b[r] for r in both], core)
        lost = sorted(f for f in fams if f not in seen)
        out.append(
            {
                "tables": f"{n0} to {n1}",
                "rows_moved": len(moved),
                "rows_moved_in_core": sum(r in core for r in moved),
                "families_moved": len(fams),
                "families_in_core": len(held),
                "families_shown": len(seen),
                "rows_of_shown_families": sum(len(v) for f, v in fams.items() if f in seen),
                "groups_moved": len(groups),
                "groups_shown": len(groups & gseen),
                "families_worse": len(wfam),
                "families_worse_shown": len(wfam & wseen),
                "not_shown": lost,
                "not_shown_rows": {f: len(fams[f]) for f in lost},
                "pairs_full": [full0, full1],
                "pairs_core": [core0, core1],
            }
        )
    return {"gap": gap, "shown": shown, "steps": out}


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m kta_perfmodel.core")
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    if a.backtest:
        got = backtest()
        print(json.dumps(got, indent=1))
        steps = got["steps"]
        rate = sum(s["families_shown"] for s in steps) / sum(s["families_moved"] for s in steps)
        print(
            f"families shown {rate:.1%}; median by step",
            statistics.median(s["families_shown"] / s["families_moved"] for s in steps),
        )
    else:
        write()


if __name__ == "__main__":
    main()
