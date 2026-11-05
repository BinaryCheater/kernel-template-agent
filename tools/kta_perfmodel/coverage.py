"""LOOP §3a: cells counted by the time they carry (D902). Every entry of a kernel row's critical
path goes to one cell of the matrix, or beside it; the share of replayed time in M, m and empty
cells replaces the cell count, and the hand placement of the misses is checked against it.

    python -m kta_perfmodel.review coverage [--rows]

The operation and target come from the entry's resource, the row's cache state and its memory
operations; the pattern from the kernel class (`PATTERN`; an own kernel with addresses from loaded
data is `hashed`); the sharing is `c` for a launch that fills the chip. A coarse map: a path entry
holds one cell, so a row whose load path crosses two cells shows in the first. It needs rows with
attribution: the core or full table.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from . import blind, matrix, table

PATTERN = (
    ("hgemm", "order"),
    ("whatif hgemm", "order"),
    ("gemm", "order"),
    ("gemv", "row"),
    ("whatif gemv", "row"),
    ("fa ", "row"),
    ("whatif fa", "row"),
    ("decode", "row"),
    ("whatif decode", "row"),
    ("rmsnorm", "row"),
    ("whatif rmsnorm", "row"),
    ("softmax", "row"),
    ("whatif softmax", "row"),
)
BESIDE = {"launch", "cta_dispatch"}


def _pattern(r: dict[str, Any]) -> str:
    s = r.get("set", "").removeprefix("testset ")
    for pre, p in PATTERN:
        if s.startswith(pre):
            return p
    gaps = (r.get("counts") or {}).get("addr_gaps", {})
    value = any(table.VALUE_CAUSE.match(k.split(" ", 2)[2]) for k in gaps if k.startswith("chip"))
    return "hashed" if value else "order"


def _classes(r: dict[str, Any]) -> Counter:
    c: Counter = Counter()
    for n, cls, _ in ((r.get("counts") or {}).get("ops") or {}).values():
        c[cls] += n
    return c


def cell_of(r: dict[str, Any], a: dict[str, Any]) -> tuple[str, str, str, str] | None:
    """The cell of one critical-path entry; None beside the matrix."""
    res, mech = a["resource"], a["mechanism"]
    if res in BESIDE:
        return None
    c = _classes(r)
    rotating = r["id"].removesuffix(" kernel").endswith("rotating")
    far = "DRAM" if rotating else "L2"
    copies = c["cp_async"] > 0
    sharing = "c"
    p = _pattern(r)
    if res in ("l2/dram", "dram_queue", "dram", "l2", "l2_return"):
        return ("copy" if copies else "read", far, p, sharing)
    if res == "lsu" and copies:
        return ("copy", far, p, sharing)
    if res in ("l1", "lsu"):
        if "local" in mech:
            return ("read", "stack", "order", sharing)
        return ("read", "L1", p if p in ("order", "row") else "order", sharing)
    if res in ("store_queue", "store_port"):
        kind = "atomic" if "red" in mech or "atomic" in mech else "write"
        return (kind, "L2", p if kind == "write" or p != "row" else "order", sharing)
    if res == "mio" or res == "smem_array":
        return ("tload", "shared", "order", "s") if c["ldsm"] else ("read", "shared", "order", "s")
    if res == "barrier":
        return ("barrier", "shared", "order", "s")
    target = "shared" if c["mma"] else "L2" if c["ldg"] else "L1"
    return ("compute", target, "order", "s" if target == "shared" else sharing)


def by_time(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per row the replayed time by state; totals over rows, each row counting as one kernel."""
    t = matrix.cells(matrix.ALL_MARKS)
    per: dict[str, dict[str, float]] = {}
    cells: Counter = Counter()
    n = 0
    for r in rows:
        if not r["id"].endswith(" kernel") or not r.get("attribution") or not r.get("predicted"):
            continue
        n += 1
        fam = _family(r)
        got = per.setdefault(fam, defaultdict(float))
        got["kernels"] += 1
        left = 1.0
        for a in r["attribution"]:
            left -= a["share"]
            key = cell_of(r, a)
            if key is None:
                got["beside"] += a["share"]
                continue
            m = t.get(key)
            st = (m.state if m else "empty") if key in t else "empty"
            got[st] += a["share"]
            if st != "M":
                cells[(st, *key)] += a["share"]
        got["other"] += max(0.0, left)
    return {"n": n, "families": per, "cells": cells}


def _family(r: dict[str, Any]) -> str:
    s = r.get("set", "").removeprefix("testset ")
    for pre in blind.DESIGN:
        if s.startswith(pre):
            return pre.strip()
    return "own and sweeps"


def report(doc: dict[str, Any]) -> str:
    res = by_time(doc["rows"])
    tot: dict[str, float] = defaultdict(float)
    for c in res["families"].values():
        for key, v in c.items():
            tot[key] += v
    k = tot["kernels"] or 1
    out = [
        f"Replayed time by cell, table {doc.get('kind')} of {doc.get('kta_commit', '?')[:7]}, "
        f"{res['n']} kernels, mean over kernels: M {100 * tot['M'] / k:.0f} %, m "
        f"{100 * tot['m'] / k:.0f} %, empty {100 * tot['empty'] / k:.0f} %, beside the matrix "
        f"{100 * tot['beside'] / k:.0f} %, under 0.5 % of a path {100 * tot['other'] / k:.0f} %"
    ]
    for f, c in sorted(res["families"].items(), key=lambda x: -x[1]["kernels"]):
        n = c["kernels"]
        out.append(
            f"  {f:16s} {n:5.0f} kernels: M {100 * c['M'] / n:3.0f}, m {100 * c['m'] / n:3.0f}, "
            f"empty {100 * c['empty'] / n:3.0f}, beside {100 * c['beside'] / n:3.0f} %"
        )
    out.append("Cells not M by the time they carry, in kernels' worth of time:")
    for key, v in res["cells"].most_common(10):
        out.append(f"  {v:7.1f} {key[0]:5s} {'×'.join(key[1:])}")
    return "\n".join(out)


def hand_check(doc: dict[str, Any], round_no: int) -> str:
    """The hand placement of a round's misses (matrix.MISSES<n>) against the cell that carries the
    most time on those rows. The rows must be in `doc` with attribution: blind rows leave the core,
    so this needs the full table that holds them."""
    from .review import _family as family_of

    table_rows = {r["id"]: r for r in doc["rows"]}
    hand = getattr(matrix, f"MISSES{round_no}", {})
    same = other = none = 0
    lines = []
    for b in blind.rows(round_no):
        r = table_rows.get(b["id"])
        fam = family_of(b)
        if fam not in hand or abs(b["error"]) <= 0.10 or b["mode"] != "kernel":
            continue
        if not r or not r.get("attribution"):
            none += 1
            continue
        got: Counter = Counter()
        for a in r["attribution"]:
            key = cell_of(r, a)
            if key:
                got[key] += a["share"]
        top = got.most_common(1)[0][0] if got else None
        if top and top[:2] == hand[fam][:2]:
            same += 1
        else:
            other += 1
            lines.append(f"  {b['unit']}: by hand {'×'.join(hand[fam][:4])}, top {top}")
    head = f"Hand placement, round {round_no}: operation and target as the top cell {same}, "
    return "\n".join([head + f"other {other}, no replay {none}", *lines[:12]])


def within_group(round_no: int) -> dict[str, float]:
    """LOOP §5: spread of log(predicted / measured) around the group's mean, blind, by family; the
    root mean square over the family's groups, and over all groups off HGEMM without warm GEMV."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in blind.rows(round_no):
        if r["mode"] == "kernel" and r["launch"].get("set", "").startswith(blind.DESIGN):
            groups.setdefault(r["launch"]["set"], []).append(r)
    fam: dict[str, list[float]] = {}
    for name, g in groups.items():
        if len(g) < 3:
            continue
        e = [math.log(r["predicted"] / r["measured"]) for r in g]
        key = name.split(" ")[0] + (" rotating" if name.endswith("rotating") else " warm")
        fam.setdefault(key, []).extend(x - statistics.fmean(e) for x in e)
    out = {f: math.sqrt(statistics.fmean(x * x for x in v)) for f, v in fam.items()}
    rest = [x for f, v in fam.items() if not f.startswith("hgemm") and f != "gemv warm" for x in v]
    if rest:
        out["off HGEMM, without warm GEMV"] = math.sqrt(statistics.fmean(x * x for x in rest))
    return out
