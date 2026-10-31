"""``partition_index``: every reduction index lands in exactly one group, and combining
the groups' partial results gives every product exactly once (completion agreement).

Bounded enumeration: lengths, group counts and group sizes up to the outline's bounds,
for each assignment strategy the outline maps to an index function. A strategy without a
mapping (e.g. runtime-balanced) is not modelled and leaves the obligation unresolved.
Each violation is replayed by combining formal partial-sum tokens.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from ..encode import Unsupported
from ..smt import enumerated
from ..types import Draft
from .base import ModelInput

KINDS = ("contiguous", "strided")


def groups_of(kind: str, length: int, n: int, size: int) -> dict[int, list[int]]:
    """group -> its indices, in group-local order."""
    out: dict[int, list[int]] = {}
    for k in range(length):
        g, local = (k // size, k % size) if kind == "contiguous" else (k % n, k // n)
        if local >= size:
            g = -1  # does not fit its group: dropped, as a program would drop it
        out.setdefault(g, []).append(k)
    return out


def replay(kind: str, length: int, n: int, size: int) -> dict[str, Any]:
    parts = groups_of(kind, length, n, size)
    combined: Counter[int] = Counter()
    for g, ks in parts.items():
        if 0 <= g < n:
            combined.update(ks)
    diffs = [
        {"index": k, "expected": 1, "actual": combined[k]}
        for k in range(length)
        if combined[k] != 1
    ]
    return {
        "agrees": not diffs,
        "differences": len(diffs),
        "sample": diffs[:4],
        "groups": {str(g): v for g, v in parts.items()},
    }


def partition_index(m: ModelInput) -> Draft:
    t0 = time.perf_counter()
    gv, sv, av = m.var("groups"), m.var("size"), m.var("assignment")
    length = m.var("length")
    mapping: dict[str, str] = dict(m.param("mapping") or {})
    bounds = dict(m.param("bounds") or {"length": 24, "groups": 8, "size": 8})
    d = m.draft("bounded", "", ("pre", "X", "program"))
    gs = m.upto(gv, int(bounds["groups"]))
    ss = m.upto(sv, int(bounds["size"]))
    unmodelled = [a for a in m.menu(av) if mapping.get(str(a)) not in KINDS]
    bounds["length"] = max(int(bounds["length"]), 2 * max(ss, default=0) + 1)
    points = 0
    for a in m.menu(av):
        kind = mapping.get(str(a))
        if kind not in KINDS:
            continue
        for n in gs:
            for size in ss:
                for ln in range(1, int(bounds["length"]) + 1):
                    point = {gv: n, sv: size, av: a, length: ln}
                    if not m.pre_holds(point):
                        continue
                    points += 1
                    rep = replay(kind, ln, int(n), int(size))
                    if not rep["agrees"]:
                        d.outcome, d.counterexample, d.replay = "counterexample", point, rep
                        d.reason = f"assignment {a}: an index is not combined exactly once"
                        break
                if d.outcome == "counterexample":
                    break
            if d.outcome == "counterexample":
                break
    d.add(
        enumerated(
            f"partition_index {gs} {ss} {bounds} {sorted(mapping.items())}",
            time.perf_counter() - t0,
        )
    )
    d.domain = (
        f"{length} in 1..{bounds['length']}, {gv} in {gs}, {sv} in {ss}, "
        f"{av} in {[a for a in m.menu(av) if a not in unmodelled]} admitted by Pre "
        f"({points} points)"
    )
    if d.outcome != "counterexample" and unmodelled:
        d.outcome = "unresolved"
        d.reason = f"{av} in {unmodelled} has no index function in the outline"
    if not points and d.outcome == "bounded":
        raise Unsupported("no point admitted by Pre within the bounds")
    return d
