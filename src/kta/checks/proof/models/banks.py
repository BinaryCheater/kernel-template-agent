"""Shared-resource models.

- ``bank_map``: the request -> location map of each binding, enumerated over tiles of
  ``rows`` x ``chunks`` (chunks a power of two). ``property: permutation`` checks the
  statement as most seed Templates write it (a permutation of the tile's addresses);
  ``injective`` checks only that distinct addresses never collide.
- ``queue_progress``: requesters enqueue into a queue of the given depth, an arbiter
  serves the head. Explores every interleaving; progress fails when a state with work left
  has no enabled step. Service of every request is the declared arbitration assumption,
  not something this model proves.
"""

from __future__ import annotations

import time
from typing import Any

from ..encode import Unsupported
from ..smt import enumerated
from ..types import Draft
from .base import ModelInput

MAPS = ("identity", "xor", "pad")


def location(kind: str, r: int, c: int, chunks: int) -> int:
    if kind == "identity":
        return r * chunks + c
    if kind == "xor":
        return r * chunks + (c ^ (r % chunks))
    return r * (chunks + 1) + c


def bank_map(m: ModelInput) -> Draft:
    t0 = time.perf_counter()
    bv = m.var("binding")
    maps: dict[str, str] = dict(m.param("maps") or {})
    prop = str(m.param("property", "permutation"))
    chunk_menu = [int(c) for c in m.param("chunks", [2, 4, 8])]
    rows = int(m.param("rows", 16))
    d = m.draft("bounded", "", ("program", "X"))
    unmodelled = [b for b in m.menu(bv) if maps.get(str(b)) not in MAPS]
    for b in m.menu(bv):
        kind = maps.get(str(b))
        if kind not in MAPS:
            continue
        for chunks in chunk_menu:
            if chunks & (chunks - 1):
                raise Unsupported("chunks must be powers of two")
            image: dict[int, tuple[int, int]] = {}
            for r in range(rows):
                for c in range(chunks):
                    loc = location(kind, r, c, chunks)
                    clash = image.get(loc)
                    outside = prop == "permutation" and not 0 <= loc < rows * chunks
                    if clash is not None or outside:
                        d.outcome = "counterexample"
                        d.counterexample = {
                            bv: b,
                            "chunks": chunks,
                            "rows": rows,
                            "address": [r, c],
                            "location": loc,
                        }
                        d.replay = {
                            "agrees": False,
                            "reason": f"collides with {list(clash)}"
                            if clash
                            else f"location {loc} outside the tile 0..{rows * chunks - 1}",
                        }
                        d.reason = f"{bv}={b} is not a {prop} of the tile's addresses"
                        break
                    image[loc] = (r, c)
                if d.outcome == "counterexample":
                    break
            if d.outcome == "counterexample":
                break
        if d.outcome == "counterexample":
            break
    d.add(
        enumerated(
            f"bank_map {prop} {chunk_menu} {rows} {sorted(maps.items())}", time.perf_counter() - t0
        )
    )
    modelled = [b for b in m.menu(bv) if b not in unmodelled]
    d.domain = f"{prop}; {bv} in {modelled}; tiles {rows} rows x chunks {chunk_menu}"
    if d.outcome != "counterexample" and unmodelled:
        d.outcome, d.reason = "unresolved", f"{bv} in {unmodelled} has no map in the outline"
    return d


def _progress(depth: int, requesters: int, each: int) -> dict[str, Any] | None:
    """A reachable stuck state (work left, nothing enabled), or None."""
    start = (tuple([each] * requesters), ())
    seen = {start}
    stack = [start]
    while stack:
        pending, queue = stack.pop()
        nxt = []
        if queue:
            nxt.append((pending, queue[1:]))
        if len(queue) < depth:
            for i, n in enumerate(pending):
                if n:
                    p2 = list(pending)
                    p2[i] -= 1
                    nxt.append((tuple(p2), (*queue, i)))
        if not nxt and (any(pending) or queue):
            return {"pending": list(pending), "queue": list(queue), "depth": depth}
        for s in nxt:
            if s not in seen:
                seen.add(s)
                stack.append(s)
    return None


def queue_progress(m: ModelInput) -> Draft:
    t0 = time.perf_counter()
    dv = m.var("depth")
    bound = int(m.param("max_depth", 4))
    requesters, each = int(m.param("requesters", 3)), int(m.param("requests", 2))
    d = m.draft("bounded", "", ("pre", "X", "assumes"))
    depths = [int(x) for x in m.menu(dv) if int(x) <= bound]
    checked: list[int] = []
    for depth in depths:
        if not m.pre_holds({dv: depth}):
            continue
        checked.append(depth)
        stuck = _progress(depth, requesters, each)
        if stuck is not None:
            d.outcome, d.counterexample = "counterexample", stuck
            d.replay = {"agrees": False, "stuck_state": stuck}
            d.reason = f"no enabled step with work left at {dv}={depth}"
            break
    d.data["without_pre"] = {"depth": 0, "stuck": _progress(0, requesters, each)}
    d.add(enumerated(f"queue_progress {depths} {requesters} {each}", time.perf_counter() - t0))
    d.domain = (
        f"{dv} in {checked} admitted by Pre (bounded to <= {bound}); {requesters} requesters x "
        f"{each} requests; every interleaving. Service of each request rests on the declared "
        "arbitration assumption"
    )
    if not checked:
        raise Unsupported(f"no {dv} within the bound is admitted by Pre")
    return d
