"""``tile_index``: coverage, exact-once and no-phantom of a tiled index space.

Per tiled dimension, with tile size T and an extent E >= 1 left symbolic, the grid has
ceil(E/T) (or floor) tiles and a point (b, o) is admitted when 0 <= b < grid, 0 <= o < T
and, with a guard, b*stride + o < E. Coverage is checked against the backward map's
witness b = i // stride; a failed witness is a counterexample only when the token replay
shows a missing term. One UNSAT result covers every extent; T is symbolic over its whole
domain when the solver decides it, else one query per value.

Reimplemented after the read-only reference
/Users/dingyi/Projects/agentic-codesign/backend/ventus_compiled/proof_templates/upstream/smem_verify.py
(``index_obligations``, ``replay_gemm_indices``); not imported.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import z3

from ..encode import Unsupported
from ..smt import run
from ..types import Draft
from .base import ModelInput

PROPERTIES = ("coverage", "exact_once", "no_phantom", "injective_origin")
MAX_REPLAY_EXTENT = 4096


def admitted_indices(extent: int, tile: int, grid: str, guard: bool, offset: int) -> list[int]:
    stride = tile + offset
    n = -(-extent // tile) if grid == "ceil" else extent // tile
    return [
        b * stride + o
        for b in range(n)
        for o in range(tile)
        if not guard or b * stride + o < extent
    ]


def replay(extent: int, tile: int, grid: str, guard: bool, offset: int) -> dict[str, Any]:
    """Token replay of one dimension: each index i < extent must be produced exactly once
    and nothing at or beyond the extent may be written."""
    got = Counter(admitted_indices(extent, tile, grid, guard, offset))
    diffs: list[dict[str, Any]] = []
    for i in range(extent):
        if got[i] != 1:
            diffs.append(
                {
                    "index": i,
                    "expected": 1,
                    "actual": got[i],
                    "reason": "missing term" if got[i] == 0 else "duplicated term",
                }
            )
    phantom = sorted(i for i in got if i >= extent)
    if phantom:
        diffs.append(
            {
                "phantom_writes": len(phantom),
                "example": phantom[0],
                "reason": "write outside the output (unguarded tail)",
            }
        )
    return {
        "extent": extent,
        "tile": tile,
        "agrees": not diffs,
        "differences": len(diffs),
        "sample": diffs[:4],
    }


def _queries(
    t: Any, tdom: Any, grid: str, guard: bool, offset: int
) -> dict[str, tuple[Any, dict[str, Any]]]:
    e, i, b, o, b2, o2 = (z3.Int(n) for n in ("E", "i", "b", "o", "b2", "o2"))
    stride = t + offset
    g = (e + t - 1) / t if grid == "ceil" else e / t  # t >= 1, so z3's division floors

    def adm(bb: Any, oo: Any) -> Any:
        c = [bb >= 0, bb < g, oo >= 0, oo < t]
        if guard:
            c.append(bb * stride + oo < e)
        return z3.And(*c)

    base = z3.And(e >= 1, t >= 1, stride >= 1, tdom)
    wb = i / stride
    return {
        "coverage": (
            z3.And(base, i >= 0, i < e, z3.Not(adm(wb, i - wb * stride))),
            {"E": e, "T": t, "i": i},
        ),
        "exact_once": (
            z3.And(
                base,
                adm(b, o),
                adm(b2, o2),
                z3.Or(b != b2, o != o2),
                b * stride + o == b2 * stride + o2,
                b * stride + o < e,
            ),
            {"E": e, "T": t, "b": b, "o": o, "b2": b2, "o2": o2},
        ),
        "no_phantom": (
            z3.And(base, adm(b, o), b * stride + o >= e),
            {"E": e, "T": t, "b": b, "o": o},
        ),
        "injective_origin": (
            z3.And(base, adm(b, 0), adm(b2, 0), b != b2, b * stride == b2 * stride),
            {"E": e, "T": t, "b": b, "b2": b2},
        ),
    }


def _one(
    m: ModelInput, d: Draft, tile_var: str, prop: str, grid: str, guard: bool, offset: int
) -> str:
    """Outcome for one dimension and property: tile symbolic first, then per value."""
    values = [int(v) for v in m.menu(tile_var)]
    t = z3.Int("T")
    attempts: list[tuple[Any, Any]] = [(t, z3.Or(*[t == v for v in values]))]
    first = run(*_queries(t, attempts[0][1], grid, guard, offset)[prop], m.timeout_ms)
    d.add(first.record())
    runs = [first]
    if first.result == "unknown":
        runs = []
        for v in values:
            q = run(
                *_queries(z3.IntVal(v), z3.BoolVal(True), grid, guard, offset)[prop], m.timeout_ms
            )
            d.add(q.record())
            if q.model:
                q.model["T"] = v
            runs.append(q)
    for q in runs:
        if q.result == "unknown":
            d.data.setdefault("unknown", []).append(
                {"tile": tile_var, "property": prop, "reason": q.reason}
            )
            return "unresolved"
        if q.result == "sat":
            extent, tile = int(q.model["E"]), int(q.model["T"])
            if extent > MAX_REPLAY_EXTENT:
                d.reason = f"{prop}[{tile_var}] witness E={extent} too large to replay"
                return "unresolved"
            rep = replay(extent, tile, grid, guard, offset)
            if rep["agrees"]:
                d.data.setdefault("failed_witness", []).append(
                    {"tile": tile_var, "property": prop, **q.model}
                )
                d.reason = f"{prop}[{tile_var}]: certificate failed but the replay shows no defect"
                return "unresolved"
            d.counterexample = {"tile_var": tile_var, "property": prop, **q.model}
            d.replay = rep
            d.reason = f"{prop} fails for {tile_var}={tile}, extent {extent}"
            return "counterexample"
    return "proved"


def tile_index(m: ModelInput) -> Draft:
    t0 = time.perf_counter()
    dims = m.param("dims")
    props = m.param("properties", ["coverage", "exact_once", "no_phantom"])
    grid, guard, offset = (
        str(m.param("grid", "ceil")),
        bool(m.param("guard", True)),
        int(m.param("stride_offset", 0)),
    )
    if not isinstance(dims, list) or not dims or any(p not in PROPERTIES for p in props):
        raise Unsupported("tile_index needs dims and known properties")
    d = m.draft("proved", "", ("program", "X"))
    for cond in m.param("only_when", []) or []:
        # a variable whose other values the model does not describe (e.g. index_map=custom)
        name, allowed = cond["var"], set(cond["values"])
        extra = [v for v in m.menu(name) if v not in allowed]
        if extra:
            d.outcome = "unresolved"
            d.reason = f"{name} in {extra} has no index model (modelled: {sorted(allowed)})"
            d.domain = f"not checked: {name} in {extra}"
            return d
    outcomes: list[str] = []
    for dim in dims:
        for prop in props:
            outcomes.append(_one(m, d, str(dim["tile"]), prop, grid, guard, offset))
            if outcomes[-1] == "counterexample":
                break
        if outcomes and outcomes[-1] == "counterexample":
            break
    d.outcome = (
        "counterexample"
        if "counterexample" in outcomes
        else "unresolved"
        if "unresolved" in outcomes
        else "proved"
    )
    tiles = ", ".join(f"{x['tile']} in {m.menu(str(x['tile']))}" for x in dims)
    d.domain = (
        f"{', '.join(props)} for every extent E >= 1, each {tiles}; grid {grid}, guard {guard}"
    )
    d.seconds = time.perf_counter() - t0
    return d
