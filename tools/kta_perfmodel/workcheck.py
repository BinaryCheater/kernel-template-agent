"""LOOP §3b step 1: the work check. Per kernel row with counters, the model's traffic and
instruction counts per block against the counters', as a ratio to the metric's median over all
rows: is the replay doing the same work as the kernel. No replay: it reads the table's rows, so it
runs on whatever table is current (older rows lack the engine's `copy_sectors` and `stg_sectors`:
they use the width of the op, 32 lanes).

    python -m kta_perfmodel.review work [--raw METRIC] [--focus REGEX]

The counters' meaning, calibrated on the rows of r13 to r18 (round 19): `l1tex ... global_op_ld`
counts the sectors of `cp.async` copies as well as of LDG, at the sectors the copy touches; so does
`lts read`, on top of the stack's reads. Their 0.03 on copy paths was the model's side: its
`ldg_sectors` counts LDG alone.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from kta.models.gb10sim.lanes import width_of

from . import blind, table, upstream

TABLE = upstream.ROOT / "docs" / "design" / "perfmodel" / "error-table.json"
BAND = 1.5
LD = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
ST = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
L2R = "lts__t_sectors_op_read.sum"
L2W = "lts__t_sectors_op_write.sum"
MEM_CLASSES = ("ldg", "stg", "cp_async", "smem_op", "local", "ldsm", "stsm", "red")
TITLE = {
    "inst": "instructions, smsp__inst_executed",
    "lsu": "memory instructions, inst_executed_pipe_lsu",
    "tensor": "tensor instructions, inst_executed_pipe_tensor",
    "ld": "global load and copy sectors, l1tex ld",
    "st": "global store sectors, l1tex st",
    "l2r": "L2 read sectors, lts read",
    "l2w": "L2 write sectors, lts write",
    "fma": "fma busy cycles, sm__pipe_fma_cycles_active",
    "alu": "alu busy cycles, sm__pipe_alu_cycles_active",
}


def _by_class(k: dict[str, Any], classes: tuple[str, ...]) -> float:
    """Sectors by the op's width: 32 lanes in a run."""
    return sum(n * width_of(op) for op, (n, cls, _) in k["ops"].items() if cls in classes)


def _count(k: dict[str, Any], classes: tuple[str, ...]) -> float:
    return sum(n for _, (n, cls, _d) in k["ops"].items() if cls in classes)


def model(r: dict[str, Any]) -> dict[str, float]:
    """The replay's work per block, by metric."""
    k = r["counts"]
    b = max(1, k.get("blocks", 1))
    copies = k.get("copy_sectors", _by_class(k, ("cp_async",)))
    stores = k.get("stg_sectors", _by_class(k, ("stg", "red")))
    cyc = k.get("cycles") or (r["predicted"] or 0.0) * table.C["clock_under_load_GHz"] * 1e3
    busy = k.get("busy") or {}
    return {
        "inst": k["issued"] / b,
        "lsu": _count(k, MEM_CLASSES) / b,
        "tensor": k.get("HMMA", 0) / b,
        "ld": (k["ldg_sectors"] + copies) / b,
        "st": stores / b,
        "l2r": (k["ldg_miss_sectors"] + copies) / b,
        "l2w": stores / b,
        "fma": busy.get("fma", 0.0) * cyc / b,
        "alu": busy.get("alu", 0.0) * cyc / b,
    }


def measured(r: dict[str, Any]) -> dict[str, float]:
    c = r["counters"]
    g = float(c.get("launch__grid_size") or 1)
    get = lambda m: float(c.get(m) or 0) / g  # noqa: E731
    return {
        "inst": get("smsp__inst_executed.sum"),
        "lsu": get("smsp__inst_executed_pipe_lsu.sum"),
        "tensor": get("smsp__inst_executed_pipe_tensor.sum"),
        "ld": get(LD),
        "st": get(ST),
        "l2r": get(L2R),
        "l2w": get(L2W),
        "fma": get("sm__pipe_fma_cycles_active.sum"),
        "alu": get("sm__pipe_alu_cycles_active.sum"),
    }


def family(r: dict[str, Any]) -> str:
    s = r.get("set", "").removeprefix("testset ")
    return s if s else re.split(r"[_ ]", r["id"])[0]


def check(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per metric every row where both sides did work, ratio to the median attached."""
    out: dict[str, list[dict[str, Any]]] = {m: [] for m in TITLE}
    for r in rows:
        if not r.get("counters") or not r.get("counts") or not r["id"].endswith(" kernel"):
            continue
        if not r.get("predicted") or "ops" not in r["counts"]:
            continue
        a, x = model(r), measured(r)
        for m in TITLE:
            if a[m] or x[m]:
                out[m].append({"id": r["id"], "family": family(r), "model": a[m], "meas": x[m]})
    for vs in out.values():
        pos = [v["model"] / v["meas"] for v in vs if v["model"] and v["meas"]]
        med = statistics.median(pos) if pos else 1.0
        for v in vs:
            v["median"] = med
            v["r"] = v["model"] / v["meas"] / med if v["model"] and v["meas"] else None
    return out


def _off(v: dict[str, Any]) -> bool:
    return v["r"] is None or not 1 / BAND <= v["r"] <= BAND


def unknown_share(rows: list[dict[str, Any]]) -> list[tuple[float, str, str]]:
    """Round 18's proposal: a load known in part of its executions (D895's class). Rows from an
    engine without `addr_mixed` give the row's share of unknown addresses instead."""
    out = []
    for r in rows:
        k = r.get("counts") or {}
        if not r["id"].endswith(" kernel"):
            continue
        for op, n, gaps, left in k.get("addr_mixed", []):
            out.append((gaps / n, r["id"], f"{op}: {gaps} of {n} unknown, {left} after its stride"))
        if "addr_mixed" in k:
            continue
        gaps = sum(
            n
            for key, n in k.get("addr_gaps", {}).items()
            if key.startswith("chip") and not table.VALUE_CAUSE.match(key.split(" ", 2)[2])
        )
        if gaps and k.get("addr_chip"):
            out.append((gaps / k["addr_chip"], r["id"], f"row: {gaps} of {k['addr_chip']}"))
    return sorted(out, reverse=True)


def report(doc: dict[str, Any], focus: str | None = None, raw: str | None = None) -> str:
    rows = doc["rows"]
    if focus:
        rows = [r for r in rows if re.search(focus, r["id"])]
    res = check(rows)
    if raw:
        vs = sorted(
            (v for v in res[raw] if _off(v)),
            key=lambda v: -abs(math.log(v["r"])) if v["r"] else -math.inf,
        )
        return "\n".join(
            f"{v['id']:48s} {v['family']:30s} model {v['model']:10.0f} measured {v['meas']:10.0f} "
            f"ratio {'-' if v['r'] is None else format(v['r'], '.2f')}"
            for v in vs
        )
    out = [
        f"Work check, table {doc.get('kind', '?')} of {doc.get('kta_commit', '?')[:7]}: model per "
        f"block over the counter's, by the metric's median, band {BAND}×; raw rows: --raw METRIC"
    ]
    for m, title in TITLE.items():
        vs = res[m]
        off = [v for v in vs if _off(v)]
        none = sum(1 for v in vs if v["r"] is None)
        med = vs[0]["median"] if vs else 1.0
        out.append(
            f"{m:7s} {title:50s} {len(vs):5d} rows, median {med:5.2f}, within "
            f"{len(vs) - len(off):5d}, off {len(off) - none:4d}, one side zero {none:4d}"
        )
        fam: dict[str, list[float]] = {}
        for v in off:
            fam.setdefault(v["family"], []).append(v["r"] or 0.0)
        top = sorted(fam.items(), key=lambda x: -len(x[1]))[:6]
        if top:
            out.append(
                "        "
                + "; ".join(f"{f} {len(x)} at {statistics.median(x):.2f}" for f, x in top)
            )
    us = unknown_share(rows)
    cl: Counter = Counter("load" if " of " in u[2] and ":" in u[2] else "row" for u in us)
    out.append(f"Loads known in part of their executions: {dict(cl)}; largest:")
    out += [f"  {s:6.3f} {i} {d}" for s, i, d in us[:6]]
    return "\n".join(out)


def load_doc(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or TABLE).read_text())


def wrong_pairs(regex: str) -> str:
    """Real pairs of the cached rows, group by group, and the wrong ones by variant over shapes:
    the ladder's work list (LOOP §3b), seen rows on this engine."""
    rows = [r for r in cached(regex)["rows"] if r.get("predicted")]
    times: dict[str, list[float]] = {}
    for n in {table.batch_order(r["note"].split()[-1])[0] for r in rows}:
        times.update(blind.passes(n))
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["set"], []).append({**r, "unit": r["id"].removesuffix(" kernel")})
    out, wrong, pairs = [], Counter(), []
    for name, g in sorted(groups.items()):
        if len(g) < 3:
            continue
        c = blind.real_pairs(g, times)
        err = [math.log(r["predicted"] / r["measured"]) for r in g]
        spread = statistics.pstdev(err)
        real = f"real {c['real_right']}/{c['real_of']}"
        out.append(f"{name:44s} {real}, within the group {100 * spread:.1f} %")
        for i, a in enumerate(g):
            for b in g[i + 1 :]:
                if abs(a["measured"] / b["measured"] - 1) < table.PAIR_GAP:
                    continue
                ta, tb = times.get(a["id"], []), times.get(b["id"], [])
                first = a["measured"] - b["measured"]
                if not all((x - y) * first > 0 for x, y in zip(ta, tb, strict=False)):
                    continue
                if first * (a["predicted"] - b["predicted"]) <= 0:
                    key = sorted(re.sub(r"^[a-z]+\d+_", "", r["unit"]) for r in (a, b))
                    wrong[" / ".join(key)] += 1
                    pairs.append(f"{a['unit']}|{b['unit']}")
    out.append(f"wrong real pairs by variant, over {len(groups)} groups:")
    out += [f"  {n:3d} {k}" for k, n in wrong.most_common(16)]
    out.append("for the sensitivity step: " + ",".join(pairs))
    return "\n".join(out)


def cached(regex: str) -> dict[str, Any]:
    """Rows a run of this engine replayed, from the cache; a row not there stops it."""
    cal, sha = upstream.calibration_dir(upstream.UPSTREAM, upstream.UPSTREAM_REV)
    want = [r for r in table.specs(cal, ("batches",)) if re.search(regex, r["id"])]
    rows = list(table.run(cal, sha, want, 1, cached_only=True, trace=False))
    return {"kind": "cache", "kta_commit": upstream.kta_commit()[0], "rows": rows}
