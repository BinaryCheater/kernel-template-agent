"""``python -m kta_perfmodel.review <round>``: one short printout of a round, computed from the raw
predicted and measured values of its batches: blind against 28 §1, real pairs with and without
warm GEMV, misses by class and cell, matrix counts, testing and batch minutes. Nothing stored; it
replaces hand-made tables in the report and the supervisor's review (D894).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from typing import Any

from . import blind, full, matrix
from .table import BATCHES, SHORT_US, batch_order, goal

MISS_US = 0.10


def _family(row: dict[str, Any]) -> str:
    """The key of the round's miss table: `decode warm`, `hgemm rotating`, or the hand kernel."""
    s = row["launch"].get("set", "")
    for d in blind.DESIGN:
        if s.startswith(d):
            return f"{d.strip()} {'rotating' if s.endswith('rotating') else 'warm'}"
    return row["unit"].split("_", 1)[1].split("_")[0] if "_" in row["unit"] else row["unit"]


def _sums(rs: list[dict[str, Any]], times: dict[str, list[float]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rs:
        groups.setdefault(r["launch"].get("set", "probe"), []).append(r)
    out = []
    for name, g in groups.items():
        if not name.startswith(blind.DESIGN) or len(g) < 3:
            continue
        s, c = goal(g), blind.real_pairs(g, times)
        out.append({"name": name, "n": len(g), **s, **c, "rows": g})
    return out


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f} %)" if b else "0/0"


def _batches(round_no: int) -> list[str]:
    ds = [d for d in BATCHES.iterdir() if batch_order(d.name)[0] == round_no]
    return [d.name for d in sorted(ds, key=lambda p: batch_order(p.name))]


def real_line(round_no: int) -> tuple[int, int, int, int]:
    rs = [r for r in blind.rows(round_no) if r["mode"] == "kernel"]
    gs = [g for g in _sums(rs, blind.passes(round_no)) if not g["name"].startswith("hgemm")]
    a, b = sum(g["real_right"] for g in gs), sum(g["real_of"] for g in gs)
    gs = [g for g in gs if g["name"] not in () and not _warm_gemv(g["name"])]
    return a, b, sum(g["real_right"] for g in gs), sum(g["real_of"] for g in gs)


def _warm_gemv(name: str) -> bool:
    return name.startswith("gemv") and not name.endswith("rotating")


def _batch_minutes(round_no: int) -> str:
    out = []
    for b in _batches(round_no):
        p = BATCHES / b / "batch.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        by: Counter = Counter()
        for s in doc["steps"]:
            k = s["step"].split("-")[0]
            kind = "profiler" if k.startswith("5") else "timing" if k.startswith("4") else "other"
            by[kind] += s["s"]
        t0 = [s["t0"] for s in doc["steps"]]
        wall = (doc["steps"][-1]["t0"] + doc["steps"][-1]["s"] - min(t0)) / 60
        out.append(
            f"{b} {wall:.1f} min: timing {by['timing'] / 60:.1f}, profiler "
            f"{by['profiler'] / 60:.1f}, dropped {len(doc.get('preflight_dropped', {}))}"
        )
    return "; ".join(out) or "none measured"


def _testing(round_no: int) -> str:
    by: Counter = Counter()
    for r in full.runs(round_no):
        by[r["mode"]] += r["cpu_s"]
    parts = ", ".join(f"{k} {v:.0f}" for k, v in sorted(by.items()))
    return (
        f"below the core {full.below_core(round_no):.0f} of {full.BELOW_CORE_S:.0f} CPU s; "
        f"{parts or 'nothing'}"
    )


def report(round_no: int) -> str:
    rs = blind.rows(round_no)
    pred, meas = {}, {}
    for b in _batches(round_no):
        d = BATCHES / b
        if not (d / "measured.json").exists():
            continue
        doc = json.loads((d / "measured.json").read_text())
        if not doc["valid"]:
            continue
        pred.update(json.loads((d / "predicted.json").read_text())["predicted"])
        meas.update({k: v["median"] for k, v in doc["results"].items()})
    kernels = [r for r in rs if r["mode"] == "kernel"]
    times = blind.passes(round_no)
    gs = _sums(kernels, times)
    tot = lambda key, sel: sum(g[key] for g in gs if sel(g["name"]))  # noqa: E731
    hg = lambda n: n.startswith("hgemm")  # noqa: E731
    off = lambda n: not n.startswith("hgemm")  # noqa: E731
    al = lambda _: True  # noqa: E731
    nog = lambda n: off(n) and not _warm_gemv(n)  # noqa: E731
    out = [f"Round {round_no}, blind, first scores of {len(kernels)} kernel rows; {_batches(round_no)}"]
    for name, a, b, sel in (
        ("Pairs ≥ 3 % right, all", "pairs_right", "pairs", al),
        ("  HGEMM", "pairs_right", "pairs", hg),
        ("  off HGEMM", "pairs_right", "pairs", off),
        ("Real pairs, all", "real_right", "real_of", al),
        ("  HGEMM", "real_right", "real_of", hg),
        ("  off HGEMM", "real_right", "real_of", off),
        ("  off HGEMM, without warm GEMV", "real_right", "real_of", nog),
    ):
        out.append(f"{name:34s} {_pct(tot(a, sel), tot(b, sel))}")
    top = sum(bool(g["best_in_top"]) for g in gs)
    losses = sorted((g["first_loss"], g["name"]) for g in gs)
    above = sum(x > 0.03 for x, _ in losses)
    out.append(f"{'Best in top 3':34s} {top}/{len(gs)}")
    if losses:
        out.append(
            f"{'First pick loss':34s} worst {100 * losses[-1][0]:.1f} % ({losses[-1][1]}); "
            f"{above} of {len(gs)} above 3 %"
        )
    k = blind._whatif(blind._pairs(rs, pred, meas), "measured", "first")
    mag = "-" if k["mag"] is None else f"{100 * k['mag']:.0f} %"
    out.append(f"{'What-if direction ≥ 3 %':34s} {_pct(k['right'], k['of'])}")
    out.append(f"{'Magnitude ≥ 10 %':34s} median {mag} over {k['mag_n']}")
    sweep = blind.sweep_rows(kernels)
    bl = [r for r in kernels if r not in sweep]
    longs = [r for r in bl if r["measured"] >= SHORT_US]
    errs = sorted((abs(r["error"]), r) for r in bl)
    if bl:
        t = blind._time(bl)
        out.append(
            f"{'Time, blind rows':34s} median {100 * t['median']:.1f} %, ≥ 30 µs "
            f"{100 * statistics.median(abs(r['error']) for r in longs):.1f} %, "
            f"above 25 %: {t['above_25']}, worst {100 * t['worst']:+.1f} % "
            f"({errs[-1][1]['unit']})"
        )
    out += ["", "Group: pairs, real, wrong real, top 3, loss, median error"]
    for g in sorted(gs, key=lambda g: g["name"]):
        med = statistics.median(abs(r["error"]) for r in g["rows"])
        out.append(
            f"  {g['name']:34s} {g['pairs_right']}/{g['pairs']:<4d} {g['real_right']}/"
            f"{g['real_of']:<4d} {g['wrong_real']:3d}  {'y' if g['best_in_top'] else 'N'} "
            f"{100 * g['first_loss']:5.1f} % {100 * med:5.1f} %"
        )
    miss = [r for r in bl if abs(r["error"]) > MISS_US]
    table = getattr(matrix, f"MISSES{round_no}", None) or {}
    fam = Counter(_family(r) for r in miss)
    out += ["", f"Misses above 10 % on blind kernels: {len(miss)} of {len(bl)}"]
    cl: Counter = Counter()
    cs: dict[str, str] = {}
    for f, n in fam.most_common():
        e = table.get(f)
        cl[("known rule" if e[4] == "k" else "no rule") if e else "not placed"] += n
        cs[f] = f"{e[0]}×{e[1]}×{e[2]}×{e[3]} {e[4]}" if e else "?"
    out.append("  classes " + ", ".join(f"{k} {v}" for k, v in cl.most_common()))
    out.append("  " + "; ".join(f"{f} {n} [{cs[f]}]" for f, n in fam.most_common()))
    worst = sorted(rs, key=lambda r: -abs(r["error"]))[:6]
    out.append(
        "  worst: "
        + ", ".join(f"{r['id'].removesuffix(' kernel')} {100 * r['error']:+.0f} %" for r in worst)
    )
    t = matrix.cells(matrix.ALL_MARKS)
    c = matrix.count(t)
    out += ["", f"Matrix: {len(t)} cells, M {c['M']}, m {c['m']}, empty {c['.']}"]
    out.append(f"Testing: {_testing(round_no)}")
    out.append(f"Batch: {_batch_minutes(round_no)}")
    line = []
    for n in range(13, round_no + 1):
        a, b, x, y = real_line(n)
        line.append(f"r{n} {100 * a / b:.0f}/{100 * x / y:.0f}" if b and y else f"r{n} -")
    out.append("Real pairs off HGEMM, % with / without warm GEMV: " + ", ".join(line))
    return "\n".join(out)


if __name__ == "__main__":
    print(report(int(sys.argv[1])))
    sys.exit(0)
