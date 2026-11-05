"""``python -m kta_perfmodel.blind <round>``: 28 §1 on the rows a round measured first, from the
predictions committed before its run and its measured values. No replay: these are the first
scores (LOOP §5), and they never change. Rows measured in an earlier batch are repeats, not blind.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from .table import (
    BATCHES,
    GOAL_DIR,
    GOAL_MAG,
    PAIR_GAP,
    SHORT_US,
    batch_order,
    first_scores,
    goal,
)

MODE = re.compile(r"(.*) (kernel|b2b|graph)$")
DESIGN = ("hgemm", "fa ", "gemv ", "decode ", "rmsnorm ", "softmax ")


def rows(round_no: int, root: Path = BATCHES) -> list[dict[str, Any]]:
    """First-scored rows of the round, with their set and what-if link from launch.json."""
    out = []
    for cid, s in first_scores(root).items():
        if batch_order(s["batch"])[0] != round_no:
            continue
        m = MODE.match(cid)
        unit, mode = (m[1], m[2]) if m else (cid, "probe")
        lj = root / s["batch"] / "build" / "testset" / unit / "launch.json"
        launch = json.loads(lj.read_text()) if lj.exists() else {}
        out.append({"id": cid, "unit": unit, "mode": mode, "launch": launch, **s})
    return out


def _time(rs: list[dict[str, Any]]) -> dict[str, Any]:
    err = sorted((r["error"] for r in rs), key=abs)
    return {
        "n": len(err),
        "median": statistics.median(abs(e) for e in err),
        "worst": err[-1],
        "above_25": sum(abs(e) > 0.25 for e in err),
    }


def _pairs(rs: list[dict[str, Any]], pred: dict[str, float], meas: dict[str, float]) -> list[dict]:
    out = []
    for r in rs:
        w = r["launch"].get("whatif")
        if not w or r["mode"] != "kernel":
            continue
        base = f"{w['base']} kernel"
        if base not in pred or base not in meas:
            continue
        short = meas[base] < SHORT_US
        a, b = (f"{w['base']} graph", f"{r['unit']} graph") if short else (base, r["id"])
        if a not in meas or b not in meas:
            continue
        out.append(
            {
                "id": r["unit"],
                "base": w["base"],
                "axis": w["axis"],
                "short": short,
                "measured": meas[r["id"]] / meas[base] - 1,
                "first": pred[r["id"]] / pred[base] - 1,
                "graph_measured": meas[b] / meas[a] - 1,
                "graph_first": pred[b] / pred[a] - 1,
            }
        )
    return out


def _whatif(ps: list[dict[str, Any]], m: str, p: str) -> dict[str, Any]:
    big = [x for x in ps if abs(x[m]) >= GOAL_DIR]
    mag = [abs(x[p] / x[m] - 1) for x in ps if abs(x[m]) >= GOAL_MAG]
    return {
        "right": sum(x[p] * x[m] > 0 for x in big),
        "of": len(big),
        "mag": statistics.median(mag) if mag else None,
        "mag_n": len(mag),
        "wrong": sorted((x for x in big if x[p] * x[m] <= 0), key=lambda x: -abs(x[m])),
    }


def passes(round_no: int, root: Path = BATCHES) -> dict[str, list[float]]:
    """Every timing of a row in the round's batches, each in its own process, in order."""
    out: dict[str, list[float]] = {}
    for d in sorted(root.iterdir(), key=lambda p: batch_order(p.name)):
        if batch_order(d.name)[0] != round_no or not (d / "measured.json").exists():
            continue
        doc = json.loads((d / "measured.json").read_text())
        if not doc["valid"]:
            continue
        for k, v in doc["results"].items():
            out.setdefault(k, []).append(v["median"])
        for k, v in doc.get("repeats", {}).items():
            out.setdefault(k, []).extend(v)
    return out


def split(p: dict[str, Any], times: dict[str, list[float]], mode: str = "kernel") -> dict[str, Any]:
    """One what-if pair over its repeats. Base and change pair up pass by pass (the same batch).
    `noise`: the measured change does not keep its sign or its size: the spread of the passes
    reaches its median. A wrong prediction on such a pair is not evidence against the model."""
    a, b = times.get(f"{p['base']} {mode}", []), times.get(f"{p['id']} {mode}", [])
    ds = [y / x - 1 for x, y in zip(a, b, strict=False)]
    if len(ds) < 2:
        return {"deltas": ds, "noise": None, "median": ds[0] if ds else None}
    mid = statistics.median(ds)
    return {"deltas": ds, "median": mid, "noise": max(ds) - min(ds) >= abs(mid)}


def real_pairs(g: list[dict[str, Any]], times: dict[str, list[float]]) -> dict[str, int]:
    """28 §1's pairs of one group, first scores, split by the repeats: a pair is `real` when its
    measured order holds in every pass. A report beside the target, not a change of it."""
    c = dict.fromkeys(("right", "of", "real_right", "real_of", "wrong_real", "wrong_flips"), 0)
    for i, a in enumerate(g):
        for b in g[i + 1 :]:
            if abs(a["measured"] / b["measured"] - 1) < PAIR_GAP:
                continue
            first = a["measured"] - b["measured"]
            right = first * (a["predicted"] - b["predicted"]) > 0
            ta, tb = times.get(a["id"], []), times.get(b["id"], [])
            holds = all((x - y) * first > 0 for x, y in zip(ta, tb, strict=False))
            c["of"] += 1
            c["right"] += right
            c["real_of"] += holds
            c["real_right"] += holds and right
            c["wrong_real"] += holds and not right
            c["wrong_flips"] += not holds and not right
    return c


def by_axis(ps: list[dict[str, Any]], times: dict[str, list[float]]) -> list[str]:
    """Direction by axis, `kernel` rows and graph rows side by side (D804 is the user's): the first
    pass as scored, then the median of the passes, and how many wrong pairs are noise."""
    axes: dict[str, dict[str, int]] = {}

    def tally(ax: str, tag: str, m: float, q: float, noise: bool | None) -> None:
        for key in (ax, "all"):
            c = axes.setdefault(key, {})
            if abs(m) < GOAL_DIR:
                continue
            c[f"{tag}_of"] = c.get(f"{tag}_of", 0) + 1
            right = m * q > 0
            c[f"{tag}_right"] = c.get(f"{tag}_right", 0) + right
            c[f"{tag}_noise"] = c.get(f"{tag}_noise", 0) + (not right and bool(noise))

    for p in ps:
        k, g = split(p, times), split(p, times, "graph")
        if k["median"] is not None:
            tally(p["axis"], "k", k["median"], p["first"], k["noise"])
        if g["median"] is not None:
            tally(p["axis"], "g", g["median"], p["graph_first"], g["noise"])
    lines = [
        "| Axis | `kernel` right / of | wrong, noise | graph right / of | wrong, noise |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ax, c in sorted(axes.items(), key=lambda kv: (kv[0] == "all", kv[0])):
        lines.append(
            f"| {ax} | {c.get('k_right', 0)}/{c.get('k_of', 0)} | {c.get('k_noise', 0)} | "
            f"{c.get('g_right', 0)}/{c.get('g_of', 0)} | {c.get('g_noise', 0)} |"
        )
    return lines


def report(round_no: int) -> str:
    rs = rows(round_no)
    pred, meas = {}, {}
    for d in BATCHES.iterdir():
        if batch_order(d.name)[0] == round_no and (d / "measured.json").exists():
            doc = json.loads((d / "measured.json").read_text())
            if not doc["valid"]:
                continue
            pred.update(json.loads((d / "predicted.json").read_text())["predicted"])
            for k, v in doc["results"].items():
                meas[k] = v["median"]
    kernels = [r for r in rs if r["mode"] == "kernel"]
    lines = [f"Blind rows of round {round_no}: {len(rs)} rows first scored, first scores only.", ""]
    lines += ["| Set | n | median | worst | above 25 % |", "| --- | ---: | ---: | ---: | ---: |"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in kernels:
        groups.setdefault(r["launch"].get("set", "probe"), []).append(r)
    sweeps = (
        "stack_path",
        "atomic_bins",
        "wr_footprint",
        "wr_rmw",
        "hash_sweep",
        "record_sweep",
        "stack_load",
        "stage_question",
    )
    blind = [r for r in kernels if r["launch"].get("set") not in sweeps]
    long_blind = [r for r in blind if r["measured"] >= SHORT_US]
    head = [("blind kernels", blind), ("those ≥ 30 µs", long_blind), ("with the sweeps", kernels)]
    for name, g in [*head, *groups.items()]:
        t = _time(g)
        lines.append(
            f"| {name} | {t['n']} | {100 * t['median']:.1f} % | {100 * t['worst']:+.1f} % | "
            f"{t['above_25']} |"
        )
    times = passes(round_no)
    lines += ["", "| Design group | pairs ≥ 3 % right | best in top 3 | first pick's loss |"]
    lines += ["| --- | ---: | --- | ---: |"]
    tp = tn = 0
    real: list[str] = []
    totals: dict[str, int] = {}
    for name, g in groups.items():
        if not name.startswith(DESIGN) or len(g) < 3:
            continue
        s = goal([{**r, "predicted": r["predicted"]} for r in g])
        tp, tn = tp + s["pairs_right"], tn + s["pairs"]
        c = real_pairs(g, times)
        for k, v in c.items():
            totals[k] = totals.get(k, 0) + v
        real.append(
            f"| {name} | {c['right']}/{c['of']} | {c['wrong_real']} | {c['wrong_flips']} | "
            f"{c['real_right']}/{c['real_of']} |"
        )
        first = min(g, key=lambda r: r["predicted"])["unit"]
        lines.append(
            f"| {name} | {s['pairs_right']}/{s['pairs']} | {s['best_in_top']} | "
            f"{100 * s['first_loss']:.1f} % ({first}) |"
        )
    ps = _pairs(rs, pred, meas)
    k = _whatif(ps, "measured", "first")
    mixed = [
        {**x, "m": x["graph_measured"] if x["short"] else x["measured"]}
        | {"p": x["graph_first"] if x["short"] else x["first"]}
        for x in ps
    ]
    g = _whatif(mixed, "m", "p")
    lines += ["", f"Pairs ordered right, all design groups: {tp}/{tn}."]
    for tag, w in (("`kernel` rows", k), ("short rows read from graph rows", g)):
        mag = "—" if w["mag"] is None else f"{100 * w['mag']:.0f} %"
        lines.append(
            f"What-if, {tag}: direction ≥ 3 % {w['right']}/{w['of']}, "
            f"magnitude ≥ 10 % median {mag} over {w['mag_n']}."
        )
    if any(len(v) > 1 for v in times.values()) and real:
        lines += ["", "The same pairs by the repeats: real when the measured order holds in every "]
        lines[-1] += "pass (a report; the target stays on all pairs):"
        lines += ["", "| Design group | all pairs right | wrong, real | wrong, order flips | "]
        lines[-1] += "real pairs right |"
        lines += ["| --- | ---: | ---: | ---: | ---: |", *real]
        lines.append(
            f"| all | {totals['right']}/{totals['of']} | {totals['wrong_real']} | "
            f"{totals['wrong_flips']} | {totals['real_right']}/{totals['real_of']} |"
        )
    if any(len(v) > 1 for v in times.values()):
        lines += ["", "What-if over the repeats, median of the passes against the first score:", ""]
        lines += by_axis(ps, times)
    lines += ["", "Wrong directions (`kernel` rows):", ""]
    lines += ["| Change | Axis | Measured | First |", "| --- | --- | ---: | ---: |"]
    lines += [
        f"| {x['id']} | {x['axis']} | {100 * x['measured']:+.1f} % | {100 * x['first']:+.1f} % |"
        for x in k["wrong"]
    ]
    lines += [
        "",
        "Worst rows:",
        "",
        "| Row | Measured | First | Error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in sorted(rs, key=lambda r: -abs(r["error"]))[:25]:
        lines.append(
            f"| {r['id']} | {r['measured']:.2f} | {r['predicted']:.2f} | "
            f"{100 * r['error']:+.1f} % |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(int(sys.argv[1])))
    sys.exit(0)
