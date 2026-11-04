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

from .table import BATCHES, GOAL_DIR, GOAL_MAG, SHORT_US, batch_order, first_scores, goal

MODE = re.compile(r"(.*) (kernel|b2b|graph)$")


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


def report(round_no: int) -> str:
    rs = rows(round_no)
    pred, meas = {}, {}
    for d in BATCHES.iterdir():
        if batch_order(d.name)[0] == round_no and (d / "measured.json").exists():
            pred.update(json.loads((d / "predicted.json").read_text())["predicted"])
            for k, v in json.loads((d / "measured.json").read_text())["results"].items():
                meas[k] = v["median"]
    kernels = [r for r in rs if r["mode"] == "kernel"]
    long_rows = [r for r in kernels if r["measured"] >= SHORT_US]
    lines = [f"Blind rows of round {round_no}: {len(rs)} rows first scored, first scores only.", ""]
    lines += ["| Set | n | median | worst | above 25 % |", "| --- | ---: | ---: | ---: | ---: |"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in kernels:
        groups.setdefault(r["launch"].get("set", "probe"), []).append(r)
    for name, g in [("all kernel rows", kernels), ("rows ≥ 30 µs", long_rows), *groups.items()]:
        t = _time(g)
        lines.append(
            f"| {name} | {t['n']} | {100 * t['median']:.1f} % | {100 * t['worst']:+.1f} % | "
            f"{t['above_25']} |"
        )
    lines += ["", "| Design group | pairs ≥ 3 % right | best in top 3 | first pick's loss |"]
    lines += ["| --- | ---: | --- | ---: |"]
    tp = tn = 0
    for name, g in groups.items():
        if not name.startswith(("hgemm", "fa ")) or len(g) < 3:
            continue
        s = goal([{**r, "predicted": r["predicted"]} for r in g])
        tp, tn = tp + s["pairs_right"], tn + s["pairs"]
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
