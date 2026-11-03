"""The error table (docs/design/28 §4.7): every validation kernel and probe replayed on CPU,
measured against predicted, one row each."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from kta.models.gb10sim import probes, validate, ws_validate
from kta.models.gb10sim.model import C
from kta.models.regression import spearman

SETS = ("gemm", "tma", "probes")


def _row(
    rid: str,
    group: str,
    level: int,
    unit: str,
    measured: float,
    run: Callable[[], float],
    excluded: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    t = time.perf_counter()
    predicted = run()
    ok = measured == measured and measured > 0
    return {
        "id": rid,
        "set": group,
        "level": level,
        "unit": unit,
        "measured": measured,
        "predicted": predicted,
        "error": predicted / measured - 1 if ok else None,
        "excluded": excluded if ok else (excluded or "no measured value"),
        "note": note,
        "wall_s": round(time.perf_counter() - t, 2),
    }


def gemm_rows(cal: Path, names: list[str] | None = None) -> Iterator[dict[str, Any]]:
    """The 22 cp.async GEMMs at 1024^3 (upstream validate.kernel_level), warm: the whole kernel,
    its loads-only variant at the grid, and its mma-only variant on one block (block_level)."""
    m = json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())
    regs = dict(
        line.split()
        for line in (cal / "ablation_cpasync" / "regs.txt").read_text().splitlines()
        if line.strip()
    )
    abl = [
        json.loads(line)
        for line in (cal / "ablation_cpasync" / "ablation.jsonl").read_text().splitlines()
        if line
    ]
    for name in names or sorted(m):
        r = m[name]
        h = r["pick"]
        tiles = r["blocks"]
        rmax = validate._rmax(h, int(regs[name]))
        yield _row(
            f"gemm1024 {name} full",
            "gemm1024 full",
            7,
            "us",
            r["full_event_us"],
            lambda name=name, h=h, tiles=tiles, rmax=rmax: validate.kernel_us(
                cal, name, "full", h, tiles, rmax
            ),
        )
        nc = [
            x
            for x in abl
            if x["name"] == name
            and x["variant"] == "nocompute"
            and x.get("ok")
            and int(x["blocks"]) == tiles
        ]
        yield _row(
            f"gemm1024 {name} nocompute",
            "gemm1024 nocompute",
            7,
            "us",
            nc[0]["median_us"] if nc else float("nan"),
            lambda name=name, h=h, tiles=tiles, rmax=rmax: validate.kernel_us(
                cal, name, "nocompute", h, tiles, rmax
            ),
        )
        yield _row(
            f"gemm1024 {name} noload 1 block",
            "gemm1024 noload 1 block",
            7,
            "us",
            r["compute_per_ktile_us"] * r["KT"] + r["overlap"]["setup_us"],
            lambda name=name: validate.simulate(cal, name, "noload", 1, 1)["us"],
        )


def tma_rows(cal: Path, names: list[str] | None = None) -> Iterator[dict[str, Any]]:
    """The 7 warp-specialised TMA picks (upstream ws_validate): whole kernel at its grid (warm),
    and the mma-only variant on one block."""
    m = json.loads((cal / "ablation" / "mechanisms.json").read_text())
    for name in names or list(m):
        r = m[name]
        h, s, kt = r["pick"], r["shape"], r["KT"]
        note = (
            "measured warm in a session where part of B stayed in the L2; upstream compares the "
            "decode picks cold against a later flush measurement (gb10 design note §7)"
            if name in ("m32", "m128")
            else None
        )
        yield _row(
            f"tma {name} full",
            "tma full",
            7,
            "us",
            r["full_event_us"],
            lambda name=name, h=h, s=s: ws_validate.kernel_us(cal, name, h, s),
            note=note,
        )
        setup = r.get("block_setup_us") or C["block_setup_us"]
        yield _row(
            f"tma {name} noload 1 block",
            "tma noload 1 block",
            7,
            "us",
            r["compute_per_ktile_us"] * kt + setup,
            lambda name=name, h=h, s=s: ws_validate.ws_sim(cal, name, "noload", h, s)["us"],
        )


def probe_rows(cal: Path) -> Iterator[dict[str, Any]]:
    for case in probes.all_cases(cal):
        yield _row(
            case.id,
            f"probe {case.probe}",
            case.level,
            case.unit,
            case.measured,
            lambda case=case: probes.run_case(case)[0],
            excluded=case.excluded,
        )


def build(cal: Path, sets: tuple[str, ...] = SETS) -> Iterator[dict[str, Any]]:
    if "gemm" in sets:
        yield from gemm_rows(cal)
    if "tma" in sets:
        yield from tma_rows(cal)
    if "probes" in sets:
        yield from probe_rows(cal)


def summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == group and not r["excluded"]]
        if not rs:
            continue
        e = [r["error"] for r in rs]
        worst = max(e, key=abs)
        item: dict[str, Any] = {
            "set": group,
            "n": len(rs),
            "excluded": sum(1 for r in rows if r["set"] == group and r["excluded"]),
            "median_abs": statistics.median(abs(x) for x in e),
            "worst": worst,
            "within_10": sum(abs(x) <= 0.1 for x in e),
        }
        if len(rs) >= 3 and group.endswith("full"):
            item["spearman"] = spearman([r["predicted"] for r in rs], [r["measured"] for r in rs])
        out.append(item)
    return out


def _fmt(x: float) -> str:
    return f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"


def _pct(x: float | None) -> str:
    return "—" if x is None or math.isnan(x) else f"{100 * x:+.1f} %"


def render(doc: dict[str, Any], top: int = 15) -> str:
    rows = doc["rows"]
    lines = [
        "# Error table",
        "",
        f"Produced by `scripts/perfmodel-table` from KTA commit `{doc['kta_commit']}`"
        f"{' (dirty)' if doc['kta_dirty'] else ''} on {doc['date']}; upstream inputs "
        f"`{doc['upstream']}` at `{doc['upstream_rev']}`. CPU only, {doc['wall_s']:.0f} s. "
        "Error = predicted / measured − 1 (positive: the model is slow). "
        "Machine-readable copy: `error-table.json`.",
        "",
        "## Summary",
        "",
        "| Set | n | excluded | median abs error | worst | within 10 % | Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in doc["summary"]:
        sp = f"{s['spearman']:.2f}" if "spearman" in s else ""
        lines.append(
            f"| {s['set']} | {s['n']} | {s['excluded']} | {100 * s['median_abs']:.1f} % | "
            f"{_pct(s['worst'])} | {s['within_10']} | {sp} |"
        )
    ranked = sorted((r for r in rows if not r["excluded"]), key=lambda r: -abs(r["error"]))
    lines += ["", f"## Top {top} rows by absolute error", ""]
    lines += _table(ranked[:top])
    lines += ["", "## All rows", ""]
    lines += _table(rows)
    return "\n".join(lines) + "\n"


def _table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| Row | Level | Unit | Measured | Predicted | Error | Attribution (critical path) | "
        "Defaults | Note |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in rows:
        att = "; ".join(
            f"{a['resource']}: {a['mechanism']} {100 * a['share']:.0f} %"
            for a in r.get("attribution", [])[:3]
        )
        defaults = ", ".join(f"{k} {v}" for k, v in sorted(r.get("defaults", {}).items()))
        note = (r["excluded"] and f"excluded: {r['excluded']}") or r["note"] or ""
        out.append(
            f"| {r['id']} | {r['level']} | {r['unit']} | {_fmt(r['measured'])} | "
            f"{_fmt(r['predicted'])} | {_pct(r['error'])} | {att} | {defaults} | {note} |"
        )
    return out
