"""The error table (docs/design/28 §4.7): every validation kernel and probe replayed on CPU,
measured against predicted, one row each.

A row is a spec (identity, measured value) plus a job (what to replay). Jobs run in a process
pool, and each result is cached under a key made of the engine's source and parameter files, the
upstream input commit and the job, so a rerun after a change only replays what the change can
affect (any engine file change replays everything)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from kta.models.gb10sim import probes, validate, ws_validate
from kta.models.gb10sim.model import C
from kta.models.regression import spearman

SETS = ("gemm", "tma", "probes")
ENGINE = Path(validate.__file__).parent
CACHE = Path(__file__).resolve().parents[2] / "data" / "perfmodel" / "cache"


def _spec(rid, group, level, unit, measured, job, excluded=None, note=None) -> dict[str, Any]:
    ok = measured == measured and measured > 0
    return {
        "id": rid,
        "set": group,
        "level": level,
        "unit": unit,
        "measured": measured,
        "excluded": excluded if ok else (excluded or "no measured value"),
        "note": note,
        "job": job,
    }


def gemm_specs(cal: Path) -> Iterator[dict[str, Any]]:
    """The 22 cp.async GEMMs at 1024^3 (upstream validate.kernel_level), warm: the whole kernel,
    its loads-only variant at the grid, and its mma-only variant on one block (block_level)."""
    m = json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())
    abl = [
        json.loads(line)
        for line in (cal / "ablation_cpasync" / "ablation.jsonl").read_text().splitlines()
        if line
    ]
    for name in sorted(m):
        r = m[name]
        g = "gemm1024"
        yield _spec(
            f"{g} {name} full", f"{g} full", 7, "us", r["full_event_us"], ["gemm", name, "full"]
        )
        nc = [
            x
            for x in abl
            if x["name"] == name
            and x["variant"] == "nocompute"
            and x.get("ok")
            and int(x["blocks"]) == r["blocks"]
        ]
        yield _spec(
            f"{g} {name} nocompute",
            f"{g} nocompute",
            7,
            "us",
            nc[0]["median_us"] if nc else float("nan"),
            ["gemm", name, "nocompute"],
        )
        yield _spec(
            f"{g} {name} noload 1 block",
            f"{g} noload 1 block",
            7,
            "us",
            r["compute_per_ktile_us"] * r["KT"] + r["overlap"]["setup_us"],
            ["gemm_block", name, "noload"],
        )


def tma_specs(cal: Path) -> Iterator[dict[str, Any]]:
    """The 7 warp-specialised TMA picks (upstream ws_validate): whole kernel at its grid (warm),
    and the mma-only variant on one block."""
    m = json.loads((cal / "ablation" / "mechanisms.json").read_text())
    for name, r in m.items():
        note = (
            "measured warm in a session where part of B stayed in the L2; upstream compares the "
            "decode picks cold against a later flush measurement (gb10 design note §7)"
            if name in ("m32", "m128")
            else None
        )
        yield _spec(
            f"tma {name} full", "tma full", 7, "us", r["full_event_us"], ["tma", name], note=note
        )
        setup = r.get("block_setup_us") or C["block_setup_us"]
        yield _spec(
            f"tma {name} noload 1 block",
            "tma noload 1 block",
            7,
            "us",
            r["compute_per_ktile_us"] * r["KT"] + setup,
            ["tma_block", name],
        )


def probe_specs(cal: Path) -> Iterator[dict[str, Any]]:
    for case in probes.all_cases(cal):
        yield _spec(
            case.id,
            f"probe {case.probe}",
            case.level,
            case.unit,
            case.measured,
            ["probe", case.id],
            excluded=case.excluded,
        )


def specs(cal: Path, sets: tuple[str, ...] = SETS) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "gemm" in sets:
        out += gemm_specs(cal)
    if "tma" in sets:
        out += tma_specs(cal)
    if "probes" in sets:
        out += probe_specs(cal)
    return out


_cases: dict[str, probes.Case] = {}


def run_job(cal: Path, job: list[Any]) -> dict[str, Any]:
    """Replay one job; the result holds at least `predicted`."""
    kind = job[0]
    if kind == "gemm":
        r = json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())[job[1]]
        regs = dict(
            line.split()
            for line in (cal / "ablation_cpasync" / "regs.txt").read_text().splitlines()
            if line.strip()
        )
        rmax = validate._rmax(r["pick"], int(regs[job[1]]))
        return {"predicted": validate.kernel_us(cal, job[1], job[2], r["pick"], r["blocks"], rmax)}
    if kind == "gemm_block":
        return {"predicted": validate.simulate(cal, job[1], job[2], 1, 1)["us"]}
    if kind in ("tma", "tma_block"):
        r = json.loads((cal / "ablation" / "mechanisms.json").read_text())[job[1]]
        if kind == "tma":
            return {"predicted": ws_validate.kernel_us(cal, job[1], r["pick"], r["shape"])}
        return {"predicted": ws_validate.ws_sim(cal, job[1], "noload", r["pick"], r["shape"])["us"]}
    if kind == "probe":
        if not _cases:
            _cases.update((c.id, c) for c in probes.all_cases(cal))
        return {"predicted": probes.run_case(_cases[job[1]])[0]}
    raise ValueError(f"unknown job {job}")


def engine_hash() -> str:
    """sha256 over the engine's code, description and parameter files and this table code."""
    h = hashlib.sha256()
    files = sorted(p for p in ENGINE.iterdir() if p.suffix in (".py", ".json", ".yaml"))
    for p in [*files, Path(__file__)]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _timed(cal: Path, job: list[Any]) -> dict[str, Any]:
    t = time.perf_counter()
    out = run_job(cal, job)
    out["wall_s"] = round(time.perf_counter() - t, 2)
    return out


def run(
    cal: Path,
    upstream_rev: str,
    rows: list[dict[str, Any]],
    jobs: int,
    use_cache: bool = True,
    hint: dict[str, float] | None = None,
) -> Iterator[dict[str, Any]]:
    """Fill each spec with its result, cached ones first, the rest from a pool of `jobs`
    processes (longest previous runs first); yields rows as they complete."""
    eng = engine_hash()
    CACHE.mkdir(parents=True, exist_ok=True)

    def key(r: dict[str, Any]) -> Path:
        k = hashlib.sha256(json.dumps([eng, upstream_rev, r["job"]]).encode()).hexdigest()
        return CACHE / f"{k}.json"

    todo = []
    for r in rows:
        p = key(r)
        if use_cache and p.exists():
            r.update(json.loads(p.read_text()), cached=True)
            yield _finish(r)
        else:
            todo.append(r)
    todo.sort(key=lambda r: -(hint or {}).get(r["id"], 0.0))
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        futs = {pool.submit(_timed, cal, r["job"]): r for r in todo}
        for f in as_completed(futs):
            r = futs[f]
            res = f.result()
            key(r).write_text(json.dumps(res))
            r.update(res, cached=False)
            yield _finish(r)


def _finish(r: dict[str, Any]) -> dict[str, Any]:
    ok = r["measured"] == r["measured"] and r["measured"] > 0
    r["error"] = r["predicted"] / r["measured"] - 1 if ok else None
    return r


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


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
    replay = sum(r["wall_s"] for r in rows)
    slowest = max(r["wall_s"] for r in rows)
    lines = [
        "# Error table",
        "",
        f"Produced by `scripts/perfmodel-table` from KTA commit `{doc['kta_commit']}`"
        f"{' (dirty)' if doc['kta_dirty'] else ''} on {doc['date']}; upstream inputs "
        f"`{doc['upstream']}` at `{doc['upstream_rev']}`. CPU only: replay time {replay:.0f} s "
        f"summed over rows (slowest row {slowest:.0f} s); this run {doc['wall_s']:.0f} s with "
        f"{doc['jobs']} processes, {doc['cached_rows']} rows from the cache. "
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
