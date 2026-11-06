"""One search round over a session's family: MIP shortlist, compile, rerank, measure, refit."""

import dataclasses
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from tools.kta_bridge import forms
from tools.kta_bridge.calibrate import nnls
from tools.kta_bridge.freeze import HARDWARE
from tools.template_chain import export, workers
from tools.template_chain.tasks import TASKS

CONSTANTS = Path(__file__).parents[1] / "kta_bridge/constants-v4.json"
SHORTLIST = 12
MEASURE = 6
CONTROLS = 2
MAX_DOMAIN = 20000
SEED = 20261106
PASSES = 3  # shortlists tried when compiles reject picks
RIDGE = 0.1  # pull of each free constant toward its start, against few rows


def load_form(workspace: Path):
    sys.path.insert(0, str(workspace))
    sys.modules.pop("form_api", None)
    spec = importlib.util.spec_from_file_location("form", workspace / "form.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def native(a) -> list[forms.Amounts]:
    items = a if isinstance(a, list | tuple) else [a]
    return [forms.Amounts(**dataclasses.asdict(k)) for k in items]


class Wrapped:
    """A form whose amounts are the bridge's dataclass."""

    def __init__(self, form) -> None:
        self.form = form
        self.FACTORS = form.FACTORS
        self.legal = form.legal

    def amounts(self, shape: dict, x: dict, records) -> list[forms.Amounts]:
        return native(self.form.amounts(shape, x, records))


def start_constants(task: dict) -> dict:
    c = json.loads(CONSTANTS.read_text())[task["constants"]]
    for name in ("scalar_w15", "scalar_w2", "scalar_w3"):
        c.setdefault(name, c.get("scalar_w1", 1.0))
    c.setdefault("spill", 10.0)
    return c


def design_row(a: list[forms.Amounts], c: dict) -> dict[str, float]:
    row: dict[str, float] = {}
    for kernel in a:
        additive, branches = forms.terms(kernel)
        active = branches[forms.active_branch(kernel, c)]
        for k, v in {**additive, **active}.items():
            row[k] = row.get(k, 0.0) + v
    return row


def refit(rows: list[dict], start: dict) -> dict:
    """Software constants on the measured rows, hardware frozen, by alternating branches."""
    c = dict(start)
    previous = None
    for _ in range(25):
        design = [design_row(r["a"], c) for r in rows]
        active = [sorted(d) for d in design]
        if active == previous:
            break
        previous = active
        names = sorted({k for d in design for k in d} - set(HARDWARE))
        a = np.zeros((len(rows) + len(names), len(names)))
        b = np.zeros(len(rows) + len(names))
        for i, (d, r) in enumerate(zip(design, rows, strict=True)):
            b[i] = 1.0 - sum(c.get(k, 0.0) * v for k, v in d.items() if k in HARDWARE) / r["us"]
            for k, v in d.items():
                if k in names:
                    a[i, names.index(k)] = v / r["us"]
        for j, k in enumerate(names):
            weight = RIDGE / max(start.get(k, 1.0), 1e-3)
            a[len(rows) + j, j] = weight
            b[len(rows) + j] = weight * start.get(k, 0.0)
        c = {**c, **dict(zip(names, map(float, nnls(a, b)), strict=True))}
    return c


def candidate(form, shape: dict, c: dict, key: str, role: str, r: dict | None) -> dict:
    x = json.loads(key)
    r = r or {"error": "no result"}
    records = r.get("compiled")
    row = {
        "x": x,
        "role": role,
        "correct": bool(r.get("correct")),
        "errors": r.get("errors"),
        "error": r.get("error"),
        "records": records,
        "pre_us": export.time_at(form.amounts(shape, x, None), c),
    }
    if row["correct"] and records:
        try:
            row["post_us"] = export.time_at(form.amounts(shape, x, records), c)
        except Exception as e:  # noqa: BLE001  # a form that fails on its records is reported
            row["error"] = f"amounts with records: {e}"
    return row


def run_round(run: Path, task_name: str, n: int) -> dict:
    task = TASKS[task_name]
    ws = run / "workspace"
    rdir = run / f"round{n}"
    rdir.mkdir(exist_ok=True)
    shape = task["shape"]
    form = Wrapped(load_form(ws))
    raw = form.form
    c_path = run / f"constants-r{n - 1}.json"
    c = json.loads(c_path.read_text()) if c_path.exists() else start_constants(task)
    measured = []
    for p in sorted(run.glob("round*/rows.json")):
        if p.parent.name != rdir.name:
            measured += json.loads(p.read_text())
    d = export.domain(form, shape)
    if not d.points or len(d.points) > MAX_DOMAIN:
        return {"status": f"domain size {len(d.points)}"}
    point = lambda x: tuple(x[k] for k in d.names)  # noqa: E731
    legal = set(d.points)
    seen = tuple(
        q for q in (point(r["x"]) for r in measured if set(r["x"]) == set(d.names)) if q in legal
    )
    structure = tuple(getattr(raw, "STRUCTURE", d.names))
    rng = random.Random(SEED + n)
    rows: list[dict] = []
    roles: dict[str, str] = {}
    failed: list[tuple] = []
    solves = []
    mip_error = None
    if n == 1:
        roles[json.dumps(raw.ANCHOR, sort_keys=True)] = "anchor"
        rest = [dict(zip(d.names, p, strict=True)) for p in d.points]
        rest = [x for x in rest if x != raw.ANCHOR]
        for x in rng.sample(rest, min(CONTROLS * 3, len(rest))):
            roles[json.dumps(x, sort_keys=True)] = "random"
    table = export.enumerate_all(d, c)
    for attempt in range(PASSES):
        solved = export.solve(d, c, SHORTLIST, structure, (*seen, *failed))
        solves.append({k: solved[k] for k in ("status", "columns", "rows", "seconds")})
        if not solved["picks"]:
            break
        if attempt == 0:
            unseen = [t for t in table if t[1] not in seen]
            mip_error = abs(solved["picks"][0]["objective"] - unseen[0][0])
        fresh = []
        for p in solved["picks"]:
            key = json.dumps(p["x"], sort_keys=True)
            if roles.get(key) != "mip" and key not in {
                json.dumps(r["x"], sort_keys=True) for r in rows
            }:
                roles[key] = "mip"
                fresh.append(key)
        pending = [k for k in roles if k not in {json.dumps(r["x"], sort_keys=True) for r in rows}]
        jobs = [
            {"id": key, "kind": "candidate", "x": json.loads(key), "shape": shape, "seed": 3}
            for key in pending
        ]
        compiled = workers.compile_jobs(task_name, ws, jobs)
        for key in pending:
            rows.append(candidate(form, shape, c, key, roles[key], compiled.get(key)))
            if not rows[-1]["correct"]:
                failed.append(point(rows[-1]["x"]))
        good = [r for r in rows if r["role"] == "mip" and r["correct"] and r.get("post_us")]
        if len(good) >= MEASURE:
            break
    usable = [r for r in rows if r["correct"] and r.get("post_us")]
    chosen = sorted((r for r in usable if r["role"] == "mip"), key=lambda r: r["post_us"])[:MEASURE]
    chosen += [r for r in usable if r["role"] == "anchor"]
    chosen += [r for r in usable if r["role"] == "random"][:CONTROLS]
    summary = {
        "round": n,
        "domain": len(d.points),
        "mip": solves,
        "mip_minus_enumeration_us": mip_error,
        "constants": c,
        "candidates": rows,
        "chosen": [r["x"] for r in chosen],
    }
    (rdir / "predictions.json").write_text(json.dumps(summary, indent=1))
    if not chosen:
        return {"status": "no correct candidate", **summary}
    batch = workers.measure(
        task,
        task_name,
        ws,
        shape,
        [
            {
                "id": json.dumps(r["x"], sort_keys=True),
                "kind": "candidate",
                "x": r["x"],
                "shape": shape,
                "seed": 3,
                "timing": True,
            }
            for r in chosen
        ],
    )
    (rdir / "batch.json").write_text(json.dumps(batch, indent=1))
    if not batch["valid"]:
        return {"status": "invalid batch", **summary}
    out = []
    for r in chosen:
        got = batch["rows"].get(json.dumps(r["x"], sort_keys=True), {})
        ok = got.get("correct") and got.get("replay_correct") and got.get("median_us")
        out.append(
            {
                **r,
                "us": got.get("median_us") if ok else None,
                "iqr_us": got.get("iqr_us"),
                "measure_error": got.get("error"),
                "replay_errors": got.get("replay_errors"),
            }
        )
    (rdir / "rows.json").write_text(json.dumps(out, indent=1))
    fit_rows = []
    for r in [*measured, *out]:
        if r.get("us") and r.get("records") and set(r["x"]) == set(d.names):
            fit_rows.append({"a": form.amounts(shape, r["x"], r["records"]), "us": r["us"]})
    fitted = refit(fit_rows, start_constants(task)) if fit_rows else c
    (run / f"constants-r{n}.json").write_text(json.dumps(fitted, indent=1, sort_keys=True))
    timed = [r for r in out if r["us"]]
    summary["comparator_us"] = [(batch.get(k) or {}).get("median_us") for k in ("before", "after")]
    summary["best_us"] = min((r["us"] for r in timed), default=None)
    summary["median_abs_error"] = (
        statistics.median(abs(r["post_us"] / r["us"] - 1) for r in timed) if timed else None
    )
    (rdir / "summary.json").write_text(json.dumps(summary, indent=1))
    return {"status": "ok", **summary}


def report(run: Path, n: int) -> str:
    s = json.loads((run / f"round{n}/summary.json").read_text())
    rows = json.loads((run / f"round{n}/rows.json").read_text())
    lines = [
        f"# Search round {n}",
        "",
        f"Domain {s['domain']}. Production comparator {s['comparator_us']} µs before and after.",
        "Predicted is the form on compile records, frozen before measuring.",
        "",
        "| Role | Assignment | Predicted µs | Measured µs | Regs | Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda r: r["us"] or 1e12):
        regs = [k.get("regs") for k in r.get("records") or []]
        note = (r.get("measure_error") or "")[:80] or ("" if r["us"] else "failed replay check")
        lines.append(
            f"| {r['role']} | `{json.dumps(r['x'])}` | {r['post_us']:.0f} | "
            f"{r['us'] and round(r['us'], 1)} | {regs} | {note} |"
        )
    bad = [r for r in s["candidates"] if not r["correct"]]
    if bad:
        lines += ["", f"{len(bad)} shortlisted assignments failed before timing:", ""]
        for r in bad[:8]:
            why = r.get("error") or r.get("errors")
            lines.append(f"- `{json.dumps(r['x'])}`: {str(why)[:300]}")
    return "\n".join(lines) + "\n"
