"""The address check (round 15, no GPU): every test kernel's one middle block under lane tracking,
without timing's L2 replay. Per kernel: memory instructions whose address the lanes do not resolve,
the opcode that broke the chain, the share of memory operations, and the replayed global sectors
against the measured ones (R1).

    scripts/perfmodel-table --addresses [--focus REGEX]

`unknown`: the lanes' addresses are not known; `opaque`: they are, but the base is a uniform value
the replay cannot link to its buffer. A cause `LDG...` is an address from loaded data: a value, not
a gap in the lanes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from kta.models.gb10sim import kernel, probeset
from kta.models.gb10sim.machine import default_machine
from kta.models.gb10sim.sm import SM

from . import full, table

ST = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
LD = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
KEEP = ("grid_dim", "block_dim", "const_bank", "machine", "stack_bytes")
VALUE = table.VALUE_CAUSE


def _one_block(ins: list, threads: int, per_sm: int, trips: int, **kw: Any) -> SM:
    grid = kw.get("grid_dim") or (1,)
    n = 1
    for g in grid:
        n *= g
    sm = SM(
        ins,
        threads,
        1,
        trips,
        block_ids=[n // 2 + 1 if n > 2 else 0],
        **{k: v for k, v in kw.items() if k in KEEP},
    )
    sm.lane_track = True
    sm.addr_sectors, sm.addr_check = {}, {}
    return sm


def check_row(job: dict) -> dict:
    """One kernel, by its row (batch, case id, counters)."""
    m = default_machine().with_overrides({"sim.l2_replay": False})
    kernel.SM = _one_block  # type: ignore[assignment,misc]
    cases = {c.id: c for c in probeset.all_cases(table.BATCHES / job["batch"])}
    sm = cases[job["case"]].build(machine=m)
    sm.run()
    sm = getattr(sm, "sm", sm)
    gaps = Counter()
    for base, n, gap in sm.addr_check.values():
        gaps[(gap[0], gap[1], base)] += n
    c = job.get("counters") or {}
    grid = float(c.get("launch__grid_size") or 0)
    sect = sm.addr_sectors or {}
    return {
        "id": job["case"],
        "issued": sm.count.get("issued", 0),
        "inst_meas": float(c.get("smsp__inst_executed.sum") or 0) / grid if grid else None,
        "family": job["set"],
        "chip": sm.count.get("addr_chip", 0),
        "local": sm.count.get("addr_local", 0),
        "gaps": [[*k, n] for k, n in gaps.most_common()],
        "pcs": len(sm.addr_check),
        "ld_model": sect.get("ld", 0),
        "st_model": sect.get("st", 0),
        "ld_meas": float(c.get(LD) or 0) / grid if grid else None,
        "st_meas": float(c.get(ST) or 0) / grid if grid else None,
    }


def jobs(focus: str | None) -> list[dict]:
    """The latest row of each kernel, one per distinct code and launch shape."""
    rows = table.latest(table.batch_specs())
    out, seen = [], set()
    for r in rows:
        if not r["id"].endswith(" kernel") or r["job"][0] != "batch":
            continue
        if focus and not re.search(focus, r["id"]):
            continue
        d = table.BATCHES / r["job"][1] / "build" / "testset" / r["id"].removesuffix(" kernel")
        sass = d / "k.sass"
        if not sass.exists():
            continue
        key = hashlib.sha256(sass.read_bytes()).hexdigest()
        key += d.name.removesuffix("_rotating")
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "batch": r["job"][1],
                "case": r["job"][2],
                "set": r["set"],
                "counters": r.get("counters"),
            }
        )
    return out


def resolved(rows: list[dict]) -> dict[str, Any]:
    """Share of memory operations with a resolved address, over kernels (each counted by its own
    share, so a long kernel does not hide a short one) and over operations."""
    ops = gap = val = 0
    cause: Counter = Counter()
    for r in rows:
        ops += r["chip"] + r["local"]
        for kind, why, base, n in r["gaps"]:
            if VALUE.match(why):
                val += n
            else:
                gap += n
                cause[f"{kind} {why} @{base}"] += n
    return {
        "kernels": len(rows),
        "ops": ops,
        "value_dependent": val,
        "gap": gap,
        "covered": 1 - gap / max(1, ops),
        "causes": cause.most_common(40),
    }


def report(rows: list[dict], worst: int = 40) -> None:
    s = resolved(rows)
    print(
        f"{s['kernels']} kernels, {s['ops']} memory ops in one block: "
        f"{100 * s['covered']:.2f} % resolved or value-dependent; "
        f"{s['value_dependent']} from loaded data, {s['gap']} gaps"
    )
    for k, n in s["causes"]:
        print(f"  {n:10d}  {k}")
    print("kernels by gap share of memory ops:")
    by = []
    for r in rows:
        g = sum(n for _, w, _, n in r["gaps"] if not VALUE.match(w))
        tot = max(1, r["chip"] + r["local"])
        by.append((g / tot, r))
    for sh, r in sorted(by, key=lambda x: -x[0])[:worst]:
        if sh:
            top = next((f"{k} {w} @{b}" for k, w, b, _ in r["gaps"] if not VALUE.match(w)), "")
            print(f"  {100 * sh:5.1f} %  {r['id']:48s} {top}")
    print("kernels by sectors, replay / measured, off by more than 5 %:")
    bad = []
    for r in rows:
        for key in ("ld", "st"):
            mo, me = r[f"{key}_model"], r[f"{key}_meas"]
            if me and abs(mo / me - 1) > 0.05:
                bad.append((abs(mo / me - 1), r["id"], key, mo, me))
    n = sum(1 for r in rows if r["ld_meas"] or r["st_meas"])
    print(f"  {len(bad)} directions off, of {n} kernels with counters")
    for _, i, key, mo, me in sorted(bad, reverse=True)[:worst]:
        print(f"  {i:48s} {key} {mo:12.0f} {me:12.1f}  {mo / me:6.2f}")


def sector_line(rows: list[dict]) -> str:
    """Replayed against measured global sectors (R1), one line for the error table."""
    n = bad_ld = bad_st = 0
    for r in rows:
        if r["id"].startswith("lib_") or not (r["ld_meas"] or r["st_meas"]):
            continue
        n += 1
        if r["ld_meas"] and abs(r["ld_model"] / r["ld_meas"] - 1) > 0.05:
            bad_ld += 1
        if r["st_meas"] and abs(r["st_model"] / r["st_meas"] - 1) > 0.05:
            bad_st += 1
    return (
        f"Sectors, replay against counters, one block per distinct kernel: {n} kernels, loads "
        f"off by more than 5 % on {bad_ld}, stores on {bad_st}. Known: FA and decode stores and "
        "copies of 16 bytes per lane at a row stride count half (`scripts/perfmodel-table "
        "--addresses`)."
    )


def main(focus: str | None, procs: int) -> list[dict]:
    todo = jobs(focus)
    with full.scratch("addresses", len(todo)):
        with ProcessPoolExecutor(procs) as ex:
            out = []
            for j, f in [(j, ex.submit(check_row, j)) for j in todo]:
                try:
                    out.append(f.result(timeout=300))
                except Exception as e:  # noqa: BLE001 - a kernel the walk cannot run is reported
                    print(f"  failed {j['case']}: {type(e).__name__} {e}")
    return out


def save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
