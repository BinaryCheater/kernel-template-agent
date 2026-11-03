"""Run the SM simulator on the 22 ablated cp.async kernels and their variants and compare with the
ablation's timings.

Copied from AI-Infra-Co-Design agentic/gb10/sim/validate.py (commit 0eef497, D681). KTA changes: the
calibration
directory is an argument (`cal` = a copy of calibration/spark-6951), results are returned instead of
printed or
written under runs/, and the rank correlation is kta.models.regression.spearman instead of
scipy's."""

from __future__ import annotations

import json
import math
import re
import statistics as st
import time
from pathlib import Path
from typing import Any

from kta.models.regression import spearman

from .kernel import occupancy
from .machine import Machine, default_machine
from .model import SMS, C
from .sass import main_loop
from .sm import SM, kernel_sass

F = C["clock_GHz"] * 1e3  # cycles per us


def trips(cal: Path, name: str, variant: str) -> int:
    src = (cal / "ablation_cpasync" / name / f"{variant}.kernel.cu").read_text()
    m = re.search(r"for \(int k = 0; k < (\d+); \+\+k\)", src)
    return int(m.group(1)) if m else 1


def sass_trips(ins, h: dict, src_trips: int) -> int:
    """The machine loop may be unrolled: its trip count is the source count over the k-tiles one
    iteration covers,
    found from the mma (or copies) one iteration holds against what one k-tile needs per warp."""
    lp = main_loop(ins)
    if not lp:
        return 1
    body = ins[lp[0] : lp[1] + 1]
    warps = h["threads"] // 32
    n_mma = sum(x.base in ("HMMA", "IMMA", "QMMA", "OMMA") for x in body)
    n_cp = sum(x.base == "LDGSTS" for x in body)
    if n_mma:
        per_kt = h["block_M"] * h["block_N"] * h["block_K"] / (16 * 8 * 16) / warps
        u = n_mma / per_kt
    elif n_cp:
        per_kt = (h["block_M"] + h["block_N"]) * h["block_K"] * 2 / 16 / h["threads"]
        u = n_cp / per_kt
    else:
        u = 1
    return max(1, round(src_trips / max(1, round(u))))


def copy_lines(h: dict) -> list:
    """Lines each copy (32 lanes x 16 B) touches, per warp, in the order TileLang issues them: the A
    tile's copies (rows
    of block_K*2 bytes) then the B tile's (rows of block_N*2 bytes). A 512-byte copy spans 512/row
    rows; each row covers
    max(1, row/128) lines."""

    def lines(row):
        rows = max(1, 512 // row)
        return rows * max(1, min(row, 512) // 128)

    n_a = h["block_M"] * h["block_K"] * 2 // 16 // h["threads"]
    n_b = h["block_K"] * h["block_N"] * 2 // 16 // h["threads"]
    return [lines(h["block_K"] * 2)] * max(1, n_a) + [lines(h["block_N"] * 2)] * max(1, n_b)


def _mechanisms(cal: Path) -> dict[str, Any]:
    return json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())


def _regs(cal: Path) -> dict[str, str]:
    text = (cal / "ablation_cpasync" / "regs.txt").read_text()
    return dict(line.split() for line in text.splitlines() if line.strip())


def simulate(
    cal: Path,
    name: str,
    variant: str,
    blocks: int,
    active: float = 1,
    resident: int | None = None,
    grid: int | None = None,
    machine: Machine | None = None,
    sm_hook=None,
    trace: bool = False,
) -> dict:
    """One SM on a variant's SASS. `sm_hook(sm)` is called with the simulator after its run."""
    m = machine or default_machine()
    ins = kernel_sass(cal / "sass" / f"{name}__{variant}.sass")
    h = _mechanisms(cal)[name]["pick"]
    sm = SM(
        ins,
        h["threads"],
        blocks,
        sass_trips(ins, h, trips(cal, name, variant)),
        active_sms=active,  # type: ignore[arg-type]  # upstream passes fractional SM equivalents
        copy_lines=copy_lines(h),
        resident=resident,
        grid_blocks=grid,
        machine=m,
        trace=trace,
    )
    t = time.time()
    r = sm.run()
    r["wall_s"] = round(time.time() - t, 2)
    r["us"] = r["cycles"] / (m.param(m.desc["clock"]["block_level_GHz"]) * 1e3)
    if sm_hook is not None:
        sm_hook(sm)
    return r


def _rmax(h: dict, regs: int, machine: Machine | None = None) -> int:
    smem = h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    return occupancy(h, regs, smem, machine)


def block_level(cal: Path, names: list[str] | None = None) -> list[dict]:
    """One block-level row per kernel: (simulated, measured) us of the whole block (`full`), the mma
    alone on one block
    (`noload`) and the loads alone at the kernel's residency (`nocompute`)."""
    m = _mechanisms(cal)
    regs = _regs(cal)
    rows = []
    for name in names or sorted(m):
        r = m[name]
        h = r["pick"]
        tiles = r["blocks"]
        rmax = _rmax(h, int(regs[name]))
        res = min(rmax, math.ceil(tiles / SMS))
        act = min(SMS, tiles)
        full = simulate(cal, name, "full", res, act)
        nol = simulate(cal, name, "noload", 1, 1)
        noc = simulate(cal, name, "nocompute", res, act)
        o = r["overlap"]
        kt = r["KT"]
        meas_full = o["measured_block_us"]
        meas_c = r["compute_per_ktile_us"] * kt + o["setup_us"]
        meas_l = o["load_part_us_under_full_grid"] + o["setup_us"]
        rows.append(
            {
                "kernel": name,
                "R": res,
                "full": (full["us"], meas_full),
                "noload": (nol["us"], meas_c),
                "nocompute": (noc["us"], meas_l),
                "wall_s": full["wall_s"],
                "assumed_latency": full["assumed_latency"],
            }
        )
    return rows


def kernel_us(
    cal: Path,
    name: str,
    variant: str,
    h: dict,
    tiles: int,
    rmax: int,
    machine: Machine | None = None,
    sm_hook=None,
    trace: bool = False,
) -> float:
    """Simulated kernel time: the block scheduler fills each SM up to Rmax blocks; each wave is
    simulated with its own
    blocks per SM and number of busy SMs; plus the launch (2.6 us, probe)."""
    from .kernel import FLOW

    m = machine or default_machine()
    C, SMS = m.C, m.sms
    if FLOW:
        # the busiest SM runs its share of the blocks, at most Rmax at a time, a new one starting
        # when one exits; the chip's L2 and store limits are shared by the blocks in flight (as
        # min(48, tiles/Rmax) equal SMs)
        return (
            C["launch_stream_us"]
            + simulate(
                cal,
                name,
                variant,
                math.ceil(tiles / SMS),
                min(tiles, SMS * rmax) / min(rmax, math.ceil(tiles / SMS)),
                resident=rmax,
                grid=tiles,
                machine=m,
                sm_hook=sm_hook,
                trace=trace,
            )["us"]
        )
    left, t = tiles, C["launch_stream_us"]
    while left > 0:
        n = min(left, SMS * rmax)
        res = math.ceil(n / SMS)
        # the chip's L2 and store limits are shared by the blocks in flight: an SM holding R blocks
        # gets R/n of them, i.e. it behaves as one of n/R equal SMs (measured: the L2 rate is the
        # same for 1 to 4 blocks per SM)
        t += simulate(cal, name, variant, res, n / res, machine=m)["us"]
        left -= n
    return t


def kernel_level(cal: Path, names: list[str] | None = None) -> list[dict]:
    """One whole-kernel row per kernel: (simulated, measured) us of the full kernel and of its
    loads-only variant.
    """
    m = _mechanisms(cal)
    regs = _regs(cal)
    abl = [
        json.loads(line)
        for line in (cal / "ablation_cpasync" / "ablation.jsonl").read_text().splitlines()
        if line
    ]
    out = []
    for name in names or sorted(m):
        r = m[name]
        h = r["pick"]
        tiles = r["blocks"]
        rmax = _rmax(h, int(regs[name]))
        meas_full = r["full_event_us"]
        nc = [
            x
            for x in abl
            if x["name"] == name
            and x["variant"] == "nocompute"
            and x.get("ok")
            and int(x["blocks"]) == tiles
        ]
        meas_nc = nc[0]["median_us"] if nc else float("nan")
        sf = kernel_us(cal, name, "full", h, tiles, rmax)
        sn = kernel_us(cal, name, "nocompute", h, tiles, rmax)
        out.append({"kernel": name, "R": rmax, "full": (sf, meas_full), "nocompute": (sn, meas_nc)})
    return out


def summary(rows: list[dict], parts: tuple[str, ...]) -> dict[str, Any]:
    """Median and worst |error|, count within 10 %, per part; the rank correlation of the `full`
    part.
    """
    out: dict[str, Any] = {}
    for part in parts:
        e = sorted(abs(r[part][0] / r[part][1] - 1) for r in rows if r[part][1] == r[part][1])
        out[part] = {
            "median": st.median(e),
            "worst": e[-1],
            "within_10": sum(x <= 0.1 for x in e),
            "n": len(e),
        }
    out["spearman_full"] = spearman([r["full"][0] for r in rows], [r["full"][1] for r in rows])
    return out
