"""The warp-specialized TMA picks (step 2c ablation) on the simulator: one block alone (mma-only,
loads-only, whole)
and the whole kernel at its grid, against the ablation's timings.

Copied from AI-Infra-Co-Design agentic/gb10/sim/ws_validate.py (commit 0eef497, D681). KTA changes:
the calibration
directory is an argument (`cal` = a copy of calibration/spark-6951) and results are returned instead
of printed."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .kernel import dram_clk, wave_dram
from .model import SMS, C, dram_split_kslice
from .sass import main_loop
from .sm import kernel_sass
from .ws import WSSM

F = C["clock_under_load_GHz"] * 1e3


def ws_sim(
    cal: Path,
    name: str,
    variant: str,
    h: dict,
    shape: dict,
    blocks_eq: float = 1.0,
    miss: float = 0.0,
    dclk: float | None = None,
    sass_path=None,
    nblocks: int = 1,
    grid: int | None = None,
) -> dict:
    ins = kernel_sass(sass_path or cal / "sass" / f"tma_{name}__{variant}.sass")
    kt = math.ceil(shape["K"] / h["block_K"])
    lp = main_loop(ins)
    warps = h["threads"] // 32
    if lp:
        body = ins[lp[0] : lp[1] + 1]
        n_mma = sum(x.base in ("HMMA", "IMMA", "QMMA", "OMMA") for x in body)
        per_kt = h["block_M"] * h["block_N"] * h["block_K"] / (16 * 8 * 16) / warps
        u = max(1, round(n_mma / per_kt)) if n_mma else 1
        trips = max(1, kt // u)
    else:
        trips = 1
    waits = [
        i for i, x in enumerate(ins) if x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op
    ]
    tmas = [i for i, x in enumerate(ins) if x.base == "UTMALDG"]
    if tmas and len(waits) >= 2:
        w0 = max(i for i in waits if i < tmas[0])
        w1 = min((i for i in waits if i > tmas[0]), default=len(ins))
        tma_per_kt = max(1, sum(1 for i in tmas if w0 < i < w1))
    else:
        tma_per_kt = 1
    # out-of-range rows and columns of a box are zero-filled by the TMA unit without moving data
    tile = (min(h["block_M"], shape["M"]) + min(h["block_N"], shape["N"])) * h["block_K"] * 2
    sm = WSSM(
        ins,
        h["threads"] + 128,
        nblocks,
        trips,
        h["threads"],
        h["num_stages"],
        tile,
        tma_per_kt,
        kt=kt,
        active_sms=blocks_eq,
        miss_frac=miss,
        dram_clk_per_512B=dclk if dclk is not None else dram_clk(0.0, 0.0, blocks_eq),
        ctile_bytes=min(h["block_M"], shape["M"]) * min(h["block_N"], shape["N"]) * 2,
        resident=1,
        grid_blocks=grid,
    )
    t = time.time()
    r = sm.run(limit=3_000_000 * nblocks)
    r["us"] = r["cycles"] / F
    r["wall_s"] = round(time.time() - t, 1)
    r["warps_done"] = sum(w.done for w in sm.warps)
    r["warps"] = len(sm.warps)
    return r


def kernel_us(cal: Path, name, h, shape, variant="full", warm=True, sass_path=None):
    tiles = math.ceil(shape["M"] / h["block_M"]) * math.ceil(shape["N"] / h["block_N"])
    kt = math.ceil(shape["K"] / h["block_K"])
    sp = dram_split_kslice(h, shape, SMS)
    from .kernel import FLOW

    if FLOW:
        rb = secs = 0.0
        for wv in sp["waves"]:
            r_, s_ = wave_dram(h, shape, wv, warm)
            rb += r_
            secs += s_
        # TMA moves only rows and columns inside the matrix (out-of-range rows of a box are
        # zero-filled, no traffic)
        cb = (
            tiles
            * kt
            * (min(h["block_M"], shape["M"]) + min(h["block_N"], shape["N"]))
            * h["block_K"]
            * 2
        )
        eq = min(float(SMS), float(tiles))
        return (
            C["launch_stream_us"]
            + ws_sim(
                cal,
                name,
                variant,
                h,
                shape,
                eq,
                min(1.0, rb / cb) if cb else 0.0,
                dram_clk(rb, secs, eq),
                sass_path,
                nblocks=math.ceil(tiles / SMS),
                grid=tiles,
            )["us"]
        )
    left, t, w = tiles, C["launch_stream_us"], 0
    while left > 0:
        n = min(left, SMS)
        cb = (
            n
            * kt
            * (min(h["block_M"], shape["M"]) + min(h["block_N"], shape["N"]))
            * h["block_K"]
            * 2
        )
        wv = (
            sp["waves"][w] if w < len(sp["waves"]) else {"A": 0.0, "B": 0.0, "cols": 1, "blocks": n}
        )
        rb, secs = wave_dram(h, shape, wv, warm)
        t += ws_sim(
            cal,
            name,
            variant,
            h,
            shape,
            n,
            min(1.0, rb / cb) if cb else 0.0,
            dram_clk(rb, secs, n),
            sass_path,
        )["us"]
        left -= n
        w += 1
    return t


def ws_level(cal: Path, names: list[str] | None = None) -> list[dict]:
    """One row per TMA pick: mma-only one block (simulated, measured), the whole block simulated,
    and the whole kernel
    at its grid (simulated, measured)."""
    m = json.loads((cal / "ablation" / "mechanisms.json").read_text())
    rows = []
    for name in names or list(m):
        r = m[name]
        h = r["pick"]
        s = r["shape"]
        kt = r["KT"]
        nol = ws_sim(cal, name, "noload", h, s)
        full1 = ws_sim(cal, name, "full", h, s)
        meas_c = r["compute_per_ktile_us"] * kt + (r.get("block_setup_us") or C["block_setup_us"])
        k = kernel_us(cal, name, h, s)
        rows.append(
            {
                "pick": name,
                "noload": (nol["us"], meas_c),
                "warps_done": [nol["warps_done"], nol["warps"]],
                "full_block_us": full1["us"],
                "full": (k, r["full_event_us"]),
            }
        )
    return rows
