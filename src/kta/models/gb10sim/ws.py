"""Warp-specialized TMA kernels on the SM simulator (TileLang's default lowering on sm_121a).

Additions to the cp.async simulator, each a mechanism in the machine code:
  roles     warps start together; comparisons of the thread index (S2R SR_TID.X) are evaluated per
  warp (warp-uniform,
            lane 0), so warps 0-3 take the producer side (warps 1-3 exit) and warps 4+ the consumer
            side; ELECT is true
  TMA       UTMALDG loads a box into shared memory: the SM's TMA engine moves the bytes at its rate
  (one block alone 81
            GB/s; chip 1.58 TB/s shared by blocks in flight), after the L2 or DRAM round trip; the
            bytes also pass the
            shared-memory array
  mbarrier  SYNCS.PHASECHK.TRYWAIT spin loops are blocking waits, matched by order: the consumer's
  n-th wait completes
            when k-tile n's TMA bytes have landed; the producer's k-th wait for a free stage
            completes when every consumer
            warp has released k-tile k - stages (SYNCS.ARRIVE...A1T0). The retry stubs are then
            never taken.
  registers USETMAXREG.TRY_ALLOC waits until the producer group's warps have executed
  USETMAXREG.DEALLOC.
The tile bytes per k-tile are split evenly over the UTMALDG instructions of one k-tile. `ws_sim` and
`kernel_us`: one block alone and the whole kernel at its grid (the step 2c ablation's TMA picks)."""

from __future__ import annotations

import math
from pathlib import Path

from .kernel import dram_clk, wave_dram
from .machine import Machine, default_machine
from .model import dram_split_kslice
from .sass import main_loop
from .sm import SM, kernel_sass


class WSSM(SM):
    def __init__(
        self,
        ins,
        threads_launched,
        blocks,
        trips,
        consumer_threads,
        stages,
        tile_bytes,
        tma_per_ktile,
        kt=None,
        **kw,
    ):
        super().__init__(ins, threads_launched, blocks, trips, **kw)
        self.kt = kt
        self.mma_loop = main_loop(ins)
        self.stages = stages
        self.ctiles = tile_bytes / max(1, tma_per_ktile)
        self.consumer_warps = consumer_threads // 32
        # a TMA box is written into the shared-memory array at the array's rate
        self.smem_B_per_clk = float(self.m.param("smem_bytes_per_clk_per_sm"))
        self.on_share(kw.get("active_sms", 1))
        self.landed = {}  # (block, k) -> time all of k-tile k's TMA bytes are in shared memory
        self.dealloc = {}  # (block) -> producer-group warps that gave registers back
        self.arrived = {}
        self.log = []
        self.pend = {}
        self.land_t = {}
        self.release_t = {}

    def on_share(self, eq):
        # the TMA engine: 81 GB/s for one SM alone, the chip's 1.86 TB/s shared by the
        # SM-equivalents in flight
        C = self.m.C
        clock = C["clock_under_load_GHz"] * 1e9
        rate_alone = C["tma_r_GBs_alone"] * 1e9 / clock
        rate_chip = C["tma_l2_aggregate_GBs"] * 1e9 / clock / max(1.0, eq)
        self.tma_clk_per_B = 1.0 / min(rate_alone, rate_chip)

    def eligible(self, w, now):
        if w.done or w.at_bar or w.ready > now:
            return False
        x = self.ins[w.pc]
        if (
            x.op.startswith("SYNCS.PHASECHK")
            and "TRYWAIT" in x.op
            and w.role is None
            and self.mma_loop is not None
        ):
            w.role = "consumer" if self.mma_loop[0] <= w.pc <= self.mma_loop[1] else "producer"
        if x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op:
            b = w.block
            if w.role == "consumer":
                # k-tile n's full barrier: the producer has arrived (with its expected byte count)
                # and every byte landed (an arrive or the last byte becomes visible to the waiting
                # warp P["mbar_wake_clk"] later, probe mbar.cu)
                k = w.n_full
                arr = self.arrived.get((b, k))
                wake = self.P["mbar_wake_clk"]
                if arr is None or arr + wake > now:
                    return False
                if (self.landed.get((b, k)) or 0.0) + wake > now:
                    return False
            elif w.role == "producer":
                k = w.k_prod + 1
                need = k - self.stages + 1
                if need > 0:
                    rel = self.release_t.get(b, [])
                    # releases are recorded in time order (simulated time only moves forward)
                    if (
                        len(rel) < need * self.consumer_warps
                        or rel[need * self.consumer_warps - 1] + self.P["mbar_wake_clk"] > now
                    ):
                        return False
        if x.base == "USETMAXREG" and "TRY_ALLOC" in x.op and self.dealloc.get(w.block, 0) < 4:
            return False
        return super().eligible(w, now)

    def issue(self, w, now):
        x = self.ins[w.pc]
        executes = x.pred != "@!PT"
        p = None
        if x.pred:
            p = self._pred(w, x.pred[1:])
        # thread-index comparisons, ELECT and predicate logic are evaluated by the base simulator
        # (SM._eval_pred_ops)
        if x.base == "USETMAXREG":
            if "DEALLOC" in x.op:
                self.dealloc[w.block] = self.dealloc.get(w.block, 0) + 1
                self.log.append(("dealloc", w.wid, w.block, now))
                w.role = w.role or "producer"
            else:
                w.role = "consumer"
                w.preds["UP0"] = True
        elif x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op:
            # the role is decided by where the warp first waits: inside the mma loop it consumes,
            # elsewhere it produces
            if w.role is None and self.mma_loop is not None:
                w.role = "consumer" if self.mma_loop[0] <= w.pc <= self.mma_loop[1] else "producer"
            dst = x.args.split(",")[0].strip()
            w.preds[dst] = True
            if w.role == "consumer":
                self.log.append(("full_pass", w.wid, w.n_full, now))
                w.n_full += 1
            elif w.role == "producer":
                w.k_prod += 1
                self.log.append(("empty_pass", w.wid, w.k_prod, now))
        elif x.op.startswith("SYNCS.ARRIVE") and w.role == "consumer":
            # a consumer warp releases its stage (the empty barrier)
            self.release_t.setdefault(w.block, []).append(now)
            self.ev(
                now + self.P["mbar_wake_clk"], "wake", w, None
            )  # the producer may get through then: re-check
            self.log.append(("release", w.wid, w.n_full - 1, now))
        elif x.op.startswith("SYNCS.ARRIVE") and w.role == "producer":
            # the producer arrives on k-tile k's full barrier (with the bytes it expects, or none if
            # loads are removed)
            self.arrived[(w.block, max(0, w.k_prod))] = now
            self.log.append(("arrive", w.wid, w.k_prod, now))
            self.ev(now + self.P["mbar_wake_clk"], "wake", w, None)
        elif x.base == "UTMALDG" and executes:
            k = max(0, w.k_prod)
            key = (w.block, k)
            start = max(now, self.t["tma_engine"])
            self.t["tma_engine"] = start + self.ctiles * self.tma_clk_per_B
            # the box's bytes that miss the L2 (this wave's share) come from DRAM at this SM's share
            # of the DRAM rate
            miss_B = self.ctiles * self.miss_frac
            if miss_B > 0:
                # (TMA's own base latencies, probe dramlat.cu: 1170 cycles from DRAM, 440 from L2,
                # before the transfer)
                t_d = max(start + self.P["tma_dram_latency_clk"], self.t["dram_queue"])
                self.t["dram_queue"] = t_d + miss_B * self.dram_clk / 512.0
                self.busy["dram"] = self.busy.get("dram", 0) + miss_B * self.dram_clk / 512.0
                land = max(
                    start + self.P["tma_l2_latency_clk"], self.t["tma_engine"], self.t["dram_queue"]
                )
            else:
                land = max(start + self.P["tma_l2_latency_clk"], self.t["tma_engine"])
            # the box is written into the shared-memory array when it arrives (an event then, not a
            # reservation now)
            self.pend[key] = self.pend.get(key, 0) + 1
            self.landed[key] = float("inf")
            self.ev(land, "tmawrite", w, key)
            self.log.append(("tma", w.wid, k, now, start, land))
            self.busy["tma"] = self.busy.get("tma", 0) + self.ctiles * self.tma_clk_per_B
        # branches whose predicate is known from computed values (loop counters, thread index)
        if x.base == "BRA" and not (x.pred and p is not None):
            bk = self.branch_known(w, x)
            if bk is not None:
                self._eval_pred_ops(w, x)
                self.follow(w, x, bk)
                w.ready = now + max(1, x.stall)
                return
        # predicated branches and exits whose predicate is known
        if x.base in ("BRA", "EXIT") and x.pred and p is not None:
            if x.base == "EXIT":
                if p:
                    w.done = True
                    self._warp_done = True
                    self.exited[w.block] = self.exited.get(w.block, 0) + 1
                    self.end = max(self.end, self.t["store_port"])
                    self.block_exit(w.block, now)
                    return
                w.pc += 1
                w.ready = now + max(1, x.stall)
                return
            if x.target is not None and self.loop_end != w.pc:
                w.pc = self.at[x.target] if p else w.pc + 1
                w.ready = now + max(1, x.stall)
                return
        # k-loops, either shape the compiler emits, decided by what the branch tests (k-tiles remain
        # for this role):  (a) a conditional back edge: taken while k-tiles remain (b) a conditional
        # forward exit followed by an unconditional jump back: the exit is taken when none remain
        if x.op.startswith("BRA.DIV"):
            w.pc += 1
            w.ready = now + max(1, x.stall)
            return
        if (
            self.kt
            and x.base == "BRA"
            and x.target is not None
            and w.role in ("producer", "consumer")
        ):
            more = (w.k_prod + 1 < self.kt) if w.role == "producer" else (w.n_full < self.kt)
            conditional = bool(x.pred) or x.op.startswith("BRA.U")
            if x.target < x.addr and self._is_work_loop(w.pc, x.target):
                if conditional:
                    w.pc = self.at[x.target] if more else w.pc + 1
                else:
                    w.pc = self.at[x.target]
                w.ready = now + max(1, x.stall)
                return
            j = self._back_jump_after(w.pc) if conditional and x.target > x.addr else None
            if j is not None and x.target > self.ins[j].addr and self.ins[j].target <= x.addr:
                # the exit test may sit a few counter updates before the jump back (4096 and 8192
                # picks)
                w.pc = self.at[x.target] if not more else w.pc + 1
                w.ready = now + max(1, x.stall)
                return
        if (
            x.base == "BRA"
            and x.op.startswith("BRA.U")
            and "UP0" in x.args
            and x.target is not None
            and x.target < x.addr
            and self.loop_end != w.pc
        ):
            # the register-allocation retry loop: TRY_ALLOC succeeded, fall through
            w.pc += 1
            w.ready = now + max(1, x.stall)
            return
        super().issue(w, now)

    def why_blocked(self, w, now):
        """Trace: the warp-specialised waits first (eligible's order), then the base reasons."""
        x = self.ins[w.pc]
        b = w.block
        if x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op and not w.at_bar:
            wake = self.P["mbar_wake_clk"]
            if w.role == "consumer":
                k = w.n_full
                arr = self.arrived.get((b, k))
                if arr is None or arr + wake > now or (self.landed.get((b, k)) or 0.0) + wake > now:
                    return ("mbar_full", b, k)
            elif w.role == "producer":
                need = w.k_prod + 1 - self.stages + 1
                if need > 0:
                    rel = self.release_t.get(b, [])
                    n = need * self.consumer_warps
                    if len(rel) < n or rel[n - 1] + wake > now:
                        return ("mbar_empty", b, n)
        if x.base == "USETMAXREG" and "TRY_ALLOC" in x.op and self.dealloc.get(b, 0) < 4:
            return ("regalloc", b)
        return super().why_blocked(w, now)

    def on_event(self, kind, w, key, now):
        if kind == "tmawrite":
            t_w = max(now, self.t["smem_array"])
            self.t["smem_array"] = t_w + self.ctiles / self.smem_B_per_clk
            self.busy["smem"] += self.ctiles / self.smem_B_per_clk
            self.ev(t_w + self.ctiles / self.smem_B_per_clk, "tmalanded", w, key)
        elif kind == "tmalanded":
            self.pend[key] -= 1
            self.land_t[key] = max(self.land_t.get(key, 0.0), now)
            if self.pend[key] == 0:
                self.landed[key] = self.land_t[key]
                # the waiting consumers see it self.P["mbar_wake_clk"] later: an event then, so the
                # simulation re-checks them
                self.ev(self.land_t[key] + self.P["mbar_wake_clk"], "wake", w, None)

    def _is_work_loop(self, end_pc, target_addr):
        key = (end_pc, target_addr)
        if not hasattr(self, "_wl"):
            self._wl = {}
        if key not in self._wl:
            s_ = self.at[target_addr]
            self._wl[key] = any(
                y.base in ("HMMA", "IMMA", "QMMA", "OMMA", "UTMALDG", "LDGSTS")
                or ("TRYWAIT" in y.op)
                for y in self.ins[s_ : end_pc + 1]
            )
        return self._wl[key]

    def _back_jump_after(self, pc):
        """The unconditional jump back that closes a work loop, if it follows pc with no other
        branch in between.
        """
        if not hasattr(self, "_bj"):
            self._bj = {}
        if pc not in self._bj:
            self._bj[pc] = None
            for j in range(pc + 1, min(pc + 9, len(self.ins))):
                y = self.ins[j]
                if y.base in ("BRA", "EXIT", "BAR") or y.base.startswith("SYNCS"):
                    if (
                        y.base == "BRA"
                        and not y.pred
                        and not y.op.startswith("BRA.U")
                        and y.target is not None
                        and y.target < y.addr
                        and self._is_work_loop(j, y.target)
                    ):
                        self._bj[pc] = j
                    break
        return self._bj[pc]


def ws_sim(
    cal: Path,
    name: str,
    variant: str,
    h: dict,
    shape: dict,
    blocks_eq: float = 1.0,
    miss: float = 0.0,
    dclk: float | None = None,
    nblocks: int = 1,
    grid: int | None = None,
    machine: Machine | None = None,
    sm_hook=None,
    trace: bool = False,
) -> dict:
    m = machine or default_machine()
    ins = kernel_sass(cal / "sass" / f"tma_{name}__{variant}.sass")
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
        dram_clk_per_512B=dclk if dclk is not None else dram_clk(0.0, 0.0, blocks_eq, m),
        ctile_bytes=min(h["block_M"], shape["M"]) * min(h["block_N"], shape["N"]) * 2,
        resident=1,
        grid_blocks=grid,
        machine=m,
        trace=trace,
    )
    r = sm.run(limit=3_000_000 * nblocks)
    r["us"] = r["cycles"] / (m.param(m.desc["clock"]["kernel_GHz"]) * 1e3)
    if sm_hook is not None:
        sm_hook(sm)
    return r


def kernel_us(
    cal: Path,
    name,
    h,
    shape,
    machine: Machine | None = None,
    sm_hook=None,
    trace: bool = False,
):
    m = machine or default_machine()
    C, SMS = m.C, m.sms
    tiles = math.ceil(shape["M"] / h["block_M"]) * math.ceil(shape["N"] / h["block_N"])
    kt = math.ceil(shape["K"] / h["block_K"])
    sp = dram_split_kslice(h, shape, SMS, C["l2_effective_bytes"])
    rb = secs = 0.0
    for wv in sp["waves"]:
        r_, s_ = wave_dram(h, shape, wv, True, m)
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
            "full",
            h,
            shape,
            eq,
            min(1.0, rb / cb) if cb else 0.0,
            dram_clk(rb, secs, eq, m),
            nblocks=math.ceil(tiles / SMS),
            grid=tiles,
            machine=m,
            sm_hook=sm_hook,
            trace=trace,
        )["us"]
    )
