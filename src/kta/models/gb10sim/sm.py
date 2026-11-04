"""An instruction-level, event-driven simulator of one GB10 SM running TileLang GEMM machine code.

It runs the real SASS (nvdisasm -hex) of the kernel, warp by warp, and computes nothing: it tracks
when each
instruction may issue and which resources it holds. Per sub-partition (4 per SM) one instruction
issues per cycle,
from a warp whose stall count has elapsed and whose wait mask is clear (the compiler's control
bits). Resources:
  tensor pipe per sub-partition (HMMA.16816 holds it 16 cycles), ldmatrix (per warp issue interval;
  4 cycles of the SM's
  shared-memory port per x4), cp.async (LDGSTS: one warp instruction per 32 cycles through the SM's
  fill path, then the
  L2 or DRAM round trip), copy groups (LDGDEPBAR / DEPBAR.LE), the block barrier (BAR.SYNC), the six
  dependency counters
  (write and read barriers). Control flow: the k-loop's back edge is taken trip-count times; EXIT
  ends a warp.
Constants: calibration/spark-6951/gb10.json["sim"], each measured by a probe except two marked as
assumptions.

KTA: the resources, their scope and the opcode -> operation-class table (which resources an
instruction gates on and holds, its latency and read-counter rule) come from the machine description
(machine.yaml, read through machine.Machine); the service rule of each class stays here (`handler`).
"""

from __future__ import annotations

import heapq
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lanes import Lanes, footprint, width_of
from .machine import Machine, OpClass, default_machine
from .sass import Ins, main_loop, parse

# the default machine's parameters, for callers that read them directly
P = default_machine().P
SMS = default_machine().sms


@dataclass
class Warp:
    wid: int
    block: int
    smsp: int
    pc: int = 0
    ready: float = 0
    done: bool = False
    sb: list[Any] = field(default_factory=lambda: [0] * 6)
    loops: int = 0
    gid: int = 0  # the copy group being filled (cp.async between two commits)
    pending: dict = field(
        default_factory=dict
    )  # group -> copies not yet written into shared memory
    committed: dict = field(default_factory=dict)  # group -> dependency counter its commit set
    at_bar: bool = False
    ldsm_next: int = 0
    ncopy: int = 0
    # KTA: attributes upstream attaches after construction (SM.__init__ and WSSM.__init__ set
    # every one of them on every warp before use), declared so they type-check
    tid0: int = 0
    regs_tid: set[Any] = field(default_factory=set)
    preds: dict[str, Any] = field(default_factory=dict)
    st_done: float = 0.0
    vals: dict[str, Any] = field(default_factory=dict)
    nldg: int = 0
    role: str | None = None
    k_prod: int = -1
    n_full: int = 0
    n_empty_arrive: int = 0
    # KTA (PERFMODEL) trace: the reason the current instruction was last found blocked; per
    # dependency counter its releases (time, producer issue time, path stages); barrier releases
    why: Any = None
    sbsrc: dict[int, list[Any]] = field(default_factory=dict)
    bar_src: list[Any] = field(default_factory=list)
    lanes: Any = None


class SM:
    def __init__(
        self,
        ins: list[Ins],
        threads: int,
        blocks: int,
        trips: int,
        mem: str = "l2",
        stats: bool = True,
        active_sms: float = 1,
        copy_lines: list | None = None,
        miss_frac: float = 0.0,
        dram_clk_per_512B: float = 0.0,
        ctile_bytes: int = 0,
        ldsm_ways: tuple = (1, 1),
        ldg_lines: list | None = None,
        red_lines: int = 8,
        loop_trips: dict | None = None,
        resident: int | None = None,
        grid_blocks: int | None = None,
        grid_dim: tuple | None = None,
        const_bank: dict[tuple[int, int], int] | None = None,
        machine: Machine | None = None,
        trace: bool = False,
        block_dim: tuple | None = None,
        l1_bytes: int | None = None,
        block_ids: list[int] | None = None,
        addr_log: list | None = None,
        miss_by_pc: dict[int, float] | None = None,
        stack_bytes: int | None = None,
    ):
        self.ins = ins
        # KTA (PERFMODEL): with `trace`, every issue and the binding reason of every wait is kept
        # for critical-path attribution (attribution.py); results are the same either way
        self.trace = trace
        self.hist: dict[int, list[tuple]] = {}
        self.block_pred: dict[int, tuple[int, float]] = {}
        # defaults used (assumption ledger): opcodes whose read counter takes the assumed release,
        # opcodes with no measured cost (issue slot only), branches decided by a fallback rule
        self.defaults: dict[str, dict[str, int]] = {
            "rbar_default": {},
            "issue_only": {},
            "branch_fallback": {},
            "front_end": {},
        }
        self.const_bank = const_bank or {}
        m = machine or default_machine()
        self.m = m
        P = self.P = m.P
        n = self.nsp = m.smsp_per_sm
        # each instruction's operation class (machine.yaml `classes`)
        self.icls: list[OpClass] = [m.classify(x.base, x.op) for x in ins]
        self.gate_smsp = m.gating("smsp")
        self.trips = trips
        self.mem = mem
        # this SM's share of the chip's L2 return bandwidth (or its own return path, if fewer SMs
        # are pulling)
        self.ret_clk = max(
            P["l2_return_clk_per_512B"], 512 * active_sms / P["l2_aggregate_B_per_clk_cpasync"]
        )
        lp = main_loop(ins)
        self.loop_end = lp[1] if lp else -1  # fully unrolled code has no back edge
        self.warps = []
        wpb = threads // 32
        for b in range(blocks):
            for w in range(wpb):
                self.warps.append(Warp(wid=len(self.warps), block=b, smsp=w % n))
        self.wpb = wpb
        self.blocks = blocks
        # block replacement: `blocks` run on this SM, at most `resident` at a time; when one exits
        # the block scheduler starts the next on the freed slot (the order blocks are launched)
        self.resident = resident or blocks
        # round 7: stack bytes per thread, from the spill offsets unless the caller knows them
        self.stack_bytes = stack_bytes if stack_bytes is not None else _stack_bytes(ins)
        self.next_block = self.resident
        self.live = {}
        self.finished = 0
        # the chip's shares follow the grid: this SM is the busiest one; with every SM progressing
        # alike, when it has finished j blocks the chip holds sum over SMs of min(resident, blocks
        # left there) blocks in flight, and this SM's share of the L2, DRAM, store and TMA limits is
        # its own blocks in flight over that
        self.grid_blocks = grid_blocks
        self.eq0 = active_sms
        self.dram_clk0 = dram_clk_per_512B
        INF = float("inf")
        for w in self.warps:
            if w.block >= self.resident:
                w.ready = INF
        for b in range(min(blocks, self.resident)):
            self.live[b] = wpb
        # per sub-partition resources (tensor, FMA, ALU, XU pipes; the memory-instruction slot):
        # the time each instance is free again; per SM resources: one time each (machine.yaml)
        self.free: dict[str, list[float]] = {r: [0] * n for r in m.resources("smsp")}
        self.t: dict[str, float] = {r: 0 for r in m.resources("sm") + m.resources("tpc")}
        # round 6 (probe mma_pipe): the tensor pipe's stream slows with the operand sets it cycles
        # through: the main loop's A register groups (the kernel's, without a loop)
        self.mma_extra = 0.0
        lp0 = main_loop(ins)
        mm = [
            x.args.split(",")
            for x in (ins[lp0[0] : lp0[1] + 1] if lp0 else ins)
            if x.base in ("HMMA", "IMMA", "QMMA", "OMMA")
        ]
        if not mm:
            mm = [x.args.split(",") for x in ins if x.base in ("HMMA", "IMMA", "QMMA", "OMMA")]
        sa = len({a[1].strip() for a in mm})
        if sa > 1:
            self.mma_extra = P["mma_operand_clk_per_set"] * sa
        # round 6 (probes branch, launch, smem_ldsts, namedbar): a branch costs its redirect only in
        # code that issues at the full rate: FFMA loops pay it, a warp's fetch has run ahead in
        # loops of memory instructions or barriers. Over the loop body, or the 8 instructions before
        # a forward branch
        self.redirect = [False] * len(ins)
        at = {y.addr: i for i, y in enumerate(ins)}
        for i, y in enumerate(ins):
            if y.base == "BRA":
                t0 = at.get(y.target, i + 1) if y.target is not None else i + 1
                body = range(t0, i) if t0 <= i else range(max(0, i - 8), i)
                self.redirect[i] = all(self.icls[j].handler in ("pipe", "other") for j in body)
        # round 6 (probe branch): instruction lines a sub-partition holds, in LRU order
        self.l0: list[OrderedDict] = [OrderedDict() for _ in range(n)]
        self.busy_key = {r: d.get("busy", r) for r, d in m.desc["resources"].items()}
        # lines each copy of a k-tile touches, in program order (A rows then B rows); cycles them
        # per warp
        self.copy_default = copy_lines is None
        self.copy_lines = copy_lines or [4]
        # the store path is shared with the partner SM of the pair when the chip runs more than one
        # block
        tpc = m.sms_per_tpc
        self.stg_share = tpc if active_sms >= tpc else 1
        self.active_sms = active_sms
        # copies that miss the L2 (this wave's share, from the launch-order L2 simulation) pay the
        # DRAM round trip and this SM's share of the DRAM rate; misses are spread evenly over the
        # copies
        self.miss_frac = miss_frac
        self.miss_acc = 0.0
        # KTA (PERFMODEL round 5, l2.py): misses per load instruction, from the kernel's addresses
        self.miss_by_pc = miss_by_pc
        self.miss_acc_pc: dict[int, float] = {}
        self.dram_clk = dram_clk_per_512B
        # TMA stores (UTMASTG): the block's C tile split evenly over the store boxes in the code;
        # the TMA unit reads the shared-memory array as fast as the pair's store port drains it
        # (probe tmastore.cu: 33 B/clk per pair, chip 482 B/clk, 457 cycles fixed; waiting for the
        # read alone costs the same as waiting for the writes)
        n_st = sum(1 for y in ins if y.base == "UTMASTG")
        self.tma_st_bytes = ctile_bytes / n_st if n_st else 0.0
        for w in self.warps:
            w.tid0 = (w.wid % wpb) * 32
            w.regs_tid = set()
            w.preds = {}
            w.st_done = 0.0
            w.vals = {
                "RZ": 0,
                "URZ": 0,
            }  # warp-uniform integer values known so far (lane 0 stands for the warp)
        self.use_values = True
        # block indices: this SM's j-th block is the grid's block j x 48 (blocks go round the SMs in
        # launch order)
        self.grid_dim = grid_dim
        # KTA (PERFMODEL round 5, l2.py): an address walk replays chosen grid blocks and logs each
        # global load's lanes
        self.block_ids = block_ids
        self.addr_log = addr_log
        # block -> warps that have exited (they count as arrived at a block-wide barrier)
        self.exited = {}
        # shared-memory bank conflicts of ldmatrix: (A reads, LDSM.M88; B reads, LDSM.MT88), from
        # the tile layout (probe smem_bw.cu: 4 cycles per x4 times the conflict ways, 1/2/4/8-way at
        # row strides 48/32/64/128+ bytes)
        self.ldsm_ways = ldsm_ways
        # the SM's store/atomic queue (probe stq.cu: about 16 warp instructions are taken before
        # issue waits for the drain): finish times of the queued STG/RED instructions
        self.stq = []
        # plain global loads (LDG): per instruction in program order, (128-byte lines the warp
        # touches, bytes it uses, whether L1 already holds them); cycled per warp. An L1 hit costs 2
        # cycles per line of the L1 data path and its latency; a miss takes the copy path (address
        # stage, L2 or DRAM, the SM's return port) into registers
        self.ldg_default = ldg_lines is None
        self.ldg_lines = ldg_lines or [(4, 512, False)]
        # KTA (PERFMODEL round 3): without a driver-supplied list, an LDG's lines and sectors come
        # from its width and its 32 lanes' addresses (lanes.py), and a line another load of this SM
        # brought into the L1 (LRU over `l1_bytes`, kernel-parameter bases only) is a hit
        self.lane_track = addr_log is not None or (
            P["ldg_lane_addresses"]
            and self.ldg_default
            and any(x.base in ("LDG", "LDS", "STS") for x in ins)
        )
        bd = tuple([*(block_dim or (threads,)), 1, 1][:3])
        self.l1: OrderedDict = OrderedDict()
        self.l1_lines = (l1_bytes if l1_bytes is not None else m.C["l1_unified_bytes"]) // 128
        # global atomics: lines one warp instruction touches (8 for a C tile added from mma
        # accumulators: 8 rows x 32 B)
        self.red_lines = red_lines
        # trip counts of other loops, by the index of their back edge (the main loop keeps `trips`)
        self.loop_trips = loop_trips or {}
        for w in self.warps:
            w.nldg = 0
            if self.lane_track:
                w.lanes = Lanes(w.tid0, bd, self.const_bank)  # type: ignore[arg-type]
        self.events = []  # (time, seq, kind, warp, arg)
        self.seq = 0
        self.bar_arrived = {}
        self.last_issue: list[Warp | None] = [None] * n
        self.count: dict[str, Any] = {"issued": 0, "HMMA": 0, "LDSM": 0, "LDGSTS": 0, "BAR": 0}
        self.count.update(ldg_sectors=0, ldg_miss_sectors=0.0)
        self.assumed_latency: dict[str, int] = {}
        self.busy: dict[str, float] = {"tensor": 0, "smem": 0, "fill": 0}

    def _ldg_lanes(self, w: Warp, x: Ins, addr: Any) -> tuple[int, float, int]:
        """KTA (PERFMODEL round 3): (lines, bytes that miss the L1, lines that hit) of one LDG from
        its width and its lanes' addresses; an address the lane tracking does not know is taken as
        32 contiguous lanes (a front-end default)."""
        width = width_of(x.op)
        if self.addr_log is not None:
            self.addr_log.append((w.wid, w.loops, w.pc, width, addr))
        if addr is None:
            self._dflt("front_end", "ldg address unknown (32 contiguous lanes)")
            n_sect = width
            keys: dict = {}
            lines = max(1, 32 * width // 128)
        else:
            keys, n_sect = footprint(addr, width)
            lines = len(keys)
        self.count["ldg_sectors"] += n_sect
        hits = 0
        if addr is not None and all(sym[0] == "c" for sym, _ in addr[0]):
            # round 5 (probe row_reduce): the L1 fills sectors; a line hits only for those it holds
            for k, mask in keys.items():
                have = self.l1.get(k, 0)
                if have and mask & ~have == 0:
                    hits += 1
                    self.l1.move_to_end(k)
                else:
                    self.l1[k] = have | mask
                    self.l1.move_to_end(k)
                    if len(self.l1) > self.l1_lines:
                        self.l1.popitem(last=False)
        miss = (lines - hits) / lines
        self.count["ldg_miss_sectors"] += n_sect * miss
        return lines, n_sect * 32 * miss, hits

    def _chip_bound(self) -> bool:
        """Round 6 (probes row_reduce, st256): this SM's reads run at its share of the chip's L2;
        stores at the chip's limit wait for `l2_chip_read_share` of the reads' time."""
        return self.P["l2_chip_read_share"] > 0 and (
            self.ret_clk > self.P["l2_return_clk_per_512B"]
        )

    def _l2_miss(self, pc: int) -> bool:
        """This load misses the L2: misses spread evenly over the load's executions."""
        if self.miss_by_pc is None:
            self.miss_acc += self.miss_frac
            if self.miss_acc >= 1.0:
                self.miss_acc -= 1.0
                return True
            return False
        a = self.miss_acc_pc.get(pc, 0.0) + self.miss_by_pc.get(pc, 0.0)
        self.miss_acc_pc[pc] = a - 1.0 if a >= 1.0 else a
        return a >= 1.0

    def ev(self, t, kind, w, arg=None):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, w, arg))

    def _dflt(self, kind: str, key: str) -> None:
        d = self.defaults[kind]
        d[key] = d.get(key, 0) + 1

    def _rel(self, w: Warp, b: int, t_rel: float, now: float, stages: list | None) -> None:
        """Trace: counter b of w is released at t_rel by an instruction issued at `now`."""
        if stages is not None:
            w.sbsrc.setdefault(b, []).append((t_rel, now, stages))

    def _return_stage(self, arrive: float, ret: float) -> tuple:
        """Trace: the wait at the L2 return port, named by the term that sets its rate."""
        if self.ret_clk > self.P["l2_return_clk_per_512B"]:
            return (
                arrive,
                ret,
                "l2_return",
                "queue (chip L2 share)",
                "sim.l2_aggregate_B_per_clk_cpasync",
            )
        return (arrive, ret, "l2_return", "queue (SM port rate)", "sim.l2_return_clk_per_512B")

    def why_blocked(self, w: Warp, now: float) -> tuple:
        """Trace: why an eligible-looking warp cannot issue now (the first failing condition in
        the order eligible() checks them); no side effects."""
        if w.at_bar:
            return ("bar",)
        x = self.ins[w.pc]
        for b in range(6):
            if (x.wait >> b) & 1 and w.sb[b] > 0:
                return ("sb", b)
        if x.base == "DEPBAR":
            m = re.search(r"SB(\d),\s*0x([0-9a-f]+)", x.args)
            if m and w.sb[int(m.group(1))] > int(m.group(2), 16):
                return ("sb", int(m.group(1)))
        cls = self.icls[w.pc]
        for g in cls.gates:
            if g == "store_queue":
                if sum(1 for q in self.stq if q > now) >= self.P["st_queue_instr"]:
                    return ("res", g, cls.name)
            elif g == "either":
                if min(self.free[r][w.smsp] for r in cls.either) > now:
                    return ("res", "/".join(cls.either), cls.name)
            elif self.free[g][w.smsp] > now:
                return ("res", g, cls.name)
        return ("unknown",)

    def _assumed(self, x) -> None:
        """KTA (WP26): count instructions whose result latency is the assumed
        default_var_latency_clk (no probe for that opcode), reported by run() as `assumed_latency`.
        """
        self.assumed_latency[x.op] = self.assumed_latency.get(x.op, 0) + 1

    def on_event(self, kind, w, key, now):
        """Events of a subclass's own kinds (the base simulator has none beyond the four above)."""

    # --- warp-uniform predicates from the thread index (lane 0 of the warp stands for the warp)
    # ----------------------
    def _pred(self, w, tok) -> Any:
        neg = tok.startswith("!")
        t = tok[1:] if neg else tok
        if t in ("PT", "UPT"):
            v = True
        elif t in w.preds:
            v = w.preds[t]
        else:
            return None
        return (not v) if neg else v

    @staticmethod
    def _imm(b) -> Any:
        return (
            int(b, 16)
            if b.lstrip("-").startswith("0x")
            else int(b)
            if re.match(r"^-?\d+$", b)
            else None
        )

    # --- warp-uniform integer values: loop counters, immediates, thread index
    # ----------------------------------------
    def _val(self, w, tok) -> Any:
        t = tok.strip()
        neg = t.startswith("-")
        t = t.lstrip("-").replace(".reuse", "")
        if t.startswith("!"):
            return None
        v = self._imm(t) if re.match(r"^-?(0x[0-9a-f]+|\d+)$", t) else w.vals.get(t)
        if v is None:
            return None
        return (-v) & 0xFFFFFFFF if neg else v & 0xFFFFFFFF

    def _eval_values(self, w, x):
        """Compute the destination of simple integer instructions from known operands; forget it
        otherwise.
        """
        args = [a.strip() for a in x.args.split(",")]
        if not args or not re.match(r"^U?R\d+$", args[0]):
            return
        d = args[0]
        op = x.op
        v = None
        try:
            if x.base == "S2R" or x.base == "S2UR":
                if "SR_TID.X" in x.args:
                    v = w.tid0
                elif "SR_LANEID" in x.args:
                    v = 0
                elif "SR_CTAID" in x.args and self.grid_dim:
                    gx, gy, gz = [*self.grid_dim, 1, 1][:3]
                    lin = (
                        self.block_ids[w.block]
                        if self.block_ids
                        else (w.block * self.m.sms) % (gx * gy * gz)
                    )
                    v = {"X": lin % gx, "Y": (lin // gx) % gy, "Z": lin // (gx * gy)}[
                        x.args.split("SR_CTAID.")[1][0]
                    ]
            elif x.base in ("MOV", "UMOV", "MOV32I") or op.startswith("IMAD.MOV"):
                v = self._val(w, args[-1])
                if op.startswith("IMAD.MOV"):
                    v = self._val(w, args[3]) if len(args) > 3 else None
            elif x.base in ("IADD3", "UIADD3"):
                terms = [a for a in args[1:] if not re.match(r"^!?U?P(T|\d)$", a)]
                vs = [self._val(w, a) for a in terms]
                if terms and None not in vs:
                    v = sum(vs) & 0xFFFFFFFF
            elif x.base in ("VIADD", "UIADD"):
                a, b = self._val(w, args[1]), self._val(w, args[2])
                v = None if None in (a, b) else (a + b)
            elif x.base in ("IMAD", "UIMAD") and ".WIDE" not in op and ".HI" not in op:
                a, b, c = (self._val(w, t) for t in args[1:4])
                if None not in (a, b, c):
                    v = a * b + c
            elif x.base in ("LEA", "ULEA") and ".HI" not in op:
                a, b, sh = self._val(w, args[1]), self._val(w, args[2]), self._val(w, args[3])
                if None not in (a, b, sh):
                    v = (a << sh) + b
            elif x.base in ("SHF", "USHF") and len(args) >= 4:
                lo, sh, hi = self._val(w, args[1]), self._val(w, args[2]), self._val(w, args[3])
                if ".L." in op and ".HI" not in op and None not in (lo, sh):
                    v = lo << sh
                elif ".R." in op and ".HI" in op and None not in (hi, sh):
                    v = hi >> sh
                elif (
                    ".R." in op
                    and None not in (lo, sh)
                    and (hi in (0, None) and args[3] in ("RZ", "URZ"))
                ):
                    v = lo >> sh
            elif x.base in ("LOP3", "ULOP3") and len(args) >= 5:
                a, b, c, lut = (
                    self._val(w, args[1]),
                    self._val(w, args[2]),
                    self._val(w, args[3]),
                    self._val(w, args[4]),
                )
                if None not in (a, b, c, lut):
                    v = 0
                    for bit in range(32):
                        i = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
                        v |= ((lut >> i) & 1) << bit
            elif x.base in ("SEL", "USEL") and len(args) >= 4:
                pv = self._pred(w, args[3])
                a, b = self._val(w, args[1]), self._val(w, args[2])
                if pv is not None:
                    v = a if pv else b
            elif x.base in ("R2UR",):
                v = self._val(w, args[1])
            elif x.base in ("LDC", "LDCU") and self.const_bank and ".64" not in op:
                # KTA: kernel arguments the caller declares (27 §2.1 dynamic facts), by
                # (bank, offset); without them a run-time loop count stays unknown
                m = re.search(r"c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]", x.args)
                if m:
                    v = self.const_bank.get((int(m.group(1), 16), int(m.group(2), 16)))
        except (ValueError, TypeError, IndexError):
            v = None
        if v is None:
            w.vals.pop(d, None)
            if ".64" in op or ".WIDE" in op:
                w.vals.pop(f"{d.rstrip('0123456789')}{int(re.sub(r'\D', '', d)) + 1}", None)
        else:
            w.vals[d] = v & 0xFFFFFFFF

    def _eval_pred_ops(self, w, x):
        if self.use_values and x.base not in ("ISETP", "UISETP"):
            self._eval_values(w, x)
        args = [a.strip() for a in x.args.split(",")]
        if x.base == "S2R" and "SR_TID.X" in x.args:
            w.regs_tid.add(args[0])
        elif x.base == "FCHK" and self.P["fchk_fast_path"]:
            # KTA (PERFMODEL round 3): FCHK sets its predicate for operands the fast division
            # cannot take (denormals, infinities, zero divisors); the replay takes them as normal,
            # so the guarded slow path is not run (a visible default, softmax / RMSNorm divisions)
            w.preds[args[0]] = False
            self._dflt("branch_fallback", "FCHK: operands normal (fast division)")
        elif x.base == "ELECT":
            w.preds[args[0]] = True  # lane 0 is the elected lane
        elif x.base in ("ISETP", "UISETP"):
            m = re.match(r"U?ISETP\.(\w+)(\.U32)?\.(AND|OR)", x.op)
            if not m or len(args) < 4:
                return
            dst = args[0]
            a, b = self._val(w, args[2]), self._val(w, args[3])
            if args[2] in w.regs_tid and a is None:
                a = w.tid0
            if a is None or b is None:
                w.preds.pop(dst, None)
                return
            if not m.group(2):  # signed compare
                a = a - (1 << 32) if a >= 1 << 31 else a
                b = b - (1 << 32) if b >= 1 << 31 else b
            v = {
                "GT": a > b,
                "GE": a >= b,
                "LT": a < b,
                "LE": a <= b,
                "NE": a != b,
                "EQ": a == b,
            }.get(m.group(1))
            if v is None:
                w.preds.pop(dst, None)
                return
            if len(args) >= 5:
                o = self._pred(w, args[4])
                if o is None:
                    w.preds.pop(dst, None)
                    return
                v = (v and o) if m.group(3) == "AND" else (v or o)
            w.preds[dst] = v
        elif x.base == "ULOP3" and args and args[0].startswith("UP"):
            # ULOP3.LUT UPd, URZ, URa, URb, URc, lut, UPq: UPd = (lut(a, b, c) != 0)
            if len(args) >= 6:
                a, b, c, lut = (self._val(w, t) for t in args[2:6])
                if None in (a, b, c, lut):
                    w.preds.pop(args[0], None)
                else:
                    r = 0
                    for bit in range(32):
                        i = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
                        r |= ((lut >> i) & 1) << bit
                    w.preds[args[0]] = r != 0
        elif x.base == "LOP3" and args and args[0].startswith("P"):
            # LOP3.LUT Pd, Rd, Ra, Rb, Rc, lut, Pq: Pd = (lut(Ra, Rb, Rc) != 0)
            if (
                len(args) >= 6
                and args[2] in w.regs_tid
                and self._imm(args[3]) is not None
                and args[4] == "RZ"
                and self._imm(args[5]) is not None
            ):
                a, b, c, lut = w.tid0, self._imm(args[3]), 0, self._imm(args[5])
                r = 0
                for bit in range(32):
                    i = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
                    r |= ((lut >> i) & 1) << bit
                w.preds[args[0]] = r != 0
            else:
                w.preds.pop(args[0], None)
        elif x.base == "PLOP3":
            # PLOP3.LUT Pd, Pq, Pa, Pb, Pc, lut, lut2: Pd = lut(Pa, Pb, Pc)
            if len(args) >= 6:
                va, vb, vc = (self._pred(w, t) for t in args[2:5])
                lut = self._imm(args[5])
                if None in (va, vb, vc) or lut is None:
                    w.preds.pop(args[0], None)
                else:
                    w.preds[args[0]] = bool(
                        (lut >> ((int(va) << 2) | (int(vb) << 1) | int(vc))) & 1
                    )
        elif (
            args
            and re.match(r"^U?P\d$", args[0])
            and x.base not in ("BRA", "EXIT")
            and not x.op.startswith("SYNCS")
        ):
            w.preds.pop(args[0], None)  # any other write to a predicate: no longer known

    def branch_known(self, w, x):
        """A branch whose predicate is known from computed values: True (taken) / False; None when
        unknown.
        """
        if not self.use_values or x.base != "BRA" or x.target is None or x.op.startswith("BRA.DIV"):
            return None
        if x.pred and x.pred != "@!PT":
            return self._pred(w, x.pred[1:])
        if x.op.startswith("BRA.U"):
            first = x.args.split(",")[0].strip()
            if re.match(r"^!?UP(T|\d)$", first):
                return self._pred(w, first)
        return None

    def follow(self, w, x, taken):
        if taken and x.target < x.addr:
            c = w.__dict__.setdefault("vb", {})
            c[w.pc] = c.get(w.pc, 0) + 1
            if c[w.pc] > 1_000_000:
                self.use_values = False  # a guard against a wrong evaluation looping forever
        w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target) if taken else w.pc + 1

    def eligible(self, w: Warp, now: float) -> bool:
        if w.done or w.at_bar or w.ready > now:
            return False
        x = self.ins[w.pc]
        for b in range(6):
            if (x.wait >> b) & 1 and w.sb[b] > 0:
                return False
        if x.base == "DEPBAR":
            m = re.search(r"SB(\d),\s*0x([0-9a-f]+)", x.args)
            if m and w.sb[int(m.group(1))] > int(m.group(2), 16):
                return False
        cls = self.icls[w.pc]
        for g in cls.gates:
            if g == "store_queue":
                while self.stq and self.stq[0] <= now:
                    heapq.heappop(self.stq)
                if len(self.stq) >= self.P["st_queue_instr"]:
                    return False
            elif g == "either":
                if min(self.free[r][w.smsp] for r in cls.either) > now:
                    return False
            elif self.free[g][w.smsp] > now:
                return False
        return True

    def _beside(self, cls: OpClass, s: int, now: float) -> None:
        """Round 6 (probe mma_issue; upstream interfere.cu for copies): an instruction issued while
        the sub-partition's resource is held delays it (operand reads beside the pipe's)."""
        for res, ref in cls.beside:
            if self.free[res][s] > now:
                self.free[res][s] += self.m.param(ref)

    def _hold(self, cls: OpClass, s: int, now: float, w: Warp) -> None:
        """A pipe class: hold its resource(s) for their parameter's cycles; `either` takes the
        first free of its resources (the earliest, ties to the first listed)."""
        for res, ref in cls.hold:
            c = self.m.param(ref)
            if res == "either":
                res = min(cls.either, key=lambda r: self.free[r][s])
            if cls.name == "mma":
                c += self.mma_extra
            self.free[res][s] = now + c
            k = self.busy_key[res]
            self.busy[k] = self.busy.get(k, 0) + c
        self._beside(cls, s, now)

    def issue(self, w: Warp, now: float):
        x = self.ins[w.pc]
        cls = self.icls[w.pc]
        P = self.P
        free, t = self.free, self.t
        s = w.smsp
        self.count["issued"] += 1
        executes = x.pred != "@!PT"
        pv = None
        if x.pred and x.pred != "@!PT":
            pv = self._pred(w, x.pred[1:])
            # a predicate known from the thread index decides; unknown ones keep the old rule (the
            # instruction runs)
            if pv is False:
                executes = False
        pend = None
        addr0 = None
        if w.lanes is not None:
            # round 6: read before the step, a load may overwrite its own address register
            if cls.handler in ("ldg", "smem_op", "cp_async", "stg"):
                addr0 = w.lanes.address(x.args, w.vals)
            # the guard as the lane tracking sees it: none, known false, known true or unknown
            runs = True if not x.pred else (False if not executes else pv)
            pend = w.lanes.step(x.base, x.op, x.args.split(","), w.vals, runs)
        self._eval_pred_ops(w, x)
        if pend is not None:
            pend(w.vals)
        lat_done = None
        lsu_done = now
        h = cls.handler
        tr = self.trace
        trw: list | None = None  # path stages to the write-counter release (trace)
        trr: list | None = None  # path stages to the read-counter release (trace)
        if executes:
            if cls.count:
                self.count[cls.count] = self.count.get(cls.count, 0) + 1
            if h == "pipe":
                # every mma.sync shape and type holds the tensor pipe 16 cycles (probe
                # mma_types.cu); FMA, ALU and XU pipes per probes alu.cu, alumix.cu, sfu.cu, cvt.cu
                self._hold(cls, s, now, w)
                if cls.latency is not None:
                    lat_done = now + self.m.param(cls.latency)
                    if x.wbar is not None and self.m.status(cls.latency) == "assumed":
                        self._assumed(x)
                    if tr:
                        res = cls.hold[0][0] if cls.hold else "issue"
                        if res == "either":
                            res = "/".join(cls.either)
                        trw = [(now, lat_done, res, f"{cls.name} result latency", cls.latency)]
            elif h == "ldsm":
                # the sub-partition's memory-instruction slot, then the SM's shared-memory array
                # (512 B per x4)
                free["mio"][s] = now + P["mio_ldsm_clk"]
                ways = self.ldsm_ways[1] if "MT88" in x.op else self.ldsm_ways[0]
                cost = P["smem_array_clk_per_512B"] * ways
                start = max(now, t["smem_array"])
                t["smem_array"] = start + cost
                self.busy["smem"] += cost
                lat_done = start + P["ldsm_latency_clk"] + cost - P["smem_array_clk_per_512B"]
                if tr:
                    trw = [
                        (now, start, "smem_array", "queue (ldsm)", None),
                        (start, lat_done, "smem_array", "ldsm latency", "sim.ldsm_latency_clk"),
                    ]
            elif h == "stsm":
                # stmatrix: like ldmatrix in the other direction (probe tmastore.cu: one warp every
                # 10.8 cycles, 4 or more warps 4 cycles per x4 on the shared-memory array); the
                # source registers are free once the array takes them
                free["mio"][s] = now + P["mio_stsm_clk"]
                start = max(now, t["smem_array"])
                t["smem_array"] = start + P["smem_array_clk_per_512B"]
                self.busy["smem"] += P["smem_array_clk_per_512B"]
                lsu_done = start + P["smem_array_clk_per_512B"]
                if tr:
                    trr = [
                        (now, start, "smem_array", "queue (stsm)", None),
                        (
                            start,
                            lsu_done,
                            "smem_array",
                            "stsm write",
                            "sim.smem_array_clk_per_512B",
                        ),
                    ]
            elif h == "smem_op":
                # KTA (PERFMODEL round 1, probes smem_ldsts.cu, shfl.cu, batch r1): LDS, STS and
                # SHFL hold the sub-partition's memory slot, then take the SM's shared-memory
                # array for max(2 cycles, bytes at 128 B/clk); a load's result follows after a
                # latency that grows with its width, a shuffle's after its own latency; a store
                # releases its source registers after the array, in proportion to its bytes
                # Round 6 (probe lds_narrow): the array is held by the lanes' extent, one 128-byte
                # line per cycle, whatever the element size; a store's slot 2 cycles per line
                nb = 0 if x.base == "SHFL" else _smem_bytes(x.op)
                span = nb
                if addr0 is not None and nb and P["smem_by_extent"]:
                    span = len(footprint(addr0, width_of(x.op))[0]) * 128
                slot = P["mio_smem_op_clk"]
                if x.base == "STS":
                    slot = max(slot, P["sts_mio_clk_per_line"] * span / 128)
                free["mio"][s] = now + slot
                cost = max(P["smem_op_min_clk"], span * P["smem_array_clk_per_512B"] / 512)
                start = max(now, t["smem_array"])
                t["smem_array"] = start + cost
                self.busy["smem"] += cost
                if x.base == "SHFL":
                    lat_done = start + P["shfl_latency_clk"]
                    lat_ref = "sim.shfl_latency_clk"
                else:
                    lat_done = (
                        start
                        + P["lds_latency_clk"]
                        + nb / P["lds_latency_B_per_clk"]
                        + max(0, span - nb) / P["lds_extent_B_per_clk"]
                    )
                    lat_ref = "sim.lds_latency_clk"
                lsu_done = start + cost + P["sts_read_clk_per_128B"] * nb / 128
                if tr:
                    q = (now, start, "smem_array", f"queue ({x.base.lower()})", None)
                    trw = [q, (start, lat_done, "smem_array", f"{x.base} latency", lat_ref)]
                    trr = [
                        q,
                        (start, lsu_done, "smem_array", "store read", "sim.sts_read_clk_per_128B"),
                    ]
            elif h == "tma_store":
                # a TMA store box: read from the shared array and sent through the pair's store port
                # (shared with plain stores) at the larger of the pair's rate and this SM's share of
                # the chip's TMA-store limit
                nb = self.tma_st_bytes
                per_B = max(
                    self.stg_share / P["tma_st_tpc_B_per_clk"],
                    self.active_sms / P["tma_st_chip_B_per_clk"],
                )
                start = max(now, t["store_port"])
                t["store_port"] = start + nb * per_B
                self.busy["stg"] = self.busy.get("stg", 0) + nb * per_B
                rd = nb / 512 * P["smem_array_clk_per_512B"]
                t["smem_array"] = max(t["smem_array"], start) + rd
                self.busy["smem"] += rd
                w.st_done = max(w.st_done, t["store_port"])
            elif h == "cp_async":
                # issue holds the sub-partition's memory slot; the data comes back through the SM's
                # L2 return path after the round trip and is written into the shared-memory array
                # (the same array ldmatrix reads)
                free["mio"][s] = now + P["mio_ldgsts_clk"]
                self._beside(cls, s, now)
                if self.addr_log is not None:
                    self.addr_log.append((w.wid, w.loops, w.pc, width_of(x.op), addr0))
                # the load unit's address stage: one 128-byte line per cycle; the copy reads its
                # address registers there
                lines = self.copy_lines[w.ncopy % len(self.copy_lines)]
                w.ncopy += 1
                if self.copy_default:
                    self._dflt("front_end", "copy_lines (4 lines per copy)")
                t_lsu = max(now, t["lsu"])
                t["lsu"] = t_lsu + lines * P["lsu_clk_per_line"]
                self.busy["lsu"] = self.busy.get("lsu", 0) + lines
                lsu_done = t["lsu"]
                if self._l2_miss(w.pc):
                    t_d = max(lsu_done + P["ldgsts_dram_latency_clk"], t["dram_queue"])
                    t["dram_queue"] = t_d + self.dram_clk
                    self.busy["dram"] = self.busy.get("dram", 0) + self.dram_clk
                    arrive = t_d
                    lat = (
                        (
                            lsu_done,
                            lsu_done + P["ldgsts_dram_latency_clk"],
                            "dram",
                            "DRAM latency",
                            "sim.ldgsts_dram_latency_clk",
                        ),
                        (
                            lsu_done + P["ldgsts_dram_latency_clk"],
                            t_d,
                            "dram_queue",
                            "queue (DRAM share)",
                            None,
                        ),
                    )
                else:
                    arrive = lsu_done + P["ldgsts_l2_latency_clk"]
                    lat = ((lsu_done, arrive, "l2", "L2 latency", "sim.ldgsts_l2_latency_clk"),)
                ret = max(arrive, t["l2_return"])
                t["l2_return"] = ret + self.ret_clk
                if self._chip_bound():
                    t["l2_chip"] = max(t["l2_chip"], now) + P["l2_chip_read_share"] * self.ret_clk
                self.busy["fill"] += self.ret_clk
                w.pending[w.gid] = w.pending.get(w.gid, 0) + 1
                if tr:
                    trr = [
                        (now, t_lsu, "lsu", "queue", None),
                        (t_lsu, lsu_done, "lsu", "address stage", "sim.lsu_clk_per_line"),
                    ]
                    path = [*trr, *lat]
                    path.append(self._return_stage(arrive, ret))
                    self.ev(ret, "write", w, (w.gid, path))
                else:
                    self.ev(ret, "write", w, w.gid)
            elif h == "ldg":
                if w.lanes is not None:
                    lines, nbytes, hit_lines = self._ldg_lanes(w, x, addr0)
                else:
                    lines, nbytes, hit = self.ldg_lines[w.nldg % len(self.ldg_lines)]
                    if self.ldg_default:
                        self._dflt("front_end", "ldg_lines (4 lines, 512 B, L1 miss)")
                    hit_lines = lines if hit else 0
                w.nldg += 1
                t_lsu = max(now, t["lsu"])
                t["lsu"] = t_lsu + lines * P["lsu_clk_per_line"]
                self.busy["lsu"] = self.busy.get("lsu", 0) + lines
                lsu_done = t["lsu"]
                if tr:
                    trr = [
                        (now, t_lsu, "lsu", "queue", None),
                        (t_lsu, lsu_done, "lsu", "address stage", "sim.lsu_clk_per_line"),
                    ]
                lat_done = lsu_done
                if hit_lines:
                    t1 = max(lsu_done, t["l1"])
                    t["l1"] = t1 + hit_lines * P["l1_clk_per_line"]
                    lat_done = t["l1"] + P["l1_hit_latency_clk"]
                    if trr is not None:
                        trw = [
                            *trr,
                            (lsu_done, t1, "l1", "queue", None),
                            (t1, t["l1"], "l1", "hit path", "sim.l1_clk_per_line"),
                            (t["l1"], lat_done, "l1", "hit latency", "sim.l1_hit_latency_clk"),
                        ]
                if hit_lines < lines:
                    if self._l2_miss(w.pc):
                        # KTA (PERFMODEL round 1, probe dram_rotate.cu, batch r1): a streaming LDG
                        # miss waits ldg_dram_latency_clk, not the copy path's chase latency
                        t_d = max(lsu_done + P["ldg_dram_latency_clk"], t["dram_queue"])
                        t["dram_queue"] = t_d + self.dram_clk * nbytes / 512
                        self.busy["dram"] = self.busy.get("dram", 0) + self.dram_clk * nbytes / 512
                        arrive = t_d
                    else:
                        arrive = lsu_done + P["ldg_l2_latency_clk"]
                    ret = max(arrive, t["l2_return"])
                    t["l2_return"] = ret + self.ret_clk * nbytes / 512
                    if self._chip_bound():
                        t["l2_chip"] = (
                            max(t["l2_chip"], now)
                            + P["l2_chip_read_share"] * self.ret_clk * nbytes / 512
                        )
                    self.busy["fill"] += self.ret_clk * nbytes / 512
                    if t["l2_return"] >= lat_done:
                        lat_done = t["l2_return"]
                        if trr is not None:
                            trw = [
                                *trr,
                                (
                                    lsu_done,
                                    arrive,
                                    "l2/dram",
                                    "load latency (incl. DRAM queue)",
                                    "sim.ldg_l2_latency_clk",
                                ),
                                self._return_stage(arrive, ret),
                                (
                                    ret,
                                    lat_done,
                                    "l2_return",
                                    "transfer",
                                    "sim.l2_return_clk_per_512B",
                                ),
                            ]
            elif h == "local":
                # local memory (register spills, one slot per warp): 4 cycles per line through L1,
                # then the L1 latency
                m_l1 = self.m.C["l1_unified_bytes"]
                # round 7 (probe spill): at the chip's scale the stack path takes
                # local_chip_B_per_clk, beyond the L1's capacity per SM local_chip_over_B_per_clk
                over = self.stack_bytes * self.wpb * 32 * self.resident > m_l1
                rate = P["local_chip_over_B_per_clk" if over else "local_chip_B_per_clk"]
                t1 = max(now, t["l1"])
                t["l1"] = t1 + max(P["local_clk_per_line"], 128 / (rate / self.active_sms))
                lat_done = t["l1"] + P["l1_hit_latency_clk"]
                lsu_done = t["l1"]
                if tr:
                    trr = [
                        (now, t1, "l1", "queue", None),
                        (t1, lsu_done, "l1", "local path", "sim.local_clk_per_line"),
                    ]
                    trw = [*trr, (lsu_done, lat_done, "l1", "latency", "sim.l1_hit_latency_clk")]
            elif h == "red":
                # global atomics without a return value: bytes-limited, per SM and for the chip
                # (probe ldg_atomic.cu) the pair's store port per line (shared with plain stores),
                # the SM's byte rate, the chip's rate for the pattern
                nb = 32 * (8 if "x2" in x.op else 16 if "x4" in x.op else 4)
                chip = (
                    P["red_chip_B_per_clk_frag"] if self.red_lines >= 8 else P["red_chip_B_per_clk"]
                )
                per = max(
                    nb / P["red_sm_B_per_clk"],
                    self.red_lines * P["red_pair_clk_per_line"] * self.stg_share,
                    nb * self.active_sms / chip,
                )
                start = max(now, t["store_port"])
                t["store_port"] = start + per
                self.busy["red"] = self.busy.get("red", 0) + per
                heapq.heappush(self.stq, t["store_port"])
                free["mio"][s] = max(
                    free["mio"][s], now + self.red_lines * P["red_smsp_clk_per_line"]
                )
                lsu_done = start
                self.end = max(getattr(self, "end", 0), t["store_port"], t["stg_chip"])
                if tr:
                    trr = [(now, start, "store_port", "queue (atomics)", None)]
            elif h == "commit":
                # commit: the group's dependency counter stays set until its last copy has landed in
                # shared memory
                g = w.gid
                w.gid += 1
                if x.wbar is not None:
                    w.sb[x.wbar] += 1
                    if w.pending.get(g, 0) == 0:
                        self.ev(now + 1, "sb", w, x.wbar)
                        if tr:
                            self._rel(
                                w, x.wbar, now + 1, now, [(now, now + 1, "issue", "commit", None)]
                            )
                    else:
                        w.committed[g] = x.wbar
            elif h == "barrier":
                # BAR.SYNC id[, threads]: a named barrier waits for the threads it names; without a
                # count, for the whole block (warps that already exited count as arrived).
                # KTA (PERFMODEL round 1, probe namedbar.cu, batch r1): BAR.ARV counts as an
                # arrival and does not wait (12 cycles flat whatever the waiters do)
                arrive_only = x.op.startswith("BAR.ARV")
                w.at_bar = not arrive_only
                ba = [a.strip() for a in x.args.split(",")]
                bid = self._imm(ba[0]) if ba and ba[0] else 0
                need = (
                    (self._imm(ba[1]) // 32)
                    if len(ba) > 1 and self._imm(ba[1])
                    else self.wpb - self.exited.get(w.block, 0)
                )
                key = (w.block, bid)
                self.bar_arrived[key] = [*self.bar_arrived.get(key, []), w]
                if len(self.bar_arrived[key]) >= need:
                    rel = now + P["bar_base_clk"] + P["bar_per_warp_clk"] * need
                    for o in self.bar_arrived.pop(key):
                        if not o.at_bar:
                            continue  # arrived with BAR.ARV, never waited
                        self.ev(rel, "bar", o)
                        if tr:
                            o.bar_src.append((rel, now, w.wid))
            elif h == "stg":
                # bytes per warp instruction and rates by store width (round 5: 32 B per lane,
                # probe st256; 8 B per lane uses the 4-byte rates)
                w_ = "32B" if ".256" in x.op else ("16B" if ".128" in x.op else "4B")
                nbytes = {"32B": 1024, "16B": 512}.get(w_, 256 if ".64" in x.op else 128)
                # the larger of the pair's shared store port and this SM's share of the chip's store
                # limit
                pair = P[f"stg_tpc_clk_per_instr_{w_}"] * self.stg_share
                # round 7 (probes l2mix, store_lines): the chip's limit counts the 32-byte sectors
                # the lanes touch, one rate for every width, behind a store buffer
                by_sector = P["stg_by_sectors"]
                if addr0 is not None and by_sector:
                    nbytes = footprint(addr0, width_of(x.op))[1] * 32
                chip = nbytes / (
                    P["stg_chip_B_per_clk" if by_sector else f"stg_chip_B_per_clk_{w_}"]
                    / self.active_sms
                )
                start = max(now, t["store_port"])
                if P["l2_chip_read_share"] > 0 and chip >= pair:
                    # at the chip's limit reads and stores share one L2 path, the stores behind
                    # the reads by at most the overlap the buffer gives
                    start = max(start, t["l2_chip"] - P["stg_read_overlap_clk"])
                done = max(t["stg_chip"], start) + chip
                start = max(start, done - P["stg_burst_bytes"] / (nbytes / chip * self.active_sms))
                t["stg_chip"] = done
                per = pair
                t["store_port"] = start + per
                self.busy["stg"] = self.busy.get("stg", 0) + per
                heapq.heappush(self.stq, t["store_port"])
                if x.rbar is not None:
                    lsu_done = start
                if tr:
                    trr = [(now, start, "store_port", "queue (stores)", None)]
            elif h == "exit":
                # the block is finished only when its stores have left the SM
                w.done = True
                self.end = max(getattr(self, "end", 0), t["store_port"], t["stg_chip"])
                self.exited[w.block] = self.exited.get(w.block, 0) + 1
                self.block_exit(w.block, now)
            elif x.wbar is not None:
                ref = cls.latency or self.m.default.latency or ""
                lat_done = now + self.m.param(ref)
                self._assumed(x)
                if tr:
                    trw = [(now, lat_done, "issue", f"default result latency ({x.base})", ref)]
            elif h == "other":
                self._dflt("issue_only", x.base)
            if x.wbar is not None and lat_done is not None and h != "commit":
                w.sb[x.wbar] += 1
                self.ev(lat_done, "sb", w, x.wbar)
                if tr:
                    self._rel(
                        w, x.wbar, lat_done, now, trw or [(now, lat_done, "issue", "?", None)]
                    )
            if x.rbar is not None:
                if cls.rbar == "lsu":
                    rel = lsu_done
                    stages = trr or [(now, rel, "issue", "operand read", None)]
                elif cls.rbar == "tma_store_flush":
                    # the bulk-store group's counter clears when every box has been read out and
                    # sent, plus the fixed delay
                    rel = max(now + P["rbar_release_clk"], w.st_done + P["tma_st_fixed_clk"])
                    stages = [
                        (
                            now,
                            rel,
                            "store_port",
                            "TMA store drain + fixed delay",
                            "sim.tma_st_fixed_clk",
                        )
                    ]
                else:
                    rel = now + P["rbar_release_clk"]
                    self._dflt("rbar_default", x.base)
                    stages = [(now, rel, "issue", "read-counter release", "sim.rbar_release_clk")]
                w.sb[x.rbar] += 1
                self.ev(rel, "sb", w, x.rbar)
                if tr:
                    self._rel(w, x.rbar, rel, now, stages)
        w.ready = now + max(1, x.stall)
        if x.base == "BRA" and self.redirect[w.pc]:
            # round 6 (probe branch): fetch resumes this long after a branch, whichever way it went
            w.ready = max(w.ready, now + P["branch_redirect_clk"])
        # next pc: the k-loop back edge is taken trips-1 times; other branches fall through unless
        # unconditional
        if x.op.startswith("BRA.DIV"):
            # taken only if the warp's lanes have diverged; warps here stay converged
            w.pc += 1
        elif self.branch_known(w, x) is not None:
            self.follow(w, x, self.branch_known(w, x))
        elif x.base == "BRA" and x.target is not None and w.pc in self.loop_trips:
            self._dflt("branch_fallback", "driver trip count")
            c = w.__dict__.setdefault("lc", {})
            c[w.pc] = c.get(w.pc, 0) + 1
            if c[w.pc] < self.loop_trips[w.pc]:
                w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target)
            else:
                c[w.pc] = 0
                w.pc += 1
        elif x.base == "BRA" and x.target is not None:
            if w.pc == self.loop_end:
                self._dflt("branch_fallback", "main-loop trip count")
                w.loops += 1
                w.pc = (
                    next(i for i, y in enumerate(self.ins) if y.addr == x.target)
                    if w.loops < self.trips
                    else w.pc + 1
                )
            elif not x.pred and (x.target > x.addr or not x.op.startswith("BRA.U")):
                # unconditional jumps are taken, forward or backward (conditional BRA.U on a uniform
                # predicate is not)
                if x.op.startswith("BRA.U") or x.args.lstrip().startswith(("!UP", "UP")):
                    self._dflt("branch_fallback", "unknown uniform predicate: taken")
                w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target)
            else:
                self._dflt("branch_fallback", "unknown predicate: fall through")
                w.pc += 1
        else:
            w.pc += 1
        if w.pc >= len(self.ins) and not w.done:
            w.done = True
            self.exited[w.block] = self.exited.get(w.block, 0) + 1
            self.block_exit(w.block, now)
        elif not w.done and P["fetch_l0_lines"]:
            # round 6 (probe branch): straight code at 9 cycles per 8 instructions past the
            # sub-partition's instruction buffer; a warp's wait, the others issue meanwhile
            line = self.ins[w.pc].addr >> 7
            l0 = self.l0[s]
            if line in l0:
                l0.move_to_end(line)
            else:
                l0[line] = True
                w.ready += P["fetch_line_clk"]
                if len(l0) > P["fetch_l0_lines"]:
                    l0.popitem(last=False)

    def share(self, finished: int) -> float:
        """SM-equivalents sharing the chip's limits once this SM has finished `finished` blocks."""
        if not self.grid_blocks:
            return self.eq0
        sms = self.m.sms
        q, r = divmod(self.grid_blocks, sms)
        R = self.resident
        counts = [q + 1] * r + [q] * (sms - r)
        mine = max(counts)
        chip = sum(max(0, min(R, c - finished)) for c in counts)
        own = max(1, min(R, mine - finished))
        return max(1.0, chip / own)

    def set_share(self, eq: float):
        self.active_sms = eq
        P = self.P
        self.ret_clk = max(
            P["l2_return_clk_per_512B"], 512 * eq / P["l2_aggregate_B_per_clk_cpasync"]
        )
        if self.eq0:
            self.dram_clk = self.dram_clk0 * eq / self.eq0
        self.on_share(eq)

    def on_share(self, eq: float):
        """Hook for a subclass whose own rates follow the chip share (WSSM's TMA engine)."""

    def block_exit(self, b, now):
        """A warp of block b exited; when the whole block has, start the next block on its slot."""
        if b not in self.live:
            return
        self.live[b] -= 1
        if self.live[b] == 0:
            del self.live[b]
            self.finished += 1
            if self.grid_blocks:
                self.set_share(self.share(self.finished))
            if self.next_block < self.blocks:
                nb = self.next_block
                self.next_block += 1
                self.block_pred[nb] = (b, now)
                self.live[nb] = self.wpb
                for o in self.warps:
                    if o.block == nb:
                        o.ready = now + 1
                self._live_dirty = True

    def run(self, limit: int = 50_000_000) -> dict:
        # only warps of started, unfinished blocks are scanned; the lists are rebuilt when a block
        # starts or a warp exits
        now = 0
        INF = float("inf")
        self._live_dirty = True
        live: list[Warp] = []
        by_smsp: list[list[Warp]] = [[] for _ in range(self.nsp)]
        while now < limit:
            if self._live_dirty:
                live = [w for w in self.warps if not w.done and w.ready < INF]
                by_smsp = [[w for w in live if w.smsp == s] for s in range(self.nsp)]
                self._live_dirty = False
            while self.events and self.events[0][0] <= now:
                _, _, kind, w, arg = heapq.heappop(self.events)
                if kind == "sb":
                    w.sb[arg] -= 1
                elif kind == "write":
                    # the returned 512 B are written into the shared-memory array (shared with
                    # ldmatrix), then counted as landed
                    c = self.P["smem_array_clk_per_512B"]
                    t_w = max(now, self.t["smem_array"])
                    self.t["smem_array"] = t_w + c
                    self.busy["smem"] += c
                    if isinstance(arg, tuple):  # trace: the copy's path so far
                        arg[1].append((now, t_w, "smem_array", "queue (copy write)", None))
                        arg[1].append(
                            (
                                t_w,
                                t_w + c,
                                "smem_array",
                                "copy write",
                                "sim.smem_array_clk_per_512B",
                            )
                        )
                    self.ev(t_w + c, "landed", w, arg)
                elif kind == "landed":
                    g, path = arg if isinstance(arg, tuple) else (arg, None)
                    w.pending[g] -= 1
                    if w.pending[g] == 0 and g in w.committed:
                        b = w.committed.pop(g)
                        w.sb[b] -= 1
                        if path is not None:
                            self._rel(w, b, now, path[0][0], path)
                elif kind == "bar":
                    w.at_bar = False
                    w.ready = max(w.ready, now)
                else:
                    self.on_event(kind, w, arg, now)
            if not live and self.next_block >= self.blocks and all(w.done for w in self.warps):
                break
            if any(w.done for w in live):
                self._live_dirty = True
            issued = False
            for s in range(self.nsp):
                cand = [w for w in by_smsp[s] if self.eligible(w, now)]
                if self.trace:
                    ok = {id(q) for q in cand}
                    for q in by_smsp[s]:
                        if not q.done and q.ready <= now and id(q) not in ok:
                            q.why = self.why_blocked(q, now)
                if not cand:
                    continue
                last = self.last_issue[s]
                pick = getattr(self, "pick", None)
                if pick is not None:
                    w = pick(cand, last)
                else:
                    w = (
                        last if (last in cand) else min(cand, key=lambda q: (q.loops, q.wid))
                    )  # greedy, then oldest
                yb = self.ins[w.pc].yld
                if self.trace:
                    self.hist.setdefault(w.wid, []).append(
                        (now, w.pc, w.ready, w.why, self.icls[w.pc].name)
                    )
                    w.why = None
                self.issue(w, now)
                self.last_issue[s] = w
                issued = True
                last_yield = getattr(self, "last_yield", None)
                if last_yield is not None:
                    last_yield[w.wid] = yb
            # jump to the next cycle where something can change
            nxt = [w.ready for w in live if not w.done and not w.at_bar and now < w.ready < INF]
            for r in self.gate_smsp:
                nxt += [t for t in self.free[r] if t > now]
            if self.events:
                nxt.append(self.events[0][0])
            nxt += [t for t in self.stq if t > now][:1]
            now = now + 1 if issued else (min(nxt) if nxt else now + 1)
        self.cycles = max(now, getattr(self, "end", 0))
        return {
            "cycles": self.cycles,
            **self.count,
            "busy": self.busy,
            "assumed_latency": self.assumed_latency,
            "defaults": self.defaults,
        }


def _stack_bytes(ins: list[Ins]) -> int:
    """Stack per thread: the highest offset a local load or store names."""
    tops = [
        int(o, 16) + 4
        for x in ins
        if x.base in ("LDL", "STL")
        for o in re.findall(r"\[\w+\+0x([0-9a-f]+)\]", x.args)
    ]
    return max(tops, default=0)


def _smem_bytes(op: str) -> int:
    """Bytes one warp instruction of a plain shared load or store moves (32 lanes; 4 B per lane
    unless the opcode names .64 or .128)."""
    parts = op.split(".")
    return 512 if "128" in parts else 256 if "64" in parts else 128


def kernel_sass(path: str | Path, kernel: str = "matmul_kernel") -> list[Ins]:
    txt = open(path).read()
    ins = parse(txt, kernel=kernel)
    return ins
