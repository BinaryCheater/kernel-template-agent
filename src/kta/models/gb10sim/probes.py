"""Replays of upstream microbenchmarks (agentic/tilelang/microbench) from their own SASS.

Each case builds the SM simulator on the probe's machine code with the probe's launch (threads,
kernel arguments) and reads the quantity the probe reports, against the value the probe printed on
the GB10 (the `*.txt` files of a calibration directory, a copy of upstream calibration/spark-6951).
Upstream replays two of these itself (`schedtest.py`, `cvttest.py`); the arithmetic, mix and SFU
replays follow the same method. Every quantity is time-like (cycles), so a positive error means the
model is slow.

A case is excluded from error statistics, with its reason, when the measurement cannot be compared
with a replay of its code: the probe's loop was folded by the compiler (no loop in the SASS), the
probe's own note withdraws the value, or the measured rate exceeds the issue limit of the machine.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sass import Ins, main_loop, parse
from .sm import SM

ALU_KINDS = [
    "FFMA",
    "IMAD",
    "IADD3",
    "LOP3",
    "SHF",
    "PRMT",
    "HFMA2",
    "F2F.F16.F32+HADD2.F32",
    "HADD2.F32(h2f)",
    "F2FP.PACK_AB",
    "I2F",
    "F2I",
]
MIX_KINDS = ["FFMA", "IMAD", "LOP3", "HFMA2", "PRMT", "HADD2.F32+FADD"]
SFU_OPS = ["ex2.f32", "rcp.f32", "rsqrt.f32", "sin.f32", "lg2.f32", "tanh.f32", "ffma.f32"]
# cvt.cu: modes and the labels of its result table (cvt.txt)
CVT_MODES = [
    "scaffold only (mov)",
    "I2F.F16 from s32",
    "I2F.F16 from u32",
    "int to fp32",
    "fp32 to fp16",
    "MUFU.EX2 ruler",
    "I2F.F16 + MUFU.EX2 alternating",
]
N_IT = 512  # alu.cu, alumix.cu: loop passes; 8 chains per thread
SFU_ITERS = 4096  # sfu.cu: the `iters` kernel argument
SFU_ITERS_ARG = (0, 0x380)  # its constant-bank slot in the SASS (LDCU UR5, c[0x0][0x380])
CVT_ITERS, CVT_E, CVT_WARPS_PER_SMSP = 4096, 8, 8
ISSUE_LIMIT_PER_SM = 4.0  # one warp instruction per cycle per sub-partition


class ClockMarked(SM):
    """The SM simulator that also records when warp 0 reads the clock (`SR_CLOCKLO`), as the
    probes time themselves with clock64 around their loop."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.marks: list[float] = []

    def issue(self, w, now):
        if w.wid == 0 and "SR_CLOCKLO" in self.ins[w.pc].args:
            self.marks.append(now)
        super().issue(w, now)


def marked_cycles(sm: ClockMarked, r: dict) -> float:
    if len(sm.marks) < 2:
        return float(r["cycles"])
    return sm.marks[-1] - sm.marks[0]


@dataclass
class Case:
    """One probe measurement and how to replay it: `build` makes the simulator, `read` turns the
    finished simulator and its run() result into the probe's quantity."""

    id: str
    probe: str
    level: int
    unit: str
    measured: float
    build: Callable[..., SM]  # keyword arguments (machine, trace) go to the simulator
    read: Callable[[SM, dict], float]
    excluded: str | None = None
    limit: int = 50_000_000


def _sass(path: Path) -> str:
    return path.read_text()


def _loop_folded(ins: list[Ins]) -> bool:
    return main_loop(ins) is None


def alu_cases(cal: Path) -> list[Case]:
    """alu.cu: 8 independent chains of one kind per thread, 4-32 warps on one SM; it reports loop
    warp-steps per clock per SM; here as cycles between its two clock reads."""
    txt = _sass(cal / "sass" / "alu" / "alu.2.sm_121a.sass")
    out = []
    for line in (cal / "alu.txt").read_text().splitlines():
        m = re.match(r"RESULT (\S+) warps=(\d+) loop_warp_steps_per_clk_per_SM=([\d.]+)", line)
        if not m:
            continue
        kind, warps, rate = m.group(1), int(m.group(2)), float(m.group(3))
        k = ALU_KINDS.index(kind)
        ins = parse(txt, kernel=f"_Z1kILi{k}EEviPyPi")
        excluded = None
        if _loop_folded(ins):
            excluded = "loop folded by the compiler (no loop in the SASS)"
        elif rate > ISSUE_LIMIT_PER_SM:
            excluded = f"measured rate {rate} exceeds the issue limit {ISSUE_LIMIT_PER_SM}/clk/SM"
        out.append(
            Case(
                id=f"alu {kind} w{warps}",
                probe="alu.cu",
                level=2,
                unit="cycles",
                measured=N_IT * 8 * warps / rate,
                build=lambda ins=ins, warps=warps, **kw: ClockMarked(ins, 32 * warps, 1, 1, **kw),
                read=marked_cycles,  # type: ignore[arg-type]
                excluded=excluded,
            )
        )
    return out


def alumix_cases(cal: Path) -> list[Case]:
    """alumix.cu: 4 chains of kind A and 4 of kind B per thread, 32 warps; cycles between clocks."""
    txt = _sass(cal / "sass" / "alu" / "alumix.2.sm_121a.sass")
    out = []
    for line in (cal / "alumix.txt").read_text().splitlines():
        m = re.match(r"RESULT mix (.+) \+ (.+) cycles=(\d+)", line)
        if not m:
            continue
        a, b = MIX_KINDS.index(m.group(1)), MIX_KINDS.index(m.group(2))
        ins = parse(txt, kernel=f"_Z1kILi{a}ELi{b}EEviPyPi")
        out.append(
            Case(
                id=f"alumix {m.group(1)}+{m.group(2)}",
                probe="alumix.cu",
                level=3,
                unit="cycles",
                measured=float(m.group(3)),
                build=lambda ins=ins, **kw: ClockMarked(ins, 1024, 1, 1, **kw),
                read=marked_cycles,  # type: ignore[arg-type]
                excluded=None if not _loop_folded(ins) else "loop folded by the compiler",
            )
        )
    return out


def sfu_cases(cal: Path) -> list[Case]:
    """sfu.cu: dependent-chain latency (1 warp, 1 chain, cycles per op) and throughput (W warps, 8
    chains, reported as lane results per clock per SM; here as cycles between clocks)."""
    txt = _sass(cal / "sass" / "sfu.sass")
    const = {SFU_ITERS_ARG: SFU_ITERS}
    out = []
    for line in (cal / "sfu.txt").read_text().splitlines():
        m = re.match(r"RESULT (\S+)\s+latency_cycles=([\d.]+)((?: \| w\d+ [\d.]+)+)", line)
        if not m:
            continue
        name = m.group(1)
        op = SFU_OPS.index(name)
        excluded = (
            "withdrawn upstream: the rcp loop was folded away (gb10 design note §9.4)"
            if name == "rcp.f32"
            else None
        )
        ins1 = parse(txt, kernel=f"_Z1kILi{op}ELi1EEviPfPx")
        out.append(
            Case(
                id=f"sfu {name} latency",
                probe="sfu.cu",
                level=1,
                unit="cycles/op",
                measured=float(m.group(2)),
                build=lambda ins=ins1, **kw: ClockMarked(ins, 32, 1, 1, const_bank=const, **kw),
                read=lambda sm, r: marked_cycles(sm, r) / SFU_ITERS,  # type: ignore[arg-type]
                excluded=excluded,
            )
        )
        ins8 = parse(txt, kernel=f"_Z1kILi{op}ELi8EEviPfPx")
        for w, rate in re.findall(r"w(\d+) ([\d.]+)", m.group(3)):
            warps = int(w)
            out.append(
                Case(
                    id=f"sfu {name} w{warps}",
                    probe="sfu.cu",
                    level=2,
                    unit="cycles",
                    measured=32 * warps * 8 * SFU_ITERS / float(rate),
                    build=lambda ins=ins8, warps=warps, **kw: ClockMarked(
                        ins, 32 * warps, 1, 1, const_bank=const, **kw
                    ),
                    read=marked_cycles,  # type: ignore[arg-type]
                    excluded=excluded,
                )
            )
    return out


def cvt_cases(cal: Path) -> list[Case]:
    """cvt.cu (upstream cvttest.py): 32 warps, 8 chains; cycles per element per sub-partition of
    the whole kernel (the probe is timed with events around the kernel)."""
    sass = cal / "sass" / "cvt.sass"
    if not sass.exists():
        return []
    txt = _sass(sass)
    measured = {}
    for line in (cal / "cvt.txt").read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 4 and parts[0] in CVT_MODES:
            measured[parts[0]] = float(parts[2])
    out = []
    per = CVT_ITERS * CVT_E * CVT_WARPS_PER_SMSP
    for mode, label in enumerate(CVT_MODES):
        ins = parse(txt, kernel=f"_Z5probeILi{mode}EEvPKjPj")
        out.append(
            Case(
                id=f"cvt {label}",
                probe="cvt.cu",
                level=3 if mode == 6 else 2,
                unit="cycles/element/smsp",
                measured=measured[label],
                build=lambda ins=ins, **kw: SM(ins, 1024, 1, 1, **kw),
                read=lambda sm, r: r["cycles"] / per,
                limit=50_000_000,
            )
        )
    return out


class SchedProbe(SM):
    """sched.cu (upstream schedtest.py): warp 0 runs a long FFMA stream, warp 4 on the same
    sub-partition runs dependent integer adds; the loops are counted by their address ranges."""

    HOG = (0x210, 0x440)
    ADD = (0x7C0, 0x8E0)

    def __init__(
        self, ins: list[Ins], with_hog: bool, n_hog: int = 6000, n_add: int = 1250, **kw: Any
    ):
        super().__init__(ins, 160, 1, 1, **kw)
        idx = {x.addr: i for i, x in enumerate(ins)}
        self.loops = {
            0: (idx[self.HOG[0]], idx[self.HOG[1]]),
            4: (idx[self.ADD[0]], idx[self.ADD[1]]),
        }
        self.iters = {0: n_hog, 4: n_add}
        self.cnt = {0: 0, 4: 0}
        self.finish: dict[int, float] = {}
        for w in self.warps:
            if w.wid == 0 and with_hog:
                w.pc = self.loops[0][0]
            elif w.wid == 4:
                w.pc = self.loops[4][0]
            else:
                w.done = True
                self._warp_done = True

    def issue(self, w, now):
        x = self.ins[w.pc]
        loop = self.loops.get(w.wid)
        if loop is not None and w.pc == loop[1]:
            self.cnt[w.wid] += 1
            self.count["issued"] += 1
            w.ready = now + max(1, x.stall)
            if self.cnt[w.wid] < self.iters[w.wid]:
                w.pc = loop[0]
            else:
                w.done = True
                self._warp_done = True
                self.finish[w.wid] = now
            return
        super().issue(w, now)


def sched_cases(cal: Path) -> list[Case]:
    ins = parse(_sass(cal / "sass" / "sched.sass"))
    out = []
    for line in (cal / "sched.txt").read_text().splitlines():
        m = re.match(r"RESULT hog=(\d) warp4_cycles=\d+ \(([\d.]+) per add\)", line)
        if not m:
            continue
        hog = m.group(1) == "1"

        def read(sm: SM, r: dict) -> float:
            assert isinstance(sm, SchedProbe)
            end = sm.finish[4] if 4 in sm.finish else float(r["cycles"])
            return end / (sm.iters[4] * 16)

        out.append(
            Case(
                id=f"sched adds {'beside FFMA' if hog else 'alone'}",
                probe="sched.cu",
                level=3 if hog else 1,
                unit="cycles/add",
                measured=float(m.group(2)),
                build=lambda hog=hog, **kw: SchedProbe(ins, hog, **kw),
                read=read,
                limit=3_000_000,
            )
        )
    return out


def all_cases(cal: Path) -> list[Case]:
    return [
        *sched_cases(cal),
        *alu_cases(cal),
        *alumix_cases(cal),
        *sfu_cases(cal),
        *cvt_cases(cal),
    ]


def run_case(case: Case, **kw: Any) -> tuple[float, SM, dict]:
    sm = case.build(**kw)
    r = sm.run(limit=case.limit)
    return case.read(sm, r), sm, r
