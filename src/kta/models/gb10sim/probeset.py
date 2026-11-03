"""Replays of KTA's own measurement batches (docs/design/perfmodel/LOOP.md products 5 and 6): the
probes under `perfmodel/probes/` and the test-set kernels under `perfmodel/testset/`, each from the
SASS the batch built (`perfmodel/batches/<batch>/build/<unit>/k.sass`) with the launch the batch
ran, against the medians the batch measured (`measured.json`, empty before the run: the replay
is then the prediction written into the batch's entry blocks).

Cycle-valued probes compare cycles directly. Microsecond-valued rows convert with the clock the
batch measured (`clock.chain_GHz`) when there is one, else with the description's kernel clock.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel import copy_lines, dram_clk, occupancy, sass_trips, simulate_kernel
from .machine import Machine, default_machine
from .probes import Case, ClockMarked, marked_cycles
from .sass import Ins, main_loop, parse
from .sm import SM

ITERS = 256  # smem_ldsts, shfl, ex2_ffma, l1hit tput
BAR_ITERS = 128  # namedbar
HOPS = 2048  # l1hit chase
DRAM_BYTES = 48 * 1024 * 16 * 85  # dram_rotate: bytes per launch
L2SHARE_PICK = {
    "block_M": 128,
    "block_N": 256,
    "block_K": 64,
    "num_stages": 2,
    "threads": 256,
}


@dataclass
class Batch:
    name: str
    dir: Path
    measured: dict[str, float]  # "<probe> <point>" -> median
    clock_GHz: float | None

    @property
    def build(self) -> Path:
        return self.dir / "build"


def load(batch_dir: Path) -> Batch:
    mp = batch_dir / "measured.json"
    measured: dict[str, float] = {}
    clock = None
    if mp.exists():
        doc = json.loads(mp.read_text())
        measured = {k: v["median"] for k, v in doc["results"].items()}
        clock = measured.get("clock chain_GHz")
    return Batch(batch_dir.name, batch_dir, measured, clock)


def _sass(b: Batch, unit: str) -> str:
    return (b.build / unit / "k.sass").read_text()


def regs_of(b: Batch, unit: str, kernel: str) -> int:
    text = (b.build / unit / "res-usage.txt").read_text()
    m = re.search(rf"Function {re.escape(kernel)}:\s*REG:(\d+)", text)
    if not m:
        raise ValueError(f"{unit}: no register count for {kernel}")
    return int(m.group(1))


def _ghz(b: Batch, m: Machine) -> float:
    return b.clock_GHz or m.param(m.desc["clock"]["kernel_GHz"])


def _k(ins_text: str, kernel: str) -> list[Ins]:
    ins = parse(ins_text, kernel=kernel)
    if not ins:
        raise ValueError(f"no SASS for {kernel}")
    return ins


def _case(b: Batch, cid: str, probe: str, level: int, unit: str, build, read, **kw) -> Case:
    return Case(
        id=cid,
        probe=probe,
        level=level,
        unit=unit,
        measured=b.measured.get(cid, float("nan")),
        build=build,
        read=read,
        **kw,
    )


def per(div: float):
    return lambda sm, r: marked_cycles(sm, r) / div


def smem_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-smem_ldsts")
    out = []
    for kind, name in enumerate(["LDS.32", "LDS.64", "LDS.128", "STS.32", "STS.128"]):
        ins = _k(txt, f"_Z1kILi{kind}EEvjPyPj")
        for w in (1, 4, 8, 16, 32):
            out.append(
                _case(
                    b,
                    f"smem_ldsts {name}_w{w}",
                    "smem_ldsts",
                    1 if w == 1 else 2,
                    "cycles/instr/SM",
                    lambda ins=ins, w=w, **kw: ClockMarked(ins, 32 * w, 1, 1, **kw),
                    per(ITERS * 8 * w),
                )
            )
    return out


def shfl_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-shfl")
    i1, i8 = _k(txt, "_Z1kILi1EEvPyPj"), _k(txt, "_Z1kILi8EEvPyPj")
    out = [
        _case(
            b,
            "shfl latency_w1",
            "shfl",
            1,
            "cycles/hop",
            lambda **kw: ClockMarked(i1, 32, 1, 1, **kw),
            per(ITERS * 5),
        )
    ]
    for w in (1, 4, 8, 16, 32):
        out.append(
            _case(
                b,
                f"shfl tput_w{w}",
                "shfl",
                2,
                "cycles/instr/SM",
                lambda w=w, **kw: ClockMarked(i8, 32 * w, 1, 1, **kw),
                per(ITERS * 5 * 8 * w),
            )
        )
    return out


def ex2_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-ex2_ffma")
    out = []
    for e, f in ((8, 0), (8, 8), (8, 16), (8, 32), (0, 32), (4, 32)):
        ins = _k(txt, f"_Z1kILi{e}ELi{f}EEvPyPf")
        out.append(
            _case(
                b,
                f"ex2_ffma E{e}_F{f}",
                "ex2_ffma",
                3,
                "cycles/pass/smsp",
                lambda ins=ins, **kw: ClockMarked(ins, 1024, 1, 1, **kw),
                per(ITERS),
            )
        )
    return out


def _pair_sum(sm: SM, r: dict) -> float:
    """namedbar arrive: warp 0 reads the clock before and after each bar.arrive."""
    marks = getattr(sm, "marks", [])
    pairs = list(zip(marks[0::2], marks[1::2], strict=False))
    return sum(t1 - t0 for t0, t1 in pairs) / BAR_ITERS


def bar_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-namedbar")
    sync = _k(txt, "_Z6sync_kPy")
    out = [
        _case(
            b,
            f"namedbar sync_w{w}",
            "namedbar",
            1,
            "cycles/barrier",
            lambda w=w, **kw: ClockMarked(sync, 32 * w, 1, 1, **kw),
            per(BAR_ITERS),
        )
        for w in (2, 4, 8, 16, 32)
    ]
    for L in (0, 64, 256):
        ins = _k(txt, f"_Z8arrive_kILi{L}EEvPyPf")
        out.append(
            _case(
                b,
                f"namedbar arrive_L{L}",
                "namedbar",
                3,
                "cycles/arrive",
                # the loop counter starts from an HFMA2 used as a move, which the value tracking
                # does not evaluate: the loop's trip count is given
                lambda ins=ins, **kw: ClockMarked(ins, 256, 1, BAR_ITERS, **kw),
                _pair_sum,
            )
        )
    return out


def l1_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-l1hit")
    chase, tput = _k(txt, "_Z5chasePKjPy"), _k(txt, "_Z4tputPK4int4PyPi")
    out = [
        _case(
            b,
            "l1hit chase",
            "l1hit",
            1,
            "cycles/hop",
            lambda **kw: ClockMarked(chase, 32, 1, 1, ldg_lines=[(1, 4, True)], **kw),
            per(HOPS),
        )
    ]
    for w in (4, 8, 16, 32):
        out.append(
            _case(
                b,
                f"l1hit tput_w{w}",
                "l1hit",
                2,
                "cycles/instr/SM",
                lambda w=w, **kw: ClockMarked(tput, 32 * w, 1, 1, ldg_lines=[(4, 512, True)], **kw),
                per(ITERS * 8 * w),
            )
        )
    return out


def store_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-store_tail")
    out = []
    for mode, name in ((0, "tl"), (1, "v16")):
        ins = _k(txt, f"_Z2stILi{mode}EEvPji")
        for blocks in (1, 2, 8, 16, 32, 48):
            out.append(
                _case(
                    b,
                    f"store_tail {name}_b{blocks}",
                    "store_tail",
                    4,
                    "us",
                    lambda ins=ins, blocks=blocks, **kw: SM(
                        ins, 256, 1, 1, active_sms=blocks, const_bank={(0, 0x388): 8}, **kw
                    ),
                    lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3),
                )
            )
    return out


def l2share_cases(b: Batch) -> list[Case]:
    ins = _k(_sass(b, "probe-l2share_copy"), "matmul_kernel")
    h = L2SHARE_PICK
    trips = sass_trips(ins, h, 15)
    return [
        _case(
            b,
            f"l2share_copy b{blocks}",
            "l2share_copy",
            4,
            "us",
            lambda blocks=blocks, **kw: SM(
                ins, 256, 1, trips, active_sms=blocks, copy_lines=copy_lines(h), **kw
            ),
            lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3),
        )
        for blocks in (1, 16, 32, 48)
    ]


def dram_cases(b: Batch) -> list[Case]:
    txt = _sass(b, "probe-dram_rotate")
    out = []
    for t in (32, 64, 128, 256, 512, 1024):
        ins = _k(txt, f"_Z6streamILi{t}EEvPK4int4Pi")

        def build(ins=ins, t=t, **kw):
            mm = kw.get("machine") or default_machine()
            secs = DRAM_BYTES / (mm.C["dram_read_GBs"] * 1e9)
            return SM(
                ins,
                t,
                1,
                1,
                active_sms=48,
                ldg_lines=[(4, 512, False)],
                miss_frac=1.0,
                dram_clk_per_512B=dram_clk(DRAM_BYTES, secs, 48, mm),
                **kw,
            )

        out.append(
            _case(
                b,
                f"dram_rotate t{t}",
                "dram_rotate",
                5,
                "us",
                build,
                lambda sm, r: (
                    r["cycles"] / (_ghz(b, sm.m) * 1e3) + sm.m.param(sm.m.desc["launch_us"])
                ),
                limit=200_000_000,
            )
        )
    return out


def _launch(b: Batch, unit: str) -> dict[str, Any]:
    return json.loads((b.build / "testset" / unit / "launch.json").read_text())


def kernel_cases(b: Batch) -> list[Case]:
    root = b.build / "testset"
    if not root.exists():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if (p / "k.sass").exists()):
        L = _launch(b, d.name)
        regs = regs_of(b, f"testset/{d.name}", L["entry"])
        if L["family"] == "hgemm":
            out.append(_hgemm_case(b, d, L, regs))
        else:
            out.append(_generic_case(b, d, L, regs))
    return out


def _hgemm_case(b: Batch, d: Path, L: dict[str, Any], regs: int) -> Case:
    def build(**kw):
        return _KernelRun(
            lambda sm_hook, machine, trace: simulate_kernel(
                d / "k.sass",
                L["pick"],
                L["shape"],
                regs,
                L["smem"],
                warm=L["cache"] == "warm",
                kernel=L["entry"],
                machine=machine,
                sm_hook=sm_hook,
                trace=trace,
            ),
            **kw,
        )

    return _case(
        b,
        f"{d.name} kernel",
        "testset hgemm2048",
        7,
        "us",
        build,
        lambda sm, r: r["us"] - _launch_diff(b, sm),
        limit=0,
    )


def _launch_diff(b: Batch, sm: Any) -> float:
    """simulate_kernel converts with the description's kernel clock; reconvert the SM part with
    the batch's clock."""
    if not b.clock_GHz:
        return 0.0
    m = sm.m
    f0 = m.param(m.desc["clock"]["kernel_GHz"])
    sim_us = sm.cycles / (f0 * 1e3)
    return sim_us - sm.cycles / (b.clock_GHz * 1e3)


def _generic_case(b: Batch, d: Path, L: dict[str, Any], regs: int) -> Case:
    """A cp.async kernel without a family front end (FA forward): the busiest SM of the grid with
    block replacement, the chip's shares by blocks in flight, warm L2 (no DRAM misses), the main
    loop's trip count from TileLang's source loop."""
    ins = _k((d / "k.sass").read_text(), L["entry"])
    src = (d / "kernel.cu").read_text()
    loops = [int(x) for x in re.findall(r"for \(int k = 0; k < (\d+); \+\+k\)", src)]
    src_trips = max(loops) if loops else 1
    tiles = math.prod(L["grid"])

    def build(**kw):
        m = kw.get("machine") or default_machine()
        h = {"threads": L["threads"]}
        rmax = occupancy(h, regs, L["smem"], m)
        per_sm = math.ceil(tiles / m.sms)
        eq = min(tiles, m.sms * rmax) / min(rmax, per_sm)
        return SM(
            ins,
            L["threads"],
            per_sm,
            src_trips if main_loop(ins) else 1,
            active_sms=eq,
            resident=rmax,
            grid_blocks=tiles,
            grid_dim=tuple(L["grid"]),
            **kw,
        )

    return _case(
        b,
        f"{d.name} kernel",
        f"testset {L['family']}",
        7,
        "us",
        build,
        lambda sm, r: r["cycles"] / (_ghz(b, sm.m) * 1e3) + sm.m.param(sm.m.desc["launch_us"]),
    )


class _KernelRun:
    """A Case build result for whole-kernel front ends: run() calls the front end and keeps the
    SM it simulated (for attribution) as `sm`; `cycles` and `m` are the SM's."""

    def __init__(self, fn, machine=None, trace: bool = False):
        self.fn = fn
        self.machine, self.trace = machine or default_machine(), trace
        self.sm: SM | None = None

    def run(self, limit: int = 0) -> dict[str, Any]:
        def hook(sm: SM) -> None:
            self.sm = sm

        return self.fn(hook, self.machine, self.trace)

    @property
    def m(self) -> Machine:
        return self.machine

    @property
    def cycles(self) -> float:
        assert self.sm is not None
        return self.sm.cycles


def all_cases(batch_dir: Path) -> list[Case]:
    b = load(batch_dir)
    out: list[Case] = []
    builders = [
        lambda: smem_cases(b),
        lambda: shfl_cases(b),
        lambda: ex2_cases(b),
        lambda: bar_cases(b),
        lambda: l1_cases(b),
        lambda: store_cases(b),
        lambda: l2share_cases(b),
        lambda: dram_cases(b),
        lambda: kernel_cases(b),
    ]
    for f in builders:
        out += f()
    return out


def run_case(case: Case, **kw: Any) -> tuple[float, Any, dict]:
    sm = case.build(**kw)
    r = sm.run(limit=case.limit) if case.limit else sm.run()
    target = sm.sm if isinstance(sm, _KernelRun) else sm
    return case.read(sm, r), target, r
