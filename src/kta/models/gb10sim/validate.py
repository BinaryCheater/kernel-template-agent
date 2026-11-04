"""Run the SM simulator on the 22 ablated cp.async kernels and their variants.

Copied from AI-Infra-Co-Design agentic/gb10/sim/validate.py (commit 0eef497, D681). KTA changes: the
calibration directory is an argument (`cal` = a copy of calibration/spark-6951) and results are
returned instead of printed or written under runs/."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .kernel import copy_lines, occupancy, sass_trips
from .machine import Machine, default_machine
from .sm import SM, kernel_sass


def trips(cal: Path, name: str, variant: str) -> int:
    src = (cal / "ablation_cpasync" / name / f"{variant}.kernel.cu").read_text()
    m = re.search(r"for \(int k = 0; k < (\d+); \+\+k\)", src)
    return int(m.group(1)) if m else 1


def _mechanisms(cal: Path) -> dict[str, Any]:
    return json.loads((cal / "ablation_cpasync" / "mechanisms.json").read_text())


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
    r = sm.run()
    r["us"] = r["cycles"] / (m.param(m.desc["clock"]["block_level_GHz"]) * 1e3)
    if sm_hook is not None:
        sm_hook(sm)
    return r


def _rmax(h: dict, regs: int, machine: Machine) -> int:
    smem = h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    return occupancy(h, regs, smem, machine)


def kernel_us(
    cal: Path,
    name: str,
    variant: str,
    tiles: int,
    rmax: int,
    machine: Machine | None = None,
    sm_hook=None,
    trace: bool = False,
) -> float:
    """Simulated kernel time, plus the launch: the busiest SM runs its share of the blocks, at most
    Rmax at a time, a new one starting when one exits; the chip's L2 and store limits are shared by
    the blocks in flight (as min(48, tiles/Rmax) equal SMs)."""
    m = machine or default_machine()
    C, SMS = m.C, m.sms
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
