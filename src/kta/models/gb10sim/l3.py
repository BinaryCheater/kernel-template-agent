"""L3 on a KTA Build: the copied GB10 simulator run on the Build's SASS (WP26, D683).

Input: the Build's `sass` blob (cuobjdump -sass text), its entry name, the TileLang holes (block
sizes, stages, threads), the problem shape, the registers and shared memory ptxas reported, and
the timing condition (warm L2, or flushed as KTA's protocol does). Output: the simulated kernel
time with the simulator's own breakdown. Only TileLang's cp.async GEMM lowering is mapped; other
kernels raise NotSupported (the warp-specialized TMA path and non-GEMM kernels need a launch
mapping of their own).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kta.core.errors import NotSupported

from .kernel import generic_kernel, machine_trips, sass_trips, tile_launch
from .machine import default_machine
from .model import PARAMS
from .sass import main_loop, parse

HOLES = ("block_M", "block_N", "block_K", "num_stages", "threads")


def simulate_build(
    sass_path: Path,
    kernel: str,
    holes: Mapping[str, int],
    shape: Mapping[str, int],
    regs: int,
    smem: int | None,
    warm: bool,
) -> dict[str, Any]:
    missing = [h for h in HOLES if h not in holes]
    if missing:
        raise NotSupported(f"L3 GB10 engine: holes {missing} not given for this Build")
    ins = parse(sass_path.read_text(), kernel=kernel)
    if not ins:
        raise NotSupported(f"L3 GB10 engine: no SASS for function {kernel!r} in the Build's dump")
    lp = main_loop(ins)
    body = ins[lp[0] : lp[1] + 1] if lp else ins
    if any(x.base == "UTMALDG" for x in body):
        raise NotSupported(
            "L3 GB10 engine: TMA warp-specialized lowering is not mapped for Builds yet"
        )
    if not any(x.base == "LDGSTS" for x in body):
        raise NotSupported("L3 GB10 engine: only TileLang's cp.async GEMM lowering is mapped")
    h = {k: int(holes[k]) for k in HOLES}
    L = tile_launch(h, dict(shape), regs, smem, warm, kernel)
    KT = math.ceil(shape["K"] / h["block_K"])
    trips = machine_trips(ins) or sass_trips(ins, h, KT - (max(1, h["num_stages"]) - 1))
    sm = generic_kernel(ins, L, regs, trips)
    r = sm.run()
    C = default_machine().C
    return {
        "total_us": C["launch_stream_us"] + r["cycles"] / (C["clock_under_load_GHz"] * 1e3),
        "detail": {"cycles": r["cycles"], "busy": r["busy"], "blocks_per_sm": sm.resident},
        "params_sha256": _sha(PARAMS),
        "warm": warm,
        "holes": h,
        "shape": dict(shape),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Job entry: `python -m kta.models.gb10sim.l3 SPEC.json SASS OUT.json` (spec: kernel, holes,
    shape, regs, smem, warm)."""

    a = argv if argv is not None else sys.argv[1:]
    spec = json.loads(Path(a[0]).read_text())
    out = simulate_build(
        Path(a[1]),
        spec["kernel"],
        spec["holes"],
        spec["shape"],
        int(spec["regs"]),
        spec.get("smem"),
        bool(spec["warm"]),
    )
    Path(a[2]).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
