"""Worker-side job for measurement and checks: ``{python} -m kta.measure.bench spec.json``.

One process loads one or two modules and either runs them once per input set (mode
``run``, used by the checkers) or times them under a measurement protocol (mode
``time``). Both modes fill outputs with NaN before every launch, so an output element
the kernel never writes stays NaN.

Timing (Protocol ``measure-v*``): every iteration rotates to the next input set,
refills the outputs with NaN, writes a flush buffer of ``l2_flush_factor`` x L2 bytes,
then times the single launch with a pair of CUDA events. Warm-up and measured phases
each run until both the iteration and the wall-time minimum are reached. With two
modules (``compare``) each round runs both in an order drawn from ``order_seed``, so
drift affects both equally.

Module kinds: ``cubin`` is loaded through the driver API like the backends' harness
(decision D186); ``so`` is a host shared library whose ``entry`` takes the launch
parameters and launches by itself (needed to exercise the library-call anti-cheat
check, decision D222).
"""

from __future__ import annotations

import ctypes
import json
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kta.measure.nvml import Nvml, visible_device
from kta.targets.cudadrv import ATTRIBUTES, CUdeviceptr, Driver

SCALARS: dict[str, type[ctypes._SimpleCData[Any]]] = {
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}
NAN_BYTE = 0xFF  # all-ones is a NaN in f16, bf16, f32 and f64


class Bench:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.drv = Driver(0)
        self.sets: list[dict[str, tuple[CUdeviceptr, int]]] = []
        self.outputs: dict[str, tuple[CUdeviceptr, int]] = {}
        self._allocate()
        self.launchers = [self._launcher(m) for m in spec["modules"]]

    def _allocate(self) -> None:
        names_in = {n for s in self.spec["inputs"] for n in s}
        nsets = max(len(self.spec["inputs"]), 1)
        for i in range(nsets):
            files: dict[str, str] = self.spec["inputs"][i] if self.spec["inputs"] else {}
            bufs: dict[str, tuple[CUdeviceptr, int]] = {}
            for name, buf in self.spec["buffers"].items():
                if name in self.spec["outputs"] or (name not in names_in and i > 0):
                    continue
                nbytes = int(buf["bytes"])
                ptr = self.drv.alloc(nbytes)
                if name in files:
                    data = Path(files[name]).read_bytes()
                    if len(data) != nbytes:
                        raise SystemExit(f"input {name}: {len(data)} bytes, expected {nbytes}")
                    self.drv.to_device(ptr, data)
                else:
                    self.drv.memset(ptr, 0, nbytes)
                bufs[name] = (ptr, nbytes)
            self.sets.append(bufs)
        for name in self.spec["outputs"]:
            nbytes = int(self.spec["buffers"][name]["bytes"])
            self.outputs[name] = (self.drv.alloc(nbytes), nbytes)

    def _args(self, module: dict[str, Any], i: int) -> list[ctypes._SimpleCData[Any]]:
        args: list[ctypes._SimpleCData[Any]] = []
        for param in module["launch"]["params"]:
            name = param["name"]
            if param["type"].endswith("*"):
                ptr = self.outputs.get(name) or self.sets[i].get(name) or self.sets[0][name]
                args.append(ctypes.c_uint64(ptr[0].value))
            else:
                args.append(SCALARS[param["type"]](self.spec["scalars"][name]))
        return args

    def _launcher(self, module: dict[str, Any]) -> Callable[[int], None]:
        launch = module["launch"]
        per_set = [self._args(module, i) for i in range(len(self.sets))]
        if module["kind"] == "so":
            lib = ctypes.CDLL(str(Path(module["file"]).resolve()))
            fn = getattr(lib, module["entry"])
            fn.restype = None
            return lambda i: fn(*per_set[i])
        func = self.drv.load_function(Path(module["file"]).read_bytes(), module["entry"])
        smem = int(launch.get("smem_bytes", 0))
        if smem > 0:
            self.drv.set_dynamic_smem(func, smem)
        grid, block = launch["grid"], launch["block"]
        return lambda i: self.drv.launch(func, grid, block, smem, per_set[i])

    def _fill_outputs(self) -> None:
        if self.spec.get("nan_fill", True):
            for ptr, nbytes in self.outputs.values():
                self.drv.memset(ptr, NAN_BYTE, nbytes)

    def run(self, out: Path) -> None:
        """Each input set once, then ``repeat`` extra runs of set 0 (determinism check)."""
        launch = self.launchers[0]
        runs = [(i, 0) for i in range(len(self.sets))]
        runs += [(0, r) for r in range(1, int(self.spec.get("repeat", 1)))]
        for i, r in runs:
            self._fill_outputs()
            launch(i)
            self.drv.synchronize()
            for name, (ptr, nbytes) in self.outputs.items():
                suffix = f".s{i}" + (f".r{r}" if r else "")
                (out / f"{name}{suffix}.bin").write_bytes(self.drv.to_host(ptr, nbytes))

    def time(self, out: Path) -> None:
        spec = self.spec
        nvml = Nvml(visible_device())
        flush_bytes = int(spec["l2_flush_factor"] * self.drv.attribute(ATTRIBUTES["l2_cache_size"]))
        flush = self.drv.alloc(flush_bytes)
        start, end = self.drv.event(), self.drv.event()
        order = random.Random(int(spec.get("order_seed", 0)))
        nmod = len(self.launchers)
        times: list[list[float]] = [[] for _ in range(nmod)]
        state = {"k": 0}

        def one(m: int, keep: bool) -> None:
            k = state["k"]
            state["k"] += 1
            self._fill_outputs()
            self.drv.memset(flush, k & 0xFF, flush_bytes)
            self.drv.record(start)
            self.launchers[m](k % len(self.sets))
            self.drv.record(end)
            ms = self.drv.elapsed_ms(start, end)
            if keep:
                times[m].append(ms)

        def phase(min_iters: int, min_ms: float, keep: bool) -> tuple[int, float]:
            rounds, t0 = 0, time.perf_counter()
            max_iters = int(spec["max_iters"])
            while rounds < max_iters and (
                rounds < min_iters or (time.perf_counter() - t0) * 1000 < min_ms
            ):
                mods = list(range(nmod))
                order.shuffle(mods)
                for m in mods:
                    one(m, keep)
                rounds += 1
            return rounds, time.perf_counter() - t0

        warm_iters, warm_s = phase(int(spec["warmup_min_iters"]), spec["warmup_min_ms"], False)
        env_start = nvml.sample()
        iters, loop_s = phase(int(spec["min_iters"]), spec["min_ms"], True)
        env_end = nvml.sample()
        nvml.close()
        self.drv.free(flush)
        result = {
            "times_ms": times,
            "iters": iters,
            "warmup_iters": warm_iters,
            "warmup_s": warm_s,
            "loop_s": loop_s,
            "flush_bytes": flush_bytes,
            "env_start": env_start,
            "env_end": env_end,
        }
        (out / "timing.json").write_text(json.dumps(result), encoding="utf-8")

    def free(self) -> None:
        for bufs in self.sets:
            for ptr, _ in bufs.values():
                self.drv.free(ptr)
        for ptr, _ in self.outputs.values():
            self.drv.free(ptr)


def main(argv: list[str]) -> int:
    spec: dict[str, Any] = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = Path("out")
    out.mkdir(exist_ok=True)
    bench = Bench(spec)
    if spec["mode"] == "time":
        bench.time(out)
    else:
        bench.run(out)
    bench.free()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
