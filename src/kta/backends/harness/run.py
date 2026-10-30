"""Worker-side harness job: load a cubin, launch it, return outputs and event timings.

Run as ``{python} -m kta.backends.harness.run spec.json``. ``spec.json`` (RunSpec)
names the cubin, entry, launch, buffers and scalars. Buffers come from input files
(raw little-endian bytes) or start zeroed; after one launch, the buffers listed in
``outputs`` are written to ``out/<name>.bin``. With ``iters > 0`` the kernel is then
timed with CUDA events (``warmup`` launches first) into ``out/timing.json``.
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path
from typing import Any

from kta.targets.cudadrv import CUdeviceptr, Driver

SCALARS: dict[str, type[ctypes._SimpleCData[Any]]] = {
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}


def main(argv: list[str]) -> int:
    spec: dict[str, Any] = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    drv = Driver(0)
    func = drv.load_function(Path(spec["module"]).read_bytes(), spec["entry"])
    launch = spec["launch"]
    if launch.get("smem_bytes", 0) > 0:
        drv.set_dynamic_smem(func, int(launch["smem_bytes"]))
    buffers: dict[str, tuple[CUdeviceptr, int]] = {}
    for name, buf in spec["buffers"].items():
        nbytes = int(buf["bytes"])
        ptr = drv.alloc(nbytes)
        if buf.get("input"):
            data = Path(buf["input"]).read_bytes()
            if len(data) != nbytes:
                raise SystemExit(f"buffer {name}: input has {len(data)} bytes, expected {nbytes}")
            drv.to_device(ptr, data)
        else:
            drv.memset(ptr, 0, nbytes)
        buffers[name] = (ptr, nbytes)
    args: list[ctypes._SimpleCData[Any]] = []
    for param in launch["params"]:
        if param["type"].endswith("*"):
            args.append(ctypes.c_uint64(buffers[param["name"]][0].value))
        else:
            args.append(SCALARS[param["type"]](spec["scalars"][param["name"]]))
    grid, block, smem = launch["grid"], launch["block"], int(launch.get("smem_bytes", 0))
    drv.launch(func, grid, block, smem, args)
    drv.synchronize()
    out = Path("out")
    out.mkdir(exist_ok=True)
    for name in spec["outputs"]:
        ptr, nbytes = buffers[name]
        (out / f"{name}.bin").write_bytes(drv.to_host(ptr, nbytes))
    iters, warmup = int(spec.get("iters", 0)), int(spec.get("warmup", 0))
    if iters > 0:
        for _ in range(warmup):
            drv.launch(func, grid, block, smem, args)
        times_ms: list[float] = []
        start, end = drv.event(), drv.event()
        for _ in range(iters):
            drv.record(start)
            drv.launch(func, grid, block, smem, args)
            drv.record(end)
            times_ms.append(drv.elapsed_ms(start, end))
        (out / "timing.json").write_text(json.dumps({"times_ms": times_ms}), encoding="utf-8")
    for ptr, _ in buffers.values():
        drv.free(ptr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
