"""Timed runs of a PyTorch baseline under a measurement protocol, inside the worker image.

Self-contained on purpose (like ``kta.checks.scripts.ref_torch``): it is shipped into the
container as a job input file, where the kta package is not installed. It follows
``kta.measure.bench.Bench.time`` step for step, so a baseline and a generated kernel are
measured the same way (decision D284):

- every iteration rotates to the next input set, fills the outputs with all-ones bytes
  (NaN), writes the flush buffer (``l2_flush_factor`` x L2 bytes), then times the single
  call with a pair of CUDA events;
- warm-up and measured phases each run until both their iteration and wall-time minimums
  are reached (capped at ``max_iters``);
- NVML is sampled at the start and end of the measured phase (a minimal ctypes copy of
  ``kta.measure.nvml``, because the package is not importable here).

Reads ``spec.json``: ``call`` (a lambda over the tensor names, for example
``lambda A, B, C: torch.matmul(A, B, out=C)``), ``tensors`` {name: {dtype, dims}},
``outputs``, ``inputs`` (list of {name: file}) and the protocol settings. Writes
``out/timing.json`` (same fields as the bench job, plus ``versions``).
"""

# pyright: reportMissingImports=false
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch

DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
NP_DTYPES = {"f16": np.float16, "f32": np.float32}


class Nvml:
    def __init__(self):
        self.lib = None
        for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        self.handle = ctypes.c_void_p()
        if self.lib is None or self.lib.nvmlInit_v2() != 0:
            self.lib = None
            return
        # The container sees only the leased GPU, as index 0.
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)):
            self.lib = None

    def _uint(self, fn, *args):
        value = ctypes.c_uint()
        code = getattr(self.lib, fn)(self.handle, *args, ctypes.byref(value))
        return (value.value if code == 0 else None), code

    def sample(self):
        if self.lib is None:
            keys = ("sm_mhz", "sm_max_mhz", "mem_mhz", "temp_c", "power_w", "throttle_reasons")
            return {**dict.fromkeys(keys), "nvml_codes": "unavailable"}
        sm, sm_rc = self._uint("nvmlDeviceGetClockInfo", 1)
        mem, mem_rc = self._uint("nvmlDeviceGetClockInfo", 2)
        sm_max, _ = self._uint("nvmlDeviceGetMaxClockInfo", 1)
        temp, temp_rc = self._uint("nvmlDeviceGetTemperature", 0)
        power, power_rc = self._uint("nvmlDeviceGetPowerUsage")
        reasons = ctypes.c_ulonglong()
        r_rc = self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(
            self.handle, ctypes.byref(reasons)
        )
        return {
            "sm_mhz": sm,
            "sm_max_mhz": sm_max,
            "mem_mhz": mem,
            "temp_c": temp,
            "power_w": power / 1000 if power is not None else None,
            "throttle_reasons": reasons.value if r_rc == 0 else None,
            "nvml_codes": {"sm": sm_rc, "mem": mem_rc, "temp": temp_rc, "power": power_rc},
        }

    def close(self):
        if self.lib is not None:
            self.lib.nvmlShutdown()


def load_sets(spec):
    sets = []
    for files in spec["inputs"] or [{}]:
        tensors = {}
        for name, t in spec["tensors"].items():
            if name in spec["outputs"]:
                continue
            raw = np.fromfile(files[name], dtype=NP_DTYPES[t["dtype"]]).reshape(t["dims"])
            tensors[name] = torch.from_numpy(raw).to("cuda")
        sets.append(tensors)
    return sets


def main(path):
    spec = json.load(open(path, encoding="utf-8"))
    call = eval(spec["call"], {"torch": torch})  # the baseline's call, from its record
    sets = load_sets(spec)
    outs = {
        n: torch.empty(
            spec["tensors"][n]["dims"], dtype=DTYPES[spec["tensors"][n]["dtype"]], device="cuda"
        )
        for n in spec["outputs"]
    }
    order = [*spec["tensors"]]
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush_bytes = int(spec["l2_flush_factor"] * l2)
    flush = torch.empty(flush_bytes, dtype=torch.uint8, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times, state = [], {"k": 0}

    def one(keep):
        k = state["k"]
        state["k"] += 1
        if spec.get("nan_fill", True):
            for t in outs.values():
                t.view(torch.uint8).fill_(0xFF)
        flush.fill_(k & 0xFF)
        args = [sets[k % len(sets)].get(n, outs.get(n)) for n in order]
        start.record()
        call(*args)
        end.record()
        end.synchronize()
        if keep:
            times.append(start.elapsed_time(end))

    def phase(min_iters, min_ms, keep):
        rounds, t0 = 0, time.perf_counter()
        while rounds < int(spec["max_iters"]) and (
            rounds < min_iters or (time.perf_counter() - t0) * 1000 < min_ms
        ):
            one(keep)
            rounds += 1
        return rounds, time.perf_counter() - t0

    nvml = Nvml()
    warm_iters, warm_s = phase(int(spec["warmup_min_iters"]), spec["warmup_min_ms"], False)
    env_start = nvml.sample()
    iters, loop_s = phase(int(spec["min_iters"]), spec["min_ms"], True)
    env_end = nvml.sample()
    nvml.close()
    result = {
        "times_ms": [times],
        "iters": iters,
        "warmup_iters": warm_iters,
        "warmup_s": warm_s,
        "loop_s": loop_s,
        "flush_bytes": flush_bytes,
        "env_start": env_start,
        "env_end": env_end,
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cublas": torch.backends.cuda.preferred_blas_library().name,
            "device": torch.cuda.get_device_name(0),
        },
    }
    os.makedirs("out", exist_ok=True)
    with open("out/timing.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
