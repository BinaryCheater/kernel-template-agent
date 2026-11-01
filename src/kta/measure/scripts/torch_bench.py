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
- NVML and host counters are sampled around the measured phase, using
  the same NVML and host-counter helpers shipped as job inputs, with periodic
  clock samples between event-timed launches.

Reads ``spec.json``: ``call`` (a lambda over the tensor names, for example
``lambda A, B, C: torch.matmul(A, B, out=C)``), ``tensors`` {name: {dtype, dims}},
``outputs``, ``inputs`` (list of {name: file}) and the protocol settings. Writes
``out/timing.json`` (same fields as the bench job, plus ``versions``).
"""

# pyright: reportMissingImports=false
import json
import os
import sys
import time

import numpy as np
import torch

from kta.measure.nvml import Nvml
from kta.measure.quiet import read_counters

DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
NP_DTYPES = {"f16": np.float16, "f32": np.float32}


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
    clock_samples = []
    sample_s = float(spec["clock_sample_ms"]) / 1000

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
        next_sample = t0 + sample_s
        while rounds < int(spec["max_iters"]) and (
            rounds < min_iters or (time.perf_counter() - t0) * 1000 < min_ms
        ):
            one(keep)
            rounds += 1
            now = time.perf_counter()
            if keep and now >= next_sample:
                clock_samples.append({"mono_s": now, **nvml.sample()})
                next_sample = now + sample_s
        return rounds, time.perf_counter() - t0

    nvml = Nvml(0)
    host_pre = read_counters(nvml)
    warm_iters, warm_s = phase(int(spec["warmup_min_iters"]), spec["warmup_min_ms"], False)
    env_start = nvml.sample()
    host_start = read_counters(nvml)
    iters, loop_s = phase(int(spec["min_iters"]), spec["min_ms"], True)
    host_end = read_counters(nvml)
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
        "host_pre": host_pre,
        "host_start": host_start,
        "host_end": host_end,
        "clock_samples": clock_samples,
        "clock_sample_ms": sample_s * 1000,
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
