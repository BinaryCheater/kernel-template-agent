"""The uniform loader and timing entry for built modules (used by measure).

Every backend produces a cubin with an ``extern "C"`` entry; the harness job loads it
with the CUDA driver API and launches it from the LaunchSpec, so launch path and
timing are identical across backends (decision D186). Inputs and outputs travel as
blobs by hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kta.backends.base import BuiltModule, LaunchSpec
from kta.core.errors import ConfigError
from kta.core.jobs import Job, JobKind, JobResult, JobRunner

RUN_TIMEOUT_S = 120


@dataclass(frozen=True)
class Buffer:
    nbytes: int
    data: bytes | None = None  # None: zero-filled on the device


@dataclass(frozen=True)
class HarnessRun:
    result: JobResult
    outputs: dict[str, bytes] = field(default_factory=lambda: dict[str, bytes]())
    times_ms: list[float] = field(default_factory=lambda: list[float]())


def run_job(
    ex: JobRunner,
    module: BuiltModule,
    launch: LaunchSpec,
    buffers: dict[str, Buffer],
    scalars: dict[str, int | float],
    outputs: list[str],
    requires: dict[str, str],
    kind: JobKind = "run",
    iters: int = 0,
    warmup: int = 0,
    timeout_s: int = RUN_TIMEOUT_S,
) -> Job:
    """The harness job for one module; inputs are uploaded as blobs."""
    if not module.sha256:
        raise ConfigError(f"build {module.build_id[:12]} produced no module")
    for p in launch.params:
        if p.is_pointer and p.name not in buffers:
            raise ConfigError(f"no buffer for pointer parameter {p.name}")
        if not p.is_pointer and p.name not in scalars:
            raise ConfigError(f"no value for scalar parameter {p.name}")
    inputs = {"module.cubin": module.sha256}
    spec_buffers: dict[str, dict[str, object]] = {}
    for name, buf in buffers.items():
        entry: dict[str, object] = {"bytes": buf.nbytes}
        if buf.data is not None:
            if len(buf.data) != buf.nbytes:
                raise ConfigError(f"buffer {name}: {len(buf.data)} bytes of data for {buf.nbytes}")
            inputs[f"in/{name}.bin"] = ex.put_blob(buf.data, "application/octet-stream")
            entry["input"] = f"in/{name}.bin"
        spec_buffers[name] = entry
    spec = {
        "module": "module.cubin",
        "entry": module.entry,
        "launch": launch.to_dict(),
        "buffers": spec_buffers,
        "scalars": scalars,
        "outputs": outputs,
        "iters": iters,
        "warmup": warmup,
    }
    inputs["spec.json"] = ex.put_blob(json.dumps(spec, sort_keys=True).encode(), "application/json")
    return Job(
        kind=kind,
        argv=["{python}", "-m", "kta.backends.harness.run", "spec.json"],
        requires=requires,
        inputs=inputs,
        timeout_s=timeout_s,
    )


def collect(ex: JobRunner, result: JobResult, outputs: list[str]) -> HarnessRun:
    got = {
        n: ex.read_blob(result.outputs[f"{n}.bin"]) for n in outputs if f"{n}.bin" in result.outputs
    }
    times: list[float] = []
    if "timing.json" in result.outputs:
        times = list(json.loads(ex.read_blob(result.outputs["timing.json"]))["times_ms"])
    return HarnessRun(result, got, times)
