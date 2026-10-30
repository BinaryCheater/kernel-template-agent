"""CUDA C++ backend: nvcc to a cubin on a worker with ``backend.cuda``.

The source must define ``extern "C" __global__`` entry ``gk.entry`` (no name mangling),
so every backend's cubin is launched the same way by the harness.
"""

from __future__ import annotations

import shlex

from kta.backends import common
from kta.backends.base import BuiltModule, GeneratedKernel
from kta.core.context import Clock
from kta.core.errors import ConfigError
from kta.core.jobs import Job, JobRunner
from kta.core.schemas.evaluation import Build
from kta.core.schemas.hardware import Target
from kta.store.base import Store
from kta.targets.base import TargetPlugin

COMPILE_TIMEOUT_S = 600
MAIN_FILE = "kernel.cu"


def compile_script(flags: list[str]) -> str:
    nvcc = "{cuda_home}/bin/nvcc"
    return (
        f"{nvcc} -cubin {shlex.join(flags)} -Xptxas -v -o out/module.cubin {MAIN_FILE} && "
        "{cuda_home}/bin/cuobjdump -sass out/module.cubin > out/module.sass"
    )


class CudaBackend:
    name = "cuda"

    def __init__(self, store: Store, clock: Clock, plugin: TargetPlugin) -> None:
        self.store = store
        self.clock = clock
        self.plugin = plugin

    def supports(self, target: Target) -> bool:
        return target.name == self.plugin.name and "cuda" in target.backends

    def build(
        self, gk: GeneratedKernel, target: Target, ex: JobRunner
    ) -> tuple[Build, BuiltModule]:
        if MAIN_FILE not in gk.files:
            raise ConfigError(f"CUDA kernels need a {MAIN_FILE} file; got {sorted(gk.files)}")
        flags = self.plugin.compile_flags(self.name)
        key = common.build_key(gk, target, flags, None, ex)
        hit = common.cached(self.store, key)
        if hit is not None:
            return hit
        inputs = {n: ex.put_blob(t.encode("utf-8"), "text/x-cuda") for n, t in gk.files.items()}
        job = Job(
            kind="compile",
            argv=["sh", "-c", compile_script(flags)],
            requires={"backend": "cuda"},
            inputs=inputs,
            timeout_s=COMPILE_TIMEOUT_S,
        )
        result = ex.run(job)
        failure = None
        cubin = result.outputs.get("module.cubin")
        if result.ok and cubin is not None and gk.entry.encode() not in ex.read_blob(cubin):
            failure = f'entry {gk.entry!r} is not in the cubin (missing extern "C"?)'
        return common.write_build(
            self.store, self.clock, gk, target, key, result, gk.entry, {}, failure
        )
