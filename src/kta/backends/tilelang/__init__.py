"""TileLang backend: codegen in the worker image, host nvcc to a cubin.

``gk.files["kernel.py"]`` defines the prim_func named ``gk.entry``. The build job runs
in the worker image (TileLang installed, no network) with the host CUDA toolkit
mounted, lowers the prim_func to CUDA source and compiles it with the same nvcc and
flags as the CUDA backend (decision D187). The generated kernel must agree with
``gk.launch``: same block size and parameter names in order; otherwise the Build fails.
"""

from __future__ import annotations

import json
import shlex
from importlib import resources

from kta.backends import common
from kta.backends.base import BuiltModule, GeneratedKernel
from kta.core.context import Clock
from kta.core.errors import ConfigError, NotSupported
from kta.core.jobs import Job, JobResult, JobRunner
from kta.core.schemas.evaluation import Build
from kta.core.schemas.hardware import Target
from kta.execution.sandbox import CONTAINER_CUDA_HOME
from kta.store.base import Store
from kta.targets.base import TargetPlugin

COMPILE_TIMEOUT_S = 900
MAIN_FILE = "kernel.py"


def _resource(name: str) -> bytes:
    return resources.files("kta.backends.tilelang").joinpath(name).read_bytes()


def launch_mismatch(gk: GeneratedKernel, gen: dict[str, object]) -> str | None:
    block = gen.get("block")
    if block is not None and block != gk.launch.threads_per_block:
        return f"TileLang kernel has {block} threads per block, launch spec {gk.launch.block}"
    names = [p.name for p in gk.launch.params]
    if gen.get("params") != names:
        return f"TileLang kernel parameters {gen.get('params')} differ from launch spec {names}"
    return None


class TileLangBackend:
    name = "tilelang"

    def __init__(self, store: Store, clock: Clock, plugin: TargetPlugin) -> None:
        self.store = store
        self.clock = clock
        self.plugin = plugin

    def supports(self, target: Target) -> bool:
        return "tilelang" in target.backends and target.image not in ("", "none")

    def _job(self, gk: GeneratedKernel, target: Target, flags: list[str], ex: JobRunner) -> Job:
        inputs = {n: ex.put_blob(t.encode("utf-8"), "text/x-python") for n, t in gk.files.items()}
        inputs["gen.py"] = ex.put_blob(_resource("gen.py"), "text/x-python")
        inputs["build.sh"] = ex.put_blob(_resource("build.sh"), "text/x-shellscript")
        return Job(
            kind="compile",
            argv=["sh", "build.sh"],
            requires={"backend": "tilelang"},
            env={
                "KTA_ENTRY": gk.entry,
                "KTA_ARCH": target.sm,
                "KTA_NVCC_FLAGS": shlex.join(flags),
                "KTA_CUDA_HOME": CONTAINER_CUDA_HOME,
            },
            inputs=inputs,
            timeout_s=COMPILE_TIMEOUT_S,
            container=target.image,
        )

    def build(
        self, gk: GeneratedKernel, target: Target, ex: JobRunner
    ) -> tuple[Build, BuiltModule]:
        if not self.supports(target):
            raise NotSupported(f"TileLang on target {target.name}: no worker image with TileLang")
        if MAIN_FILE not in gk.files:
            raise ConfigError(f"TileLang kernels need a {MAIN_FILE} file; got {sorted(gk.files)}")
        flags = self.plugin.compile_flags(self.name)
        key = common.build_key(gk, target, flags, target.image, ex)
        hit = common.cached(self.store, key)
        if hit is not None:
            return hit
        result: JobResult = ex.run(self._job(gk, target, flags, ex))
        gen: dict[str, object] = {}
        extra: dict[str, str] = {}
        if "gen.json" in result.outputs:
            gen = json.loads(ex.read_blob(result.outputs["gen.json"]))
            extra["codegen"] = result.outputs["gen.json"]
        if "kernel.cu" in result.outputs:
            extra["generated_source"] = result.outputs["kernel.cu"]
        failure = launch_mismatch(gk, gen) if result.ok else None
        entry = str(gen.get("kernel_entry", gk.entry))
        return common.write_build(
            self.store, self.clock, gk, target, key, result, entry, extra, failure
        )
