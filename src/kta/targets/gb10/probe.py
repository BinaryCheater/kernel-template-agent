"""GB10 probe: device attributes, toolchain versions and per-feature support.

Every step is an Executor job on a worker with the ``gpu=sm_121`` label (compiles go to
any worker with ``backend.cuda``). The probe writes, all with origin ``real``:

- one Target record (GPU, driver, CUDA, ptxas, image and backend versions from the
  worker's EnvManifest and the device query);
- one HwFact (method ``documented``) per ``hw.*`` limit;
- one HwFeature per entry of ``features.yaml`` with ``support[gb10]`` and the raw
  compiler / run output as blobs (decision D188).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import Any, Literal

import yaml

from kta.core.context import Clock
from kta.core.errors import ConfigError, ExecutionError
from kta.core.jobs import Job, JobResult, JobRunner
from kta.core.schemas import Actor, Record
from kta.core.schemas.hardware import HwFact, HwFeature, Target
from kta.store.base import Store
from kta.targets.base import ProbeReport
from kta.targets.gb10 import limits

log = logging.getLogger("kta.targets.gb10")
ACTOR = Actor(type="tool", name="kta target probe")
PROBE_PRELUDE = (
    "#include <cuda_fp16.h>\n#include <stdint.h>\n"
    'extern "C" __global__ void kta_probe(unsigned* out) {\n'
)
COMPILE_TIMEOUT_S = 120
RUN_TIMEOUT_S = 60


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    instructions: list[str]
    description: str
    body: str
    mode: str  # "run" or "compile"

    @property
    def source(self) -> str:
        return PROBE_PRELUDE + self.body + "}\n"


def load_features() -> list[FeatureSpec]:
    text = resources.files("kta.targets.gb10").joinpath("features.yaml").read_text("utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    out: list[FeatureSpec] = []
    for item in data["features"]:
        mode = str(item.get("mode", "run"))
        if mode not in ("run", "compile"):
            raise ConfigError(f"feature {item['name']}: mode must be run or compile")
        out.append(
            FeatureSpec(
                name=str(item["name"]),
                unit=str(item["unit"]),
                instructions=[str(i) for i in item.get("instructions", [])],
                description=str(item["description"]).strip(),
                body=str(item["body"]),
                mode=mode,
            )
        )
    names = [f.name for f in out]
    if len(set(names)) != len(names):
        raise ConfigError("features.yaml has duplicate names")
    return out


Support = Literal["yes", "no", "unknown"]


def _decide(spec: FeatureSpec, compiled: JobResult, ran: JobResult | None) -> Support:
    if not compiled.ok:
        return "no" if compiled.error_kind == "nonzero" else "unknown"
    if spec.mode == "compile" or ran is None:
        return "unknown"
    return "yes" if ran.ok else "unknown"


class Gb10Probe:
    def __init__(
        self, store: Store, ex: JobRunner, clock: Clock, target: str, sm: str, arch_flag: str
    ) -> None:
        self.store = store
        self.clock = clock
        self.ex = ex
        self.target = target
        self.gpu_requires = {"gpu": sm}
        self.arch_flag = arch_flag

    def device_info(self) -> tuple[JobResult, dict[str, Any]]:
        job = Job(
            kind="run",
            argv=["{python}", "-m", "kta.targets.gb10.devinfo"],
            requires=self.gpu_requires,
            timeout_s=RUN_TIMEOUT_S,
        )
        result = self.ex.run(job)
        if not result.ok or "devinfo.json" not in result.outputs:
            err = self.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")[-2000:]
            raise ExecutionError(
                result.error_kind if result.error_kind != "none" else "infra",
                f"device query failed on {result.worker_name}: {err}",
            )
        return result, json.loads(self.ex.read_blob(result.outputs["devinfo.json"]))

    def write_target(self, result: JobResult, info: dict[str, Any]) -> str:
        manifest = self.store.get(result.env_manifest).body
        body = Target(
            name=self.target,
            gpu=str(info["name"]),
            sm=self.gpu_requires["gpu"],
            driver=str(manifest.get("driver") or "unknown"),
            cuda=str(manifest.get("cuda") or "unknown"),
            ptxas=str(manifest.get("ptxas") or "unknown"),
            image=str(manifest.get("image_digest") or "none"),
            backends=dict(manifest.get("backends") or {}),
        )
        refs: dict[str, str | list[str]] = {"env": result.env_manifest, "worker": result.worker}
        rec = Record.new(
            "Target",
            body,
            ACTOR,
            self._now(),
            refs=refs,
            blobs={"devinfo": result.outputs["devinfo.json"]},
        )
        return self.store.put(rec)

    def write_facts(
        self, target_id: str, result: JobResult, info: dict[str, Any]
    ) -> dict[str, str]:
        attrs: dict[str, int] = info["attributes"]
        values: dict[str, int] = {hw: attrs[a] for hw, a in limits.FROM_ATTRIBUTES.items()}
        values["global_mem_bytes"] = int(info["total_mem_bytes"])
        values.update(limits.DOCUMENTED_CONSTANTS)
        refs: dict[str, str | list[str]] = {
            "target": target_id,
            "env": result.env_manifest,
            "worker": result.worker,
        }
        source = self.ex.put_blob(limits.DOCUMENTED_SOURCE.encode(), "text/plain")
        out: dict[str, str] = {}
        for quantity, value in sorted(values.items()):
            fact = HwFact(quantity=quantity, value=float(value), method="documented")
            blob = (
                source
                if quantity in limits.DOCUMENTED_CONSTANTS
                else result.outputs["devinfo.json"]
            )
            rec = Record.new("HwFact", fact, ACTOR, self._now(), refs=refs, blobs={"source": blob})
            out[quantity] = self.store.put(rec)
        return out

    def _compile_job(self, source_sha: str) -> Job:
        return Job(
            kind="compile",
            argv=[
                "{cuda_home}/bin/nvcc",
                "-cubin",
                self.arch_flag,
                "-o",
                "out/probe.cubin",
                "probe.cu",
            ],
            requires={"backend": "cuda"},
            inputs={"probe.cu": source_sha},
            timeout_s=COMPILE_TIMEOUT_S,
        )

    def _run_job(self, cubin_sha: str) -> Job:
        return Job(
            kind="run",
            argv=["{python}", "-m", "kta.targets.gb10.featrun", "probe.cubin"],
            requires=self.gpu_requires,
            inputs={"probe.cubin": cubin_sha},
            timeout_s=RUN_TIMEOUT_S,
        )

    def features(
        self, target_id: str, specs: list[FeatureSpec]
    ) -> tuple[dict[str, str], dict[str, str]]:
        sources = {s.name: self.ex.put_blob(s.source.encode(), "text/x-cuda") for s in specs}
        compile_ids = {s.name: self.ex.submit(self._compile_job(sources[s.name])) for s in specs}
        compiled = {name: self.ex.wait(jid) for name, jid in compile_ids.items()}
        run_ids = {
            s.name: self.ex.submit(self._run_job(compiled[s.name].outputs["probe.cubin"]))
            for s in specs
            if s.mode == "run" and compiled[s.name].ok and "probe.cubin" in compiled[s.name].outputs
        }
        ran = {name: self.ex.wait(jid) for name, jid in run_ids.items()}
        ids: dict[str, str] = {}
        support: dict[str, Support] = {}
        for s in specs:
            c, r = compiled[s.name], ran.get(s.name)
            support[s.name] = _decide(s, c, r)
            blobs = {"source": sources[s.name], "compile_stderr": c.stderr_blob}
            refs: dict[str, str | list[str]] = {
                "target": target_id,
                "env": c.env_manifest,
                "worker": c.worker,
            }
            if r is not None:
                blobs.update(run_stdout=r.stdout_blob, run_stderr=r.stderr_blob)
                if "result.json" in r.outputs:
                    blobs["run_result"] = r.outputs["result.json"]
                refs["run_worker"] = r.worker
            body = HwFeature(
                name=s.name,
                unit=s.unit,
                instructions=s.instructions,
                description=s.description,
                support={self.target: support[s.name]},
            )
            ids[s.name] = self.store.put(
                Record.new("HwFeature", body, ACTOR, self._now(), refs=refs, blobs=blobs)
            )
            log.info("feature %s: %s", s.name, support[s.name])
        return ids, dict(support)

    def run(self) -> ProbeReport:
        result, info = self.device_info()
        target_id = self.write_target(result, info)
        facts = self.write_facts(target_id, result, info)
        feature_ids, support = self.features(target_id, load_features())
        return ProbeReport(
            target_id, result.worker, result.env_manifest, facts, feature_ids, support
        )

    def _now(self) -> datetime:
        return self.clock.now()
