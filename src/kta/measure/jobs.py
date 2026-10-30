"""Bench jobs: the control-plane side of ``kta.measure.bench`` (also used by kta.checks).

A BuiltModule does not carry its LaunchSpec; it is read from the Build record's
``build_key`` blob, which every backend writes (decision D189). Problem tensors give
the buffers; scalar launch parameters are taken from the shape by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kta.backends.base import BuiltModule, LaunchSpec
from kta.core.errors import ConfigError
from kta.core.ids import is_record_id
from kta.core.jobs import Job, JobKind, JobRunner
from kta.core.schemas.evaluation import Case, Problem
from kta.measure import inputs as inp
from kta.store.base import Store

BENCH_ARGV = ["{python}", "-m", "kta.measure.bench", "spec.json"]


def launch_of(store: Store, module: BuiltModule) -> LaunchSpec:
    rec = store.get(module.build_id)
    if "build_key" not in rec.blobs:
        raise ConfigError(f"Build {module.build_id[:12]} has no build_key blob with its launch")
    key = json.loads(store.blob_path(rec.blobs["build_key"]).read_bytes())
    return LaunchSpec.from_dict(key["launch"])


def problem_of(store: Store, case: Case) -> Problem:
    if not is_record_id(case.problem):
        raise ConfigError(f"Case.problem must be a Problem record ID, got {case.problem!r}")
    return Problem.model_validate(store.get(case.problem).body)


def target_sm(store: Store, target: str) -> str:
    """The ``gpu`` label value jobs require: Case.target is a Target record ID or name."""
    if is_record_id(target):
        return str(store.get(target).body["sm"])
    found = store.query("Target", {"name": target}, order="-created_at", limit=1)
    if not found:
        raise ConfigError(f"no Target record named {target!r}")
    return str(found[0].body["sm"])


@dataclass(frozen=True)
class BenchJob:
    job: Job
    spec: dict[str, Any]
    spec_blob: str


def bench_job(
    ex: JobRunner,
    store: Store,
    modules: list[BuiltModule],
    problem: Problem,
    shape: dict[str, Any],
    input_sets: list[dict[str, bytes]],
    mode: str,
    kind: JobKind,
    requires: dict[str, str],
    settings: dict[str, Any] | None = None,
    wrap: list[str] | None = None,
    timeout_s: int = 600,
) -> BenchJob:
    """A job running ``modules`` on ``input_sets`` (mode ``run`` or ``time``)."""
    job_inputs: dict[str, str] = {}
    mods: list[dict[str, Any]] = []
    scalars: dict[str, int | float] = {}
    for i, module in enumerate(modules):
        if not module.sha256:
            raise ConfigError(f"build {module.build_id[:12]} produced no module")
        launch = launch_of(store, module)
        name = f"m{i}.{module.kind}"
        job_inputs[name] = module.sha256
        mods.append(
            {"file": name, "kind": module.kind, "entry": module.entry, "launch": launch.to_dict()}
        )
        for p in launch.params:
            if not p.is_pointer:
                if p.name not in shape:
                    raise ConfigError(f"scalar parameter {p.name} is not a shape variable")
                scalars[p.name] = shape[p.name]
    buffers = {t.name: {"bytes": inp.nbytes(t, shape)} for t in [*problem.inputs, *problem.outputs]}
    files: list[dict[str, str]] = []
    for i, data in enumerate(input_sets):
        entry: dict[str, str] = {}
        for name, raw in data.items():
            if name not in buffers or len(raw) != buffers[name]["bytes"]:
                raise ConfigError(f"input {name}: does not match the Problem's tensor")
            path = f"in/s{i}_{name}.bin"
            job_inputs[path] = ex.put_blob(raw, "application/octet-stream")
            entry[name] = path
        files.append(entry)
    spec: dict[str, Any] = {
        "modules": mods,
        "buffers": buffers,
        "inputs": files,
        "outputs": [t.name for t in problem.outputs],
        "scalars": scalars,
        "mode": mode,
        **(settings or {}),
    }
    blob = ex.put_blob(json.dumps(spec, sort_keys=True).encode(), "application/json")
    job_inputs["spec.json"] = blob
    job = Job(
        kind=kind,
        argv=[*(wrap or []), *BENCH_ARGV],
        requires=requires,
        inputs=job_inputs,
        timeout_s=timeout_s,
    )
    return BenchJob(job, spec, blob)


def random_inputs(
    problem: Problem, shape: dict[str, Any], dist: str, rng: Any, table: Any = None
) -> dict[str, bytes]:
    return {t.name: inp.generate(t, shape, dist, rng, table).tobytes() for t in problem.inputs}
