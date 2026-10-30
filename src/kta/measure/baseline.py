"""Library baselines (torch.matmul / cuBLAS) measured under the same Protocol (decision D284).

A baseline is fixed before anything is measured: ``ensure_baseline`` writes a Report
(``subject = "baseline"``) holding the call, the library and the worker image. A Case
names it with ``refs.baseline``, which ``full_check`` keeps to one record.

``measure_baseline`` runs ``scripts/torch_bench.py`` in the worker image on the GPU (pool
``gpu-exclusive``, job kind ``measure``). The inputs are drawn exactly as for a kernel
(``Measurer.input_sets``: same distribution, set count and seeded streams), the loop is
the bench job's, and the Measurement is computed by the same code. The Measurement's
``refs.baseline`` replaces ``refs.build``.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from kta.core.context import Clock
from kta.core.errors import ConfigError, ExecutionError
from kta.core.ids import canonical_json, record_id
from kta.core.jobs import Job
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case
from kta.core.schemas.process import Report
from kta.measure import inputs as inp
from kta.measure import jobs
from kta.measure.protocol import load_params
from kta.measure.runner import Measured, Measurer, Timed
from kta.store.base import Store

ACTOR = Actor(type="tool", name="kta measure baseline")
SCRIPT = "torch_bench.py"
TIMEOUT_S = 900
TORCH_MATMUL = "lambda A, B, C: torch.matmul(A, B, out=C)"
SETTINGS = (
    "l2_flush_factor",
    "nan_fill",
    "warmup_min_iters",
    "warmup_min_ms",
    "min_iters",
    "min_ms",
    "max_iters",
)


def ensure_baseline(
    store: Store, clock: Clock, problem_id: str, call: str, library: str, image: str
) -> str:
    """The baseline Report for (problem, call, library, image); reused when present."""
    data = {"problem": problem_id, "call": call, "library": library, "image": image}
    for rec in store.query("Report", {"subject": "baseline"}, order="created_at", limit=500):
        if rec.body.get("data") == data:
            return record_id(rec)
    body = Report(
        title="baseline",
        subject="baseline",
        summary=f"{library}: {call} in worker image {image[:19]}",
        data=data,
    )
    return store.put(Record.new("Report", body, ACTOR, clock.now(), refs={"problem": problem_id}))


def _job(
    m: Measurer, baseline: dict[str, Any], case: Case, shape: dict[str, Any], protocol_id: str
) -> tuple[Job, list[str]]:
    params = load_params(m.store, protocol_id)
    problem = jobs.problem_of(m.store, case)
    tag = f"{canonical_json(shape).decode()}:baseline:{baseline['call']}"
    streams = {"inputs": f"measure.inputs:{tag}"}
    rng = m.rng(streams["inputs"])
    sets = [jobs.random_inputs(problem, shape, "normal", rng) for _ in range(params.min_input_sets)]
    script = resources.files("kta.measure.scripts").joinpath(SCRIPT).read_bytes()
    job_inputs = {SCRIPT: m.ex.put_blob(script, "text/x-python")}
    files: list[dict[str, str]] = []
    for i, data in enumerate(sets):
        entry: dict[str, str] = {}
        for name, raw in data.items():
            path = f"in/s{i}_{name}.bin"
            job_inputs[path] = m.ex.put_blob(raw, "application/octet-stream")
            entry[name] = path
        files.append(entry)
    tensors = {
        t.name: {"dtype": t.dtype, "dims": list(inp.dims(t, shape))}
        for t in [*problem.inputs, *problem.outputs]
    }
    spec: dict[str, Any] = {
        "call": baseline["call"],
        "tensors": tensors,
        "outputs": [t.name for t in problem.outputs],
        "inputs": files,
        **{k: getattr(params, k) for k in SETTINGS},
    }
    raw = json.dumps(spec, sort_keys=True).encode()
    job_inputs["spec.json"] = m.ex.put_blob(raw, "application/json")
    job = Job(
        kind="measure",
        argv=["python", SCRIPT, "spec.json"],
        requires={"gpu": jobs.target_sm(m.store, case.target)},
        inputs=job_inputs,
        timeout_s=TIMEOUT_S,
        container=baseline["image"],
    )
    meta = {
        "spec": job_inputs["spec.json"],
        "shape": shape,
        "ctx_seed": m.seed,
        "rng_streams": streams,
        "input_dist": "normal",
        "input_sets": params.min_input_sets,
    }
    return job, [m.ex.put_blob(json.dumps(meta, sort_keys=True).encode(), "application/json")]


def measure_baseline(
    m: Measurer,
    baseline_id: str,
    case: Case,
    shape: dict[str, Any],
    protocol_id: str,
    case_id: str | None = None,
) -> Measured:
    """Time the baseline on ``shape`` under ``protocol_id``: a Measurement with
    refs.baseline (and refs.case when ``case_id`` is given)."""
    rec = m.store.get(baseline_id)
    if rec.kind != "Report" or rec.body.get("subject") != "baseline":
        raise ConfigError(f"{baseline_id[:12]} is not a baseline Report")
    baseline: dict[str, Any] = rec.body["data"]
    job, (params_blob,) = _job(m, baseline, case, shape, protocol_id)
    result = m.ex.run(job)
    if not result.ok or "timing.json" not in result.outputs:
        stderr = m.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")[-2000:]
        raise ExecutionError(
            "nonzero" if result.error_kind == "none" else result.error_kind,
            f"baseline job {result.job_id} failed: {stderr}",
        )
    timing = json.loads(m.ex.read_blob(result.outputs["timing.json"]))
    t = Timed(result, timing, params_blob, load_params(m.store, protocol_id))
    body, times_us = m.measurement_body(t, 0, baseline_id, protocol_id)
    body.clocks["library_versions"] = timing.get("versions", {})
    refs: dict[str, str | list[str]] = {
        "env": result.env_manifest,
        "worker": result.worker,
        "baseline": baseline_id,
        "protocol": protocol_id,
    }
    if case_id is not None:
        refs["case"] = case_id
    return m.write_measurement(t, body, times_us, refs, case)
