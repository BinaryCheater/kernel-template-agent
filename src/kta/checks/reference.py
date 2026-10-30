"""Reference outputs on the GPU (docs/design/50 §3).

``Problem.reference_impl`` is a Python statement over torch tensors named like the
Problem's inputs and outputs, with the shape variables in scope (``C = A @ B``). It runs
in the worker image (PyTorch, no network) on the GPU, in FP64 by default; FP32 is used
only when the caller says so and the reason is recorded (decision D223). Inputs are the
exact bytes given to the kernel, widened on the GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

from kta.checks.base import Checker
from kta.core.errors import CheckerError, NotSupported
from kta.core.jobs import Job, JobResult
from kta.core.schemas.evaluation import Problem
from kta.measure import inputs as inp

REF_TIMEOUT_S = 900
SCRIPT = "ref_torch.py"


@dataclass(frozen=True)
class GpuTarget:
    sm: str
    image: str


def gpu_target(c: Checker) -> GpuTarget:
    found = c.store.query("Target", {}, order="-created_at", limit=1)
    if not found:
        raise NotSupported("no Target record; run `kta target probe` first")
    body = found[0].body
    image = c.image or str(body.get("image") or "")
    if image in ("", "none"):
        raise NotSupported(f"target {body['name']}: no worker image with PyTorch for references")
    return GpuTarget(str(body["sm"]), image)


@dataclass(frozen=True)
class Reference:
    result: JobResult
    outputs: list[dict[str, np.ndarray[Any, Any]]]  # per input set
    compute: str


def reference(
    c: Checker,
    problem: Problem,
    shape: dict[str, Any],
    sets: list[dict[str, bytes]],
    compute: str = "f64",
) -> Reference:
    if problem.reference_impl.startswith("none"):
        raise NotSupported(f"problem {problem.name} has no reference: {problem.reference_impl}")
    target = gpu_target(c)
    script = resources.files("kta.checks.scripts").joinpath(SCRIPT).read_bytes()
    job_inputs = {SCRIPT: c.ex.put_blob(script, "text/x-python")}
    files: list[dict[str, str]] = []
    for i, data in enumerate(sets):
        entry: dict[str, str] = {}
        for name, raw in data.items():
            path = f"in/s{i}_{name}.bin"
            job_inputs[path] = c.ex.put_blob(raw, "application/octet-stream")
            entry[name] = path
        files.append(entry)
    spec = {
        "statement": problem.reference_impl,
        "compute": compute,
        "shape": shape,
        "inputs": {
            t.name: {"dtype": np.dtype(inp.np_dtype(t.dtype)).name, "dims": inp.dims(t, shape)}
            for t in problem.inputs
        },
        "outputs": [t.name for t in problem.outputs],
        "sets": files,
    }
    job_inputs["ref.json"] = c.ex.put_blob(json.dumps(spec).encode(), "application/json")
    job = Job(
        kind="check",
        argv=["python", SCRIPT, "ref.json"],
        requires={"gpu": target.sm},
        inputs=job_inputs,
        timeout_s=REF_TIMEOUT_S,
        container=target.image,
    )
    result = c.ex.run(job)
    if not result.ok:
        tail = c.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")[-1500:]
        raise CheckerError(f"reference job {result.job_id} failed ({result.error_kind}): {tail}")
    outs: list[dict[str, np.ndarray[Any, Any]]] = []
    for i in range(len(sets)):
        got: dict[str, np.ndarray[Any, Any]] = {}
        for t in problem.outputs:
            raw = c.ex.read_blob(result.outputs[f"{t.name}.s{i}.bin"])
            got[t.name] = np.frombuffer(raw, dtype=np.float64).reshape(inp.dims(t, shape))
        outs.append(got)
    return Reference(result, outs, compute)
