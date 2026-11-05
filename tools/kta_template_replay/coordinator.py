"""Executor-only evaluator submission and append-only candidate/budget ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kta.core.errors import BudgetExceeded, ExecutionError
from kta.core.jobs import Job
from kta.execution.base import Executor

from .manifest import identity, load, write_new


def reserve(run: Path, manifest: dict[str, Any], agent_active_s: float) -> Path:
    if not 0 <= agent_active_s < manifest["active_budget_s"]:
        raise BudgetExceeded("Agent active time exhausted or invalid")
    attempts = sorted((run / "attempts").glob("*"))
    consumed = 0.0
    for path in attempts:
        result = path / "execution.json"
        if not result.exists():
            raise BudgetExceeded("Previous evaluation is unfinished; reconcile it before retry")
        consumed += load(result)["wall_s"]
    if consumed + agent_active_s >= manifest["active_budget_s"]:
        raise BudgetExceeded("Active execution budget exhausted")
    if len(attempts) >= manifest["max_batches"]:
        raise BudgetExceeded("Candidate evaluation batch budget exhausted")
    attempt = run / "attempts" / f"{len(attempts) + 1:03d}"
    attempt.mkdir(parents=True)
    write_new(
        attempt / "reservation.json",
        {
            "agent_active_s": agent_active_s,
            "remaining_s": manifest["active_budget_s"] - consumed - agent_active_s,
        },
    )
    return attempt


def dispatch(
    executor: Executor, manifest: dict[str, Any], candidate: bytes | None, timeout_s: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    inputs = {
        "worker.py": Path(__file__).with_name("worker.py").read_bytes(),
        "manifest.json": json.dumps(manifest, allow_nan=False).encode(),
    }
    if candidate is not None:
        inputs["candidate.py"] = candidate
    environment = manifest["environment"]
    requires = {**environment["requires"], "pool": "gpu-exclusive"}
    job = Job(
        kind="measure",
        argv=[
            environment.get("python", "python"),
            "worker.py",
            *(["--preflight"] if candidate is None else []),
        ],
        requires=requires,
        container=environment["container"],
        timeout_s=timeout_s,
        inputs={
            name: executor.put_blob(data, "application/octet-stream")
            for name, data in inputs.items()
        },
    )
    result = executor.run(job)
    execution = {
        **result.to_dict(),
        "payload_identities": {name: identity(data) for name, data in inputs.items()},
        "snapshot": "uncommitted experimental payload; not clean WP evidence",
    }
    parsed = None
    output = result.outputs.get("result.json") or result.outputs.get("out/result.json")
    if output:
        parsed = json.loads(executor.read_blob(output))
    elif result.ok:
        raise ExecutionError("infra", "Evaluator job omitted result.json")
    return execution, parsed


def smoke(executor: Executor, run: Path) -> dict[str, Any]:
    manifest = load(run / "manifest.json")
    execution, result = dispatch(executor, manifest, None, manifest["active_budget_s"])
    write_new(run / "preflight-execution.json", execution)
    result = result or {"passed": False, "reason": execution["error_kind"]}
    result["manifest_identity"] = identity((run / "manifest.json").read_bytes())
    write_new(run / "preflight.json", result)
    return result


def evaluate(executor: Executor, run: Path, source: Path, agent_active_s: float) -> dict[str, Any]:
    manifest = load(run / "manifest.json")
    preflight = load(run / "preflight.json")
    if not preflight.get("passed") or preflight["manifest_identity"] != identity(
        (run / "manifest.json").read_bytes()
    ):
        raise ValueError("An unchanged manifest with successful preflight is required")
    if len(manifest["shapes"]) > manifest["max_shapes_per_batch"]:
        raise ValueError("Shape budget exceeded")
    # Atomic directory creation prevents concurrent evaluators from sharing an attempt.
    attempt = reserve(run, manifest, agent_active_s)
    candidate = source.read_bytes()
    (attempt / "candidate.py").write_bytes(candidate)
    write_new(attempt / "identity.json", {"candidate": identity(candidate)})
    remaining_s = load(attempt / "reservation.json")["remaining_s"]
    execution, result = dispatch(executor, manifest, candidate, max(1, int(remaining_s)))
    write_new(attempt / "execution.json", execution)
    result = result or {"passed": False, "reason": execution["error_kind"]}
    write_new(attempt / "result.json", result)
    return {"attempt": attempt.name, **result}
