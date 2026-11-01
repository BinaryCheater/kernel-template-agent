"""Resumable evaluation of selected candidates, with cooperative pause boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from kta.core.errors import SchemaError
from kta.core.schemas.evaluation import Candidate
from kta.orchestrator import control, pipeline, proofgate
from kta.orchestrator.pipeline import Services, bucket_shape, case_view
from kta.orchestrator.transition import Loop


def paused(svc: Services, loop: Loop) -> bool:
    return control.paused_scope(svc.ops, [f"loop:{loop.id}", "process:solve", "global"]) is not None


def pending(loop: Loop, done: Mapping[str, str]) -> bool:
    return any(cid not in done for ids in loop.data["top"].values() for cid in ids)


def checkpoint(svc: Services, loop: Loop, **updates: Any) -> None:
    # Only the runner holding this loop's lock writes progress. Read fresh data so a
    # filtered measurement view cannot overwrite the complete selected candidate set.
    row = svc.ops.get_loop(loop.id)
    svc.ops.put_loop(loop.id, loop.kind, loop.subject, loop.state, {**row["data"], **updates})


def each_top(svc: Services, loop: Loop, fn: Callable[..., str], key: str) -> dict[str, str]:
    wanted = {cid for ids in loop.data["top"].values() for cid in ids}
    out = {cid: rid for cid, rid in loop.data.get(key, {}).items() if cid in wanted}
    if not pending(loop, out) or paused(svc, loop):
        return out
    cv = case_view(svc.store, loop.subject)
    family = str(loop.data["families"][0])
    fp = pipeline.family_package(svc, family)
    _, hw = svc.machine(cv.case.target)
    for name, ids in loop.data["top"].items():
        bucket = next(b for b in loop.data["buckets"] if str(b["name"]) == name)
        shape = bucket_shape(bucket)
        controls = [
            c for c in ids if loop.data.get("arm_of", {}).get(name, {}).get(c, "model") != "model"
        ]
        if proofgate.control_gate(svc, family, fp, hw, bucket, controls) != controls:
            raise SchemaError(f"bucket {name}: control candidate failed C/Gamma before execution")
        for cid in ids:
            if paused(svc, loop):
                return out
            if cid in out:
                continue
            a = Candidate.model_validate(svc.store.get(cid).body).assignment
            gk = pipeline.generate(fp, a, shape)
            _, module = pipeline.build(svc, gk)
            out[cid] = fn(cv, fp, gk, module, shape, a)
            checkpoint(svc, loop, **{key: dict(out)})
    return out
