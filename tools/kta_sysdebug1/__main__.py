"""Record the runbook's deterministic probes without changing calibrated controls.

Run with ``PYTHONPATH=tools uv run python -m kta_sysdebug1 consistency|interference|tests``.
The ordinary ``kta`` commands perform setup, noise measurements and hardware calibration.
"""

from __future__ import annotations

import argparse
import json
import shlex
from typing import Any

import yaml

from kta.cli.exec_cmds import command
from kta.cli.main import context_from_environ
from kta.core.config import repo_root
from kta.core.hostlock import heavy
from kta.core.jobs import Job
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case
from kta.core.schemas.process import Report
from kta.formulation import assemble
from kta.knowledge import load_family
from kta.measure import interference
from kta.orchestrator.pipeline import Services, templates_of
from kta.solve.consistency import consistency_check

ACTOR = Actor(type="tool", name="SYSDEBUG-1 deterministic probes")


def consistency(svc: Services) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    root = repo_root() / "knowledge" / "seed"
    fp = load_family(root / "families" / "gemm_fp16_tilelang")
    doc = yaml.safe_load((root / "cases" / "gemm_fp16.yaml").read_text())
    case = Case(problem="gemm_fp16", target="gb10", **doc["case"])
    hid, mm = svc.machine(case.target)
    f = assemble(fp, fp.chain, templates_of(svc.store, fp), case, mm.hw)
    results: list[dict[str, Any]] = []
    with heavy(svc.ctx.config.section("host"), "SYSDEBUG-1 MIP-L2"):
        for bucket in case.shape_dist:
            shape = dict(bucket["shape"])
            result = consistency_check(f, shape, mm, **svc.ctx.controls.mip_l2_check)
            results.append({"shape": shape, **result.summary()})
    return {"results": results, "controls_unchanged": True}, {"hwmodel": hid}, {}


def pairwise(svc: Services) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    target = svc.target()
    tid = str(svc.ctx.registry.resolve("target", svc.ctx).latest_target_id())
    pid, _ = svc.protocol()
    cells = []
    for a, b in (("mma_sync_m16n8k16_f16", "global_load"), ("ffma", "mufu")):
        cell = interference.measure_pair(
            svc.measurer(), svc.backend("cuda"), target, tid, a, b, pid
        )
        cells.append(cell.model_dump(mode="json"))
    measurements = [rid for cell in cells for rid in cell["measurements"]]
    return (
        {"cells": cells, "scope": "two initial pairs; not an exhaustive interference matrix"},
        {"target": tid, "protocol": pid, "measurements": measurements},
        {},
    )


def tests(svc: Services) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    root = repo_root()
    invocation = (
        f"cd {shlex.quote(str(root))} && .venv/bin/python -m pytest -q tests/unit tests/contract"
    )
    result = svc.executor.run(Job(kind="shell", argv=["sh", "-c", invocation], timeout_s=600))
    return (
        {"ok": result.ok, "exit_code": result.exit_code, "job_id": result.job_id},
        {"env": result.env_manifest, "worker": result.worker},
        {"stdout": result.stdout_blob, "stderr": result.stderr_blob},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=("consistency", "interference", "tests"))
    args = parser.parse_args()
    with command(context_from_environ, "sysdebug1", f"SYSDEBUG-1 {args.probe}") as ctx:
        svc = Services.open(ctx)
        probe = {"consistency": consistency, "interference": pairwise, "tests": tests}[args.probe]
        data, refs, blobs = probe(svc)
        refs["controls"] = svc.ctx.controls.version_id
        report = Report(
            title=f"SYSDEBUG-1 {args.probe}",
            subject="SYSDEBUG1",
            summary=f"Deterministic {args.probe} probe; decisions require runbook card review",
            data=data,
        )
        rec = Record.new("Report", report, ACTOR, ctx.clock.now(), refs=refs, blobs=blobs)
        rid = svc.store.put(rec)
    commit = svc.store.get(rid).actor.commit
    print(json.dumps({"report": rid, "code_commit": commit, "data": data}, indent=2))
    return 0 if data.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
