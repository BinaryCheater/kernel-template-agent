"""Nsight Compute profiles of one launch (``profile`` jobs in the gpu-exclusive pool).

The bench job runs once under ``ncu --csv``; the metric rows of the module's kernel
become a Profile record. A worker whose driver restricts performance counters to
administrators (``RmProfilingAdminOnly=1``, ncu's ``ERR_NVGPUCTRPERM``) cannot profile:
that raises NotSupported instead of returning partial numbers (decision D228).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from kta.backends.base import BuiltModule
from kta.core.context import Context
from kta.core.errors import ExecutionError, NotSupported
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case, Profile
from kta.measure import jobs
from kta.measure.runner import Measurer

log = logging.getLogger("kta.measure.profile")
ACTOR = Actor(type="tool", name="kta profile@1")
DEFAULT_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]
NO_PERMISSION = "ERR_NVGPUCTRPERM"


def parse_ncu_csv(text: str, kernel: str) -> dict[str, float]:
    rows = [line for line in text.splitlines() if line.startswith('"')]
    out: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO("\n".join(rows))):
        if kernel not in (row.get("Kernel Name") or ""):
            continue
        value = (row.get("Metric Value") or "").replace(",", "")
        try:
            out[str(row["Metric Name"])] = float(value)
        except ValueError:
            log.warning("ncu metric %s has a non-numeric value %r", row.get("Metric Name"), value)
    return out


def run_profile(
    m: Measurer,
    module: BuiltModule,
    case: Case,
    shape: dict[str, Any],
    metrics: list[str] | None = None,
) -> tuple[str, Profile]:
    names = metrics or DEFAULT_METRICS
    problem = jobs.problem_of(m.store, case)
    sets = [jobs.random_inputs(problem, shape, "normal", m.rng("profile.inputs"))]
    wrap = [
        "{cuda_home}/bin/ncu",
        "--csv",
        "--metrics",
        ",".join(names),
        "--kernel-name",
        module.entry,
        "--launch-count",
        "1",
    ]
    requires = {"gpu": jobs.target_sm(m.store, case.target)}
    bench = jobs.bench_job(
        m.ex, m.store, [module], problem, shape, sets, "run", "profile", requires, wrap=wrap
    )
    result = m.ex.run(bench.job)
    stdout = m.ex.read_blob(result.stdout_blob).decode("utf-8", "replace")
    stderr = m.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")
    if NO_PERMISSION in stdout or NO_PERMISSION in stderr:
        raise NotSupported(
            f"profiling on worker {result.worker_name}: GPU performance counters are restricted "
            "to administrators (ncu ERR_NVGPUCTRPERM)"
        )
    if not result.ok:
        raise ExecutionError(
            "nonzero" if result.error_kind == "none" else result.error_kind,
            f"profile job {result.job_id} failed: {stderr[-2000:]}",
        )
    values = parse_ncu_csv(stdout, module.entry)
    if not values:
        raise ExecutionError("nonzero", f"ncu reported no metrics for kernel {module.entry}")
    body = Profile(metrics=values)
    meta = {"spec": bench.spec_blob, "metrics": names, "shape": shape}
    blobs = {
        "ncu_csv": result.stdout_blob,
        "stderr": result.stderr_blob,
        "params": m.ex.put_blob(json.dumps(meta, sort_keys=True).encode(), "application/json"),
    }
    refs: dict[str, str | list[str]] = {
        "env": result.env_manifest,
        "worker": result.worker,
        "build": module.build_id,
    }
    rec = Record.new("Profile", body, ACTOR, m.clock.now(), refs=refs, blobs=blobs)
    return m.store.put(rec), body


def profile(
    ctx: Context,
    module: BuiltModule,
    case: Case,
    shape: dict[str, Any],
    metrics: list[str] | None = None,
) -> Profile:
    return run_profile(Measurer.from_ctx(ctx), module, case, shape, metrics)[1]
