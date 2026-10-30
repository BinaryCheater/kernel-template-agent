"""compute-sanitizer memcheck and racecheck on a small shape (docs/design/50 §3).

The bench job runs once under each tool with one normal input set. The verdict comes
from the tool's summary line; a run without a summary is a checker error, not a
kernel failure.
"""

from __future__ import annotations

import re
from typing import Any

from kta.backends.base import BuiltModule
from kta.checks import reference
from kta.checks.base import Checked, Checker, Evidence, stderr_tail, write_check
from kta.core.context import Context
from kta.core.ids import canonical_json
from kta.core.schemas.evaluation import Problem
from kta.measure import jobs

NAME = "sanitizer_checker"
TOOLS = ("memcheck", "racecheck")
SUMMARY = {
    "memcheck": re.compile(r"ERROR SUMMARY: (\d+) error"),
    "racecheck": re.compile(r"RACECHECK SUMMARY: \d+ hazards? displayed \((\d+) errors?"),
}
TIMEOUT_S = 900


def first_reports(text: str, limit: int = 5) -> list[str]:
    lines = [ln.removeprefix("========= ").strip() for ln in text.splitlines()]
    hits = [ln for ln in lines if ln.startswith(("Invalid", "Error", "Race", "Program hit"))]
    return hits[:limit]


def sanitizer_check(
    c: Checker, module: BuiltModule, problem: Problem, shape: dict[str, Any]
) -> Checked:
    ev = Evidence()
    rng = c.rng(f"checks.sanitizer:{canonical_json(shape).decode()}")
    sets = [jobs.random_inputs(problem, shape, "normal", rng)]
    sm = reference.gpu_target(c).sm
    for tool in TOOLS:
        wrap = ["{cuda_home}/bin/compute-sanitizer", "--tool", tool, "--print-limit", "20"]
        bench = jobs.bench_job(
            c.ex,
            c.store,
            [module],
            problem,
            shape,
            sets,
            "run",
            "check",
            {"gpu": sm},
            {"nan_fill": True},
            wrap=wrap,
            timeout_s=TIMEOUT_S,
        )
        result = c.ex.run(bench.job)
        ev.job(result, tool)
        text = c.ex.read_blob(result.stdout_blob).decode("utf-8", "replace")
        text += c.ex.read_blob(result.stderr_blob).decode("utf-8", "replace")
        match = SUMMARY[tool].search(text)
        if result.error_kind in ("infra",) or match is None:
            ev.error(
                f"sanitizer.{tool}",
                f"no {tool} summary ({result.error_kind}): {stderr_tail(c, result, 600)}",
            )
            continue
        errors = int(match.group(1))
        ev.detail[tool] = {"errors": errors, "exit_code": result.exit_code}
        if errors:
            ev.failures.append(
                {
                    "check": f"sanitizer.{tool}",
                    "shape": shape,
                    "errors": errors,
                    "reports": first_reports(text),
                }
            )
    return write_check(c, NAME, ev, module.build_id)


def check_sanitizer(
    ctx: Context, module: BuiltModule, problem: Problem, shape: dict[str, Any]
) -> Checked:
    return sanitizer_check(Checker.from_ctx(ctx), module, problem, shape)
