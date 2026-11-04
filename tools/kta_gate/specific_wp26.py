"""WP26 specific checks (spec/workplan.md "WP26", D680-D683): one hardware-model engine.

Static part (both stages): the named WP26 tests passed; the spec sections exist; the copied
simulator's README names its upstream commit and every copied file; every parameter has a
status; no code under src/ imports the upstream repository.

Real-run part: the L3 Executor job has not run yet, so the full stage fails until it has (WARN in
the code stage).
"""

from __future__ import annotations

import json
from pathlib import Path

from kta_gate.result import from_findings

WP26_TESTS = (
    "test_every_parameter_has_a_status",
    "test_code_constants_are_parameters",
    "test_reader_finds_the_function_and_its_loop",
    "test_l3_refuses_a_lowering_it_does_not_map",
)
SPEC_MARKS = (
    ("spec/workplan.md", "## WP26 "),
    ("spec/modules/formulation-solve-models.md", "models/gb10sim/l3.py (WP26, D683)"),
    ("docs/design/25-hardware-model-sources.md", "## 4. Direction after user review"),
    ("src/kta/models/gb10sim/README.md", "0eef4979390559f7cca72eac2b06e9cfc249ea30"),
)
COPIED = ("sass.py", "sm.py", "ws.py", "kernel.py", "model.py")
STATUSES = {"measured", "fitted", "structural", "assumed", "withdrawn"}


def check_tests(passed: set[str]):
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP26_TESTS if t not in names]
    return from_findings("WP26 tests", absent, f"{len(WP26_TESTS)} named tests passed")


def check_static(root: Path):
    findings = [
        f"{path}: missing {mark!r}"
        for path, mark in SPEC_MARKS
        if mark not in (root / path).read_text(encoding="utf-8")
    ]
    pkg = root / "src/kta/models/gb10sim"
    readme = (pkg / "README.md").read_text(encoding="utf-8")
    findings += [f"README does not list {f}" for f in COPIED if f"`{f}`" not in readme]
    raw = json.loads((pkg / "gb10.json").read_text(encoding="utf-8"))
    findings += [
        f"gb10.json {k}: status {v.get('status')!r}"
        for k, v in raw.items()
        if not k.startswith("_") and v.get("status") not in STATUSES
    ]
    for py in (root / "src").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "import agentic" in text or "from agentic" in text:
            findings.append(f"{py.relative_to(root)} imports the upstream package")
    return from_findings("WP26 provenance, parameters, spec", findings, "copy and parameters")


def wp26(root: Path, passed: set[str]):
    from kta_gate.specific import code_stage, pending_real

    results = [check_tests(passed), check_static(root)]
    if code_stage():
        what = "the L3 Executor job on a Build in dev/WP26, and a measured Build to compare"
        return [*results, pending_real("WP26 real L3 job", what)]
    return [*results, from_findings("WP26 real L3 job", ["the L3 Executor job is not built yet"])]
