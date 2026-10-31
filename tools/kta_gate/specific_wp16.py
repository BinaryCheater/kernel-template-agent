"""WP16 specific checks (spec/gate.md; spec/workplan.md WP16: transaction-level simulator,
what-if and regression scoreboard). Kept apart from specific.py (500-line limit); registered
there (D512).

- The named WP16 tests passed (the GEMM seed family at L2 with a timeline, L1 as the
  steady-state limit of L2 on a loop-dominated fixture, a non-zero shadow price on a walled
  fixture, the L3 adapter's NotSupported, scoring and the scoreboard on fixture data).
- No code under src/kta imports the research repository's simulator; the L3 adapter only
  names it as an external tool.
- The models spec documents L2, what-if, L3 and the regression scoreboard; the API spec
  lists the two model views; the web app routes /model/scoreboard.
- The admission knob is in the controls registry; the scoreboard fixture is marked.

No real-run check: WP16 is a code-stage WP (D320); its real runs are listed in the delivery
note under "Pending real runs" and belong to the full gate.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from kta_gate.result import CheckResult, from_findings

WP16_TESTS = (
    "test_gemm_seed_family_simulated_at_l2_with_a_timeline",
    "test_the_l2_result_is_stored_as_a_report",
    "test_l1_equals_the_steady_state_limit_of_l2_on_a_loop_dominated_fixture",
    "test_with_shallow_pipelines_l2_exceeds_l1_by_the_recurrence",
    "test_what_if_shows_a_non_zero_shadow_price_on_a_walled_fixture",
    "test_what_if_on_a_peak_and_on_an_unknown_name",
    "test_a_no_launch_wall_is_infeasible_not_a_cost",
    "test_unpriced_operation_classes_are_carried_into_l2",
    "test_l3_adapter_defines_an_executor_job_and_is_not_supported_until_run",
    "test_spearman_ties_and_degenerate_inputs",
    "test_scoring_errors_per_resource_and_admission",
    "test_scoreboard_renders_on_fixture_data_marked_fixture",
)
EXTERNAL = re.compile(r"^\s*(?:import|from)\s+(?:calibration_tools|simulate|replay|gb10_sim)", re.M)


def _no_external_import(root: Path) -> list[str]:
    hits = []
    for path in sorted((root / "src" / "kta").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if EXTERNAL.search(text) or ("sys.path.insert" in text and "kernel-lab" in text):
            hits.append(f"{path.relative_to(root)} imports the external simulator")
    return hits


def _docs(root: Path) -> list[str]:
    def read(rel: str) -> str:
        return (root / rel).read_text(encoding="utf-8")

    need = [
        ("spec/modules/formulation-solve-models.md", "### L2 transaction-level simulator"),
        (
            "spec/modules/formulation-solve-models.md",
            "### What-if, L3 and the regression scoreboard",
        ),
        ("spec/modules/api-web.md", "/model/scoreboard"),
        ("spec/modules/api-web.md", "/model/versions/{id}/timeline"),
        ("web/src/App.tsx", 'path="/model/scoreboard"'),
        ("tools/kta_devseed/models.py", "# kta:fixture"),
    ]
    return [f"{rel} lacks {text!r}" for rel, text in need if text not in read(rel)]


def _knob(root: Path) -> list[str]:
    doc = yaml.safe_load((root / "src" / "kta" / "core" / "controls.yaml").read_text("utf-8"))
    knobs = doc["knobs"] if isinstance(doc, dict) else doc
    ids = {k["id"] for k in knobs}
    return [] if "model.version_admission" in ids else ["model.version_admission not in registry"]


def wp16(root: Path, passed: set[str]) -> list[CheckResult]:
    names = {t.split("::")[-1] for t in passed}
    absent = [f"{t} did not pass" for t in WP16_TESTS if t not in names]
    return [
        from_findings("WP16 tests", absent, f"{len(WP16_TESTS)} named tests passed"),
        from_findings("L3 simulator not imported", _no_external_import(root), "external tool only"),
        from_findings(
            "WP16 spec, views and fixture marks", _docs(root), "models + api spec, route"
        ),
        from_findings("admission knob registered", _knob(root), "model.version_admission"),
    ]
