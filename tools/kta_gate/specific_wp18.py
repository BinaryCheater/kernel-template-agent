"""WP18 specific checks (spec/gate.md "WP18", D530): gap decomposition, routing and the
loop disciplines. Kept apart from specific.py (500-line limit); registered there.

- The named WP18 tests passed (the O5 fault-injection harness among them).
- The routing table loads; every class of its attribution order has a route; every route's
  (rule, target) is a judging edge of machines.yaml; every canary expects a routed class.
- Thresholds come from the controls registry: the WP18 knobs no longer say "pending WP18",
  and the WP18 modules hold no fractional literal other than 0.0 and 1.0.
- spec/schemas.md describes GapDecomposition; the orchestrator spec has the WP18 section.

No real-run check: WP18 is a code-stage WP (D320); its real runs are listed in the delivery
note under "Pending real runs".
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from kta_gate.result import CheckResult, from_findings

WP18_TESTS = (
    "test_fault_injection_routes_to_the_right_class",
    "test_the_base_bucket_is_accepted",
    "test_terms_sum_to_the_gap_and_sol_bounds_l_rep",
    "test_unassessed_terms_are_labelled_not_zeroed_silently",
    "test_mip_gap_bounds_delta_and_attribution_stops_at_it",
    "test_routing_table_is_versioned_and_recorded",
    "test_accept_rule_reads_the_registry",
    "test_mechanisms_count_coupled_templates_once",
    "test_gain_is_credited_only_beyond_the_band_and_attributed",
    "test_a_regressing_round_is_rolled_back_to_the_incumbent",
    "test_model_error_after_calibration_escalates_to_knowledge",
    "test_selection_controls_are_recorded_per_bucket",
    "test_selection_arms_draw_random_feasible_points_and_copy_a_solution",
    "test_an_identical_resubmission_is_rejected_without_running_checks",
    "test_canaries_rotate_kinds_and_are_caught",
    "test_a_missed_canary_escalates_to_the_user",
    "test_no_decomposition_means_no_base_not_a_pass",
    "test_summarize_counts_routed_classes_and_escalates_recurring_ones",
    "test_judging_rule_5_uncertified_solver_closes_and_escalates",
    "test_judging_rule_6_null_rounds_close_the_case",
)
ORCH = Path("src") / "kta" / "orchestrator"
MODULES = ("gap.py", "disciplines.py", "canary.py")
KNOBS = (
    "guard.improvement_band",
    "guard.mechanism_attribution",
    "guard.null_round_limit",
    "guard.one_mechanism_per_round",
    "guard.exploration_share",
    "guard.duplicate_proposal_rejection",
    "guard.selection_controls",
    "guard.canary_rate",
    "escalation.accept",
    "escalation.case_to_knowledge",
    "escalation.to_user",
)
FRACTION = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")


def _routing(root: Path) -> list[str]:
    table = yaml.safe_load((root / ORCH / "routing.yaml").read_text(encoding="utf-8"))
    machines = yaml.safe_load((root / ORCH / "machines.yaml").read_text(encoding="utf-8"))
    solve = next(m for m in machines["machines"] if m["id"] == "solve")
    edges = {(t["rule"], t["to"]) for t in solve["transitions"] if t["from"] == "judging"}
    routes = {**table["classes"], **table.get("escalations", {})}
    findings = [f"class {c} has no route" for c in table["order"] if c not in table["classes"]]
    for name, r in routes.items():
        if (r["rule"], r["to"]) not in edges:
            findings.append(f"route {name}: rule {r['rule']} -> {r['to']} is not a judging edge")
    for c in table.get("canaries", []):
        if c["expect"] not in table["classes"]:
            findings.append(f"canary {c['kind']} expects unrouted class {c['expect']}")
    if not table.get("canaries"):
        findings.append("routing.yaml defines no canary")
    return findings


def _registry(root: Path) -> list[str]:
    raw = yaml.safe_load((root / "src" / "kta" / "core" / "controls.yaml").read_text("utf-8"))
    knobs = {k["id"]: k for k in raw["knobs"]}
    findings = [
        f"{k} still says {knobs[k]['consumer']!r}"
        for k in KNOBS
        if k not in knobs or "pending WP18" in str(knobs[k].get("consumer", ""))
    ]
    for name in MODULES:
        for n, line in enumerate((root / ORCH / name).read_text("utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if code.strip().startswith(('"', "'")):
                continue
            hits = [x for x in FRACTION.findall(code) if x not in ("0.0", "1.0")]
            if hits:
                findings.append(
                    f"{ORCH / name}:{n}: literal {hits} (a knob belongs in the registry)"
                )
    return findings


def _spec(root: Path) -> list[str]:
    schemas = (root / "spec" / "schemas.md").read_text(encoding="utf-8")
    orch = (root / "spec" / "modules" / "orchestrator-cli.md").read_text(encoding="utf-8")
    need = [
        ("spec/schemas.md", schemas, "GapDecomposition"),
        ("spec/modules/orchestrator-cli.md", orch, "## WP18: gap decomposition"),
    ]
    return [f"{where} lacks {text!r}" for where, doc, text in need if text not in doc]


def wp18(root: Path, passed: set[str]) -> list[CheckResult]:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP18_TESTS if t not in names]
    return [
        from_findings("WP18 tests", absent, f"{len(WP18_TESTS)} named tests passed"),
        from_findings("WP18 routing table", _routing(root), "routes match judging edges"),
        from_findings("WP18 thresholds in the registry", _registry(root), f"{len(KNOBS)} knobs"),
        from_findings("WP18 spec sections", _spec(root), "GapDecomposition; orchestrator WP18"),
    ]
