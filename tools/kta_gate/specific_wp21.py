"""WP21 specific checks (spec/gate.md "WP21", D489): controls registry and staged release.

Static part (both stages): the named WP21 tests passed; the registry loads and holds every
row of docs/design/12 with a matching `set_by` cell; X1–X9 are defined; the knob constants
the registry replaced are gone from `src/`; stages S0–S4 are defined with exit criteria; the
Settings and Release pages are routed and the web build passes.

Real-run part: a hand-stepped S1 run with a real agent (a released call, its AgentRun naming
the controls version). WARN in the code stage (D320); FAIL in the full gate until WP21's
real-run follow-up adds the record check.
"""

from __future__ import annotations

import re
from pathlib import Path

from kta_gate.result import CheckResult, from_findings

WP21_TESTS = (
    "test_registry_holds_every_row_of_12_with_set_by",
    "test_every_knob_has_one_typed_accessor",
    "test_experiments_x1_to_x9_are_defined_with_the_knobs_they_set",
    "test_d325_defaults_are_registry_values",
    "test_settings_are_a_view_of_the_controls",
    "test_invalid_values_and_invariants_are_refused",
    "test_moved_configuration_keys_are_refused",
    "test_a_value_change_is_a_new_version_with_author_and_reason",
    "test_definitions_upgrade_writes_a_system_version",
    "test_task_card_and_loop_record_the_controls_version",
    "test_s0_runs_no_agent_calls_and_skips_processes_outside_it",
    "test_s1_gate_holds_a_fake_adapter_task_until_released",
    "test_s1_runs_one_agent_call_at_a_time",
    "test_stage_change_is_a_controls_version",
    "test_settings_api_reads_open_and_writes_a_version_with_the_token",
    "test_release_api_shows_stages_and_releases_a_held_call",
    "test_operator_token_file_is_required_for_writes",
    "test_value_change_is_versioned_and_visible_in_the_agent_run",
    "test_a_task_token_cannot_write_a_controls_version",
)
# Knob constants the registry replaced (D482, D483): none may come back in src/.
REPLACED = (
    (r"^\s*LEASE_S\s*=", "LEASE_S constant (budget.lease_s)"),
    (r"^\s*RENEW_EVERY_S\s*=", "RENEW_EVERY_S constant (lease_s / 6)"),
    (r"DEFAULT_RESET_RATIO", "DEFAULT_RESET_RATIO (session.fresh_session_threshold)"),
    (r"^\s*BUDGET_WARN\s*=", "BUDGET_WARN constant (inbox.budget_warn)"),
    (r"^\s*REPEAT_FAILURES\s*=", "REPEAT_FAILURES constant (inbox.repeat_failures)"),
    (r"^\s*QUOTAS\s*=", "QUOTAS constant (budget.process_quotas)"),
    (r"STEP_HEADER_MAX_CHARS", "STEP_HEADER_MAX_CHARS (session.step_header_size)"),
    (r"\bSettings\.from_config", "Settings.from_config (Settings.of(controls))"),
    (r"^\s*max_rounds:\s*\d", "a numeric max_rounds in machines.yaml"),
    (r"^\s*max_retries:\s*\d", "a numeric max_retries in machines.yaml"),
)


def check_tests(passed: set[str]) -> CheckResult:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP21_TESTS if t not in names]
    return from_findings("WP21 tests", absent, f"{len(WP21_TESTS)} named tests passed")


def _doc_rows(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    text = (root / "docs" / "design" / "12-agent-controls.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith(("| Knob", "| ---")):
            cells = [c.strip() for c in line.strip("|").split(" | ")]
            out[cells[0]] = cells[-1]
    return out


def check_registry(root: Path) -> CheckResult:
    from kta.core.controls import definitions

    findings: list[str] = []
    defs = definitions()
    rows = _doc_rows(root)
    by_row = {k.row: k for k in defs.knobs}
    findings += [f"row of 12 without a knob: {r}" for r in sorted(set(rows) - set(by_row))]
    findings += [f"knob without a row in 12: {r}" for r in sorted(set(by_row) - set(rows))]
    for row, cell in rows.items():
        k = by_row.get(row)
        if k is not None and cell != (", ".join(k.set_by) or "—"):
            findings.append(f"{row}: set_by cell {cell!r} != registry {list(k.set_by)}")
    ids = [e.id for e in defs.experiments]
    if ids != [f"X{i}" for i in range(1, 10)]:
        findings.append(f"experiments are {ids}, not X1–X9")
    return from_findings("WP21 registry", findings, f"{len(defs.knobs)} knobs, X1–X9, 12 matches")


def check_replaced(root: Path) -> CheckResult:
    findings: list[str] = []
    files = [
        *root.joinpath("src", "kta").rglob("*.py"),
        *root.joinpath("src", "kta").rglob("*.yaml"),
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern, what in REPLACED:
            for m in re.finditer(pattern, text, re.M):
                line = text.count("\n", 0, m.start()) + 1
                findings.append(f"{path.relative_to(root)}:{line}: {what}")
    return from_findings("WP21 no scattered knob constants", findings, "none found")


def check_stages() -> CheckResult:
    from kta.orchestrator import release

    stages = release.stages()
    findings = [] if [s.id for s in stages] == ["S0", "S1", "S2", "S3", "S4"] else ["not S0–S4"]
    findings += [f"{s.id} has no exit criteria" for s in stages[:4] if not s.exit]
    if stages[1].agent_calls != "gated" or stages[0].agent_calls != "none":
        findings.append("S0 must run no agent calls and S1 must gate them")
    return from_findings("WP21 stages", findings, "S0–S4 with exit criteria")


def check_pages(root: Path) -> CheckResult:
    app = (root / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    findings = [
        f"page {p} is not routed" for p in ('path="/settings"', 'path="/release"') if p not in app
    ]
    for name in ("Settings.tsx", "Release.tsx"):
        if not (root / "web" / "src" / "pages" / name).exists():
            findings.append(f"web/src/pages/{name} missing")
    return from_findings("WP21 pages", findings, "/settings and /release routed")


def wp21(root: Path, passed: set[str]) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real, web_build

    results = [
        check_tests(passed),
        check_registry(root),
        check_replaced(root),
        check_stages(),
        check_pages(root),
        web_build(root),
    ]
    name = "WP21 hand-stepped S1 run"
    what = "a released S1 call with a real agent; its AgentRun names the controls version"
    if code_stage():
        return [*results, pending_real(name, what)]
    return [*results, CheckResult(name, "FAIL", f"needs a real run: {what}")]
