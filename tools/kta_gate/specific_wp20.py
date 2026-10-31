"""WP20 specific checks (spec/gate.md; spec/workplan.md WP20: learning-loop runner).
Kept apart from specific.py, which is over the 500-line limit; registered there.

- The named WP20 tests passed (admit with counted language gaps, unexplained warning
  refuses, T2 rejection back to the same fake-adapter session, coverage levels, ablation,
  dry-run plan).
- The seven WP10a decompositions import through admit into a scratch store (catalog and
  seed first, as ``kta init`` does): all admitted, language gaps counted, coverage shows
  the four SOTA levels and a system-feature count.
- The learning table has tiers L0-L4 with resolvable rules, four SOTA levels, the O12
  protocol (defined, not run), and T2 runs the ``admission`` done-check.
- The delivery note carries the design-interrogation artefacts and the pending real runs.
- Full stage only: real intake / reproduce / ablate runs; WARN in the code stage.
"""

from __future__ import annotations

import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from kta_gate.result import CheckResult, from_findings

WP20_TESTS = (
    "test_seven_decompositions_admitted_with_counted_language_gaps",
    "test_an_unexplained_warning_refuses_the_batch",
    "test_language_gaps_are_classified_from_the_table",
    "test_system_features_tiers_priority_and_core",
    "test_ablation_design_and_coupling_evidence",
    "test_fractional_design_when_large",
    "test_dry_run_plan_of_a_runnable_reference",
    "test_t2_admission_rejection_goes_back_to_the_same_session",
)


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 10, 31, tzinfo=UTC)


def _admit_seven(root: Path) -> tuple[list[str], str]:
    from kta.core.config import config_from_mapping
    from kta.core.context import Context
    from kta.core.errors import KtaError
    from kta.core.registry import Registry
    from kta.knowledge import coverage_report, import_hw_catalog, import_seed
    from kta.knowledge.admission import admit_references
    from kta.knowledge.coverage import coverage_page
    from kta.store import CodeState, open_store
    from kta.targets.gb10 import catalog

    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(
            env="test",
            config=config_from_mapping({}),
            data_dir=Path(tmp),
            registry=Registry(),
            clock=_Clock(),
        )
        store = open_store(ctx, CodeState("0" * 40, False))
        seed = root / "knowledge" / "seed"
        try:
            import_hw_catalog(ctx, store, catalog())
            import_seed(ctx, store, seed)
            result, reports = admit_references(ctx, store, seed / "references")
        except KtaError as exc:
            return [f"admit failed: {str(exc).splitlines()[0][:300]}"], ""
        findings: list[str] = []
        if result.counts.get("Reconstruction") != 7 or len(reports) != 7:
            findings.append(f"expected 7 admitted references, got {result.counts}")
        if not result.language_gaps:
            findings.append("no language gap counted")
        page = coverage_page(coverage_report(store, None, random.Random(0)), {})
        learning = page.get("learning", {})
        levels = [r["name"] for r in learning.get("sota_levels", [])]
        if levels != ["expressible", "reproducible", "reachable", "surpassed"]:
            findings.append(f"coverage SOTA levels are {levels}")
        sys_total = learning.get("system_features", {}).get("total")
        if not sys_total:
            findings.append("coverage has no system-feature count")
        return findings, (
            f"7 references admitted, {len(result.language_gaps)} language gaps, "
            f"{sys_total} system features"
        )


def _tables() -> list[str]:
    from kta.knowledge.learning import TIER_RULES, o12_protocol, tables
    from kta.orchestrator.done_checks import DONE_CHECKS

    t = tables()
    findings = [
        f"tier rule {r['enter']!r} unknown" for r in t["tiers"] if r["enter"] not in TIER_RULES
    ]
    if [r["id"] for r in t["tiers"]] != ["L0", "L1", "L2", "L3", "L4"]:
        findings.append("tiers are not L0-L4")
    if [r["level"] for r in t["sota_levels"]] != [1, 2, 3, 4]:
        findings.append("SOTA levels are not 1-4")
    if "not run" not in str(o12_protocol().get("status", "")):
        findings.append("O12 protocol status does not say it is not run")
    if "admission" not in DONE_CHECKS["T2"]:
        findings.append("T2 does not run the admission done-check")
    return findings


def _delivery(root: Path) -> list[str]:
    note = root / "deliveries" / "WP20.md"
    if not note.exists():
        return [f"{note.relative_to(root)} is missing"]
    text = note.read_text(encoding="utf-8")
    return [
        f"WP20.md has no '{h}' section"
        for h in ("## Design interrogation", "## Pending real runs")
        if h not in text
    ]


def wp20(root: Path, passed: set[str]) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real

    names = {t.split("::")[-1].split("[")[0] for t in passed}
    admit_findings, admit_detail = _admit_seven(root)
    results = [
        from_findings(
            "WP20 tests",
            [f"{t} did not pass" for t in WP20_TESTS if t not in names],
            f"{len(WP20_TESTS)} tests passed",
        ),
        from_findings("WP20 admit of the seven references", admit_findings, admit_detail),
        from_findings("WP20 learning tables", _tables(), "tiers, levels, O12, T2 admission"),
        from_findings("WP20 delivery artefacts", _delivery(root), "interrogation, pending runs"),
    ]
    owed = "intake, reproduce and ablate run for real on spark (GB10); O12 blind reconstruction"
    if code_stage():
        return [*results, pending_real("WP20 real learning-loop runs", owed)]
    return [
        *results,
        CheckResult("WP20 real learning-loop runs", "FAIL", f"not implemented yet: {owed}"),
    ]
