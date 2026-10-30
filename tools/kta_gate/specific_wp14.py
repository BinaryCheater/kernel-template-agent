"""WP14 specific checks (spec/gate.md; spec/workplan.md WP14: knowledge model v2). Kept
apart from specific.py, which is over the 500-line limit; registered there.

- The named WP14 tests passed (upcast of old records, the unprobed filter, promotion and
  demotion with stale propagation, pool diff, idempotent seed migration).
- The seed imports under the new schema into a scratch store, with Capabilities, GB10
  declarations and Realisations, and without errors.
- The seed migration reports nothing pending; no seed Template still has ``requires_hw``.
- Every object kind of 02 §2 is a registered record kind.
- The delivery note carries the design-interrogation artefacts and a pool-diff screenshot.
- Full stage only: a probe-backed CapabilityDecl and a non-empty ``probed_on`` from a real
  run; WARN in the code stage.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from kta_gate.result import CheckResult, from_findings

WP14_TESTS = (
    "test_old_template_reads_back_upcast_with_its_stored_id",
    "test_old_pattern_gap_and_reconstruction_levels_are_mapped",
    "test_a_newer_schema_than_the_code_is_refused",
    "test_unprobed_template_is_filtered_on_gb10_and_counted",
    "test_target_declaring_a_capability_wakes_the_template_up",
    "test_promotion_needs_the_evidence_of_its_step",
    "test_demotion_steps_down_and_propagates_stale",
    "test_only_the_system_demotes_and_never_up",
    "test_review_proposes_promotions_out_of_the_finding_layer",
    "test_pool_diff_between_two_snapshots",
    "test_seed_migration_is_idempotent",
)
# 02 §2 "Objects": Pattern, Template, Realisation, Capability (declared per Target), HwModel,
# HwFact, Finding, Chain, Family, Solution; plus the coupled_with relation (02 §8.4).
OBJECT_KINDS = (
    "Pattern",
    "Template",
    "Realisation",
    "Capability",
    "CapabilityDecl",
    "HwModel",
    "HwFact",
    "Finding",
    "Coupling",
    "Chain",
    "Family",
    "Solution",
)
MIGRATION = Path("scripts") / "migrations" / "WP14_knowledge_v2.py"
SCREENSHOT = Path("deliveries") / "WP14-pool-diff.png"


def _seed_imports(root: Path) -> list[str]:
    from kta.core.config import config_from_mapping
    from kta.core.context import Context
    from kta.core.errors import KtaError
    from kta.core.registry import Registry
    from kta.knowledge import import_references, import_seed
    from kta.knowledge.capabilities import target_capabilities
    from kta.store import CodeState, open_store

    with tempfile.TemporaryDirectory() as tmp:
        from datetime import UTC, datetime

        class _Clock:
            def now(self) -> datetime:
                return datetime(2026, 10, 31, tzinfo=UTC)

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
            first = import_seed(ctx, store, seed)
            import_references(ctx, store, seed / "references")
        except KtaError as exc:  # a SeedError lists the invalid records
            return [f"seed import failed: {str(exc).splitlines()[0][:300]}"]
        findings = [
            f"seed wrote no {k} record"
            for k in ("Capability", "CapabilityDecl", "Realisation")
            if not first.counts.get(k)
        ]
        caps = target_capabilities(store, "gb10")
        if not any(c.basis == "documented" for c in caps.values()):
            findings.append("no documented gb10 CapabilityDecl")
        return findings


def _migration(root: Path) -> list[str]:
    findings: list[str] = []
    proc = subprocess.run(
        [sys.executable, str(root / MIGRATION), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        findings.append(f"migration --check: {proc.stdout.strip() or proc.stderr.strip()[:300]}")
    seed = root / "knowledge" / "seed"
    for path in sorted(
        [*seed.glob("templates/*.yaml"), *seed.glob("references/*/templates/*.yaml")]
    ):
        if "requires_hw" in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}):
            findings.append(f"{path.relative_to(root)} still has requires_hw")
    return findings


def _kinds() -> list[str]:
    from kta.core.schemas import KINDS

    return [f"{k} is not a record kind" for k in OBJECT_KINDS if k not in KINDS]


def _delivery(root: Path) -> list[str]:
    note = root / "deliveries" / "WP14.md"
    if not note.exists():
        return [f"{note.relative_to(root)} is missing"]
    text = note.read_text(encoding="utf-8")
    findings = [
        f"WP14.md has no '{h}' section"
        for h in ("## Design interrogation", "## Pending real runs")
        if h not in text
    ]
    if not (root / SCREENSHOT).exists() or SCREENSHOT.name not in text:
        findings.append(f"{SCREENSHOT} missing or not referenced in WP14.md")
    return findings


def wp14(root: Path, passed: set[str]) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real

    names = {t.split("::")[-1].split("[")[0] for t in passed}
    results = [
        from_findings(
            "WP14 tests",
            [f"{t} did not pass" for t in WP14_TESTS if t not in names],
            f"{len(WP14_TESTS)} tests passed",
        ),
        from_findings(
            "WP14 seed under schema v2", _seed_imports(root), "seed and references import"
        ),
        from_findings("WP14 seed migration", _migration(root), "nothing pending"),
        from_findings("02 §2 object kinds", _kinds(), f"{len(OBJECT_KINDS)} kinds registered"),
        from_findings("WP14 delivery artefacts", _delivery(root), "interrogation and screenshot"),
    ]
    if code_stage():
        return [
            *results,
            pending_real(
                "WP14 probe-backed capabilities",
                "a CapabilityDecl with basis probed and a non-empty probed_on from a real run",
            ),
        ]
    return [
        *results,
        CheckResult(
            "WP14 probe-backed capabilities",
            "FAIL",
            "not implemented for the full stage yet: needs a real GB10 probe (D320)",
        ),
    ]
