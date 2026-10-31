"""WP15 specific checks (spec/gate.md; spec/workplan.md WP15: Template effects and the
declarative hardware model). Kept apart from specific.py (500-line limit); registered there.

- The named WP15 tests passed (effects on every seed Template, upcast of schema-1 cost
  records, the GEMM seed chain priced by L1 with per-unit SOL, wall and peak changes
  without code, unpriced classes and unmodelled resources counted, regression set and
  interference job definitions, the migration's idempotence).
- Every seed Template has ``effects`` and no ``cost``; the migration script's ``--check``
  reports no change.
- The GB10 class and instance under knowledge/seed/hardware validate and resolve, and the
  instance gives provenance for every value.
- spec/schemas.md has §4.2.1 and §4.6.1; the models spec documents the L1 interface.

No real-run check: WP15 is a code-stage WP (D320); its real runs are listed in the
delivery note under "Pending real runs" and belong to the full gate (D473).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from kta_gate.result import CheckResult, from_findings

WP15_TESTS = (
    "test_every_seed_template_validates_with_effects",
    "test_schema_1_cost_record_upcasts_on_read_and_keeps_its_id",
    "test_gemm_seed_chain_priced_by_l1_with_per_unit_sol",
    "test_changing_one_peak_or_wall_changes_the_prediction",
    "test_unpriced_class_and_unmodelled_resource_are_counted_not_free",
    "test_pair_and_recurrence_bounds_and_serial_terms",
    "test_mip_rows_match_l1_amounts",
    "test_regression_set_needs_several_configurations_and_model_version_is_a_record",
    "test_interference_plan_defines_three_jobs_per_pair",
    "test_migration_is_idempotent_and_keeps_the_rest_of_the_file",
    "test_effects_without_architecture_class_are_refused",
)
SEED = Path("knowledge") / "seed"
MIGRATION = Path("scripts") / "migrations" / "WP15_effects.py"


def _templates(root: Path) -> list[str]:
    seed = root / SEED
    files = [*seed.glob("templates/*.yaml"), *seed.glob("references/*/templates/*.yaml")]
    findings: list[str] = []
    for path in sorted(files):
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "cost" in body or "effects" not in body:
            findings.append(f"{path.relative_to(root)} still has cost or lacks effects")
    if not files:
        findings.append("no seed Templates found")
    run = subprocess.run(
        [sys.executable, str(root / MIGRATION), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        findings.append(f"{MIGRATION} --check: {run.stdout.strip() or run.stderr.strip()}")
    return findings


def _hardware(root: Path) -> list[str]:
    from kta.core.schemas.hardware import HwModel
    from kta.core.schemas.hwmodel import HwArch
    from kta.models import MachineModel

    hw_dir = root / SEED / "hardware"
    archs = {}
    findings: list[str] = []
    for path in sorted(hw_dir.glob("arch_*.yaml")):
        try:
            arch = HwArch.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except ValueError as exc:
            findings.append(f"{path.name}: {str(exc).splitlines()[0]}")
            continue
        archs[arch.ref] = arch
    models = sorted(hw_dir.glob("model_*.yaml"))
    if not archs or not models:
        findings.append("knowledge/seed/hardware needs an arch_*.yaml and a model_*.yaml")
    for path in models:
        try:
            hw = HwModel.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            if hw.arch is None or hw.arch not in archs:
                findings.append(f"{path.name}: class {hw.arch} not in the seed")
                continue
            MachineModel.build(hw, archs[hw.arch])
        except ValueError as exc:
            findings.append(f"{path.name}: {str(exc).splitlines()[0]}")
    return findings


def _spec(root: Path) -> list[str]:
    schemas = (root / "spec" / "schemas.md").read_text(encoding="utf-8")
    models = (root / "spec" / "modules" / "formulation-solve-models.md").read_text("utf-8")
    need = [
        ("spec/schemas.md", schemas, "#### 4.2.1 Effects"),
        ("spec/schemas.md", schemas, "#### 4.6.1 Declarative hardware model"),
        ("spec/modules/formulation-solve-models.md", models, "### Hardware model and L1"),
    ]
    return [f"{where} lacks {text!r}" for where, doc, text in need if text not in doc]


def wp15(root: Path, passed: set[str]) -> list[CheckResult]:
    names = {t.split("::")[-1] for t in passed}
    absent = [f"{t} did not pass" for t in WP15_TESTS if t not in names]
    return [
        from_findings("WP15 tests", absent, f"{len(WP15_TESTS)} named tests passed"),
        from_findings(
            "seed Templates declare effects", _templates(root), "37 Templates, migration idempotent"
        ),
        from_findings("GB10 class and instance resolve", _hardware(root), "sm12x@1 + gb10@1"),
        from_findings("WP15 spec sections", _spec(root), "schemas §4.2.1, §4.6.1; L1 interface"),
    ]
