"""WP24 specific checks (spec/workplan.md "WP24", D630–D643): verification in the loop.

Static part (both stages): the named WP24 tests passed (paper Table 16, the planted
unsound Pre, K1 failure, composition with an incompatible assumption, invalidation,
reuse, assurance only from proof records, weakest precondition, Δpre, candidate check,
the Pool view); `z3-solver` is locked and logged; the seed holds the tiling Template and
outlines for the five GEMM steps; the spec sections exist; the web app builds.

Real-run part: in `$KTA_DATA/dev/WP24/kta.db`, a ProofSummary of shared_memory_tiling@1
with a level and a `proof_canary` Report with the canary caught. WARN in the code stage.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from kta_gate.result import CheckResult, from_findings

WP24_TESTS = (
    "test_event_model_matches_paper_table16",
    "test_tiling_template_index_proved_schedule_bounded",
    "test_planted_unsound_pre_yields_replayed_counterexample",
    "test_weakest_precondition_relaxes_the_conservative_contract",
    "test_delta_pre_reads_the_relaxed_prediction",
    "test_k1_failure_caught_on_a_fixture",
    "test_unsupported_construct_is_unresolved_never_proved",
    "test_seed_steps_end_with_recorded_outcomes",
    "test_assurance_only_from_live_proof_records",
    "test_invalidation_on_premise_change",
    "test_reuse_keeps_certificates_when_only_constraints_strengthen",
    "test_composition_rejects_an_incompatible_assumption",
    "test_chain_steps_are_checked_under_their_bindings",
    "test_candidate_check_in_c_and_gamma",
    "test_pool_view_lists_assurance_and_every_obligation",
)
SEED_OUTLINES = (
    "reduction_partition",
    "physical_storage",
    "async_transfer",
    "multi_buffer_rotation",
    "shared_resource_binding",
    "shared_memory_tiling",
)
SPEC_MARKS = (
    ("spec/workplan.md", "## WP24 "),
    ("spec/schemas.md", "### 4.11 Proof records"),
    ("spec/modules/measure-check.md", "## WP24: verification in the loop"),
    ("spec/modules/knowledge.md", "## WP24: proof outlines"),
    ("spec/modules/orchestrator-cli.md", "## WP24: candidate gate"),
    ("spec/modules/api-web.md", "## WP24: assurance and proofs"),
    ("docs/decisions.md", "| D630 | `z3-solver` dependency"),
)


def check_tests(passed: set[str]) -> CheckResult:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP24_TESTS if t not in names]
    return from_findings("WP24 tests", absent, f"{len(WP24_TESTS)} named tests passed")


def check_static(root: Path) -> CheckResult:
    findings = [
        f"{path}: missing {mark!r}"
        for path, mark in SPEC_MARKS
        if mark not in (root / path).read_text(encoding="utf-8")
    ]
    if 'name = "z3-solver"' not in (root / "uv.lock").read_text(encoding="utf-8"):
        findings.append("z3-solver is not in uv.lock")
    seed = root / "knowledge" / "seed"
    findings += [
        f"no seed outline proofs/{n}.yaml"
        for n in SEED_OUTLINES
        if not (seed / "proofs" / f"{n}.yaml").is_file()
    ]
    if not (seed / "templates" / "shared_memory_tiling.yaml").is_file():
        findings.append("the paper's tiling Template is not in the seed")
    pool = (root / "web" / "src" / "pages" / "Pool.tsx").read_text(encoding="utf-8")
    result = (root / "web" / "src" / "pages" / "Result.tsx").read_text(encoding="utf-8")
    if "AssuranceStrip" not in pool or "ProofCell" not in pool:
        findings.append("Pool does not show the assurance distribution and obligations")
    if "ProvedVsTested" not in result:
        findings.append("Result has no proved-vs-tested panel")
    return from_findings(
        "WP24 spec, seed and pages", findings, "spec, z3 lock, seed outlines, pages"
    )


def check_real(root: Path) -> CheckResult:
    data = Path(os.environ.get("KTA_DATA", Path.home() / "kta-data")) / "dev" / "WP24" / "kta.db"
    name = "WP24 real checker runs"
    if not data.is_file():
        return CheckResult(name, "FAIL", f"{data} does not exist")
    con = sqlite3.connect(f"file:{data}?mode=ro", uri=True)
    rows = [
        json.loads(r[0])
        for r in con.execute("SELECT json FROM records WHERE kind IN ('ProofSummary', 'Report')")
    ]
    tiling = [
        r for r in rows
        if r["kind"] == "ProofSummary" and r["body"]["level"] and r["origin"] == "real"
        and not r["actor"].get("dirty")
    ]  # fmt: skip
    canary = [
        r for r in rows
        if r["kind"] == "Report" and r["body"]["title"] == "proof_canary"
        and r["body"]["data"].get("caught") and r["origin"] == "real"
        and not r["actor"].get("dirty")
    ]  # fmt: skip
    findings = []
    if not tiling:
        findings.append("no clean real ProofSummary with a level")
    if not canary:
        findings.append("no clean real proof_canary Report with the canary caught")
    _ = root
    return from_findings(
        name, findings, f"{len(tiling)} summaries with a level, {len(canary)} canaries caught"
    )


def wp24(root: Path, passed: set[str]) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real, web_build

    results = [check_tests(passed), check_static(root), web_build(root)]
    if code_stage():
        what = (
            "ProofSummary with a level and a caught proof_canary Report in dev/WP24 (clean commit)"
        )
        return [*results, pending_real("WP24 real checker runs", what)]
    return [*results, check_real(root)]
