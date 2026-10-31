"""WP23 specific checks (spec/gate.md "WP23", D600–D619): modes, host lock, quiet host,
stage queues and role work units.

Static part (both stages): the named WP23 tests passed (the contract tests for host-lock
exclusion across processes, writer preference, interfered measurement requeue, WIP
backpressure and work-unit roles among them); both mode files exist and name one compute
host with a host lock; the role table is kernel T1/T6, model T4/T5, knowledge T2/T3; the
spec sections exist; the web app builds.

Real-run part: a measurement window under the host lock on spark-hk with quiet-host
verdicts recorded. WARN in the code stage (D320).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from kta_gate.result import CheckResult, from_findings

WP23_TESTS = (
    "test_host_lock_excludes_across_two_processes",
    "test_writer_preference_blocks_new_readers",
    "test_heavy_takes_the_lock_shared_only_on_the_compute_host",
    "test_cpu_job_waits_for_another_control_planes_measurement",
    "test_unified_memory_cap_defers_cpu_jobs",
    "test_interfered_measurement_is_requeued_and_not_counted",
    "test_a_host_that_never_quiets_raises_and_writes_no_measurement",
    "test_wip_backpressure_holds_upstream_when_downstream_is_full",
    "test_measurement_window_collects_until_full_or_due",
    "test_dequeue_prefers_work_that_unblocks_waiting_cases",
    "test_work_unit_roles",
    "test_cards_of_one_loop_split_by_role",
    "test_workflow_shows_the_work_unit_rule",
    "test_stage_view_derives_depth_gpu_busy_and_lock_waits",
    "test_backup_restore_and_handover",
)
ROLES = {"kernel": ["T1", "T6"], "model": ["T4", "T5"], "knowledge": ["T2", "T3"]}
SPEC_MARKS = (
    ("spec/workplan.md", "## WP23 "),
    ("spec/modules/target-exec.md", "## WP23: host lock"),
    ("spec/modules/orchestrator-cli.md", "## WP23: stage queues"),
    ("spec/modules/agents.md", "## WP23: where agents run"),
    ("spec/modules/api-web.md", "## WP23: stage timeline"),
    ("spec/modules/measure-check.md", "## WP23: quiet host"),
    ("docs/design/11-processes-resources-concurrency.md", "### 4.3 Stage queues"),
)


def check_tests(passed: set[str]) -> CheckResult:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP23_TESTS if t not in names]
    return from_findings("WP23 tests", absent, f"{len(WP23_TESTS)} named tests passed")


def check_modes(root: Path) -> CheckResult:
    findings: list[str] = []
    for mode, transport in (("dev-mac", "ssh"), ("spark-hk", "local")):
        path = root / "config" / "modes" / f"{mode}.toml"
        if not path.is_file():
            findings.append(f"{path} missing")
            continue
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        hk = cfg.get("workers", {}).get("spark-hk", {})
        if (
            hk.get("transport") != transport
            or not hk.get("host_lock")
            or not hk.get("unified_memory")
        ):
            findings.append(
                f"{mode}: [workers.spark-hk] must be {transport}, host_lock, unified_memory"
            )
        gpu = [
            n for n, w in cfg.get("workers", {}).items() if "gpu-exclusive" in w.get("pools", {})
        ]
        if gpu != ["spark-hk"]:
            findings.append(f"{mode}: GPU workers {gpu}, expected only spark-hk")
    from kta.orchestrator.workunits import roles

    if {k: list(v) for k, v in roles().items()} != ROLES:
        findings.append(f"role table {roles()} != {ROLES}")
    return from_findings("WP23 modes and roles", findings, "two modes, one compute host, roles")


def check_docs(root: Path) -> CheckResult:
    findings = [
        f"{path}: missing {mark!r}"
        for path, mark in SPEC_MARKS
        if mark not in (root / path).read_text(encoding="utf-8")
    ]
    if "PipelineStages" not in (root / "web/src/pages/Pipeline.tsx").read_text(encoding="utf-8"):
        findings.append("the Pipeline page does not show the stage queues")
    return from_findings("WP23 spec and pages", findings, "spec sections, Pipeline stage view")


def wp23(root: Path, passed: set[str]) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real, web_build

    results = [check_tests(passed), check_modes(root), check_docs(root), web_build(root)]
    name = "WP23 measurement window under the host lock"
    what = "Measurements on spark-hk taken in a window with quiet-host verdicts recorded"
    if code_stage():
        return [*results, pending_real(name, what)]
    return [*results, CheckResult(name, "FAIL", f"needs a real run: {what}")]
