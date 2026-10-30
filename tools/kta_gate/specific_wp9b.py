"""WP9b specific checks (spec/gate.md: web build, API tests, screenshots; spec/workplan.md WP9b).

Kept apart from specific.py, which is at the 500-line limit; registered there.

- The named API tests passed: every operation writes a correct HumanAction, a missing
  required reason is a 400, the WP6/WP8 operations go through the real task queue, profile
  pointer, batch verify and task tokens, and operations not wired yet are a 501 that writes
  nothing.
- Every operation endpoint of spec/modules/api-web.md is exercised by the action tests.
- The web build of HEAD in a clean directory passes; web/src holds no sample data.
- Full stage only (D346): real HumanAction records in ``dev/WP9b`` for the three operations the
  workplan names (start a T1, cancel a task, reject a result), listed in the evidence manifest
  and written by a clean commit of this branch; the delivery note shows screenshots.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kta_gate.result import CheckResult, from_findings

WP9B_TESTS = (
    "test_every_spec_operation_is_served",
    "test_reject_writes_human_action_and_marks_dependents_stale",
    "test_reject_and_deprecate_need_a_reason",
    "test_deprecate_does_not_propagate",
    "test_flag_marks_suspect_and_reports_the_missing_reverification",
    "test_budget_needs_reason_and_merges_limits",
    "test_approve_protocol_and_improvement",
    "test_confirm_pattern",
    "test_cancel_goes_through_the_task_queue",
    "test_retry_requeues_an_escalated_task",
    "test_create_task_queues_a_human_task",
    "test_priority_is_not_supported_yet",
    "test_pause_and_resume_are_not_supported_yet",
    "test_approve_profile_moves_the_pointer",
    "test_unwired_port_answers_501_for_every_delegated_operation",
    "test_capabilities_say_what_is_wired",
    "test_write_record_with_a_valid_task_token",
    "test_batch_verify_regenerates_and_compares",
    "test_skill_catalogue_comes_from_the_agents_readme",
    "test_inbox_comes_from_the_orchestrator",
    "test_task_detail_of_a_running_task",
    "test_task_detail_lists_rejections",
    "test_agents_lists_runtime_skills_profiles_and_batches",
)
SPEC_OPERATIONS = (
    "pause", "resume", "cancel", "retry", "budget", "priority", "create-task", "reject", "flag",
    "approve", "confirm",
)  # fmt: skip
ACTION_TESTS = Path("tests/unit/test_api_actions.py")
# HumanAction ``action`` prefixes for the workplan's three real operations. The API writes
# ``reject ...``; WP6's TaskQueue writes ``task cancel <id>`` (and, once wired, the create).
REAL_OPERATIONS: dict[str, tuple[str, ...]] = {
    "start a task": ("create_task", "task create"),
    "cancel a task": ("cancel", "task cancel"),
    "reject a result": ("reject",),
}
SCREENSHOT = re.compile(r"\S+\.png\b")


def _names(passed: set[str]) -> set[str]:
    return {t.split("::")[-1].split("[")[0] for t in passed}


def _covered(root: Path) -> list[str]:
    path = root / ACTION_TESTS
    if not path.exists():
        return [f"{ACTION_TESTS} is missing"]
    text = path.read_text(encoding="utf-8")
    return [
        f"/actions/{op} is not exercised in {ACTION_TESTS}"
        for op in SPEC_OPERATIONS
        if f"/actions/{op}" not in text and f'"{op}"' not in text
    ]


def _sample_data(root: Path) -> list[str]:
    from kta_gate.specific import WEB_FIXTURE

    return [
        f"{p.relative_to(root)} contains sample-data markers"
        for p in sorted((root / "web" / "src").rglob("*"))
        if p.is_file() and WEB_FIXTURE.search(p.read_text(encoding="utf-8", errors="replace"))
    ]


def real_operations(
    con: sqlite3.Connection, listed: set[str], clean: Callable[[str], bool]
) -> list[str]:
    """Findings for the workplan's three operations among the listed, real HumanActions."""
    found: dict[str, bool] = dict.fromkeys(REAL_OPERATIONS, False)
    for rid in sorted(listed):
        row = con.execute(
            "SELECT origin, json FROM records WHERE id = ? AND kind = 'HumanAction'", (rid,)
        ).fetchone()
        if row is None or row[0] != "real":
            continue
        rec: dict[str, Any] = json.loads(row[1])
        actor = rec["actor"]
        if actor.get("type") != "human" or actor.get("dirty") is not False:
            continue
        if not clean(str(actor.get("commit"))):
            continue
        action = str(rec["body"]["action"])
        for what, prefixes in REAL_OPERATIONS.items():
            if action.startswith(prefixes):
                found[what] = True
    return [f"no listed, real, clean HumanAction to {what}" for what, ok in found.items() if not ok]


def _full_stage(root: Path, data_root: Path | None) -> list[CheckResult]:
    from kta_gate.specific import _git_ok, _listed

    note = root / "deliveries" / "WP9b.md"
    shots = SCREENSHOT.findall(note.read_text(encoding="utf-8")) if note.exists() else []
    out = [from_findings("screenshots", [] if shots else ["no .png path in deliveries/WP9b.md"],
                         f"{len(shots)} screenshot(s) referenced")]  # fmt: skip
    base = data_root or (Path(os.environ["KTA_DATA"]) if os.environ.get("KTA_DATA") else None)
    if base is None:
        return [*out, CheckResult("WP9b real operations", "FAIL", "set KTA_DATA")]
    db = base / "dev" / "WP9b" / "kta.db"
    if not db.exists():
        return [*out, CheckResult("WP9b real operations", "FAIL", f"{db} not found")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        findings = real_operations(
            con,
            _listed(root, "WP9b"),
            lambda c: _git_ok(root, "merge-base", "--is-ancestor", c, "HEAD"),
        )
    finally:
        con.close()
    return [
        *out,
        from_findings("WP9b real operations", findings, "start, cancel and reject recorded"),
    ]


def wp9b(
    root: Path,
    passed: set[str],
    data_root: Path | None = None,
    build: Callable[[Path], CheckResult] | None = None,
) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real, web_build

    names = _names(passed)
    results = [
        from_findings(
            "WP9b API tests",
            [f"{t} did not pass" for t in WP9B_TESTS if t not in names],
            f"{len(WP9B_TESTS)} tests passed",
        ),
        from_findings("operations tested", _covered(root), "every spec operation is exercised"),
        (build or web_build)(root),
        from_findings("no sample data in web/", _sample_data(root), "web/src reads the API only"),
    ]
    if code_stage():
        return [
            *results,
            pending_real(
                "WP9b real operations",
                "start a T1, cancel a task and reject a result from the UI against real "
                "records in dev/WP9b; screenshots in the delivery note",
            ),
        ]
    return [*results, *_full_stage(root, data_root)]
