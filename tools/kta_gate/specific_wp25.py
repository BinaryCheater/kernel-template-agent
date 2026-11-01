"""WP25 specific checks (spec/workplan.md "WP25", D660–D667): supervision views.

Static part (both stages): the named WP25 tests passed; the spec sections exist; the web
app routes /activity, the home page draws the loop maps with loop controls, Settings edits
mappings key by key; journal.yaml has its `other` rows; the web app builds.

Real-run part: in `$KTA_DATA/dev/WP25/kta.db`, a clean real HumanAction of a hand advance
(`advance {...}`) followed by a loop step. WARN in the code stage.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import yaml

from kta_gate.result import CheckResult, from_findings

WP25_TESTS = (
    "test_manual_step_ignores_a_loop_pause_but_not_a_process_pause",
    "test_advance_round_stops_where_the_loop_waits",
    "test_advance_step_takes_exactly_one_step",
    "test_advance_run_resumes_the_loop_and_says_no_process_runs",
    "test_retry_clears_the_block_and_steps_again",
    "test_advance_refuses_an_unknown_mode_and_a_non_human",
    "test_advance_endpoint_writes_the_action_and_lists_the_capability",
    "test_journal_phrases_records_and_counts_unlisted_kinds",
    "test_journal_groups_a_burst_even_when_interleaved",
    "test_journal_milestones_drop_detail_and_page_backwards",
    "test_every_phrase_renders_with_its_placeholders_missing",
    "test_active_view_and_trace_mark_a_blocked_paused_loop",
    "test_lock_holder_ignores_an_expired_lock",
)
SPEC_MARKS = (
    ("spec/workplan.md", "## WP25 "),
    ("spec/modules/api-web.md", "## WP25: supervision views"),
    ("web/DESIGN.md", "## 7. Supervision views (WP25"),
)
PAGE_MARKS = (
    ("web/src/App.tsx", 'path="/activity"'),
    ("web/src/pages/Overview.tsx", "LoopMap"),
    ("web/src/pages/Overview.tsx", "LoopControls"),
    ("web/src/pages/Case.tsx", "RoundCard"),
    ("web/src/pages/Settings.tsx", "ValueEditor"),
    ("web/src/components/Swimlanes.tsx", "attrs.blocked"),
)


def check_tests(passed: set[str]) -> CheckResult:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP25_TESTS if t not in names]
    return from_findings("WP25 tests", absent, f"{len(WP25_TESTS)} named tests passed")


def check_static(root: Path) -> CheckResult:
    findings = [
        f"{path}: missing {mark!r}"
        for path, mark in (*SPEC_MARKS, *PAGE_MARKS)
        if mark not in (root / path).read_text(encoding="utf-8")
    ]
    table = yaml.safe_load((root / "src/kta/api/journal.yaml").read_text(encoding="utf-8"))
    for section in ("records", "spans"):
        if "other" not in table[section]:
            findings.append(f"journal.yaml has no {section}.other row")
    return from_findings("WP25 spec, pages and phrase table", findings, "spec, pages, journal.yaml")


def check_real(root: Path) -> CheckResult:
    data = Path(os.environ.get("KTA_DATA", Path.home() / "kta-data")) / "dev" / "WP25" / "kta.db"
    name = "WP25 real hand advance"
    if not data.is_file():
        return CheckResult(name, "FAIL", f"{data} does not exist")
    con = sqlite3.connect(f"file:{data}?mode=ro", uri=True)
    acts = [
        json.loads(r[0]) for r in con.execute("SELECT json FROM records WHERE kind = 'HumanAction'")
    ]
    clean = [
        a for a in acts
        if a["body"]["action"].startswith("advance ") and a["origin"] == "real"
        and not a["actor"].get("dirty")
    ]  # fmt: skip
    steps = con.execute("SELECT COUNT(*) FROM spans WHERE kind = 'loop_step'").fetchone()[0]
    findings = [] if clean else ["no clean real advance HumanAction"]
    if clean and not steps:
        findings.append("no loop step recorded")
    _ = root
    return from_findings(name, findings, f"{len(clean)} hand advances, {steps} loop steps")


def wp25(root: Path, passed: set[str]) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real, web_build

    results = [check_tests(passed), check_static(root), web_build(root)]
    if code_stage():
        what = "a clean real hand advance (step or round) and its loop step in dev/WP25"
        return [*results, pending_real("WP25 real hand advance", what)]
    return [*results, check_real(root)]
