"""WP11 specific checks (spec/workplan.md WP11, D408): processes, skills, the afternoon.

Static part: the named WP11 tests passed; every process of docs/design/11 §1 has a tick;
the runtime skills ``diagnose-model`` and ``write-microbench`` exist with front matter,
are catalogued as written and are in profile ``codex-v2``.

Record part (full stage only; WARN in the code stage, D320), in ``$KTA_DATA/dev/WP11``:
listed, real, clean records of a microbenchmark HwFact (measured, with its
Measurements), an ExperimentRun, a T4 TaskCard created by the calibrate process and an
improvement Report; and spans of the experiment, solve, learn and calibrate lanes that
overlap one window of at most four hours (docs/design/11 §7 "an ordinary afternoon").
The frontend view of it is a screenshot named in the delivery note.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from kta_gate.result import CheckResult, from_findings

WP11_TESTS = (
    "test_experiment_quota_is_redistributed_when_no_batch_runs",
    "test_apply_quotas_follows_running_experiments",
    "test_budget_period_rolls_over_with_a_report",
    "test_queued_tasks_take_turns_by_quota",
    "test_learn_writes_a_snapshot_once_and_starts_backlog_loops",
    "test_calibrate_records_missing_hardware_and_alerts_on_model_error",
    "test_hw_fact_gaps_get_one_t5",
    "test_microbench_format_and_value",
    "test_experiment_runs_only_approved_protocols_in_their_batch",
    "test_summarize_counts_attribution_creates_t3_and_waits_for_you",
    "test_every_process_runs_a_tick",
    "test_profile_v2_adds_the_t4_t5_skills_and_keeps_v1",
)
SKILLS = ("diagnose-model", "write-microbench")
LANES = ("process:experiment", "process:solve", "process:learn", "process:calibrate")
MICROBENCH_FACTS = ("l2_effective_bytes", "dram_latency_s", "launch_latency_s")
WINDOW = timedelta(hours=4)


def check_tests(passed: set[str]) -> CheckResult:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP11_TESTS if t not in names]
    return from_findings("WP11 tests", absent, f"{len(WP11_TESTS)} named tests passed")


def check_processes() -> CheckResult:
    from kta.orchestrator import processes

    missing = [n for n in processes.ALL if n not in processes.TICKS]
    if set(processes.IMPLEMENTED) != set(processes.ALL):
        missing.append(f"IMPLEMENTED = {processes.IMPLEMENTED}")
    return from_findings("WP11 processes", missing, f"{len(processes.ALL)} processes have ticks")


def check_skills(root: Path) -> CheckResult:
    from kta.agents.profile import CODEX_V2

    findings: list[str] = []
    base = root / "src" / "kta" / "agents" / "skills"
    readme = (base / "README.md").read_text(encoding="utf-8")
    in_v2 = {s for names in CODEX_V2.skills_by_type.values() for s in names}
    for name in SKILLS:
        path = base / name / "SKILL.md"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not text.startswith(f"---\nname: {name}\ndescription: "):
            findings.append(f"{name}: missing or without front matter")
        if f"| `{name}` |" not in readme or "| written |" not in readme.split(f"`{name}`")[1][:40]:
            findings.append(f"{name}: not catalogued as written")
        if name not in in_v2:
            findings.append(f"{name}: not in profile codex-v2")
    return from_findings("WP11 runtime skills", findings, ", ".join(SKILLS))


def _clean_real(rec: dict[str, Any] | None) -> bool:
    return bool(rec) and rec["_origin"] == "real" and rec["actor"].get("dirty") is False  # type: ignore[index]


def record_findings(con: sqlite3.Connection, listed: set[str]) -> list[str]:
    from kta_gate.specific import _record

    recs = [(rid, _record(con, rid)) for rid in sorted(listed)]
    good = [(rid, r) for rid, r in recs if r is not None and _clean_real(r)]
    findings: list[str] = []

    def any_of(what: str, pred: Any) -> None:
        if not any(pred(r) for _, r in good):
            findings.append(f"no listed, real, clean {what}")

    any_of(
        "microbenchmark HwFact",
        lambda r: (
            r["kind"] == "HwFact"
            and r["body"].get("quantity") in MICROBENCH_FACTS
            and r["body"].get("method") == "measured"
            and (r["refs"].get("measurement") or r["refs"].get("measurements"))
        ),
    )
    any_of("ExperimentRun", lambda r: r["kind"] == "ExperimentRun")
    any_of(
        "T4 TaskCard from the calibrate process",
        lambda r: (
            r["kind"] == "TaskCard"
            and r["body"].get("type") == "T4"
            and "calibrate" in str(r["actor"].get("name"))
        ),
    )
    any_of(
        "improvement Report",
        lambda r: r["kind"] == "Report" and r["body"].get("title") == "improvement",
    )
    return findings + afternoon_findings(con)


def afternoon_findings(con: sqlite3.Connection) -> list[str]:
    """Some window of WINDOW has a span of every lane in LANES (11 §7)."""
    rows = con.execute(
        "SELECT lane, start_ts, COALESCE(end_ts, start_ts) FROM spans WHERE lane IN (?,?,?,?)",
        LANES,
    ).fetchall()
    spans = [(lane, datetime.fromisoformat(a), datetime.fromisoformat(b)) for lane, a, b in rows]
    for _, start, _ in spans:
        end = start + WINDOW
        if {lane for lane, a, b in spans if a <= end and b >= start} >= set(LANES):
            return []
    return [f"no {WINDOW} window with spans of every lane {', '.join(LANES)}"]


def screenshot_findings(root: Path) -> list[str]:
    note = root / "deliveries" / "WP11.md"
    text = note.read_text(encoding="utf-8") if note.exists() else ""
    shots = [w.strip("`()") for w in text.split() if w.strip("`()").endswith(".png")]
    if not any((root / s).exists() for s in shots):
        return ["the delivery note names no existing .png screenshot of the afternoon"]
    return []


def wp11(root: Path, passed: set[str], data_root: Path | None = None) -> list[CheckResult]:
    from kta_gate.specific import _listed, code_stage, pending_real

    results = [check_tests(passed), check_processes(), check_skills(root)]
    name = "WP11 afternoon"
    if code_stage():
        what = "microbenchmark HwFacts, an ordinary afternoon of all processes, screenshot"
        return [*results, pending_real(name, what)]
    if data_root is None and not os.environ.get("KTA_DATA"):
        return [*results, CheckResult(name, "FAIL", "set KTA_DATA")]
    db = (data_root or Path(os.environ["KTA_DATA"])) / "dev" / "WP11" / "kta.db"
    if not db.exists():
        return [*results, CheckResult(name, "FAIL", f"{db} not found")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        findings = record_findings(con, _listed(root, "WP11")) + screenshot_findings(root)
    finally:
        con.close()
    return [*results, from_findings(name, findings, "records, spans and screenshot")]
