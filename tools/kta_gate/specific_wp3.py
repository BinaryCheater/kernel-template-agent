"""WP3 specific checks (spec/gate.md): GPU acceptance run and real measure/check records.

Kept apart from specific.py, which is at the 500-line limit; registered there.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from kta_gate.result import CheckResult, from_findings

WP3_ACCEPTANCE = (
    "test_checker_passes_the_correct_kernel_and_names_each_fault",
    "test_noise_probe_and_revised_protocol",
    "test_compare_separates_a_five_percent_kernel_and_not_an_identical_one",
    "test_profile_writes_a_profile_or_reports_not_supported",
)
WP3_CODE = "src/kta/measure/bench.py"
WP3_KINDS = ("Measurement", "Check", "Protocol")
VARIANTS = ("correct", "oob_write", "missing_row", "cublas")
NOISE_GROUPS = 30


def _protocols(good: list[dict[str, Any]]) -> tuple[dict[int, str], list[str]]:
    found = {
        int(r["body"]["version"]): r["_id"]
        for r in good
        if r["kind"] == "Protocol" and r["body"].get("name") == "measure"
    }
    findings = [f"no listed Protocol measure-v{v}" for v in (1, 2) if v not in found]
    return found, findings


def _noise_findings(
    con: sqlite3.Connection, good: list[dict[str, Any]], v2_id: str | None
) -> list[str]:
    from kta_gate.specific import _record

    if v2_id is None:
        return []
    v2 = next(r for r in good if r["_id"] == v2_id)
    reports = v2["refs"].get("noise_reports") or []
    listed = {r["_id"] for r in good}
    out = [f"noise report {r[:12]} of measure-v2 is not listed" for r in reports if r not in listed]
    for rid in reports:
        rec = _record(con, rid)
        if rec is None or int(rec["body"]["data"].get("groups", 0)) < NOISE_GROUPS:
            out.append(f"noise report {rid[:12]} has fewer than {NOISE_GROUPS} groups")
    return out or ([] if reports else ["measure-v2 references no noise report"])


def _report_findings(good: list[dict[str, Any]], v2_id: str | None) -> list[str]:
    out: list[str] = []
    selftests = [r for r in good if r["kind"] == "Report" and r["body"].get("subject") == "checks"]
    if not any(
        all(r["body"]["data"].get(v, {}).get("as_expected") is True for v in VARIANTS)
        for r in selftests
    ):
        out.append("no listed checker-selftest Report with all four variants as expected")
    compares = [
        r["body"]["data"]
        for r in good
        if r["kind"] == "Report"
        and r["body"].get("subject") == "compare"
        and r["body"]["data"].get("protocol") == v2_id
    ]
    if not any(d.get("significant") is True for d in compares):
        out.append("no listed significant compare Report under measure-v2")
    if not any(d.get("significant") is False for d in compares):
        out.append("no listed non-significant compare Report under measure-v2")
    return out


def wp3_records(git_root: Path, con: sqlite3.Connection, listed: set[str]) -> list[str]:
    from kta_gate.specific import _git_ok, _record

    def clean(rec: dict[str, Any]) -> bool:
        commit = str(rec["actor"].get("commit"))
        return (
            rec["_origin"] == "real"
            and rec["actor"].get("dirty") is False
            and _git_ok(git_root, "merge-base", "--is-ancestor", commit, "HEAD")
            and _git_ok(git_root, "cat-file", "-e", f"{commit}:{WP3_CODE}")
        )

    def on_gpu(rec: dict[str, Any]) -> bool:
        worker = _record(con, rec["refs"].get("worker"))
        return worker is not None and "gpu" in worker["body"].get("labels", {})

    good: list[dict[str, Any]] = []
    for rid in sorted(listed):
        rec = _record(con, rid)
        if rec is not None and clean(rec):
            good.append({**rec, "_id": rid})
    findings = [
        f"no listed, real, clean WP3 {kind} record"
        for kind in WP3_KINDS
        if not any(r["kind"] == kind for r in good)
    ]
    for kind in ("Measurement", "Check"):
        if not any(r["kind"] == kind and on_gpu(r) for r in good):
            findings.append(f"no listed {kind} from a GPU-labelled worker")
    protocols, missing = _protocols(good)
    findings += missing
    ids = set(protocols.values())
    if any(r["kind"] == "Measurement" and r["body"]["protocol"] not in ids for r in good):
        findings.append("a listed Measurement names a Protocol that is not listed")
    findings += _noise_findings(con, good, protocols.get(2))
    findings += _report_findings(good, protocols.get(2))
    return findings


def wp3(
    root: Path,
    passed: set[str],
    data_root: Path | None = None,
    git_root: Path | None = None,
    gpu_passed: set[str] | None = None,
) -> list[CheckResult]:
    """GPU acceptance tests run on spark; real Measurement, Check, Protocol records exist."""
    from kta_gate.specific import _listed

    results: list[CheckResult] = []
    if gpu_passed is None:
        from kta_gate.runners import run_pytest

        run = run_pytest(root, "WP3 GPU acceptance run", ["-m", "gpu", "tests/acceptance/WP3"])
        results.append(run.result)
        gpu_passed = run.passed
    names = {t.split("::")[-1].split("[")[0] for t in gpu_passed if "acceptance.WP3" in t}
    absent = [f"{t} did not pass" for t in WP3_ACCEPTANCE if t not in names]
    results.append(from_findings("WP3 acceptance tests", absent, f"{len(names)} tests passed"))
    if data_root is None and not os.environ.get("KTA_DATA"):
        return [*results, CheckResult("WP3 records", "FAIL", "set KTA_DATA")]
    db = (data_root or Path(os.environ["KTA_DATA"])) / "dev" / "WP3" / "kta.db"
    if not db.exists():
        return [*results, CheckResult("WP3 records", "FAIL", f"{db} not found")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        findings = wp3_records(git_root or root, con, _listed(root, "WP3"))
    finally:
        con.close()
    detail = "Measurement, Check, Protocol v1/v2, noise, compare, checker selftest listed"
    return [*results, from_findings("WP3 records", findings, detail)]
