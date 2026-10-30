"""WP10b specific checks (spec/gate.md; spec/workplan.md WP10b: automatic decomposition and
coverage). Kept apart from specific.py, which is at the 500-line limit; registered there.

- The named tests passed: the expand machine agrees with machines.yaml, WP10a's references
  replay from ``frozen`` to ``covered`` (Python and CLI paths), a cancelled extraction and an
  inflated ``matched`` count behave as specified.
- Every expand transition that creates tasks creates only T1, T2 or T3, and every T2/T3
  done-check named in done_checks.py exists.
- Full stage only: at least 3 references beyond WP10a's 7 under knowledge/seed/references,
  and a ``coverage`` Report in ``dev/WP10b`` listed in the evidence manifest, written by a
  clean commit of this branch, whose ``references`` include an expand loop in ``covered``.
"""

from __future__ import annotations

import importlib
import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from kta_gate.result import CheckResult, from_findings

WP10B_TESTS = (
    "test_step_produces_exactly_the_yaml_transitions",
    "test_created_task_types_are_listed_on_the_edge",
    "test_expand_creates_one_t3_per_template_to_extract",
    "test_expand_gap_waits_until_every_task_settles",
    "test_expand_rematch_is_bounded_by_the_round_limit",
    "test_wp10a_references_go_from_frozen_to_covered",
    "test_cancelled_extraction_leaves_the_template_open",
    "test_t2_with_inflated_matched_count_is_rejected",
    "test_expand_cutlass_through_the_cli",
    "test_coverage_counts_the_latest_reconstruction_per_reference",
    "test_chao1_interval_and_page_fields",
)
MACHINES = Path("src") / "kta" / "orchestrator" / "machines.yaml"
WP10A_REFERENCES = 7
NEW_REFERENCES = 3


def _expand_edges(root: Path) -> list[str]:
    path = root / MACHINES
    if not path.exists():
        return [f"{path} is missing"]
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    expand = next((m for m in doc.get("machines", []) if m.get("id") == "expand"), None)
    if expand is None:
        return ["machines.yaml has no expand machine"]
    findings = [
        f"{t['from']} -> {t['to']} creates {sorted(set(t['creates']) - {'T1', 'T2', 'T3'})}"
        for t in expand.get("transitions", [])
        if set(t.get("creates", [])) - {"T1", "T2", "T3"}
    ]
    if not any(t.get("to") == "covered" for t in expand.get("transitions", [])):
        findings.append("no expand transition reaches covered")
    return findings


def _done_checks() -> list[str]:
    try:
        mod = importlib.import_module("kta.orchestrator.done_checks")
    except ImportError as exc:
        return [f"kta.orchestrator.done_checks does not import: {exc}"]
    CHECKS, DONE_CHECKS = mod.CHECKS, mod.DONE_CHECKS
    return [
        f"{t} names an unknown done-check {n}"
        for t in ("T2", "T3")
        for n in DONE_CHECKS.get(t, ())
        if n not in CHECKS
    ] + [f"{t} has no done-checks" for t in ("T2", "T3") if not DONE_CHECKS.get(t)]


def _covered_reports(con: sqlite3.Connection, listed: set[str], clean: Any) -> list[str]:
    from kta_gate.specific import _record

    for rid in sorted(listed):
        rec = _record(con, rid)
        if rec is None or rec.get("kind") != "Report" or rec["body"].get("title") != "coverage":
            continue
        actor = rec.get("actor", {})
        if actor.get("dirty") or not clean(str(actor.get("commit", ""))):
            continue
        refs = rec["body"].get("data", {}).get("references", [])
        if any(r.get("state") == "covered" for r in refs):
            return []
    return ["no listed coverage Report from a clean commit with a covered reference"]


def _full_stage(root: Path, data_root: Path | None) -> list[CheckResult]:
    from kta_gate.specific import _git_ok, _listed

    base = root / "knowledge" / "seed" / "references"
    n = len([p for p in base.iterdir() if p.is_dir()]) if base.exists() else 0
    want = WP10A_REFERENCES + NEW_REFERENCES
    out = [
        from_findings(
            "new references",
            [] if n >= want else [f"{n} reference directories; need at least {want}"],
            f"{n} references",
        )
    ]
    data = data_root or (Path(os.environ["KTA_DATA"]) if os.environ.get("KTA_DATA") else None)
    if data is None:
        return [*out, CheckResult("WP10b real expand loop", "FAIL", "set KTA_DATA")]
    db = data / "dev" / "WP10b" / "kta.db"
    if not db.exists():
        return [*out, CheckResult("WP10b real expand loop", "FAIL", f"{db} not found")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        findings = _covered_reports(
            con,
            _listed(root, "WP10b"),
            lambda c: _git_ok(root, "merge-base", "--is-ancestor", c, "HEAD"),
        )
    finally:
        con.close()
    return [*out, from_findings("WP10b real expand loop", findings, "coverage Report found")]


def wp10b(root: Path, passed: set[str], data_root: Path | None = None) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real

    names = {t.split("::")[-1].split("[")[0] for t in passed}
    results = [
        from_findings(
            "WP10b tests",
            [f"{t} did not pass" for t in WP10B_TESTS if t not in names],
            f"{len(WP10B_TESTS)} tests passed",
        ),
        from_findings("expand transitions", _expand_edges(root), "T1-T3 only; covered reached"),
        from_findings("T2/T3 done-checks", _done_checks(), "every named check exists"),
    ]
    if code_stage():
        return [
            *results,
            pending_real(
                "WP10b real expand loop",
                f"at least {NEW_REFERENCES} new references taken from frozen to covered (or a "
                "Gap) by agent T2/T3 runs in dev/WP10b; coverage Report in the evidence",
            ),
        ]
    return [*results, *_full_stage(root, data_root)]
