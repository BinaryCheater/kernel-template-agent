"""Checks on the delivery note, decision log, acceptance lock and evidence manifest."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from kta_gate.result import CheckResult, from_findings
from kta_gate.scans import fixture_markers

# Canonical English headings, with the Chinese names AGENTS.md uses accepted as aliases.
SECTIONS: dict[str, tuple[str, ...]] = {
    "done": ("what was done", "做了什么"),
    "fixtures": ("fixtures and simplifications", "fixture 与简化"),
    "issues": ("known issues", "已知问题"),
    "decisions": ("new decisions", "新增的决策"),
    "changes": ("needed changes", "需要的变更"),
    "gate": ("gate output", "gate 输出"),
}
HEADING = re.compile(r"^#{2,3}\s+(.+?)\s*$")
DECISION_ROW = re.compile(r"^\|\s*(D\d+)\s*\|")
DECISION_ID = re.compile(r"\bD\d{2,}\b")
# Work packages whose claims must point at records in their dev store (conventions §5).
STORE_EVIDENCE_WPS = frozenset(
    {"WP1", "WP2", "WP3", "WP7", "WP8", "WP9a", "WP9b", "WP10b", "WP11", "WP12"}
)
EVIDENCE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "claim": str,
    "command": str,
    "git_commit": str,
    "dirty": bool,
    "env": str,
    "control_host": str,
    "workers": list,
    "record_ids": list,
    "blob_ids": list,
    "env_manifests": list,
    "started_at": str,
}


def delivery_path(root: Path, wp: str) -> Path:
    return root / "deliveries" / f"{wp}.md"


def split_sections(text: str) -> dict[str, str]:
    """Map section key -> body text for the known delivery headings."""
    out: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            title = m.group(1).strip().lower()
            current = next((k for k, names in SECTIONS.items() if title in names), None)
            if current is not None:
                out.setdefault(current, "")
            continue
        if current is not None:
            out[current] += line + "\n"
    return out


def check_delivery_sections(root: Path, wp: str) -> CheckResult:
    path = delivery_path(root, wp)
    if not path.exists():
        return CheckResult("delivery note", "FAIL", f"{path.relative_to(root)} is missing")
    sections = split_sections(path.read_text(encoding="utf-8"))
    missing = [SECTIONS[k][0] for k in SECTIONS if k not in sections]
    return from_findings("delivery note", [f"missing section: {m}" for m in missing])


def check_fixture_declarations(root: Path, wp: str) -> CheckResult:
    markers = fixture_markers(root)
    if not markers:
        return CheckResult("fixture declarations", "PASS", "no # kta:fixture markers")
    path = delivery_path(root, wp)
    declared = (
        split_sections(path.read_text(encoding="utf-8")).get("fixtures", "")
        if path.exists()
        else ""
    )
    findings = [
        f"{f}:{n} not listed in 'Fixtures and simplifications'"
        for f, n in markers
        if f not in declared
    ]
    return from_findings("fixture declarations", findings, f"{len(markers)} declared")


def decision_ids(root: Path) -> tuple[set[str], list[str]]:
    """IDs in docs/decisions.md and format problems found while reading them."""
    ids: set[str] = set()
    problems: list[str] = []
    path = root / "docs" / "decisions.md"
    if not path.exists():
        return ids, ["docs/decisions.md is missing"]
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = DECISION_ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not all(cells):
            problems.append(f"decisions.md:{no}: expected 4 non-empty cells, got {len(cells)}")
        if m.group(1) in ids:
            problems.append(f"decisions.md:{no}: duplicate ID {m.group(1)}")
        ids.add(m.group(1))
        if "needs user review" in line:
            queue = root / "docs" / "review-queue.md"
            queue_text = queue.read_text(encoding="utf-8") if queue.exists() else ""
            if not re.search(rf"\b{m.group(1)}\b", queue_text):
                problems.append(f"{m.group(1)} needs user review but is not in review-queue.md")
    return ids, problems


def check_decisions(root: Path, wp: str) -> CheckResult:
    ids, findings = decision_ids(root)
    path = delivery_path(root, wp)
    if path.exists():
        listed = split_sections(path.read_text(encoding="utf-8")).get("decisions", "")
        findings += [
            f"delivery lists {d}, which is not in decisions.md"
            for d in sorted(set(DECISION_ID.findall(listed)) - ids)
        ]
    return from_findings("decision log", findings, f"{len(ids)} decisions")


def check_lock(root: Path) -> CheckResult:
    lock = root / "tests" / "acceptance" / "LOCK"
    if not lock.exists():
        return CheckResult("acceptance lock", "FAIL", "tests/acceptance/LOCK is missing")
    findings: list[str] = []
    count = 0
    for no, line in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            findings.append(f"LOCK:{no}: expected '<sha256>  <path>'")
            continue
        digest, rel = parts[0], parts[1].strip()
        target = root / rel
        count += 1
        if not target.exists():
            findings.append(f"{rel}: locked file was removed")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            findings.append(f"{rel}: locked file was modified")
    return from_findings("acceptance lock", findings, f"{count} locked files unchanged")


def _validate_claim(index: int, claim: Any) -> list[str]:
    if not isinstance(claim, dict):
        return [f"claim {index}: not an object"]
    problems: list[str] = []
    for key, typ in EVIDENCE_FIELDS.items():
        if key not in claim:
            problems.append(f"claim {index}: missing {key}")
        elif not isinstance(claim[key], typ):
            problems.append(f"claim {index}: {key} has wrong type")
    if problems:
        return problems
    if not re.fullmatch(r"[0-9a-f]{40}", claim["git_commit"]):
        problems.append(f"claim {index}: git_commit is not a full sha")
    if claim["dirty"]:
        problems.append(f"claim {index}: dirty runs are not evidence")
    if claim["env"] != "dev":
        problems.append(f"claim {index}: env must be dev")
    try:
        datetime.fromisoformat(claim["started_at"])
    except ValueError:
        problems.append(f"claim {index}: started_at is not ISO 8601")
    return problems


def _commit_exists(root: Path, sha: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _verify_records(db: Path, claim: dict[str, Any], index: int) -> list[str]:
    """Each record exists, is real, and was written by the claimed clean commit."""
    problems: list[str] = []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for rid in claim["record_ids"]:
            row = con.execute("SELECT origin, json FROM records WHERE id = ?", (rid,)).fetchone()
            if row is None:
                problems.append(f"claim {index}: record {rid} not in {db}")
                continue
            origin, body = row[0], json.loads(row[1])
            actor = body.get("actor", {})
            if origin != "real":
                problems.append(f"claim {index}: record {rid} origin {origin}")
            if actor.get("commit") != claim["git_commit"] or actor.get("dirty") is not False:
                problems.append(f"claim {index}: record {rid} not from clean {claim['git_commit']}")
            if claim.get("requires_gpu") and not _from_gpu_worker(con, body):
                problems.append(f"claim {index}: record {rid} not from a GPU-labelled worker")
    finally:
        con.close()
    return problems


def _from_gpu_worker(con: sqlite3.Connection, record: dict[str, Any]) -> bool:
    worker_id = record.get("refs", {}).get("worker")
    if not isinstance(worker_id, str):
        return False
    row = con.execute("SELECT json FROM records WHERE id = ?", (worker_id,)).fetchone()
    if row is None:
        return False
    labels = json.loads(row[0]).get("body", {}).get("labels", {})
    return isinstance(labels, dict) and "gpu" in labels


def check_evidence(
    root: Path, wp: str, data_root: Path | None, path: Path | None = None
) -> CheckResult:
    path = path or root / "deliveries" / f"{wp}.evidence.json"
    if not path.exists():
        return CheckResult("evidence", "FAIL", f"{path} is missing")
    try:
        claims = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult("evidence", "FAIL", f"invalid JSON: {exc}")
    if not isinstance(claims, list):
        return CheckResult("evidence", "FAIL", "top level must be a list of claims")
    findings: list[str] = []
    for i, claim in enumerate(claims):
        problems = _validate_claim(i, claim)
        findings += problems
        if problems:
            continue
        if not _commit_exists(root, claim["git_commit"]):
            findings.append(f"claim {i}: commit {claim['git_commit'][:12]} not in this repo")
        if wp in STORE_EVIDENCE_WPS:
            if not claim["record_ids"]:
                findings.append(f"claim {i}: {wp} claims need record_ids")
                continue
            db = (data_root / "dev" / wp / "kta.db") if data_root else None
            if db is None or not db.exists():
                findings.append(f"claim {i}: dev store {db} not found (set KTA_DATA)")
                continue
            findings += _verify_records(db, claim, i)
    if wp in STORE_EVIDENCE_WPS and not claims:
        findings.append(f"{wp} requires at least one evidenced claim")
    if not findings and not claims:
        return CheckResult("evidence", "WARN", "evidence list is empty")
    return from_findings("evidence", findings, f"{len(claims)} claim(s)")


def check_skips(root: Path, wp: str, skipped: list[str]) -> CheckResult:
    if not skipped:
        return CheckResult("skipped tests", "PASS", "none")
    path = delivery_path(root, wp)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    unexplained = [s for s in skipped if s.split("::")[-1] not in text]
    if unexplained:
        return CheckResult(
            "skipped tests", "WARN", "not explained in the delivery note", unexplained
        )
    return CheckResult("skipped tests", "PASS", f"{len(skipped)} explained")
