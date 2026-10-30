"""WP8 specific checks (spec/gate.md): agent runs, resume, archives, runtime skills.

Kept apart from specific.py (at the 500-line limit); registered there. The real-run
part (spec/gate.md: "at least one real AgentRun and one resumed session") is pending in
the code stage (D320).
"""

from __future__ import annotations

import ast
import io
import os
import sqlite3
import tarfile
from pathlib import Path
from typing import Any

from kta_gate.result import CheckResult, from_findings

WP8_TESTS = (
    "test_start_resume_violation_and_close",
    "test_refuses_an_agents_md_above_the_workspace",
    "test_parse_codex_0117_session",
    "test_session_rules",
    "test_token_roundtrip_expiry_tamper_and_task_state",
    "test_compliance_from_events",
    "test_pack_excludes_credentials_and_links_and_is_deterministic",
    "test_fake_accepts_adapter_argv_and_refuses_non_persistent",
)
RUNTIME_SKILLS = ("use-kta", "find-knowledge", "write-family", "write-template", "decompose-kernel")
# Options that disable session persistence (conventions §7.1); only the capability table
# that defines the refusal may spell them.
FORBIDDEN_OPTIONS = ("--ephemeral", "--no-session", "--no-session-persistence")
OPTION_TABLE = "src/kta/agents/adapters/capabilities.py"
CREDENTIAL_NAMES = ("auth.json", ".credentials.json")
INDUCED_KINDS = ("write_outside_workspace", "denied")


def check_skills(root: Path) -> CheckResult:
    findings: list[str] = []
    for name in RUNTIME_SKILLS:
        path = root / "src" / "kta" / "agents" / "skills" / name / "SKILL.md"
        if not path.exists():
            findings.append(f"missing runtime skill {name}")
            continue
        head = path.read_text(encoding="utf-8").split("---")
        front = head[1] if len(head) > 2 else ""
        if f"name: {name}" not in front or "description:" not in front:
            findings.append(f"{name}: front matter needs name: {name} and a description")
    return from_findings("runtime skills", findings, f"{len(RUNTIME_SKILLS)} skills")


def check_options(root: Path) -> CheckResult:
    """No string literal in src/ spells a non-persistent session option."""
    findings: list[str] = []
    for path in sorted((root / "src").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == OPTION_TABLE:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in FORBIDDEN_OPTIONS:
                findings.append(f"{rel}:{node.lineno}: {node.value}")
    return from_findings("non-persistent session options", findings, "none in src/")


def check_tests(passed: set[str]) -> CheckResult:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP8_TESTS if t not in names]
    return from_findings("WP8 tests", absent, f"{len(WP8_TESTS)} named tests passed")


def _blob(data: Path, sha: str) -> bytes | None:
    path = data / "objects" / sha[:2] / sha[2:]
    return path.read_bytes() if len(sha) == 64 and path.exists() else None


def _archive_findings(data: Path, run: dict[str, Any]) -> list[str]:
    rid = run["_id"][:12]
    body = run["body"]
    raw = _blob(data, str(body.get("session_archive", "")))
    if raw is None:
        return [f"AgentRun {rid}: session archive blob missing"]
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        names = tar.getnames()
    out = [f"AgentRun {rid}: archive holds {n}" for n in names if Path(n).name in CREDENTIAL_NAMES]
    if not any(n.startswith("sessions/") for n in names):
        out.append(f"AgentRun {rid}: archive has no session file")
    if _blob(data, str(body.get("context_manifest", ""))) is None:
        out.append(f"AgentRun {rid}: context manifest blob missing")
    return out


def run_findings(con: sqlite3.Connection, data: Path, listed: set[str]) -> list[str]:
    from kta_gate.specific import _record

    runs: list[dict[str, Any]] = []
    for rid in sorted(listed):
        rec = _record(con, rid)
        if (
            rec
            and rec["kind"] == "AgentRun"
            and rec["_origin"] == "real"
            and rec["actor"].get("dirty") is False
        ):
            runs.append({**rec, "_id": rid})
    if not runs:
        return ["no listed, real, clean AgentRun"]
    findings = [f for r in runs for f in _archive_findings(data, r)]
    resumed = False
    for r in runs:
        prev = _record(con, r["refs"].get("previous"))
        if prev and prev["kind"] == "AgentRun" and prev["body"]["session"] == r["body"]["session"]:
            resumed = True
    if not resumed:
        findings.append("no AgentRun that resumed the session of its previous run")
    if not any(
        v.get("kind") in INDUCED_KINDS for r in runs for v in r["body"].get("violations", [])
    ):
        findings.append("no listed AgentRun recorded an out-of-scope write (induced violation)")
    return findings


def wp8(root: Path, passed: set[str], data_root: Path | None = None) -> list[CheckResult]:
    from kta_gate.specific import _listed, code_stage, pending_real

    results = [check_tests(passed), check_skills(root), check_options(root)]
    if code_stage():
        return [*results, pending_real("WP8 agent runs", "real T1, resume, induced write")]
    if data_root is None and not os.environ.get("KTA_DATA"):
        return [*results, CheckResult("WP8 agent runs", "FAIL", "set KTA_DATA")]
    data = (data_root or Path(os.environ["KTA_DATA"])) / "dev" / "WP8"
    db = data / "kta.db"
    if not db.exists():
        return [*results, CheckResult("WP8 agent runs", "FAIL", f"{db} not found")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        findings = run_findings(con, data, _listed(root, "WP8"))
    finally:
        con.close()
    return [*results, from_findings("WP8 agent runs", findings, "AgentRun, resume, archives")]
