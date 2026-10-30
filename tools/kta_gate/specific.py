"""Work-package specific checks (spec/gate.md, second table).

A work package whose specific checks are not implemented here fails the gate: the
WP that owns them adds them, so no check is silently skipped.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import yaml

from kta_gate.delivery import split_sections
from kta_gate.result import CheckResult, from_findings

MODULES = (
    "core",
    "store",
    "targets",
    "execution",
    "backends",
    "knowledge",
    "models",
    "measure",
    "formulation",
    "checks",
    "solve",
    "orchestrator",
    "agents",
    "cli",
    "api",
)
WP0_VIOLATION_TESTS = (
    "test_gate_blocks_layer_violation",
    "test_gate_blocks_fake_class_in_src",
    "test_gate_blocks_bare_except",
    "test_gate_blocks_print_in_src",
    "test_gate_blocks_fake_api_key",
    "test_gate_blocks_fixture_impl_in_dev",
)
REFERENCE_KINDS = ("Reference", "Chain", "Reconstruction")
CITATION = re.compile(r"\[S\d+\]")
SOURCE_DEF = re.compile(r"^\s*[-*]?\s*\[(S\d+)\]:?\s+(\S+)")
MEASURED_NUMBER = re.compile(r"\d+\.\d+|\d+(?:\.\d+)?\s*(?:%|×|x\b|s\b|ms\b|us\b|µs\b|min\b|h\b)")


def wp0(root: Path, passed: set[str]) -> list[CheckResult]:
    missing = [
        f"src/kta/{m}/README.md"
        for m in MODULES
        if not (root / "src" / "kta" / m / "README.md").exists()
    ]
    names = {t.split("::")[-1] for t in passed}
    absent = [f"{t} did not pass" for t in WP0_VIOLATION_TESTS if t not in names]
    return [
        from_findings("module READMEs", missing, f"{len(MODULES)} modules"),
        from_findings(
            "gate violation samples", absent, f"{len(WP0_VIOLATION_TESTS)} samples blocked"
        ),
    ]


WP1_ACCEPTANCE = (
    "test_main_env_rejects_fixture_and_simulated_records",
    "test_main_env_rejects_dirty_writes",
    "test_registry_rejects_fixture_store_in_dev",
    "test_record_id_is_stable_across_processes",
    "test_update_and_delete_are_refused_by_triggers",
    "test_put_fails_when_refs_are_missing",
    "test_status_promotion_rules",
    "test_stale_propagation_on_five_level_graph",
    "test_concurrent_cas_pointer_update_has_one_winner",
)
BENCH_SUBJECT = "store-benchmark"
BENCH_MIN_RECORDS = 100_000
BENCH_P95_MS = 50.0
STORE_SOURCE = "src/kta/store/sqlite.py"


def _p95(values: list[float]) -> float:
    """Same rule as kta.store.bench: nearest rank on the sorted samples."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def _git_ok(root: Path, *args: str) -> bool:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    return proc.returncode == 0


def _bench_problems(
    root: Path, data: Path, rid: str, record: dict[str, Any], con: Any
) -> list[str]:
    """Why a store-benchmark Report does not count as WP1 evidence (empty when it does)."""
    problems: list[str] = []
    actor, body = record.get("actor", {}), record.get("body", {})
    commit = str(actor.get("commit"))
    if actor.get("dirty") is not False:
        problems.append("written by dirty code")
    # "Created during WP1": the code that ran contains the store and is in this branch.
    if not (
        _git_ok(root, "merge-base", "--is-ancestor", commit, "HEAD")
        and _git_ok(root, "cat-file", "-e", f"{commit}:{STORE_SOURCE}")
    ):
        problems.append(f"commit {commit[:12]} is not a WP1 commit in this branch")
    env_row = con.execute(
        "SELECT json FROM records WHERE id = ? AND kind = 'EnvManifest'",
        (record.get("refs", {}).get("env"),),
    ).fetchone()
    if env_row is None or json.loads(env_row[0])["body"].get("role") != "worker":
        problems.append("refs.env is not a worker EnvManifest (the run belongs on spark)")
    if body.get("data", {}).get("records_in_store", 0) < BENCH_MIN_RECORDS:
        problems.append(f"fewer than {BENCH_MIN_RECORDS} records in the benchmarked store")
    sha = str(record.get("blobs", {}).get("latencies_ms", ""))
    blob = data / "objects" / sha[:2] / sha[2:]
    if not sha or not blob.exists():
        return [*problems, "raw latencies blob is missing"]
    samples: dict[str, list[float]] = json.loads(blob.read_text(encoding="utf-8"))
    for name, values in sorted(samples.items()):
        p95 = _p95(values) if values else math.inf
        if not p95 < BENCH_P95_MS:
            problems.append(f"{name}: p95 {p95:.3f} ms >= {BENCH_P95_MS} ms")
    if not any(n.startswith("query") for n in samples) or not any(
        n.startswith("search") for n in samples
    ):
        problems.append("samples lack query or search timings")
    return [f"Report {rid[:12]}: {p}" for p in problems]


def wp1(
    root: Path, passed: set[str], data_root: Path | None = None, git_root: Path | None = None
) -> list[CheckResult]:
    """Acceptance tests passed; a real store-benchmark Report from spark is in the dev store.

    WP1 has no GPU work, so "pytest -m gpu tests/acceptance/WP1" is the acceptance run
    inside the unit tests plus this record check (decision D154).
    """
    names = {t.split("::")[-1].split("[")[0] for t in passed if "acceptance.WP1" in t}
    absent = [f"{t} did not pass" for t in WP1_ACCEPTANCE if t not in names]
    results = [from_findings("WP1 acceptance tests", absent, f"{len(names)} tests passed")]
    if data_root is None and not os.environ.get("KTA_DATA"):
        return [*results, CheckResult("store benchmark record", "FAIL", "set KTA_DATA")]
    data = (data_root or Path(os.environ["KTA_DATA"])) / "dev" / "WP1"
    db = data / "kta.db"
    if not db.exists():
        return [*results, CheckResult("store benchmark record", "FAIL", f"{db} not found")]
    evidence_path = root / "deliveries" / "WP1.evidence.json"
    listed: set[str] = set()
    if evidence_path.exists():
        for claim in json.loads(evidence_path.read_text(encoding="utf-8")):
            listed |= set(claim.get("record_ids", []))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT id, json FROM records WHERE kind = 'Report' AND origin = 'real' "
            "AND json_extract(json, '$.body.subject') = ?",
            (BENCH_SUBJECT,),
        ).fetchall()
        findings = {
            rid: _bench_problems(git_root or root, data, rid, json.loads(js), con)
            for rid, js in rows
        }
    finally:
        con.close()
    good = sorted(rid for rid, p in findings.items() if not p and rid in listed)
    if good:
        detail = f"Report {good[0][:12]} (listed in the evidence manifest)"
        return [*results, CheckResult("store benchmark record", "PASS", detail)]
    problems = [p for ps in findings.values() for p in ps] or [
        f"no real {BENCH_SUBJECT} Report in {db}"
    ]
    problems += [
        f"Report {r[:12]} passes but is not in WP1.evidence.json"
        for r, p in findings.items()
        if not p and r not in listed
    ]
    return [*results, CheckResult("store benchmark record", "FAIL", "no valid record", problems)]


def _yaml_docs(path: Path) -> list[dict[str, Any]]:
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d is not None]
    return [d for d in docs if isinstance(d, dict)]


def _kind_of(path: Path, doc: dict[str, Any]) -> str:
    """A document's kind: its ``kind`` field, else the file name (``gap-*.yaml`` -> Gap)."""
    if isinstance(doc.get("kind"), str):
        return str(doc["kind"])
    stem = path.stem.lower()
    for kind in (*REFERENCE_KINDS, "Gap", "Pattern", "Template"):
        if stem.startswith(kind.lower()) or path.parent.name.lower() == f"{kind.lower()}s":
            return kind
    return "unknown"


def _model_validator() -> Callable[[str, dict[str, Any]], None] | None:
    """kta.core.schemas validation once WP1 provides it."""
    try:
        schemas = importlib.import_module("kta.core.schemas")
    except ModuleNotFoundError:
        return None
    validate = getattr(schemas, "validate_body", None)
    if not callable(validate):
        return None
    return cast(Callable[[str, dict[str, Any]], None], validate)


def wp10a(root: Path, passed: set[str]) -> list[CheckResult]:
    base = root / "knowledge" / "seed" / "references"
    refs = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if not refs:
        return [CheckResult("references", "FAIL", f"no reference directories under {base}")]
    validate = _model_validator()
    findings: list[str] = []
    for ref in refs:
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for path in sorted([*ref.rglob("*.yaml"), *ref.rglob("*.yml")]):
            try:
                docs = _yaml_docs(path)
            except yaml.YAMLError as exc:
                findings.append(f"{path.relative_to(root)}: YAML error {exc}")
                continue
            for doc in docs:
                kind = _kind_of(path, doc)
                by_kind.setdefault(kind, []).append(doc)
                if validate is not None and kind != "unknown":
                    try:
                        validate(kind, doc.get("body", doc))
                    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
                        findings.append(f"{path.relative_to(root)}: {kind} invalid: {exc}")
        findings += [f"{ref.name}: no {k}" for k in REFERENCE_KINDS if k not in by_kind]
        recon = by_kind.get("Reconstruction", [{}])[0]
        body = recon.get("body", recon)
        unmatched = body.get("unmatched_transactions", []) if isinstance(body, dict) else []
        gaps = len(by_kind.get("Gap", []))
        if len(unmatched) != gaps:
            findings.append(f"{ref.name}: {len(unmatched)} unmatched transactions, {gaps} Gaps")
    results = [from_findings("references", findings, f"{len(refs)} references")]
    if validate is None:
        results.append(
            CheckResult(
                "reference schemas", "WARN", "kta.core.schemas not available yet; YAML parsed only"
            )
        )
    return results


def wp13(root: Path, passed: set[str]) -> list[CheckResult]:
    """Every measured number in the note cites a source defined in a Sources list."""
    note = root / "deliveries" / "WP13.md"
    if not note.exists():
        return [CheckResult("number sources", "FAIL", "deliveries/WP13.md is missing")]
    text = note.read_text(encoding="utf-8")
    gate_body = split_sections(text).get("gate", "")
    sources = {
        m.group(1): m.group(2) for line in text.splitlines() if (m := SOURCE_DEF.match(line))
    }
    findings: list[str] = []
    for no, line in enumerate(text.splitlines(), 1):
        if line in gate_body or SOURCE_DEF.match(line) or line.lstrip().startswith("#"):
            continue
        if MEASURED_NUMBER.search(line) and not CITATION.search(line):
            findings.append(f"WP13.md:{no}: number without [S<n>] citation")
        findings += [
            f"WP13.md:{no}: {c} has no source"
            for c in CITATION.findall(line)
            if c.strip("[]") not in sources
        ]
    # Remote sources ("host:/path") cannot be checked from here; local ones must exist.
    findings += [
        f"source {sid} path does not exist: {src}"
        for sid, src in sorted(sources.items())
        if src.startswith("/") and not Path(src).exists()
    ]
    return [from_findings("number sources", findings, f"{len(sources)} sources")]


WP9A_TESTS = (
    "test_active_maps_ops_states_onto_the_task_lifecycle",
    "test_case_list_and_timeline",
    "test_lineage_builds_the_full_evidence_chain",
    "test_families_follow_snapshot_membership",
    "test_coverage_reads_the_stored_report_or_reports_missing",
    "test_trace_window_includes_spans_started_before_it",
    "test_app_serves_every_spec_read_endpoint",
    "test_machine_is_well_formed[solve]",
    "test_judging_rules_are_ordered_one_to_four",
)
WEB_FIXTURE = re.compile(r"kta:fixture|SAMPLE DATA")


def wp9a(root: Path, passed: set[str]) -> list[CheckResult]:
    """API tests pass, the web build (type check + bundle) passes, no sample data in web/."""
    names = {t.split("::")[-1] for t in passed}
    absent = [f"{t} did not pass" for t in WP9A_TESTS if t not in names]
    web = root / "web"
    if not (web / "node_modules").is_dir():
        build = CheckResult("web build", "FAIL", "web/node_modules missing: run `npm ci` in web/")
    else:
        proc = subprocess.run(
            ["npm", "run", "--silent", "build"],
            cwd=web,
            capture_output=True,
            text=True,
            check=False,
        )
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-10:]
        build = (
            CheckResult("web build", "PASS", "tsc + vite build")
            if proc.returncode == 0
            else CheckResult("web build", "FAIL", "\n".join(tail))
        )
    sample = [
        f"{p.relative_to(root)} contains sample-data markers"
        for p in sorted((web / "src").rglob("*"))
        if p.is_file() and WEB_FIXTURE.search(p.read_text(encoding="utf-8", errors="replace"))
    ]
    if (web / "src" / "sample").exists():
        sample.append("web/src/sample still exists (D95: deleted in phase B)")
    return [
        from_findings("WP9a API tests", absent, f"{len(WP9A_TESTS)} tests passed"),
        build,
        from_findings("no sample data in web/", sample, "web/src reads the API only"),
    ]


SPECIFIC: dict[str, Callable[[Path, set[str]], list[CheckResult]]] = {
    "WP9a": wp9a,
    "WP0": wp0,
    "WP1": wp1,
    "WP10a": wp10a,
    "WP13": wp13,
}


def run_specific(wp: str, root: Path, passed: set[str]) -> list[CheckResult]:
    check = SPECIFIC.get(wp)
    if check is None:
        return [
            CheckResult(
                f"{wp} specific checks",
                "FAIL",
                "not implemented in tools/kta_gate/specific.py; this WP must add them",
            )
        ]
    return check(root, passed)
