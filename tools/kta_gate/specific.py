"""Work-package specific checks (spec/gate.md, second table).

A work package whose specific checks are not implemented here fails the gate: the
WP that owns them adds them, so no check is silently skipped.
"""

from __future__ import annotations

import importlib
import re
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


SPECIFIC: dict[str, Callable[[Path, set[str]], list[CheckResult]]] = {
    "WP0": wp0,
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
