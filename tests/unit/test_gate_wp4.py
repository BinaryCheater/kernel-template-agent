"""The WP4 gate check: seed validation and named acceptance tests (decision D215)."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from kta.core.config import repo_root
from kta_gate.result import CheckResult
from kta_gate.specific import WP4_ACCEPTANCE, wp4

PASSED = {f"tests.acceptance.WP4.test_knowledge_acceptance::{t}" for t in WP4_ACCEPTANCE}


def _root(tmp: Path) -> Path:
    shutil.copytree(
        repo_root() / "knowledge" / "seed",
        tmp / "knowledge" / "seed",
        ignore=shutil.ignore_patterns("references", "families"),
    )
    return tmp


def _status(results: list[CheckResult], name: str) -> str:
    return next(r.status for r in results if r.name == name)


def test_gate_wp4_passes_on_the_real_seed(tmp_path: Path) -> None:
    results = wp4(_root(tmp_path), PASSED)
    assert [r.status for r in results] == ["PASS", "PASS"], [r.render() for r in results]


def test_gate_wp4_fails_without_acceptance_tests(tmp_path: Path) -> None:
    results = wp4(_root(tmp_path), set())
    assert _status(results, "WP4 acceptance tests") == "FAIL"


def test_gate_wp4_fails_on_an_invalid_seed_template(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / "knowledge" / "seed" / "templates" / "async_transfer.yaml"
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    body["constraints"][0]["expr"] = "granularity_bytes <= undeclared_name"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    results = wp4(root, PASSED)
    seed = next(r for r in results if r.name == "seed validates")
    assert seed.status == "FAIL" and any("undeclared_name" in f for f in seed.findings)


def test_gate_wp4_fails_when_a_seed_template_is_missing(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "knowledge" / "seed" / "templates" / "physical_storage.yaml").unlink()
    # multi_buffer_rotation and shared_resource_binding point at physical_storage, so the
    # import itself is refused.
    assert _status(wp4(root, PASSED), "seed validates") == "FAIL"


def test_gate_wp4_reports_broken_references_as_warning(tmp_path: Path) -> None:
    root = _root(tmp_path)
    ref = root / "knowledge" / "seed" / "references" / "r1"
    ref.mkdir(parents=True)
    (ref / "source.yaml").write_text("name: s\ntype: git\nlocator: x\nrevision: r\n")
    (ref / "problem.yaml").write_text("name: p\nreference_impl: torch\n")
    (ref / "reference.yaml").write_text("name: r1\nproblem: p\nbuild: make\n")
    (ref / "chain.yaml").write_text(
        "workload: w\nitems:\n- {key: a, template: nope@1, applies_to: x}\n"
    )
    (ref / "reconstruction.yaml").write_text("matched: 0\ntotal: 1\nnotes: n\n")
    results = wp4(root, PASSED)
    assert _status(results, "seed validates") == "PASS"
    assert _status(results, "reference decompositions") == "WARN"
