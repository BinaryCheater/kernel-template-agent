"""The WP5 gate check: named acceptance tests, seed validation, family assembly (D252)."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from kta.core.config import repo_root
from kta_gate.result import CheckResult
from kta_gate.specific import WP5_ACCEPTANCE, WP5_TEST_FAMILY, wp5

PASSED = {f"tests.acceptance.WP5.test_formulation_solve_acceptance::{t}" for t in WP5_ACCEPTANCE}


def _root(tmp: Path) -> Path:
    shutil.copytree(
        repo_root() / "knowledge" / "seed",
        tmp / "knowledge" / "seed",
        ignore=shutil.ignore_patterns("references", "families"),
    )
    shutil.copytree(repo_root() / WP5_TEST_FAMILY, tmp / WP5_TEST_FAMILY)
    return tmp


def _result(results: list[CheckResult], name: str) -> CheckResult:
    return next(r for r in results if r.name == name)


def test_gate_wp5_passes_on_the_real_seed(tmp_path: Path) -> None:
    results = wp5(_root(tmp_path), PASSED)
    assert [r.status for r in results] == ["PASS", "PASS", "PASS"], [r.render() for r in results]


def test_gate_wp5_fails_without_acceptance_tests(tmp_path: Path) -> None:
    assert _result(wp5(_root(tmp_path), set()), "WP5 acceptance tests").status == "FAIL"


def test_gate_wp5_fails_when_a_family_does_not_assemble(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / WP5_TEST_FAMILY / "chain.yaml"
    chain = yaml.safe_load(path.read_text(encoding="utf-8"))
    del chain["items"][3]["bindings"]["p.iterations"]
    path.write_text(yaml.safe_dump(chain, sort_keys=False), encoding="utf-8")
    check = _result(wp5(root, PASSED), "seed formulations assemble")
    assert check.status == "FAIL" and any("p.iterations" in f for f in check.findings)


def test_gate_wp5_fails_when_a_seed_template_is_unused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / WP5_TEST_FAMILY / "chain.yaml"
    chain = yaml.safe_load(path.read_text(encoding="utf-8"))
    chain["items"] = [i for i in chain["items"] if i["key"] != "ab_swizzle"]
    path.write_text(yaml.safe_dump(chain, sort_keys=False), encoding="utf-8")
    gen = root / WP5_TEST_FAMILY / "gen.py"
    gen.write_text(gen.read_text(encoding="utf-8").replace(', "ab_swizzle"', ""), encoding="utf-8")
    check = _result(wp5(root, PASSED), "seed formulations assemble")
    assert check.status == "FAIL"
    assert any("shared_resource_binding" in f for f in check.findings)
