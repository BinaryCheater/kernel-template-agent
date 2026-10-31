"""The WP21 gate checks catch what they claim to: a knob constant back in src/, and a row of
docs/design/12 that disagrees with the registry."""

from __future__ import annotations

import shutil
from pathlib import Path

from kta_gate.specific_wp21 import check_registry, check_replaced, check_stages

ROOT = Path(__file__).resolve().parents[2]


def test_gate_wp21_flags_a_replaced_constant(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "kta" / "agents"
    pkg.mkdir(parents=True)
    (pkg / "runner.py").write_text("LEASE_S = 1800\n")
    assert check_replaced(tmp_path).status == "FAIL"
    (pkg / "runner.py").write_text("lease = controls.lease_s\n")
    assert check_replaced(tmp_path).status == "PASS"


def test_gate_wp21_flags_a_set_by_mismatch(tmp_path: Path) -> None:
    doc = tmp_path / "docs" / "design"
    doc.mkdir(parents=True)
    shutil.copy(ROOT / "docs" / "design" / "12-agent-controls.md", doc)
    assert check_registry(tmp_path).status == "PASS"
    path = doc / "12-agent-controls.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("| spec (10 §3.2) | X5 |", "| spec (10 §3.2) | — |"))
    result = check_registry(tmp_path)
    assert result.status == "FAIL" and "retries per task" in result.findings[0]
    assert check_stages().status == "PASS"
