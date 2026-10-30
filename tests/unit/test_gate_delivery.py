from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kta.core.config import repo_root
from kta_gate import delivery, scans
from kta_gate.specific import wp13

HEAD = subprocess.run(
    ["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()


def _note(root: Path, wp: str, body: str) -> None:
    (root / "deliveries").mkdir(exist_ok=True)
    (root / "deliveries" / f"{wp}.md").write_text(body)


FULL_NOTE = """# WPx
## What was done
x
## Fixtures and simplifications
None
## Known issues
None
## New decisions
D01
## Needed changes
None
## Gate output
...
"""


def _decisions(root: Path, rows: str, queue: str = "") -> None:
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "decisions.md").write_text(
        "| ID | a | b | c |\n| --- | --- | --- | --- |\n" + rows
    )
    (root / "docs" / "review-queue.md").write_text(queue)


def test_delivery_sections(tmp_path: Path) -> None:
    _note(tmp_path, "WPx", FULL_NOTE)
    assert delivery.check_delivery_sections(tmp_path, "WPx").status == "PASS"
    _note(tmp_path, "WPx", FULL_NOTE.replace("## Known issues", "## Issues"))
    assert delivery.check_delivery_sections(tmp_path, "WPx").status == "FAIL"
    chinese = FULL_NOTE.replace("What was done", "做了什么")
    _note(tmp_path, "WPx", chinese)
    assert delivery.check_delivery_sections(tmp_path, "WPx").status == "PASS"


def test_decision_rows_and_review_queue(tmp_path: Path) -> None:
    _note(tmp_path, "WPx", FULL_NOTE)
    _decisions(tmp_path, "| D01 | a | b | WPx |\n")
    assert delivery.check_decisions(tmp_path, "WPx").status == "PASS"
    _decisions(tmp_path, "| D01 | a | b |\n")
    assert delivery.check_decisions(tmp_path, "WPx").status == "FAIL"
    _decisions(tmp_path, "| D01 | a | default — needs user review | WPx |\n")
    assert delivery.check_decisions(tmp_path, "WPx").status == "FAIL"
    _decisions(tmp_path, "| D01 | a | default — needs user review | WPx |\n", "## D01 ...")
    assert delivery.check_decisions(tmp_path, "WPx").status == "PASS"
    _note(tmp_path, "WPx", FULL_NOTE.replace("D01", "D99"))
    assert delivery.check_decisions(tmp_path, "WPx").status == "FAIL"


def test_fixture_marker_must_be_declared(tmp_path: Path) -> None:
    (tmp_path / "src" / "kta").mkdir(parents=True)
    (tmp_path / "src" / "kta" / "x.py").write_text("A = 1  # kta:fixture no hardware yet\n")
    _note(tmp_path, "WPx", FULL_NOTE)
    assert delivery.check_fixture_declarations(tmp_path, "WPx").status == "FAIL"
    _note(tmp_path, "WPx", FULL_NOTE.replace("None\n## Known", "src/kta/x.py: why\n## Known"))
    assert delivery.check_fixture_declarations(tmp_path, "WPx").status == "PASS"
    assert scans.fixture_markers(tmp_path) == [("src/kta/x.py", 1)]
    # Declared by the WP that added it: a later WP's note need not repeat it.
    _note(tmp_path, "WPy", FULL_NOTE)
    assert delivery.check_fixture_declarations(tmp_path, "WPy").status == "PASS"


def test_lock_detects_modification(tmp_path: Path) -> None:
    import hashlib

    (tmp_path / "tests" / "acceptance").mkdir(parents=True)
    locked = tmp_path / "tests" / "acceptance" / "t.py"
    locked.write_text("assert True\n")
    digest = hashlib.sha256(locked.read_bytes()).hexdigest()
    (tmp_path / "tests" / "acceptance" / "LOCK").write_text(f"{digest}  tests/acceptance/t.py\n")
    assert delivery.check_lock(tmp_path).status == "PASS"
    locked.write_text("assert 1\n")
    assert delivery.check_lock(tmp_path).status == "FAIL"


def _claim(**over: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "claim": "c",
        "command": "kta doctor",
        "git_commit": HEAD,
        "dirty": False,
        "env": "dev",
        "control_host": "h",
        "workers": [],
        "record_ids": [],
        "blob_ids": [],
        "env_manifests": [],
        "started_at": "2026-10-30T00:00:00+00:00",
    }
    claim.update(over)
    return claim


def _check(tmp_path: Path, wp: str, claims: list[dict[str, object]]) -> str:
    evidence = tmp_path / f"{wp}.evidence.json"
    evidence.write_text(json.dumps(claims))
    return delivery.check_evidence(repo_root(), wp, tmp_path, path=evidence).status


def test_evidence_format(tmp_path: Path) -> None:
    assert _check(tmp_path, "WP0", [_claim()]) == "PASS"
    assert _check(tmp_path, "WP0", []) == "WARN"
    assert _check(tmp_path, "WP0", [_claim(dirty=True)]) == "FAIL"
    assert _check(tmp_path, "WP0", [_claim(git_commit="0" * 40)]) == "FAIL"
    assert _check(tmp_path, "WP0", [_claim(env="main")]) == "FAIL"
    assert _check(tmp_path, "WP0", [{"claim": "only"}]) == "FAIL"


def test_store_backed_wp_needs_records(tmp_path: Path) -> None:
    assert _check(tmp_path, "WP1", [_claim()]) == "FAIL"
    assert _check(tmp_path, "WP1", [_claim(record_ids=["abc"])]) == "FAIL"
    assert _check(tmp_path, "WP1", []) == "FAIL"


def test_wp13_numbers_need_sources(tmp_path: Path) -> None:
    src = tmp_path / "raw.json"
    src.write_text("{}")
    _note(tmp_path, "WP13", FULL_NOTE + f"\n## Sources\n- [S1] {src}\n")
    assert wp13(tmp_path, set())[0].status == "PASS"
    _note(tmp_path, "WP13", FULL_NOTE.replace("x\n", "Reading took 41.5% of time.\n"))
    assert wp13(tmp_path, set())[0].status == "FAIL"
    cited = FULL_NOTE.replace("x\n", "Reading took 41.5% of time [S1].\n")
    _note(tmp_path, "WP13", cited + f"\n## Sources\n- [S1] {src}\n")
    assert wp13(tmp_path, set())[0].status == "PASS"


def test_code_stage_requires_pending_section(tmp_path: Path) -> None:
    body = "\n".join(f"## {names[0]}\nx\n" for names in delivery.SECTIONS.values())
    (tmp_path / "deliveries").mkdir()
    note = tmp_path / "deliveries" / "WPx.md"
    note.write_text(body, encoding="utf-8")
    assert delivery.check_delivery_sections(tmp_path, "WPx").status == "PASS"
    assert delivery.check_delivery_sections(tmp_path, "WPx", code_stage=True).status == "FAIL"
    note.write_text(body + "## Pending real runs\n- the solve run\n", encoding="utf-8")
    assert delivery.check_delivery_sections(tmp_path, "WPx", code_stage=True).status == "PASS"


def test_code_stage_evidence_needs_no_records(tmp_path: Path) -> None:
    evidence = tmp_path / "e.json"
    evidence.write_text("[]", encoding="utf-8")
    assert delivery.check_evidence(repo_root(), "WP7", None, path=evidence).status == "FAIL"
    result = delivery.check_evidence(repo_root(), "WP7", None, path=evidence, code_stage=True)
    assert result.status == "WARN"


def test_stage_context() -> None:
    from kta_gate import specific

    assert not specific.code_stage()
    with specific.stage("code"):
        assert specific.code_stage()
        assert specific.pending_real("x", "y").status == "WARN"
    assert not specific.code_stage()
