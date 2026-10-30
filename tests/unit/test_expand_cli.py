"""The expand loop through the ``kta`` commands a human uses (WP10b): WP10a's CUTLASS
decomposition is resubmitted as the T2 and T3 deliveries, from ``frozen`` to ``covered``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from kta.cli.main import EXIT_OK, main
from replayhelp import NOTE, REFERENCES, TEMPLATE_IN_GOAL, in_dependency_order, t2_files

REF = "cutlass_ampere_tf32_gemm"


@pytest.fixture
def kta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    monkeypatch.setenv("KTA_ENV", "test")
    monkeypatch.setenv("KTA_TEST_DATA", str(tmp_path / "data"))

    def run(*argv: str) -> Any:
        code = main([*argv, "--json"])
        out = capsys.readouterr().out
        assert code == EXIT_OK, out
        return json.loads(out) if out.strip() else None

    return run


def _write(root: Path, files: dict[str, bytes], what: str) -> Path:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    (root / "DELIVERY.md").write_text(NOTE.format(ref=REF, what=what), encoding="utf-8")
    return root


def _t3(source: str, goal: str) -> dict[str, bytes]:
    m = TEMPLATE_IN_GOAL.search(goal)
    assert m
    draft = yaml.safe_load((REFERENCES / REF / "templates" / f"{m.group(1)[:-2]}.yaml").read_text())
    doc = {"template": draft, "sources": [source]}
    return {"template.yaml": yaml.safe_dump(doc, sort_keys=False).encode()}


def test_expand_cutlass_through_the_cli(kta: Any, tmp_path: Path) -> None:
    kta("init")
    d = REFERENCES / REF
    doc = {
        "source": yaml.safe_load((d / "source.yaml").read_text()),
        "reference": yaml.safe_load((d / "reference.yaml").read_text()),
    }
    (tmp_path / "ref.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    added = kta("reference", "add", str(tmp_path / "ref.yaml"))
    started = kta(
        "expand", added["reference"], "--analysis-only", "real runs paused (D320)", "--steps", "2"
    )
    loop = started["loop"]
    assert started["state"] == "analyzing" and started["data"]["reference_point"] is None
    n = 0

    def deliver(state: str) -> None:
        nonlocal n
        tasks = [t for t in kta("task", "list", "--loop", loop)["tasks"] if t["state"] == "queued"]
        files = {}
        for t in tasks:
            files[t["id"]] = t2_files(REF) if t["type"] == "T2" else _t3(added["source"], t["goal"])
        for tid in in_dependency_order(files):
            n += 1
            out = _write(tmp_path / f"out{n}", files[tid], tid)
            assert kta("task", "submit", tid, "--dir", str(out), "--check")["state"] == "accepted"
        assert kta("loop", "step", loop)["state"] == state

    deliver("gap")  # T2 accepted: 9 Templates to extract
    assert len(kta("task", "list", "--loop", loop, "--state", "queued")["tasks"]) == 9
    deliver("analyzing")  # every T3 accepted: a re-match T2
    deliver("reconstructed")
    assert kta("loop", "step", loop)["state"] == "covered"
    cov = kta("coverage")
    assert cov["reconstructions"] == 1 and cov["templates_missing"] == []
    shown = kta("loop", "show", loop)
    assert shown["terminal"] is True and shown["data"]["analysis_only"].startswith("set by you")
