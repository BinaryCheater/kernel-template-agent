"""Source replay estimator, coverage, matched seeds and retry-bound invariants."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kta_source_replay.evaluate import allowed_retry, initialize, request_for
from kta_source_replay.worker import drift, run, stats, window_summary

DESIGN = (
    Path(__file__).resolve().parents[2]
    / "docs/runbooks/learndemo-1/controlled-replay/campaign.json"
)


def test_ratio_of_medians_inclusive_iqr_and_drift() -> None:
    campaign = json.loads(DESIGN.read_text())
    original = [10.0] * 30
    candidate = [20.0] * 15 + [5.0] * 15
    result = window_summary(original, candidate, campaign["timing"])
    assert result["ratio"] == 0.8
    assert result["time_difference_percent"] == 25
    assert result["valid"]
    assert stats([1.0, 2.0, 3.0, 4.0, 100.0])["iqr_us"] == 2
    assert drift([100.0, 105.0]) == pytest.approx(0.05)


def test_original_evidence_alone_controls_window_validity() -> None:
    campaign = json.loads(DESIGN.read_text())
    assert not window_summary([10.0] * 15 + [11.0] * 15, [1.0] * 30, campaign["timing"])["valid"]
    noisy = [8.0, 12.0] * 15
    assert not window_summary(noisy, [100.0] * 30, campaign["timing"])["valid"]
    for values in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, math.nan], [1.0, 2.0, 3.0, 0.0]):
        with pytest.raises(ValueError):
            stats(values)


def test_schedule_matches_arms_by_block_submission_and_withholds_hidden_seeds(
    tmp_path: Path,
) -> None:
    initialize(tmp_path, DESIGN)
    (tmp_path / "baseline.json").write_text(
        json.dumps(
            {"remote_path": "/baseline", "identity": {"binding.py": "a", "original.so": "b"}}
        )
    )
    campaign = json.loads(DESIGN.read_text())
    a = request_for(tmp_path, campaign, "run-01", "development", 1, 0)
    b = request_for(tmp_path, campaign, "run-03", "development", 1, 0)
    c = request_for(tmp_path, campaign, "run-01", "development", 2, 0)
    assert a["seeds"] == b["seeds"] != c["seeds"]
    final = request_for(tmp_path, campaign, "run-01", "final", 1, 1)
    seeds = json.loads((tmp_path / "seeds.json").read_text())
    development = {
        seed
        for block in seeds["development"].values()
        for submission in block.values()
        for values in submission.values()
        for seed in values
    }
    hidden = {seed for values in final["seeds"].values() for seed in values}
    assert not hidden & development
    assert len(hidden) == 40


def test_invalid_coverage_rejected_before_any_gpu_or_model(tmp_path: Path) -> None:
    campaign = json.loads(DESIGN.read_text())
    for request in (
        {"mode": "unknown"},
        {"mode": "development", "seeds": {}},
        {"mode": "final", "seeds": {p: [1, 2] for p in campaign["validation"]["patterns"]}},
    ):
        with pytest.raises(ValueError):
            run(campaign, request)


def test_external_retry_limits_cover_entire_slot_and_window(tmp_path: Path) -> None:
    assert allowed_retry(tmp_path, 0)
    assert not allowed_retry(tmp_path, 2)
    for name in ("discovery", "confirmation"):
        path = tmp_path / name
        path.mkdir()
        (path / "external-retry.json").write_text("{}")
    assert not allowed_retry(tmp_path, 0)


def test_candidate_rejects_delegation_and_cpu_extraction() -> None:
    from kta_source_replay.worker import audit_candidate

    audit_candidate("import torch\nimport tilelang as tl\nx = torch.empty((1,), device='cuda')")
    for source in (
        "import flash_attn",
        "import torch as th\nth.matmul(q,k)",
        "from torch import einsum as contract\ncontract('bij,bjk->bik',q,k)",
        "x.cpu()",
        "open('/baseline/binding.py')",
    ):
        with pytest.raises(ValueError):
            audit_candidate(source)
    audit_candidate("import os, sys, pathlib, inspect, importlib, subprocess")
    audit_candidate("from torch import einsum\nx = 1")
    audit_candidate("exec('x = 1')")


def test_competing_gpu_processes_requires_full_monitor_evidence(tmp_path: Path) -> None:
    from kta_source_replay.remote import competing_processes

    (tmp_path / "processes-before.txt").write_text("")
    (tmp_path / "process-monitor.txt").write_text(
        "sample 1 allowed 101\n101\nsample 2 allowed \n101\n202\n"
    )
    assert competing_processes(tmp_path) == ["202"]
    (tmp_path / "processes-before.txt").write_text("303\n")
    assert competing_processes(tmp_path) == ["202", "303"]


def test_worker_remaining_budget_counts_transport_and_excludes_queue_once() -> None:
    from kta_source_replay.remote import remaining_after_queue

    assert remaining_after_queue(20, 12, 10) == 18
    assert remaining_after_queue(3, 8, 1) == 0
    assert remaining_after_queue(20, 20, 20) == 20
    for values in ((math.inf, 1, 0), (20, -1, 0), (20, 1, math.nan)):
        with pytest.raises(ValueError):
            remaining_after_queue(*values)
