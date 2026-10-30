"""The WP6 ``kta`` commands through ``kta.cli.main`` in the test environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from kta.cli.main import EXIT_NOT_SUPPORTED, EXIT_OK, main

CASE = {
    "problem": {
        "name": "vadd_test",
        "reference_impl": "A + B",
        "params": {"min_work": {"dram": "12 * case.n"}},
    },
    "contract": {"timing_boundary": "kernel"},
    "case": {
        "shape_dist": [{"shape": {"n": 1024}, "weight": 1.0}],
        "objective": "weighted_latency",
        "target": "nowhere-gpu",
    },
}


@pytest.fixture
def kta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    monkeypatch.setenv("KTA_ENV", "test")
    monkeypatch.setenv("KTA_TEST_DATA", str(tmp_path / "data"))

    def run(*argv: str) -> tuple[int, Any]:
        code = main([*argv, "--json"])
        out = capsys.readouterr().out
        return code, json.loads(out) if out.strip() else None

    return run


def test_case_solve_loop_task_inbox(kta: Any, tmp_path: Path) -> None:
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(CASE), encoding="utf-8")
    code, added = kta("case", "add", str(path))
    assert code == EXIT_OK
    code, started = kta("solve", added["case"], "--steps", "3")
    assert code == EXIT_OK and started["state"] == "bounded"
    assert started["data"]["blocked"]["error"] == "NotSupported"  # no HwModel for the target
    _, loops = kta("loop", "list")
    assert [lp["blocked"] for lp in loops["loops"]] == [True]
    _, shown = kta("loop", "show", started["loop"])
    assert shown["kind"] == "solve"
    _, items = kta("inbox")
    assert [i["trigger"] for i in items["items"]] == ["Loop blocked in bounded"]
    _, tasks = kta("task", "list")
    assert tasks["tasks"] == []
    _, rec = kta("show", added["case"])
    assert rec["kind"] == "Case"
    _, why = kta("why", added["case"])
    assert set(why["nodes"]) >= {added["case"], added["contract"]}


def test_budget_command(kta: Any) -> None:
    code, b = kta("budget", "global")
    assert code == EXIT_OK and b["limit"] is None
    _, b = kta("budget", "global", "--set", "agent_min=60")
    assert b["limit"] == {"agent_min": 60.0} and b["remaining"] == {"agent_min": 60.0}


def test_agent_run_and_src_are_not_supported_yet(kta: Any) -> None:
    assert kta("run", "x")[0] == EXIT_NOT_SUPPORTED
    assert kta("src", "x")[0] == EXIT_NOT_SUPPORTED


def test_unimplemented_process_exits_not_supported(kta: Any) -> None:
    assert kta("process", "start", "learn", "--ticks", "1")[0] == EXIT_NOT_SUPPORTED


def test_human_task_create_submit_check(kta: Any, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "DELIVERY.md").write_text("# What was done\nnothing\n", encoding="utf-8")
    code, created = kta("task", "create", "T3", "--goal", "extract")
    assert code == EXIT_OK and created["work_unit"].startswith("human:")
    code, verdict = kta("task", "submit", created["task"], "--dir", str(out), "--check")
    assert code == 1 and verdict["state"] == "queued"  # rejected: no template.yaml, short note
    failed = {c["name"] for c in verdict["checks"] if not c["passed"]}
    assert failed == {"delivery_note", "validate_template", "template_fills_chain"}
    _, shown = kta("task", "show", created["task"])
    assert shown["attempts"] == 1 and shown["state"] == "queued"


def test_client_mode_refuses_local_store_commands(
    kta: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KTA_TASK_TOKEN", "kta1.test")
    assert kta("show", "a" * 64)[0] == EXIT_NOT_SUPPORTED
    monkeypatch.delenv("KTA_API", raising=False)
    assert kta("note", "hello", "--task", "T")[0] == 2  # usage: KTA_API is not set


def test_agents_runner_profile_unit_commands_exist() -> None:
    from kta.cli.main import build_parser

    p = build_parser()
    for argv in (
        ["agents", "env", "verify", "b"],
        ["runner", "start", "--ticks", "1"],
        ["profile", "approve", "p", "--reason", "r"],
        ["unit", "close", "w"],
    ):
        assert callable(p.parse_args(argv).func)


def test_seed_case_with_a_baseline_needs_a_target(kta: Any) -> None:
    code, _ = kta("case", "add", "gemm_fp16")  # knowledge/seed/cases/gemm_fp16.yaml
    assert code == EXIT_NOT_SUPPORTED  # the baseline needs the worker image of a Target


def test_runner_skips_tasks_without_an_approved_profile(kta: Any) -> None:
    from kta.cli.agent_runner import next_task
    from kta.cli.main import context_from_environ
    from kta.orchestrator.pipeline import Services
    from kta.orchestrator.tasks import TaskQueue

    kta("task", "create", "T3", "--goal", "g")
    svc = Services.open(context_from_environ("test"))
    q = TaskQueue(svc.ctx, svc.store, svc.ops, svc)
    assert next_task(q) is None  # card profile is "unassigned"
