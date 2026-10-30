"""The WP6 ``kta`` commands through ``kta.cli.main`` in the test environment."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

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


def test_learn_process_runs_one_tick(kta: Any) -> None:
    assert kta("process", "start", "learn", "--ticks", "1", "--interval", "0")[0] == EXIT_OK


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


class _Api(BaseHTTPRequestHandler):
    """Records requests to the API's agent endpoints and answers like WP9b's agentapi."""

    seen: ClassVar[list[dict[str, Any]]] = []

    def _answer(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        self.seen.append(
            {
                "method": self.command,
                "path": self.path,
                "body": body,
                "auth": self.headers.get("Authorization"),
                "parent": self.headers.get("X-KTA-Trace-Parent"),
            }
        )
        code, out = (501, {"detail": "no"}) if "/build" in self.path else (200, {"ok": True})
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _answer

    def log_message(self, *args: Any) -> None:  # quiet test server
        return None


def test_client_mode_talks_to_the_agent_api(
    kta: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("KTA_TASK_TOKEN", "kta1.test")
        monkeypatch.setenv("KTA_TRACE_PARENT", "trace:span")
        monkeypatch.setenv("KTA_API", f"http://127.0.0.1:{server.server_port}")
        assert kta("show", "a" * 64)[0] == EXIT_OK
        out = tmp_path / "out"
        out.mkdir()
        (out / "DELIVERY.md").write_text("# What was done\n", encoding="utf-8")
        (out / "family.yaml").write_text("name: f\n", encoding="utf-8")
        assert kta("submit", "--dir", str(out))[0] == EXIT_OK
        args = ["fam", "--assignment", "{}", "--shape", "{}"]
        assert kta("build", *args)[0] == EXIT_NOT_SUPPORTED  # the API answered 501
        assert kta("run", "x")[0] == EXIT_NOT_SUPPORTED  # no endpoint at all
    finally:
        server.shutdown()
    show, submit, build = _Api.seen[:3]
    assert (show["method"], show["path"]) == ("GET", "/agent/records/" + "a" * 64)
    assert show["auth"] == "Bearer kta1.test" and show["parent"] == "trace:span"
    assert submit["path"] == "/agent/submit" and list(submit["body"]["files"]) == ["family.yaml"]
    assert base64.b64decode(submit["body"]["files"]["family.yaml"]) == b"name: f\n"
    assert submit["body"]["note"].startswith("# What was done")
    assert build["body"] == {"family": "fam", "assignment": {}, "shape": {}}


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
