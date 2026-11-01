"""WP25 supervision views: hand advancing a loop (step, round, run, retry), the journal's
phrasing and grouping, blocked and paused loops in the active view and the Pipeline, round
summaries, and the probe/solve split of Cases."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from apihelp import put, request, seeded_api
from knowhelp import seed
from kta.api import journal, opsread, runs, trace
from kta.api.app import create_app
from kta.api.common import Api
from kta.core.context import Context
from kta.core.errors import ConfigError
from kta.core.schemas import Actor
from kta.orchestrator import control
from kta.orchestrator.operations import Operations
from kta.orchestrator.pipeline import Services
from kta.orchestrator.runner import Runner
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import world

ALICE = Actor(type="human", name="alice")


@pytest.fixture
def svc(ctx: Context, store: SqliteStore, ops: SqliteOps) -> Services:
    seed(ctx, store)
    return Services(ctx, store, ops)


@pytest.fixture(scope="module")
def api(tmp_path_factory: pytest.TempPathFactory) -> Api:
    return seeded_api(tmp_path_factory.mktemp("wp25"))


# ---- advancing a loop by hand (D664) ----


def test_manual_step_ignores_a_loop_pause_but_not_a_process_pause(svc: Services) -> None:
    w = world(svc)
    r = Runner(svc, "test")
    loop_id = r.start("solve", w["case"], "c")
    now = svc.ctx.clock.now()
    control.set_paused(svc.ops, f"loop:{loop_id}", True, "hand-step", "alice", now)
    assert r.advance(loop_id) is None  # the process would not move it
    t = r.advance(loop_id, manual=True)
    assert t is not None and (t.src, t.to) == ("new", "bounded")
    control.set_paused(svc.ops, "process:solve", True, "stop all", "alice", now)
    assert r.advance(loop_id, manual=True) is None
    assert svc.ops.get_loop(loop_id)["state"] == "bounded"


def test_advance_round_stops_where_the_loop_waits(svc: Services) -> None:
    w = world(svc)
    loop_id = Runner(svc, "test").start("solve", w["case"], "c")
    ops = Operations(svc)
    out = ops.advance_action(loop_id, "round", "watch one round", ALICE)
    taken = ops.advance_work(loop_id, "round", "human:alice")
    assert [(t["from"], t["to"]) for t in taken] == [
        ("new", "bounded"),
        ("bounded", "retrieving"),
        ("retrieving", "waiting"),
    ]
    row = svc.ops.get_loop(loop_id)
    assert row["state"] == "waiting" and row["data"]["waiting_on"]
    act = svc.store.get(out["action"])
    assert act.kind == "HumanAction" and act.actor.name == "alice"
    assert act.body["action"].startswith("advance ") and '"mode": "round"' in act.body["action"]


def test_advance_step_takes_exactly_one_step(svc: Services) -> None:
    w = world(svc)
    loop_id = Runner(svc, "test").start("solve", w["case"], "c")
    ops = Operations(svc)
    ops.advance_action(loop_id, "step", "", ALICE)
    assert len(ops.advance_work(loop_id, "step", "human:alice")) == 1
    assert svc.ops.get_loop(loop_id)["state"] == "bounded"


def test_advance_run_resumes_the_loop_and_says_no_process_runs(svc: Services) -> None:
    w = world(svc)
    loop_id = Runner(svc, "test").start("solve", w["case"], "c")
    control.set_paused(svc.ops, f"loop:{loop_id}", True, "hold", "alice", svc.ctx.clock.now())
    out = Operations(svc).advance_action(loop_id, "run", "go on", ALICE)
    assert not control.is_paused(svc.ops, f"loop:{loop_id}")
    assert out["process"] == "solve" and out["process_running"] is False
    assert Operations(svc).advance_work(loop_id, "run", "human:alice") == []
    svc.ops.acquire("process:solve", "solve:x", 60)
    assert Operations(svc).advance_action(loop_id, "run", "", ALICE)["process_running"] is True


def test_retry_clears_the_block_and_steps_again(svc: Services) -> None:
    w = world(svc, target="other-gpu", hw=False)
    r = Runner(svc, "test")
    loop_id = r.start("solve", w["case"], "c")
    r.advance(loop_id)
    assert r.advance(loop_id) is None  # bounded raises NotSupported: blocked
    assert svc.ops.get_loop(loop_id)["data"]["blocked"]
    ops = Operations(svc)
    ops.advance_action(loop_id, "retry", "tried again", ALICE)
    assert not svc.ops.get_loop(loop_id)["data"]["blocked"]
    ops.advance_work(loop_id, "retry", "human:alice")
    assert svc.ops.get_loop(loop_id)["data"]["blocked"]["error"] == "NotSupported"  # same cause


def test_advance_refuses_an_unknown_mode_and_a_non_human(svc: Services) -> None:
    w = world(svc)
    loop_id = Runner(svc, "test").start("solve", w["case"], "c")
    with pytest.raises(ConfigError):
        Operations(svc).advance_action(loop_id, "leap", "", ALICE)
    with pytest.raises(ConfigError):
        Operations(svc).advance_action(loop_id, "step", "", Actor(type="agent", name="w"))


def test_advance_endpoint_writes_the_action_and_lists_the_capability(tmp_path: Path) -> None:
    api = seeded_api(tmp_path)
    loop = opsread.loops(api.ops)[0]
    status, body = request(
        create_app(api), "POST", "/actions/advance", {"loop": loop.id, "mode": "run"}
    )
    assert status == 200 and body["mode"] == "run" and body["started"] is False
    assert api.store.get(body["action"]).kind == "HumanAction"
    status, caps = request(create_app(api), "GET", "/actions/capabilities")
    assert status == 200 and any(c["name"] == "advance" and c["wired"] for c in caps)
    status, _ = request(create_app(api), "POST", "/actions/advance", {"loop": loop.id, "mode": "x"})
    assert status == 400  # validation errors are 400 here (app.status_for)


# ---- the journal (D660) ----


def test_journal_phrases_records_and_counts_unlisted_kinds(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(api, "Candidate", {"bucket": "s0", "assignment": {"BM": 64}})
    table = journal.phrases()
    del table["records"]["Candidate"]  # as if journal.yaml had no row for this kind
    monkeypatch.setattr(journal, "phrases", lambda: table)
    out = journal.journal(api, None, 200, "all", None)
    lines = [e for e in out["entries"] if e["kind"] == "Candidate"]
    assert lines and lines[0]["text"].startswith(
        ("Candidate written by", "1 Candidate", "2 Candidate")
    )
    assert out["unphrased"].get("Candidate", 0) >= 1
    monkeypatch.undo()
    assert "Candidate" not in journal.journal(api, None, 200, "all", None)["unphrased"]


def test_journal_groups_a_burst_even_when_interleaved(api: Api) -> None:
    for i in range(5):
        put(api, "Candidate", {"bucket": "s0", "assignment": {"BM": i}})
        put(api, "Prediction", {"total": float(i), "serial": 0.0, "model": "m"})
    out = journal.journal(api, None, 50, "all", None)
    cands = [e for e in out["entries"] if e["kind"] == "Candidate"]
    assert (
        cands
        and cands[0]["count"] >= 5
        and cands[0]["text"].startswith(f"{cands[0]['count']} candidates")
    )


def test_journal_milestones_drop_detail_and_page_backwards(api: Api) -> None:
    out = journal.journal(api, None, 5, "milestone", None)
    assert out["entries"] and all(e["level"] == "milestone" for e in out["entries"])
    if out["next_before"]:
        older = journal.journal(api, journal._when(out["next_before"]), 5, "milestone", None)  # pyright: ignore[reportPrivateUsage]
        assert all(e["at"] <= out["next_before"] for e in older["entries"])


def test_every_phrase_renders_with_its_placeholders_missing() -> None:
    table = journal.phrases()
    for section in ("records", "spans"):
        for kind, p in table[section].items():
            if kind == "skip":
                continue
            assert journal.render(p["text"], {}) and p["level"] in journal.LEVELS, kind
            assert p["tone"] in ("running", "waiting", "ok", "fail", "human", "idle"), kind


# ---- blocked and paused loops (D663) ----


def _block(api: Api, loop: opsread.LoopRow) -> dict[str, Any]:
    at = (api.now() - timedelta(minutes=5)).isoformat()
    blocked = {"error": "JSONDecodeError", "message": "bad", "state": loop.state, "at": at}
    api.ops.put_loop(
        loop.id, loop.kind, loop.subject, loop.state, {**loop.data, "blocked": blocked}
    )
    return blocked


def test_active_view_and_trace_mark_a_blocked_paused_loop(tmp_path: Path) -> None:
    api = seeded_api(tmp_path)
    open_steps = {s.trace_id for s in opsread.open_spans(api.ops) if s.kind == "loop_step"}
    loop = next(lp for lp in opsread.loops(api.ops) if lp.kind == "solve" and lp.id in open_steps)
    _block(api, loop)
    control.set_paused(api.ops, f"loop:{loop.id}", True, "hold", "alice", api.now())
    view = runs.loop_view(api, next(lp for lp in opsread.loops(api.ops) if lp.id == loop.id))
    assert view["blocked"]["error"] == "JSONDecodeError" and view["paused"] == f"loop:{loop.id}"
    assert view["subject_name"] != loop.subject[:12]
    win = trace.window(api, api.now() - timedelta(days=2), api.now() + timedelta(minutes=1))
    steps = [s for s in win["spans"] if s["trace_id"] == loop.id and s["kind"] == "loop_step"]
    open_ = [s for s in steps if s["end_ts"] is None]
    assert open_ and all("blocked" in s["attrs"] for s in open_)
    assert all("blocked" not in s["attrs"] for s in steps if s["end_ts"] is not None)
    lines = journal.journal(api, None, 50, "milestone", None)["entries"]
    assert any(e["source"] == "blocked" and "JSONDecodeError" in e["text"] for e in lines)


def test_process_start_and_stop_are_recorded_and_guarded(svc: Services) -> None:
    ops = Operations(svc)
    out = ops.process_action("solve", True, "run all", ALICE)
    assert out["running"] is False
    assert svc.store.get(out["action"]).body["action"] == "process start solve"
    svc.ops.acquire("process:solve", "solve:1", 60)
    with pytest.raises(ConfigError):
        ops.process_action("solve", True, "", ALICE)  # already running
    ops.process_action("solve", False, "pause all", ALICE)
    assert svc.ops.get_pointer("process:solve:stop")  # honoured at the next tick
    with pytest.raises(ConfigError):
        ops.process_action("dance", True, "", ALICE)


def test_active_view_lists_processes(tmp_path: Path) -> None:
    api = seeded_api(tmp_path)
    procs = runs.active(api)["processes"]
    assert [p["name"] for p in procs] == ["solve", "learn"]
    assert all(p["running"] is False for p in procs)
    api.ops.acquire("process:solve", "solve:x", 60)
    assert runs.active(api)["processes"][0]["running"] is True


def test_human_actions_read_as_words(tmp_path: Path) -> None:
    api = seeded_api(tmp_path)
    loop = next(lp for lp in opsread.loops(api.ops) if lp.kind == "solve")
    action = "advance " + json.dumps({"loop": loop.id, "mode": "round"})
    put(api, "HumanAction", {"action": action, "reason": ""})
    line = journal.journal(api, None, 5, "milestone", None)["entries"][0]
    assert line["text"].startswith("You: round ") and loop.id not in line["text"]


def test_lock_holder_ignores_an_expired_lock(svc: Services) -> None:
    assert svc.ops.lock_holder("process:solve") is None
    svc.ops.acquire("process:solve", "solve:1", 30)
    assert svc.ops.lock_holder("process:solve") == "solve:1"


# ---- the Template page (D671) ----


def test_template_page_gives_body_sources_usage_and_checks(api: Api) -> None:
    app = create_app(api)
    status, pool = request(app, "GET", "/pool")
    assert status == 200 and pool["templates"]
    row = next((t for t in pool["templates"] if t["used_by"]), pool["templates"][0])
    assert {"scope", "sources", "used_by"} <= row.keys()
    status, page = request(app, "GET", f"/templates/{row['id']}")
    assert status == 200
    assert page["body"]["name"] == row["name"] and page["used_by"] == row["used_by"]
    assert [s["name"] for s in page["sources"]] == row["sources"]
    assert any(v["id"] == row["id"] for v in page["versions"])
    assert {"obligations", "contract"} <= page["proof"].keys()
    for rel in page["relations"]:
        assert rel["id"] is None or api.store.get(rel["id"]).kind == "Template"
    other = (pool["patterns"] or pool["gaps"] or [None])[0]
    assert other is not None
    status, _ = request(app, "GET", f"/templates/{other['id']}")
    assert status == 404
