"""Operation endpoints (spec/modules/api-web.md acceptance): each writes a correct HumanAction;
a missing required reason is a 400; operations not wired yet are a 501 that writes nothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apihelp import put, request, seeded_api
from kta.agents.profile import write_profile
from kta.api import operate, opsread
from kta.api.app import create_app
from kta.api.common import Api
from kta.api.wiring import HUMAN_PRIORITY, NEEDS


@pytest.fixture
def api(tmp_path: Path) -> Api:
    return seeded_api(tmp_path)


def call(
    api: Api, path: str, body: Any = None, method: str = "POST", **headers: str
) -> tuple[int, Any]:
    return request(create_app(api), method, path, body, headers)


def actions(api: Api) -> list[dict[str, Any]]:
    return [r.to_json_dict() for r in api.store.query("HumanAction", order="created_at")]


def _id(api: Api, kind: str, **where: Any) -> str:
    with api.store.db.read() as con:
        rows = con.execute("SELECT id FROM records WHERE kind = ? ORDER BY created_at", (kind,))
        ids = [r[0] for r in rows.fetchall()]
    for rid in ids:
        body = api.store.get_json(rid)["body"]
        if all(body.get(k) == v for k, v in where.items()):
            return rid
    raise LookupError(f"no {kind} {where}")


def test_every_spec_operation_is_served(api: Api) -> None:
    paths = set(create_app(api).openapi()["paths"])
    spec = {
        f"/actions/{n}"
        for n in (
            "pause", "resume", "cancel", "retry", "budget", "priority", "create-task", "reject",
            "flag", "approve", "confirm",
        )
    }  # fmt: skip
    assert spec | {"/write/record", "/actions/deprecate"} <= paths


def test_reject_writes_human_action_and_marks_dependents_stale(api: Api) -> None:
    tid = _id(api, "Template")
    status, out = call(api, "/actions/reject", {"record": tid, "reason": "wrong bound on K"})
    assert status == 200, out
    [act] = actions(api)
    assert act["actor"]["type"] == "human" and act["actor"]["name"] == "user"
    assert act["body"]["action"] == 'reject {"kind":"Template"}'
    assert act["body"]["reason"] == "wrong bound on K"
    assert act["refs"] == {"subject": tid}
    assert out["action"] and api.store.status(tid)[0] == "rejected"
    assert out["stale"] and all(api.store.status(d)[0] == "stale" for d in out["stale"])


@pytest.mark.parametrize("reason", [None, "", "   "])
@pytest.mark.parametrize("path", ["/actions/reject", "/actions/deprecate"])
def test_reject_and_deprecate_need_a_reason(api: Api, path: str, reason: str | None) -> None:
    tid = _id(api, "Template")
    body: dict[str, Any] = {"record": tid} if reason is None else {"record": tid, "reason": reason}
    status, out = call(api, path, body)
    assert status == 400 and "reason" in out["detail"]
    assert actions(api) == [] and api.store.status(tid)[0] != "rejected"


def test_deprecate_does_not_propagate(api: Api) -> None:
    tid = _id(api, "Template")
    status, out = call(api, "/actions/deprecate", {"record": tid, "reason": "superseded"})
    assert status == 200, out
    assert api.store.status(tid)[0] == "deprecated" and out["stale"] == []
    assert actions(api)[0]["body"]["action"].startswith("deprecate ")


def test_flag_marks_suspect_and_reports_the_missing_reverification(api: Api) -> None:
    sol = _id(api, "Solution")
    status, out = call(api, "/actions/flag", {"record": sol})
    assert status == 200, out
    assert api.store.status(sol)[0] == "suspect"
    [act] = actions(api)
    assert act["body"] == {"action": 'flag {"kind":"Solution"}', "reason": ""}
    assert out["followups"] == [
        {"what": "re-verification task", "state": "not_supported", "needs": NEEDS["reverify"]}
    ]


def test_decision_records_cannot_be_judged(api: Api) -> None:
    sc = _id(api, "StatusChange")
    status, _ = call(api, "/actions/reject", {"record": sc, "reason": "x"})
    assert status == 400 and actions(api) == []


def test_unknown_record_is_404(api: Api) -> None:
    status, _ = call(api, "/actions/flag", {"record": "0" * 40, "reason": "x"})
    assert status == 404


def test_budget_needs_reason_and_merges_limits(api: Api) -> None:
    status, out = call(api, "/actions/budget", {"scope": "global", "limit": {"agent_min": 600}})
    assert status == 400 and "reason" in out["detail"]
    status, _ = call(
        api, "/actions/budget", {"scope": "loop:x", "limit": {"agent_min": 1}, "reason": "r"}
    )
    assert status == 404
    status, _ = call(
        api, "/actions/budget", {"scope": "global", "limit": {"Agent": 1}, "reason": "r"}
    )
    assert status == 400
    status, _ = call(
        api, "/actions/budget", {"scope": "global", "limit": {"agent_min": -1}, "reason": "r"}
    )
    assert status == 400
    assert actions(api) == []
    status, out = call(
        api,
        "/actions/budget",
        {"scope": "global", "limit": {"agent_min": 600}, "reason": "long run"},
    )
    assert status == 200, out
    [act] = actions(api)
    assert act["body"] == {
        "action": 'budget {"limit":{"agent_min":600.0},"previous":{"agent_min":480.0,'
        '"gpu_min":120.0},"scope":"global"}',
        "reason": "long run",
    }
    row = next(b for b in opsread.budgets(api.ops) if b.scope == "global")
    assert row.limit == {"agent_min": 600.0, "gpu_min": 120.0}


def test_approve_protocol_and_improvement(api: Api) -> None:
    proto = put(
        api,
        "Protocol",
        {"name": "arms", "version": 1, "type": "experiment", "description": "d", "params": {}},
    )
    status, out = call(api, "/actions/approve", {"protocol": proto, "reason": "frozen"})
    assert status == 200, out
    [act] = actions(api)
    assert act["body"] == {"action": "approve_protocol", "reason": "frozen"}
    assert act["refs"] == {"protocol": proto} and act["actor"]["type"] == "human"
    judgement = _id(api, "Report", title="judgement")
    status, _ = call(api, "/actions/approve", {"improvement": judgement})
    assert status == 400
    status, _ = call(api, "/actions/approve", {"protocol": proto, "improvement": judgement})
    assert status == 400
    status, _ = call(api, "/actions/approve", {"protocol": judgement})
    assert status == 400
    assert len(actions(api)) == 1


def test_confirm_pattern(api: Api) -> None:
    pat = put(
        api,
        "Pattern",
        {
            "name": "warp_shuffle_reduce",
            "level": 3,
            "category": "reduce",
            "description": "d",
            "examples": ["a", "b"],
        },
    )
    status, out = call(api, "/actions/confirm", {"pattern": pat})
    assert status == 200, out
    assert actions(api)[0]["body"]["action"] == "confirm_pattern"
    assert actions(api)[0]["refs"] == {"pattern": pat}
    status, _ = call(api, "/actions/confirm", {"structure": pat})
    assert status == 400


def _task(api: Api, state: str) -> str:
    return next(t.id for t in opsread.tasks(api.ops) if t.state == state)


def _actions_like(api: Api, prefix: str) -> list[dict[str, Any]]:
    return [a for a in actions(api) if a["body"]["action"].startswith(prefix)]


def test_cancel_goes_through_the_task_queue(api: Api) -> None:
    task = _task(api, "running")
    status, out = call(api, "/actions/cancel", {"task": task, "reason": "stuck"})
    assert status == 200, out
    assert out["state"] == "cancelled" and api.ops.task(task).state == "cancelled"
    [act] = _actions_like(api, "task cancel")
    assert act["body"] == {"action": f"task cancel {task}", "reason": "stuck"}
    assert act["actor"]["type"] == "human"
    status, _ = call(api, "/actions/cancel", {"task": task, "reason": "again"})
    assert status == 400  # already terminal: the lifecycle has no such edge
    assert call(api, "/actions/cancel", {"task": "01NOSUCHTASK"})[0] == 404


def test_retry_requeues_an_escalated_task(api: Api) -> None:
    task = _task(api, "escalated")
    assert call(api, "/actions/retry", {"task": task})[0] == 400  # the note is required
    status, out = call(api, "/actions/retry", {"task": task, "note": "declare swz's domain"})
    assert status == 200, out
    assert api.ops.task(task).state == "queued"
    [act] = _actions_like(api, "task retry")
    assert act["body"]["reason"] == "declare swz's domain"


def test_priority_is_not_supported_yet(api: Api) -> None:
    status, out = call(api, "/actions/priority", {"task": _task(api, "queued"), "priority": 5})
    assert status == 501 and NEEDS["priority"] in out["detail"]
    assert call(api, "/actions/priority", {"task": "01NOSUCHTASK", "priority": 5})[0] == 404
    assert actions(api) == []


def test_create_task_queues_a_human_task(api: Api) -> None:
    family = _id(api, "Family")
    status, _ = call(api, "/actions/create-task", {"type": "T9", "goal": "g", "inputs": []})
    assert status == 400
    status, _ = call(api, "/actions/create-task", {"type": "T1", "goal": "g", "inputs": ["0" * 40]})
    assert status == 404
    assert actions(api) == []
    body = {"type": "T1", "goal": "bf16 -> fp16 variant", "inputs": [family], "reason": "try"}
    status, out = call(api, "/actions/create-task", body)
    assert status == 200, out
    row = api.ops.task(out["task"])
    assert row.state == "queued" and row.priority == HUMAN_PRIORITY
    assert row.work_unit == out["work_unit"] and out["work_unit"].startswith("human:")
    card = api.store.get(out["card"])
    assert card.body["goal"] == "bf16 -> fp16 variant" and card.body["inputs"] == [family]
    [act] = actions(api)
    assert act["body"]["action"].startswith('create_task {"task":"' + out["task"])
    assert act["body"]["reason"] == "try" and act["refs"] == {"card": out["card"]}
    assert act["actor"] == {**act["actor"], "type": "human", "name": "user"}


@pytest.mark.parametrize("path", ["/actions/pause", "/actions/resume"])
def test_pause_and_resume_are_not_supported_yet(api: Api, path: str) -> None:
    assert call(api, path, {"scope": "global"})[0] == 501
    assert call(api, path, {"scope": "somewhere"})[0] == 400


def test_approve_profile_moves_the_pointer(api: Api) -> None:
    status, _ = call(api, "/actions/approve", {"profile": _id(api, "Template")})
    assert status == 400
    assert actions(api) == []
    profile = write_profile(api.store, api.now())
    status, out = call(api, "/actions/approve", {"profile": profile, "reason": "first profile"})
    assert status == 200, out
    assert out["current"] == profile == api.ops.get_pointer("current_profile")
    [act] = actions(api)
    assert act["body"] == {"action": "approve_profile", "reason": "first profile"}
    assert act["refs"] == {"profile": profile} and act["actor"]["name"] == "user"


def test_capabilities_say_what_is_wired(api: Api) -> None:
    status, caps = call(api, "/actions/capabilities", method="GET")
    assert status == 200
    by = {c["name"]: c for c in caps}
    for name in ("reject", "budget", "confirm", "cancel", "retry", "create-task", "approve-profile",
                 "env-verify", "write-record"):  # fmt: skip
        assert by[name]["wired"], name
    for name in ("priority", "pause", "resume"):
        assert not by[name]["wired"] and by[name]["needs"] == NEEDS[name]
    assert "not stopped" in by["cancel"]["partial"]
    assert "re-verification" in by["flag"]["partial"]


def test_unwired_port_answers_501_for_every_delegated_operation(tmp_path: Path) -> None:
    bare = seeded_api(tmp_path, wired=False)
    task = _task(bare, "running")
    assert call(bare, "/actions/cancel", {"task": task})[0] == 501
    assert call(bare, "/actions/create-task", {"type": "T1", "goal": "g", "inputs": []})[0] == 501
    assert actions(bare) == []


def test_action_log_is_newest_first_with_parameters(api: Api) -> None:
    tid = _id(api, "Template")
    call(api, "/actions/flag", {"record": tid, "reason": "odd"})
    call(api, "/actions/budget", {"scope": "global", "limit": {"gpu_min": 30}, "reason": "cap"})
    status, log = call(api, "/actions/log?limit=10", method="GET")
    assert status == 200
    assert [x["verb"] for x in log] == ["budget", "flag"]
    assert log[0]["params"]["scope"] == "global" and log[1]["refs"] == {"subject": tid}


def test_parse_action_reads_every_writer() -> None:
    assert operate.parse_action('budget {"scope":"global"}')["params"] == {"scope": "global"}
    assert operate.parse_action("task cancel 01ABC") == {
        "verb": "cancel",
        "params": None,
        "args": "01ABC",
    }
    assert operate.parse_action("approve_profile")["verb"] == "approve_profile"
