"""Operation endpoints (spec/modules/api-web.md acceptance): each writes a correct HumanAction;
a missing required reason is a 400; operations not wired yet are a 501 that writes nothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apihelp import put, request, seeded_api
from kta.api import opsread
from kta.api.app import create_app
from kta.api.common import Api
from kta.api.wiring import NEEDS


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


@pytest.mark.parametrize(
    ("path", "body", "need"),
    [
        ("/actions/cancel", {"reason": "stuck"}, "cancel"),
        ("/actions/retry", {"note": "use cp.async"}, "retry"),
        ("/actions/priority", {"priority": 5}, "priority"),
    ],
)
def test_task_operations_are_not_supported_until_wp6(
    api: Api, path: str, body: dict[str, Any], need: str
) -> None:
    task = _task(api, "leased")
    status, out = call(api, path, {"task": task, **body})
    assert status == 501 and NEEDS[need] in out["detail"]
    status, _ = call(api, path, {"task": "01NOSUCHTASK", **body})
    assert status == 404
    assert actions(api) == []


def test_retry_needs_a_note(api: Api) -> None:
    status, _ = call(api, "/actions/retry", {"task": _task(api, "failed")})
    assert status == 400


def test_create_task_validates_then_needs_wp6(api: Api) -> None:
    family = _id(api, "Family")
    status, _ = call(api, "/actions/create-task", {"type": "T9", "goal": "g", "inputs": []})
    assert status == 400
    status, _ = call(api, "/actions/create-task", {"type": "T1", "goal": "g", "inputs": ["0" * 40]})
    assert status == 404
    body = {"type": "T1", "goal": "bf16 -> fp16 variant", "inputs": [family]}
    status, out = call(api, "/actions/create-task", body)
    assert status == 501 and "TaskQueue.create" in out["detail"]
    assert actions(api) == []


@pytest.mark.parametrize("path", ["/actions/pause", "/actions/resume"])
def test_pause_and_resume_need_wp6(api: Api, path: str) -> None:
    assert call(api, path, {"scope": "global"})[0] == 501
    assert call(api, path, {"scope": "somewhere"})[0] == 400


def test_approve_profile_needs_wp8(api: Api) -> None:
    status, _ = call(api, "/actions/approve", {"profile": _id(api, "Template")})
    assert status == 400
    assert actions(api) == []


def test_write_record_needs_a_task_token(api: Api) -> None:
    rec = {"kind": "Gap"}
    assert call(api, "/write/record", rec)[0] == 403
    status, out = call(api, "/write/record", rec, authorization="Bearer kta1.x.y.z")
    assert status == 501 and "task tokens" in out["detail"]


def test_capabilities_say_what_is_wired(api: Api) -> None:
    status, caps = call(api, "/actions/capabilities", method="GET")
    assert status == 200
    by = {c["name"]: c for c in caps}
    assert by["reject"]["wired"] and by["budget"]["wired"] and by["confirm"]["wired"]
    assert not by["cancel"]["wired"] and by["cancel"]["needs"] == NEEDS["cancel"]
    assert "re-verification" in by["flag"]["partial"]


def test_action_log_is_newest_first_with_parameters(api: Api) -> None:
    tid = _id(api, "Template")
    call(api, "/actions/flag", {"record": tid, "reason": "odd"})
    call(api, "/actions/budget", {"scope": "global", "limit": {"gpu_min": 30}, "reason": "cap"})
    status, log = call(api, "/actions/log?limit=10", method="GET")
    assert status == 200
    assert [x["verb"] for x in log] == ["budget", "flag"]
    assert log[0]["params"]["scope"] == "global" and log[1]["refs"] == {"subject": tid}
