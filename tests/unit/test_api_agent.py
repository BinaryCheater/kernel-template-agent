"""Token endpoints for the agent-facing commands in client mode (WP6 D370, src/kta/api/agentapi.py),
with a real task token for a running fixture task; plus the WP11 improvement decisions,
experiment state and quotas views."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from apihelp import put, request, seeded_api
from kta.agents import guard
from kta.api import agentapi, opsread, runs, studies, taskview
from kta.api.app import create_app
from kta.api.common import Api

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "knowledge" / "seed" / "families" / "gemm_fp16_tilelang"


@pytest.fixture
def api(tmp_path: Path) -> Api:
    return seeded_api(tmp_path / "data", {"token_key_file": str(tmp_path / "key")})


def _auth(api: Api, tmp_path: Path, task: str) -> dict[str, str]:
    key = guard.ensure_key(tmp_path / "key")
    token = guard.issue_token(key, task, api.now() + timedelta(hours=1))
    return {"authorization": f"Bearer {token}", "x-kta-trace-parent": ""}


def _running(api: Api) -> str:
    return next(t.id for t in opsread.tasks(api.ops) if t.state == "running")


def _template(api: Api) -> tuple[str, dict[str, Any]]:
    tid = _first_id(api, "Template")
    return tid, api.store.get_json(tid)["body"]


def _first_id(api: Api, kind: str) -> str:
    with api.store.db.read() as con:
        row = con.execute("SELECT id FROM records WHERE kind = ? LIMIT 1", (kind,)).fetchone()
    return str(row[0])


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/agent/records/x", None),
        ("POST", "/agent/query", {"what": "tmpl"}),
        ("POST", "/agent/validate-template", {"template": {}}),
        ("POST", "/agent/validate-family", {"files": {}}),
        ("POST", "/agent/build", {"family": "f", "assignment": {}, "shape": {}}),
        ("POST", "/agent/submit", {"files": {}, "note": "n"}),
    ],
)
def test_every_agent_endpoint_needs_a_token(api: Api, method: str, path: str, body: Any) -> None:
    assert request(create_app(api), method, path, body)[0] == 403


def test_show_query_and_why_as_the_task(api: Api, tmp_path: Path) -> None:
    app, task = create_app(api), _running(api)
    auth = _auth(api, tmp_path, task)
    tid, body = _template(api)
    status, out = request(app, "GET", f"/agent/records/{tid}", None, auth)
    assert status == 200 and out["kind"] == "Template" and out["body"]["name"] == body["name"]
    status, out = request(app, "POST", "/agent/query", {"what": "tmpl", "term": body["name"]}, auth)
    assert status == 200 and out["cards"], out
    status, out = request(app, "GET", f"/agent/why/{tid}?depth=2", None, auth)
    assert status == 200 and out["root"] == tid
    status, _ = request(app, "POST", "/agent/query", {"what": "pool"}, auth)
    assert status == 400  # pool and chain need the Case
    calls = [c for c in taskview.task_detail(api, task)["calls"] if c["kind"] == "cli"]
    # the fixture clock is frozen, so the spans share a start time: compare as a multiset
    assert sorted(c["name"].split()[1] for c in calls) == ["query", "query", "show", "why"]
    assert all(c["lane"] == "process:api" and c["attrs"]["task_id"] == task for c in calls)


def test_validate_template_and_family(api: Api, tmp_path: Path) -> None:
    app = create_app(api)
    auth = _auth(api, tmp_path, _running(api))
    _, body = _template(api)
    status, out = request(app, "POST", "/agent/validate-template", {"template": body}, auth)
    assert status == 200 and "issues" in out and isinstance(out["ok"], bool)
    status, _ = request(app, "POST", "/agent/validate-template", {"template": {"name": 1}}, auth)
    assert status == 400
    files = {p.name: p.read_bytes() for p in FAMILY.iterdir() if p.is_file()}
    payload = {"files": agentapi.encode_files(files)}
    status, out = request(app, "POST", "/agent/validate-family", payload, auth)
    assert status == 200 and "issues" in out, out
    bad = {"files": {"../escape.py": agentapi.encode_files({"x": b"1"})["x"]}}
    assert request(app, "POST", "/agent/validate-family", bad, auth)[0] == 400


def test_submit_uploads_the_delivery(api: Api, tmp_path: Path) -> None:
    app, task = create_app(api), _running(api)
    auth = _auth(api, tmp_path, task)
    files = {"family.yaml": b"name: x\n", "DELIVERY.md": b"ignored here", "gen/a.py": b"pass\n"}
    body = {"files": agentapi.encode_files(files), "note": "Done: x. Fixtures: none. Issues: none."}
    status, out = request(app, "POST", "/agent/submit", body, auth)
    assert status == 200, out
    assert out["files"] == ["family.yaml", "gen/a.py"]
    assert api.ops.task(task).state == "submitted"
    sub = api.store.get(out["submission"])
    assert sub.kind == "Submission" and sub.actor.task == task and sorted(sub.blobs) == out["files"]
    # The token is no longer valid once the task left leased/running.
    assert request(app, "POST", "/agent/submit", body, auth)[0] == 403


def test_improvement_inbox_items_and_decisions(api: Api) -> None:
    item = {"key": "b1:verification", "title": "1 x verification", "kind": "system",
            "description": "system-level (verification)"}  # fmt: skip
    data = {"batch": "b1", "attribution": {"verification": 1}, "items": [item]}
    rid = put(
        api, "Report", {"title": "improvement", "subject": "b1", "summary": "s", "data": data}
    )
    pending = [i for i in runs.inbox(api) if i["trigger"] == "improvement_pending"]
    assert [(i["record"], i["actions"]) for i in pending] == [
        (rid, ["approve-improvement", "reject-improvement"])
    ]
    app = create_app(api)
    assert request(app, "POST", "/actions/reject-improvement", {"improvement": rid})[0] == 400
    status, out = request(
        app, "POST", "/actions/reject-improvement", {"improvement": rid, "reason": "not now"}
    )
    assert status == 200, out
    assert not [i for i in runs.inbox(api) if i["trigger"] == "improvement_pending"]
    [rep] = studies.improvements(api)["reports"]
    assert rep["rejected"] and rep["system_items"] == 1


def test_experiment_state_and_quotas(api: Api) -> None:
    proto = put(
        api,
        "Protocol",
        {"name": "e", "version": 1, "type": "experiment", "description": "d",
         "params": {"arms": ["a", "b"], "runs_per_arm": 2, "cases": ["c1", "c2"]}},
    )  # fmt: skip
    state = {"status": "running", "slots": {"a/c1/0": {"loop": "L1", "run": "R1"}, "b/c1/0": {}}}
    api.ops.set_pointer(f"experiment:{proto}", None, json.dumps(state))
    [p] = [x for x in studies.experiments(api)["protocols"] if x["id"] == proto]
    assert p["progress"] == {"status": "running", "started": 2, "finished": 1, "report": None,
                           "error": None}  # fmt: skip
    assert p["arms"][0]["planned"] == 4  # runs_per_arm x cases (WP11 D403)
    status, q = request(create_app(api), "GET", "/quotas")
    assert status == 200 and set(q) == {"configured", "in_force", "experiments"}
    assert q["in_force"]
