"""WP22 Part D: the agent workflow view, draft profile versions, preview and approval."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from apihelp import request, seeded_api
from kta.agents import batches
from kta.agents.profile import CODEX_V2, write_profile
from kta.api.app import create_app
from kta.api.common import Api
from kta.api.workflow import NODE_KNOBS, session_rules
from kta.core.controls import definitions
from kta.orchestrator.done_checks import DONE_CHECKS

TOKEN = "secret-operator-token"


@pytest.fixture
def api(tmp_path: Path) -> Api:
    token = tmp_path / "operator-token"
    token.write_text(TOKEN + "\n")
    a = seeded_api(tmp_path / "d", sections={"api": {"operator_token_file": str(token)}})
    pid = write_profile(a.store, datetime.now(UTC), CODEX_V2)
    a.ops.set_pointer("current_profile", a.ops.get_pointer("current_profile"), pid)
    return a


def _call(api: Api, method: str, path: str, body: Any = None, token: str | None = None):
    headers = {"X-KTA-Operator-Token": token} if token else {}
    return request(create_app(api), method, path, body, headers)


def test_workflow_is_derived_from_machines_checks_and_sessions(api: Api) -> None:
    status, w = _call(api, "GET", "/workflow")
    assert status == 200, w
    assert w["profile"]["name"] == "codex-v2" and w["profile"]["current"]
    by = {t["id"]: t for t in w["task_types"]}
    assert set(by) == {"T1", "T2", "T3", "T4", "T5", "T6"}
    for t, names in DONE_CHECKS.items():
        assert [c["name"] for c in by[t]["done_checks"]] == list(names)
        assert all(
            c["source"].startswith("kta/orchestrator/done_checks.py:") for c in by[t]["done_checks"]
        )
    kinds = {(c["kind"], c.get("machine")) for c in by["T4"]["creators"]}
    assert ("loop", "solve") in kinds and ("process", None) in kinds  # judging rule 2, calibrate
    assert any(c["kind"] == "loop" and c["machine"] == "expand" for c in by["T2"]["creators"])
    assert by["T4"]["skills"] == ["use-kta", "find-knowledge", "diagnose-model"]
    assert w["templates"]["step:T1"]["text"] and "goal" in w["templates"]["step:T1"]["placeholders"]
    assert "diagnose-model" in w["skills"] and w["skills"]["diagnose-model"]["used_by"] == ["T4"]
    assert {r["mode"] for r in w["session"]["rules"]} == {"start", "resume"}
    assert w["session"]["rules"][-1]["mode"] == "resume"
    assert w["proxy"]["proxy"] == "required"
    ids = {k.id for k in definitions().knobs}
    assert all(i in ids for v in NODE_KNOBS.values() for i in v)
    assert [r["mode"] for r in session_rules()][:1] == ["start"]


def test_preview_flags_a_missing_placeholder_and_renders_task_md(api: Api) -> None:
    w = _call(api, "GET", "/workflow")[1]
    base = w["profile"]["id"]
    text = w["templates"]["step:T1"]["text"]
    missing = {"base": base, "changes": {"templates": {"step:T1": text + "\n$owner"}}}
    status, out = _call(api, "POST", "/workflow/preview", missing)
    assert status == 200 and any("$owner is not provided" in e for e in out["errors"])
    assert "task_md" not in out
    short = {"base": base, "changes": {"templates": {"step:T1": "Goal: $goal\n"}}}
    status, out = _call(api, "POST", "/workflow/preview", short)
    assert out["errors"] == [] and out["task_md"].startswith("Goal: (sample) a T1 goal")
    assert any("$task_id is provided but not used" in x for x in out["warnings"])


def test_a_draft_is_a_new_version_and_approval_moves_the_pointer(api: Api) -> None:
    w = _call(api, "GET", "/workflow")[1]
    base = w["profile"]["id"]
    before = api.store.get(base).to_json_dict()
    skill = w["skills"]["use-kta"]["text"] + "\nOne more rule.\n"
    changes = {
        "templates": {"message:rejection": "Rejected $task_id:\n$failures\nFix it.\n"},
        "skills": {"use-kta": skill},
        "fields": {"model": "gpt-5.5", "reasoning_effort": "high"},
    }
    body = {"base": base, "changes": changes, "reason": "tighten the rejection message"}
    assert _call(api, "POST", "/workflow/drafts", body)[0] == 403  # operator token needed
    bad = {**body, "changes": {"templates": {"message:rejection": "$nope"}}}
    assert _call(api, "POST", "/workflow/drafts", bad, TOKEN)[0] == 400
    status, out = _call(api, "POST", "/workflow/drafts", body, TOKEN)
    assert status == 200, out
    draft = out["id"]
    assert draft != base and api.store.get(base).to_json_dict() == before  # base unchanged
    rec = api.store.get(draft)
    assert rec.body["version"] == 3 and rec.refs["base"] == base and rec.actor.type == "human"
    whats = {d["what"] for d in out["diff"]}
    assert {"model", "reasoning_effort", "message:rejection", "skill:use-kta"} <= whats
    assert api.ops.get_pointer("current_profile") == base  # a draft is not in force
    status, out = _call(
        api, "POST", f"/workflow/drafts/{draft}/approve", {"reason": "reviewed"}, TOKEN
    )
    assert status == 200, out
    assert api.ops.get_pointer("current_profile") == draft and out["batch"] == "codex-v2-v3"
    root = batches.batches_root(api.data_dir) / "codex-v2-v3"
    assert batches.read_lock(root)["profile"] == draft
    assert batches.verify(api.store, root).ok
    w2 = _call(api, "GET", "/workflow")[1]
    assert w2["profile"]["id"] == draft and [v["version"] for v in w2["versions"]] == [3, 2]
