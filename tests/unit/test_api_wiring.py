"""The API over WP8's functions: task tokens on POST /write/record (D305) and batch
environment verification (D322), both with real keys, tokens and batch roots."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from apihelp import request, seeded_api
from kta.agents import batches, guard
from kta.agents.profile import write_profile
from kta.api import agentsview, opsread
from kta.api.app import create_app
from kta.api.common import Api
from kta.core.schemas import Actor, Record
from kta.core.schemas.common import ActorType


@pytest.fixture
def api(tmp_path: Path) -> Api:
    return seeded_api(tmp_path / "data", {"token_key_file": str(tmp_path / "key")})


def _gap(api: Api, task: str, actor_type: ActorType = "agent") -> dict[str, Any]:
    body = {"missing": "template", "description": "d", "attribution": "space"}
    rec = Record.new("Gap", body, Actor(type=actor_type, name="kta note", task=task), api.now())
    return rec.to_json_dict()


def _running(api: Api) -> str:
    return next(t.id for t in opsread.tasks(api.ops) if t.state == "running")


def test_write_record_with_a_valid_task_token(api: Api, tmp_path: Path) -> None:
    app, task = create_app(api), _running(api)
    status, out = request(app, "POST", "/write/record", _gap(api, task))
    assert status == 403 and "Bearer" in out["detail"]
    status, out = request(
        app, "POST", "/write/record", _gap(api, task), {"authorization": "Bearer x"}
    )
    assert status == 403 and "no task token key" in out["detail"]  # never created by the API
    assert not (tmp_path / "key").exists()
    key = guard.ensure_key(tmp_path / "key")
    token = guard.issue_token(key, task, api.now() + timedelta(hours=1))
    auth = {"authorization": f"Bearer {token}"}
    status, out = request(app, "POST", "/write/record", _gap(api, task), auth)
    assert status == 200, out
    assert api.store.get(out["id"]).actor.task == task
    other = next(t.id for t in opsread.tasks(api.ops) if t.state == "queued")
    assert request(app, "POST", "/write/record", _gap(api, other), auth)[0] == 403
    assert request(app, "POST", "/write/record", _gap(api, task, "human"), auth)[0] == 403
    expired = guard.issue_token(key, task, api.now() - timedelta(seconds=1))
    assert (
        request(
            app, "POST", "/write/record", _gap(api, task), {"authorization": f"Bearer {expired}"}
        )[0]
        == 403
    )


def test_batch_verify_regenerates_and_compares(api: Api) -> None:
    profile = write_profile(api.store, api.now())
    root = batches.build(api.store, api.data_dir, "b1", profile, "0" * 40)
    app = create_app(api)
    status, out = request(app, "POST", "/agents/batches/b1/verify")
    assert status == 200 and out == {"batch": "b1", "state": "ok", "problems": []}
    [row] = agentsview.batches(api)
    assert (
        row["lock"]["profile"] == profile and row["files"] and row["verify"]["state"] == "unknown"
    )
    prompt = next((root / "prompts").iterdir())
    prompt.write_text(prompt.read_text() + "\nedited\n")
    status, out = request(app, "POST", "/agents/batches/b1/verify")
    assert status == 200 and out["state"] == "failed"
    assert any(p.startswith("changed: prompts/") for p in out["problems"])
    assert request(app, "POST", "/agents/batches/nope/verify")[0] == 404


def test_skill_catalogue_comes_from_the_agents_readme(api: Api) -> None:
    out = agentsview.agents(api)["skills"]
    assert out["has_catalogue"]
    names = {s["name"] for s in out["skills"]}
    assert {
        "use-kta",
        "find-knowledge",
        "write-family",
        "decompose-kernel",
        "write-template",
        "diagnose-model",
        "write-microbench",
    } <= names
    # WP11 wrote the two skills that were planned when this test was added.
    assert all(r["skill"].strip("`") not in names for r in out["planned"])
    use = next(s for s in out["skills"] if s["name"] == "use-kta")
    assert use["catalogue"] and use["catalogue"]["status"] == "written"
