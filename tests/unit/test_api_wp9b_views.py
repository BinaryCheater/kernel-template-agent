"""WP9b read views: task detail, blobs, agents (skills, profiles, batches), pool, experiments,
improvement. Store: the WP9a fixture world plus records written here (origin fixture)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apihelp import put, request, seeded_api
from kta.api import agentsview, opsread, studies, taskview
from kta.api.app import create_app
from kta.api.common import Api
from kta.core.errors import NotSupported


@pytest.fixture(scope="module")
def api(tmp_path_factory: pytest.TempPathFactory) -> Api:
    return seeded_api(Path(tmp_path_factory.mktemp("wp9b")))


def test_task_detail_of_a_running_task(api: Api) -> None:
    t = next(t for t in opsread.tasks(api.ops) if t.card_type == "T1" and t.state == "leased")
    d = taskview.task_detail(api, t.id)
    assert d["task"]["state"] == "running" and d["card"]["type"] == "T1"
    assert d["loop"] is not None and d["loop"]["id"] == t.work_unit
    kinds = {s["kind"] for s in d["spans"]}
    assert "agent_session" in kinds
    assert d["time"]["source"] in ("AgentRun", "spans")


def test_task_detail_lists_rejections(api: Api) -> None:
    t = next(t for t in opsread.tasks(api.ops) if t.state == "failed")
    d = taskview.task_detail(api, t.id)
    assert d["task"]["state"] == "escalated"
    assert d["rejections"] and all(not c["body"]["passed"] for c in d["rejections"])


def test_unknown_task_is_404(api: Api) -> None:
    status, _ = request(create_app(api), "GET", "/tasks/01NOSUCH")
    assert status == 404


def test_blob_text_and_binary(api: Api) -> None:
    sha = api.store.put_blob(b"# TASK\nwrite a family\n", "text/markdown")
    assert taskview.blob_text(api, sha)["text"].startswith("# TASK")
    binary = api.store.put_blob(b"\x1f\x8b\x00\x00", "application/gzip")
    with pytest.raises(NotSupported):
        taskview.blob_text(api, binary)


def test_markdown_table() -> None:
    rows = agentsview.markdown_table("x\n| skill | used by |\n| --- | --- |\n| `a` | T1 |\n\nafter")
    assert rows == [{"skill": "`a`", "used by": "T1"}]


def test_agents_lists_runtime_skills_profiles_and_batches(api: Api) -> None:
    assert api.data_dir is not None
    batch = api.data_dir / "batches" / "b1"
    (batch / "work" / "wu1").mkdir(parents=True)
    lock = {"batch": "b1", "profile": "p", "commit": "c", "files": {"a": "1", "b": "2"}}
    (batch / "env.lock.json").write_text(json.dumps(lock))
    out = agentsview.agents(api)
    names = [s["name"] for s in out["skills"]["skills"]]
    assert "decompose-kernel" in names
    assert all(s["description"] for s in out["skills"]["skills"])
    assert out["current"] is None and out["profiles"] == []
    [b] = out["batches"]
    assert b["batch"] == "b1" and b["files"] == 2 and b["work_units"] == ["wu1"]
    assert b["lock"] == {"batch": "b1", "profile": "p", "commit": "c"}
    assert b["verify"]["state"] == "not_supported"
    status, _ = request(create_app(api), "POST", "/agents/batches/b1/verify")
    assert status == 501
    status, _ = request(create_app(api), "POST", "/agents/batches/nope/verify")
    assert status == 404


def test_pool_lists_knowledge_and_snapshot_membership(api: Api) -> None:
    full = studies.pool(api, None)
    assert full["counts"]["templates"] == 12 and full["counts"]["gaps"] == 1
    assert full["references"] and full["references"][0]["name"]
    first = studies.pool(api, _first_snapshot(api))
    assert 0 < first["counts"]["templates"] <= full["counts"]["templates"]


def _first_snapshot(api: Api) -> str:
    with api.store.db.read() as con:
        row = con.execute(
            "SELECT id FROM records WHERE kind = 'PoolSnapshot' ORDER BY created_at LIMIT 1"
        ).fetchone()
    return str(row[0])


def test_experiments_group_runs_by_arm(api: Api) -> None:
    proto = put(
        api,
        "Protocol",
        {
            "name": "pool-vs-none",
            "version": 1,
            "type": "experiment",
            "description": "with and without the pool",
            "params": {"arms": ["pool", "none"], "runs_per_arm": 3},
        },
    )
    for i, cond in enumerate(("pool", "pool", "none")):
        put(
            api,
            "ExperimentRun",
            {"name": f"run {i}", "condition": cond, "params": {}, "note": ""},
            {"protocol": proto},
        )
    exp = studies.experiments(api)
    [p] = [x for x in exp["protocols"] if x["id"] == proto]
    assert p["arms"] == [
        {"name": "pool", "runs": 2, "planned": 3},
        {"name": "none", "runs": 1, "planned": 3},
    ]
    assert p["approved"] == []


def test_improvements_count_gap_attribution(api: Api) -> None:
    out = studies.improvements(api)
    assert out["attribution"] == {"space": 1} and out["missing"] == {"template": 1}
    assert out["attribution_by_day"][0]["space"] == 1
