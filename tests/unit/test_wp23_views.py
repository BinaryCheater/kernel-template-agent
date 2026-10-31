"""WP23: the card's work unit through TaskQueue, and the /workflow and /trace/stages views."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from apihelp import request, seeded_api
from kta.agents.profile import CODEX_V2, write_profile
from kta.api.app import create_app
from kta.api.common import Api
from kta.core.context import Context
from kta.core.trace import SpanStart
from kta.orchestrator.tasks import TaskQueue
from kta.orchestrator.transition import TaskRequest
from kta.store.sqlite import SqliteStore
from kta.store.sqlite_ops import SqliteOps

pytestmark = pytest.mark.contract


def test_cards_of_one_loop_split_by_role(ctx: Context, store: SqliteStore, ops: SqliteOps) -> None:
    q = TaskQueue(ctx, store, ops)
    ids = {t: q.create(TaskRequest(t, f"goal {t}", []), "L1", "solve") for t in ("T1", "T4", "T6")}
    unit = {t: q.card(q.get(i)).work_unit for t, i in ids.items()}
    assert unit == {"T1": "L1:kernel", "T4": "L1:model", "T6": "L1:kernel"}
    # the ops row keeps the loop: budgets, pauses and views key on it
    assert {q.get(i).work_unit for i in ids.values()} == {"L1"}
    human = q.create(TaskRequest("T4", "by hand", []), None, "human")
    assert q.card(q.get(human)).work_unit == q.get(human).work_unit
    assert q.get(human).work_unit.startswith("human:")


@pytest.fixture
def api(tmp_path: Path) -> Api:
    a = seeded_api(tmp_path / "d")
    pid = write_profile(a.store, datetime.now(UTC), CODEX_V2)
    a.ops.set_pointer("current_profile", a.ops.get_pointer("current_profile"), pid)
    return a


def _get(api: Api, path: str) -> Any:
    status, body = request(create_app(api), "GET", path)
    assert status == 200, body
    return body


def test_workflow_shows_the_work_unit_rule(api: Api) -> None:
    w = _get(api, "/workflow")
    wu = w["session"]["work_units"]
    assert wu["rule"] == "case_role" and wu["roles"]["model"] == ["T4", "T5"]
    assert wu["example"] == {"T1": "<loop>:kernel", "T4": "<loop>:model", "T2": "<loop>:knowledge"}
    assert "session.work_unit_rule" in [k["id"] for k in w["knobs"]["nodes"]["session"]]


def test_stage_view_derives_depth_gpu_busy_and_lock_waits(api: Api) -> None:
    now = api.now()
    t0 = now - timedelta(minutes=30)

    def sp(i: str, kind: str, lane: str, name: str, a: int, b: int | None, **attrs: Any) -> None:
        api.ops.start_span(
            SpanStart(i, None, "L1", kind, lane, name, t0 + timedelta(minutes=a), attrs)
        )
        if b is not None:
            api.ops.end_span(i, t0 + timedelta(minutes=b), "ok", {})

    sp("s1", "loop_step", "process:solve", "building", 0, 10)
    sp("s2", "loop_step", "process:solve", "measuring", 10, None)
    sp("j1", "job_run", "worker:spark-hk:gpu-exclusive", "measure j", 12, 18)
    sp("w1", "lock_wait", "host:spark-hk", "host lock shared", 2, 3, mode="shared")
    api.ops.put_loop("L1", "solve", "case-1", "measuring",
                     {"deferred": "collecting for a measurement window (1/8)"})  # fmt: skip
    q = f"from={t0.isoformat()}&to={now.isoformat()}".replace("+", "%2B")
    d = _get(api, f"/trace/stages?{q}&bucket_s=600")
    assert len(d["buckets"]) == 3
    assert d["stages"]["build"]["depth"][0] == pytest.approx(1.0)
    assert d["stages"]["build"]["throughput_per_h"][1] == pytest.approx(6.0)
    assert d["stages"]["ready_to_measure"]["depth"][2] == pytest.approx(1.0, abs=0.01)
    assert d["gpu_busy"][1] == pytest.approx(0.6)
    assert d["host_lock"]["waits"][0] == 1 and d["host_lock"]["wait_s"][0] == pytest.approx(60)
    held = d["now"]["held"]
    assert held and held[0]["reason"].startswith("collecting")
    assert {s["id"] for s in d["now"]["stages"]} >= {"build", "measure", "analyse"}
    assert datetime.fromisoformat(d["to"]).tzinfo is not None


def test_backup_restore_and_handover(tmp_path: Path) -> None:
    from kta.core.errors import ConfigError
    from kta.store import backup

    a = seeded_api(tmp_path / "old")
    blob = a.store.put_blob(b"evidence", "text/plain")
    now = datetime.now(UTC)
    with pytest.raises(ConfigError, match="no store"):
        backup.backup(tmp_path / "nothing", tmp_path / "b0", now)
    m = backup.backup(a.ctx.data_dir, tmp_path / "b1", now)  # a periodic backup changes nothing
    backup.check_owner(a.ctx.data_dir)
    assert m["blobs"] >= 1 and m["handover_to"] is None
    backup.backup(a.ctx.data_dir, tmp_path / "b2", now, handover_to="spark-hk")
    with pytest.raises(ConfigError, match="handed over to spark-hk"):
        backup.check_owner(a.ctx.data_dir)
    new = tmp_path / "new" / "data"
    out = backup.restore(tmp_path / "b2", new)
    assert out["restored_to"] == str(new)
    assert (new / "objects").rglob(blob[2:]) is not None
    with pytest.raises(ConfigError, match="already holds a store"):
        backup.restore(tmp_path / "b2", new)
    (tmp_path / "b1" / "kta.db").write_bytes(b"corrupt")
    with pytest.raises(ConfigError, match="manifest hash"):
        backup.restore(tmp_path / "b1", tmp_path / "other")


def test_direct_check_of_an_off_provider_reports_each_host() -> None:
    from kta.core import proxy

    def refuse(addr: tuple[str, int], timeout_s: float) -> Any:
        raise ConnectionRefusedError(f"refused {addr[0]}")

    out = proxy.direct_check(["api.openai.com:443", "chatgpt.com"], connect=refuse)
    assert not out["ok"] and out["proxy"] is None
    assert [(h["host"], h["port"]) for h in out["hosts"]] == [
        ("api.openai.com", 443),
        ("chatgpt.com", 443),
    ]
    assert "refused api.openai.com" in out["error"]
    assert proxy.direct_check([])["error"].startswith("no API host")
