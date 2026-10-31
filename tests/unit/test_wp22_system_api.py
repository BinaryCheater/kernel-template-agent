"""WP22: Settings → System endpoints (read open, write with the operator token, Tests)."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from apihelp import request, seeded_api
from kta.agents import proxyplan
from kta.api.app import create_app
from kta.api.common import Api
from kta.core import proxy
from kta.execution.registry import register, worker_configs
from kta.store import code_state

TOKEN = "secret-operator-token"


@pytest.fixture
def api(tmp_path: Path) -> Api:
    token = tmp_path / "operator-token"
    token.write_text(TOKEN + "\n")
    sections = {
        "api": {"operator_token_file": str(token)},
        "workers": {"local": {"transport": "local", "pools": {"cpu": 1}}},
    }
    return seeded_api(tmp_path / "d", sections=sections)


def _call(api: Api, method: str, path: str, body: Any = None, token: str | None = None):
    headers = {"X-KTA-Operator-Token": token} if token else {}
    return request(create_app(api), method, path, body, headers)


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_system_view_shows_proxy_providers_workers_and_stage(api: Api) -> None:
    status, view = _call(api, "GET", "/system")
    assert status == 200, view
    assert view["proxy"]["detected"]["source"] in ("scutil", "env", "default")
    assert view["proxy"]["in_force"]["endpoint"]["url"].startswith(("http://", "socks5"))
    by = {p["name"]: p for p in view["providers"]}
    assert by["codex"]["proxy"] == "required" and by["pi"]["proxy"] == "off"
    assert set(by["codex"]["auth"]) >= {"kind", "logged_in"}
    assert "auth.json" not in str(by["codex"]["auth"].get("logged_in"))
    (w,) = view["workers"]
    assert w["name"] == "local" and w["registered"] is None and w["commit_match"] is False
    assert view["control"]["data_dir"] == str(api.data_dir)
    assert view["release"]["page"] == "/release" and view["release"]["id"].startswith("S")
    assert view["config"]["revision"] == 0


def test_system_write_needs_the_token_and_is_audited(api: Api) -> None:
    body = {
        "expected": 0,
        "changes": {"proxy": {"override": "http://127.0.0.1:7897"}},
        "reason": "pin Clash",
    }
    assert _call(api, "POST", "/system", body)[0] == 403
    status, out = _call(api, "POST", "/system", body, TOKEN)
    assert status == 200 and out["revision"] == 1, out
    action = api.store.get(out["action"])
    assert action.actor.type == "human" and action.body["action"].startswith("set_system")
    assert action.body["reason"] == "pin Clash"
    assert _call(api, "POST", "/system", body, TOKEN)[0] == 400  # stale revision
    creds = "u" + ":" + "p"
    bad = {**body, "expected": 1, "changes": {"proxy": {"override": f"http://{creds}@h:1"}}}
    assert _call(api, "POST", "/system", bad, TOKEN)[0] == 400
    view = _call(api, "GET", "/system")[1]
    assert view["proxy"]["in_force"]["source"] == "override"
    assert view["audit"][0]["reason"] == "pin Clash" and view["audit"][0]["by"]


def test_proxy_test_reports_failure_and_a_pass_clears_the_breaker(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"expected": 0, "changes": {"proxy": {"override": f"http://127.0.0.1:{_closed_port()}"}}}
    _call(api, "POST", "/system", {**body, "reason": "down"}, TOKEN)
    status, out = _call(api, "POST", "/system/proxy/test", {"provider": "codex"})
    assert (
        status == 200 and out["ok"] is False and "not reachable" in out["results"]["codex"]["error"]
    )
    proxyplan.mark(api.ops, "codex", False, api.now(), error="down")

    def passes(ep: proxy.Endpoint, hosts: list[str], userinfo: str | None = None):
        return {
            "ok": True,
            "proxy": ep.to_dict(),
            "hosts": [{"host": h, "ok": True} for h in hosts],
        }

    monkeypatch.setattr(proxy, "preflight", passes)  # the network boundary
    status, out = _call(api, "POST", "/system/proxy/test", {})
    assert status == 200 and out["ok"] and set(out["results"]) == {"codex", "claude_code"}
    assert proxyplan.blocked(api.ops, "codex") is None
    off = _call(api, "POST", "/system/proxy/test", {"provider": "pi"})[1]
    assert off["results"]["pi"]["skipped"] == "policy off" and "report" not in off["results"]["pi"]
    rep = api.store.get(out["results"]["codex"]["report"])
    assert rep.body["title"] == "proxy_preflight" and rep.body["data"]["ok"] is True


def test_worker_test_runs_a_trivial_job_through_the_executor(api: Api) -> None:
    status, out = _call(api, "POST", "/system/workers/local/test")
    assert status == 200 and out["ran"] is False and "never registered" in out["error"]
    assert _call(api, "POST", "/system/workers/nowhere/test")[0] == 400
    commit = code_state().commit
    register(api.ctx, api.store, api.ops, worker_configs(api.ctx)["local"], commit)
    status, out = _call(api, "POST", "/system/workers/local/test")
    assert status == 200 and out["ok"] and out["ran"], out
    assert '"python"' in out["stdout"]
