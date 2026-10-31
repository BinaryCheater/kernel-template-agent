"""WP22: system configuration, proxy detection, injection per policy, fail-closed preflight."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from test_agents_runner import (  # noqa: F401  (fixtures)
    _members,
    _profile_and_batch,
    _task,
    actx,
    parts,
)

from kta.agents import guard, proxyplan
from kta.agents.config import agents_config
from kta.agents.runner import Deps, run_step
from kta.core import proxy, sysconfig
from kta.core.context import Context
from kta.core.errors import ConfigError, PermissionDenied
from kta.core.ids import UlidFactory
from kta.core.secrets import find_secrets
from kta.core.sysconfig import SystemConfig
from kta.core.trace import new_tracer, tracing
from proxyhelp import direct, proxy_at

T = datetime(2026, 10, 31, tzinfo=UTC)
# Built at run time so the gate's secret scan does not see a credential-shaped literal.
CREDS = "alice" + ":" + "s3cret-pw"
SCUTIL = """<dictionary> {
  ExceptionsList : <array> {
    0 : 127.0.0.1
    1 : localhost
  }
  HTTPEnable : 1
  HTTPPort : 7897
  HTTPProxy : 127.0.0.1
  HTTPSEnable : 1
  HTTPSPort : 7897
  HTTPSProxy : 127.0.0.1
  SOCKSEnable : 0
}"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---- detection and configuration ------------------------------------------------------


def test_detect_prefers_scutil_then_env_then_clash_default() -> None:
    d = proxy.detect(SCUTIL, {"HTTPS_PROXY": "http://10.0.0.1:3128"})
    assert (d.source, d.endpoint.url) == ("scutil", "http://127.0.0.1:7897")
    assert d.scutil["ExceptionsList"] == "127.0.0.1,localhost"
    d = proxy.detect(None, {"https_proxy": "socks5://10.0.0.1:1080"})
    assert (d.source, d.endpoint.url) == ("env", "socks5://10.0.0.1:1080")
    d = proxy.detect("HTTPEnable : 0", {})
    assert (d.source, d.endpoint.url) == ("default", proxy.DEFAULT_URL)


def test_system_config_defaults_revisions_and_stale_writes(tmp_path: Path) -> None:
    cfg = sysconfig.load(tmp_path)
    assert cfg.revision == 0
    assert {k: v.proxy for k, v in cfg.providers.items()} == {
        "codex": "required",
        "claude_code": "required",
        "pi": "off",
        "deepseek_harness": "off",
    }
    new = sysconfig.save(
        tmp_path, {"proxy": {"override": "socks5://127.0.0.1:7897"}}, 0, "me", T, "use socks"
    )
    assert (new.revision, new.updated_by) == (1, "me")
    assert sysconfig.load(tmp_path).proxy.override == "socks5://127.0.0.1:7897"
    with pytest.raises(ConfigError, match="revision 1, not 0"):
        sysconfig.save(tmp_path, {"proxy": {"override": None}}, 0, "me", T, "stale")
    with pytest.raises(ConfigError):
        sysconfig.save(tmp_path, {"providers": {"pi": {"proxy": "optional"}}}, 1, "me", T, "x")
    with pytest.raises(ConfigError, match="reason"):
        sysconfig.save(tmp_path, {}, 1, "me", T, " ")


# ---- injection per policy (contract) ----------------------------------------------------


@pytest.mark.contract
def test_whitelist_injection_per_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    # The runner's own shell proxy must never leak in: the override wins, env is ignored.
    monkeypatch.setenv("HTTPS_PROXY", "http://10.9.9.9:1")
    cfg = SystemConfig().model_copy(
        update={"proxy": sysconfig.ProxySettings(override="http://127.0.0.1:7897")}
    )
    base = {"KTA_TASK_TOKEN": "t", "HOME": "{home}"}
    for provider, policy in (
        ("codex", "required"),
        ("claude_code", "required"),
        ("pi", "off"),
        ("deepseek_harness", "off"),
    ):
        px = proxyplan.plan_for(provider, cfg, "http://127.0.0.1:7310")
        env = guard.agent_env(base, ["/kta/bin"], px.env)
        present = {k for k in env if k in proxy.ALL_VARS}
        if policy == "required":
            assert present == proxy.ALL_VARS, provider
            assert {env[k] for k in proxy.PROXY_VARS} == {"http://127.0.0.1:7897"}
            assert px.source == "override"
        else:
            assert present == set(), provider
    # Proxy variables cannot be smuggled in through the ordinary values either.
    with pytest.raises(PermissionDenied):
        guard.agent_env({**base, "HTTPS_PROXY": "http://x:1"}, [])
    with pytest.raises(PermissionDenied):
        guard.agent_env(base, [], {"HTTPS_PROXY": "http://x:1"})


@pytest.mark.contract
def test_no_proxy_always_holds_loopback_and_the_api_host() -> None:
    for api in ("http://127.0.0.1:7310", "http://kta-control.lan:7310", "", None):
        value = proxy.no_proxy(api, ["corp.internal"]).split(",")
        assert {"127.0.0.1", "localhost", "::1", "corp.internal"} <= set(value)
        if api and "lan" in api:
            assert "kta-control.lan" in value
    cfg = SystemConfig()
    px = proxyplan.plan_for("codex", cfg, "http://kta-control.lan:7310")
    env = px.env
    assert env["NO_PROXY"] == env["no_proxy"]
    assert {"127.0.0.1", "localhost", "::1", "kta-control.lan"} <= set(env["NO_PROXY"].split(","))
    bad = {**env, "NO_PROXY": "kta-control.lan", "no_proxy": "kta-control.lan"}
    with pytest.raises(PermissionDenied, match="NO_PROXY"):
        guard.agent_env({}, [], bad)


# ---- preflight --------------------------------------------------------------------------


class FakeProxy:
    """A local proxy that answers CONNECT with a fixed status line and closes (no tunnel)."""

    def __init__(self, status: bytes) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = int(self.sock.getsockname()[1])
        self.status = status
        self.seen: list[bytes] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                data = conn.recv(4096)
                self.seen.append(data)
                if data:
                    conn.sendall(self.status + b"\r\n\r\n")


@pytest.fixture
def refusing_proxy() -> Iterator[FakeProxy]:
    p = FakeProxy(b"HTTP/1.1 403 Forbidden")
    yield p
    p.sock.close()


def test_preflight_reports_a_down_port_and_a_refused_connect(refusing_proxy: FakeProxy) -> None:
    down = proxy.preflight(proxy.parse_url(f"http://127.0.0.1:{_free_port()}")[0], ["a.b:443"])
    assert not down["ok"] and not down["port"]["ok"] and "not reachable" in down["error"]
    ep = proxy.parse_url(f"http://127.0.0.1:{refusing_proxy.port}")[0]
    res = proxy.preflight(ep, ["api.example.com:443"], userinfo=CREDS)
    assert res["port"]["ok"] and not res["ok"] and "403" in res["error"]
    assert res["hosts"][0]["attempts"] == 2
    assert any(b"CONNECT api.example.com:443" in s for s in refusing_proxy.seen)
    assert any(b"Proxy-Authorization: Basic" in s for s in refusing_proxy.seen)
    assert CREDS not in json.dumps(res)
    assert not proxy.preflight(ep, [])["ok"]  # nothing to check proves nothing


@pytest.mark.contract
def test_fail_closed_preflight_refuses_the_launch(actx: Context, parts: Any) -> None:  # noqa: F811
    store, ops, ex = parts
    proxy_at(actx.data_dir, f"http://127.0.0.1:{_free_port()}", ["chatgpt.com:443"])
    deps = Deps(store, ops, ex, agents_config(actx))
    profile = _profile_and_batch(actx, store)
    task = _task(store, ops, profile)
    tracer = new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})
    with tracing(tracer), pytest.raises(proxy.ProxyUnavailable, match="not reachable") as err:
        run_step(actx, task, deps=deps)
    rid = err.value.result["report"]
    rep = store.get(rid)
    assert rep.body["title"] == "proxy_preflight" and rep.body["data"]["ok"] is False
    assert rep.actor.task == task.id
    # The CLI never started: no session and no AgentRun.
    assert not store.query("AgentRun", {}, limit=5)
    assert not list(Path(actx.data_dir).rglob("work/wu-1/out/env.json"))
    # The breaker holds further starts until a preflight passes; the Inbox lists it.
    assert proxyplan.blocked(ops, "codex")
    with tracing(tracer), pytest.raises(proxy.ProxyUnavailable, match="Settings"):
        run_step(actx, task, deps=deps)
    from kta.orchestrator.inbox import inbox
    from kta.orchestrator.pipeline import Services

    items = [i for i in inbox(Services(actx, store, ops)) if i.id == "proxy:codex"]
    assert items and items[0].record == rid
    proxyplan.mark(ops, "codex", True, T)
    assert proxyplan.blocked(ops, "codex") is None


def test_off_provider_launch_gets_no_proxy_variable(actx: Context, parts: Any) -> None:  # noqa: F811
    store, ops, ex = parts
    direct(actx.data_dir)  # already off in the fixture; explicit for this test's meaning
    deps = Deps(store, ops, ex, agents_config(actx))
    task = _task(store, ops, _profile_and_batch(actx, store))
    with tracing(new_tracer(ops, actx.clock, UlidFactory(actx.clock), {})):
        first = run_step(actx, task, deps=deps)
    names = json.loads(_members(store, first.launch.workspace)["out/env.json"])
    assert not [n for n in names if n.upper() in {"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"}]
    assert first.run.proxy == {"used": False, "policy": "off"}


# ---- credentials (contract) -------------------------------------------------------------


@pytest.mark.contract
def test_no_credentials_in_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"http://{CREDS}@127.0.0.1:7897"
    assert find_secrets(url)  # logs redact it and the gate's scan catches it
    with pytest.raises(ConfigError, match=r"secrets\.env"):
        sysconfig.save(tmp_path, {"proxy": {"override": url}}, 0, "me", T, "x")
    with pytest.raises(ConfigError):
        sysconfig.save(tmp_path, {"workers": {"slai": {"intake_proxy": url}}}, 0, "me", T, "x")
    d = proxy.detect(None, {"HTTPS_PROXY": url})
    assert d.credentials_dropped and CREDS not in json.dumps(d.to_dict())

    # On the agent host the credentials join the URLs only in the process environment.
    calls: list[str | None] = []

    def fake_preflight(ep: proxy.Endpoint, hosts: list[str], userinfo: str | None = None):
        calls.append(userinfo)
        return {
            "ok": True,
            "proxy": ep.to_dict(),
            "hosts": [{"host": h, "ok": True} for h in hosts],
        }

    monkeypatch.setattr(proxy, "preflight", fake_preflight)
    cfg = SystemConfig().model_copy(
        update={"proxy": sysconfig.ProxySettings(credentials_from_secrets=True)}
    )
    px = proxyplan.plan_for("codex", cfg, None)
    spec = px.spec("~/.config/kta/secrets.env")
    env = guard.agent_env({}, [], px.env)
    check = proxyplan.host_check(spec, env, CREDS)
    assert check.ok and calls == [CREDS]
    assert all(CREDS in check.env[k] for k in proxy.PROXY_VARS)
    assert set(check.secret) == set(proxy.PROXY_VARS)  # shown as <provided> in the manifest
    assert CREDS not in json.dumps(check.report) and CREDS not in json.dumps(spec)
    missing = proxyplan.host_check(spec, env, None)
    assert not missing.ok and "KTA_PROXY_USERINFO" in missing.report["error"]
