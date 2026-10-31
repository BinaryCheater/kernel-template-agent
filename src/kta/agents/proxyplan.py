"""The proxy of one agent step (WP22): policy, injected variables, preflight outcome.

The runner builds a :class:`ProxyPlan` from the system configuration (never from its own
shell environment), injects its variables through ``guard.agent_env`` and hands the
launcher the preflight to run on the agent host, where the CLI will connect from. A
failed preflight trips a breaker (ops pointer ``proxy:status:<provider>``): no further
task of that provider is leased until a later preflight passes (the settings page's
Test, or the next launch after the user cleared it). The Inbox lists the tripped breaker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from kta.core import proxy
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.core.sysconfig import SystemConfig
from kta.core.sysconfig import load as load_system
from kta.store.base import Ops, Store

ACTOR = Actor(type="system", name="kta.agents.proxyplan")
REPORT_TITLE = "proxy_preflight"


@dataclass(frozen=True)
class ProxyPlan:
    provider: str
    policy: str  # required | off
    endpoint: proxy.Endpoint | None = None
    source: str | None = None
    detail: str = ""
    no_proxy: str = ""
    check_hosts: list[str] = field(default_factory=lambda: list[str]())
    credentials: bool = False  # KTA_PROXY_USERINFO is added on the agent host

    @property
    def env(self) -> dict[str, str]:
        """Variables injected into the CLI's environment (no credentials)."""
        if self.policy != "required" or self.endpoint is None:
            return {}
        return proxy.proxy_env(self.endpoint, self.no_proxy)

    def spec(self, secrets_file: str) -> dict[str, Any]:
        """What the launcher needs to run the preflight and add credentials."""
        out: dict[str, Any] = {"provider": self.provider, "policy": self.policy}
        if self.endpoint is not None:
            out.update(
                url=self.endpoint.url,
                source=self.source,
                check_hosts=list(self.check_hosts),
                no_proxy=self.no_proxy,
                credentials=self.credentials,
                secrets_file=secrets_file,
            )
        return out


def plan_for(provider: str, cfg: SystemConfig, api_url: str | None) -> ProxyPlan:
    pol = cfg.provider(provider)
    if pol.proxy == "off":
        return ProxyPlan(provider, "off", detail="policy off: no proxy variable is passed")
    found = proxy.resolve(cfg.proxy.override)
    return ProxyPlan(
        provider,
        "required",
        endpoint=found.endpoint,
        source=found.source,
        detail=found.detail,
        no_proxy=proxy.no_proxy(api_url, cfg.proxy.no_proxy_extra),
        check_hosts=list(pol.check_hosts),
        credentials=cfg.proxy.credentials_from_secrets,
    )


def for_step(data_dir: Path, ops: Ops, provider: str, api_url: str | None) -> ProxyPlan:
    """The step's proxy from the system configuration; a tripped breaker refuses (D585)."""
    px = plan_for(provider, load_system(data_dir), api_url)
    why = blocked(ops, provider) if px.policy == "required" else None
    if why:
        raise proxy.ProxyUnavailable(f"{why}; run Test in Settings → System once it is up")
    return px


def outcome(
    store: Store,
    ops: Ops,
    px: ProxyPlan,
    exit: str,
    report: dict[str, Any],
    at: datetime,
    task: str,
) -> None:
    """Fail closed on a refused preflight: Report, breaker, ProxyUnavailable (D584)."""
    rep: dict[str, Any] = report.get("proxy") or {}
    if exit == "refused" and rep.get("error"):
        rid = record_failure(store, ops, px.provider, rep, at, task)
        msg = f"{px.provider}: proxy preflight failed (Report {rid[:12]}): {rep['error']}"
        raise proxy.ProxyUnavailable(msg, {**rep, "report": rid})
    if px.policy == "required" and rep.get("used"):
        mark(ops, px.provider, True, at, task=task)


# ---- breaker -----------------------------------------------------------------------


def status(ops: Ops, provider: str) -> dict[str, Any] | None:
    raw = ops.get_pointer(proxy.STATUS_POINTER + provider)
    return json.loads(raw) if raw else None


def mark(ops: Ops, provider: str, ok: bool, at: datetime, **detail: Any) -> None:
    name = proxy.STATUS_POINTER + provider
    value = json.dumps({"ok": ok, "at": at.isoformat(), **detail}, sort_keys=True)
    for _ in range(5):  # compare-and-set: another writer may have moved it meanwhile
        if ops.set_pointer(name, ops.get_pointer(name), value):
            return


def blocked(ops: Ops, provider: str) -> str | None:
    """Why tasks of ``provider`` must not be leased now, or None."""
    st = status(ops, provider)
    if st is None or st.get("ok"):
        return None
    return f"proxy preflight for {provider} failed at {st.get('at')}: {st.get('error')}"


def record_failure(
    store: Store, ops: Ops, provider: str, result: dict[str, Any], at: datetime, task: str | None
) -> str:
    """A Report with the failed preflight (evidence) and the tripped breaker (Inbox)."""
    body = Report(
        title=REPORT_TITLE,
        subject=provider,
        summary=f"{provider}: proxy preflight failed; the CLI was not started",
        data={"ok": False, **result},
    )
    actor = Actor(type=ACTOR.type, name=ACTOR.name, task=task)
    rid = store.put(Record.new("Report", body, actor, at))
    mark(ops, provider, False, at, error=str(result.get("error", "")), report=rid, task=task)
    return rid


# ---- on the agent host (the launcher) --------------------------------------------------


@dataclass(frozen=True)
class HostCheck:
    ok: bool
    report: dict[str, Any]  # AgentRun.proxy: no credentials
    env: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # replaces values
    secret: list[str] = field(default_factory=lambda: list[str]())  # hidden in the manifest


def host_check(spec: dict[str, Any] | None, env: dict[str, str], userinfo: str | None) -> HostCheck:
    """Run where the CLI runs, just before it starts: fail closed for `required`.

    ``off`` (or no proxy section): no proxy variable may be present at all. ``required``:
    every variable must be present and the preflight must pass; ``userinfo`` (read from
    the host's secrets.env when the configuration asks for it) is added to the URLs here
    and only here.
    """
    leaked = sorted(k for k in env if k in proxy.ALL_VARS)
    if not spec or spec.get("policy") != "required":
        report: dict[str, Any] = {"used": False, "policy": (spec or {}).get("policy", "off")}
        if leaked:
            return HostCheck(False, {**report, "error": f"proxy variables present: {leaked}"})
        return HostCheck(True, report)
    ep, _ = proxy.parse_url(str(spec["url"]))
    report = {
        "used": True,
        "policy": "required",
        "scheme": ep.scheme,
        "host": ep.host,
        "port": ep.port,
        "source": spec.get("source"),
        "no_proxy": spec.get("no_proxy"),
        "credentials": bool(spec.get("credentials")),
    }
    missing = sorted(proxy.ALL_VARS - set(leaked))
    if missing:
        return HostCheck(False, {**report, "error": f"proxy variables missing: {missing}"})
    if spec.get("credentials") and not userinfo:
        err = f"{proxy.USERINFO_SECRET} is not set in this host's secrets.env"
        return HostCheck(False, {**report, "error": err})
    result = proxy.preflight(ep, list(spec.get("check_hosts") or []), userinfo)
    report["preflight"] = result
    if not result["ok"]:
        return HostCheck(False, {**report, "error": result.get("error")})
    if not userinfo:
        return HostCheck(True, report)
    with_creds = {k: ep.with_userinfo(userinfo) for k in proxy.PROXY_VARS}
    return HostCheck(True, report, with_creds, list(proxy.PROXY_VARS))
