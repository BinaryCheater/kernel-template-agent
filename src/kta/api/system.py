"""Settings → System (WP22, D580–D590): the control plane's infrastructure settings.

Reading is open. Writing the system configuration needs the operator token (as the
controls do, WP21) and leaves a HumanAction ``set_system``. The proxy Test runs the
preflight from the control host and, when it passes, clears that provider's breaker; the
worker Test runs one trivial job on the named worker through the Executor. Provider rows
show the CLI version and whether credentials are present — never the credentials.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict, StringConstraints

from kta.agents import proxyplan
from kta.agents.config import agents_config
from kta.api import operate
from kta.api.common import Api, controls, iso
from kta.api.controls import TOKEN_HEADER, check_operator, token_file
from kta.core import proxy, sysconfig
from kta.core.envprobe import run_probe
from kta.core.errors import ConfigError, KtaError
from kta.core.jobs import Job
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.execution.executor import JobExecutor
from kta.execution.registry import latest_registration, worker_configs
from kta.orchestrator import release
from kta.store import code_state

Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
VERB = "set_system"
TEST_ACTOR = Actor(type="system", name="kta.api.system")
# Version command and credential location of each provider's CLI on the control host
# (read-only probes, D78). API providers carry a key in secrets.env, checked at launch.
CLI_VERSION: dict[str, list[str] | None] = {
    "codex": ["codex", "--version"],
    "claude_code": ["claude", "--version"],
    "pi": ["pi", "--version"],
    "deepseek_harness": None,
}
AUTH: dict[str, str] = {
    "codex": "subscription",
    "claude_code": "subscription",
    "pi": "api",
    "deepseek_harness": "api",
}
AUTH_ROOT = "~/.config/kta/auth"  # conventions §8
WORKER_PROBE = (
    "import json, os, platform, sys; "
    "print(json.dumps({'host': platform.node(), 'python': sys.version.split()[0], "
    "'cwd': os.getcwd()}))"
)


class SystemBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected: int  # the revision the author read
    changes: dict[str, Any]
    reason: Reason


class ProxyTestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str | None = None  # None: every `required` provider


def _auth(api: Api, provider: str) -> dict[str, Any]:
    kind = AUTH.get(provider, "api")
    if kind == "api":
        return {
            "kind": "api",
            "logged_in": None,
            "note": "API key in secrets.env, checked at launch",
        }
    where = _auth_dir(api, provider)
    path = where.expanduser()
    present = path.is_dir() and any(p.is_file() for p in path.iterdir())
    return {"kind": kind, "logged_in": present, "where": str(where)}


def _auth_dir(api: Api, provider: str) -> Path:
    """Codex's directory from ``[agents.hosts.<worker>]``; else conventions §8's path."""
    if provider == "codex":
        try:
            return Path(agents_config(api.ctx).host.auth_dir)
        except ConfigError:
            return Path(AUTH_ROOT, provider)  # no [agents] host section configured
    return Path(AUTH_ROOT, provider)


def providers(api: Api, cfg: sysconfig.SystemConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, pol in cfg.providers.items():
        argv = CLI_VERSION.get(name)
        version = run_probe(argv) if argv else None
        out.append(
            {
                "name": name,
                "proxy": pol.proxy,
                "check_hosts": pol.check_hosts,
                "cli": {"command": argv[0] if argv else None, "version": version},
                "auth": _auth(api, name),
                "status": proxyplan.status(api.ops, name),
            }
        )
    return out


def workers(api: Api, cfg: sysconfig.SystemConfig) -> list[dict[str, Any]]:
    mine = code_state().commit
    out: list[dict[str, Any]] = []
    for name, wc in worker_configs(api.ctx).items():
        reg = latest_registration(api.store, name)
        labels = dict(reg.worker.labels) if reg else {}
        out.append(
            {
                "name": name,
                "transport": wc.transport,
                "address": wc.ssh or "local",
                "root": wc.root,
                "pools": dict(wc.pools),
                "registered": reg.worker_id if reg else None,
                "commit": reg.worker.commit if reg else None,
                "control_commit": mine,
                "commit_match": bool(reg and reg.worker.commit == mine),
                "gpu": labels.get("gpu_name") or labels.get("gpu"),
                "healthy": api.ops.worker_health(name),
                "intake_proxy": cfg.workers.get(name, sysconfig.WorkerProxy()).intake_proxy,
            }
        )
    return out


def _proxy_view(cfg: sysconfig.SystemConfig) -> dict[str, Any]:
    detected = proxy.detect_system()
    view: dict[str, Any] = {"detected": detected.to_dict(), "override": cfg.proxy.override}
    try:
        view["in_force"] = proxy.resolve(cfg.proxy.override).to_dict()
    except ConfigError as exc:
        view["in_force"] = None
        view["error"] = str(exc)
    view["no_proxy"] = cfg.proxy.no_proxy_extra
    view["credentials_from_secrets"] = cfg.proxy.credentials_from_secrets
    return view


def _audit(api: Api, limit: int = 20) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in api.store.query("HumanAction", order="-created_at", limit=500):
        if str(rec.body["action"]).startswith(VERB):
            parsed = operate.parse_action(rec.body["action"])
            out.append(
                {"at": iso(rec.created_at), "by": rec.actor.name, "reason": rec.body["reason"]}
                | parsed
            )
            if len(out) >= limit:
                break
    return out


def _agents(api: Api) -> dict[str, Any]:
    try:
        ac = agents_config(api.ctx)
    except ConfigError as exc:
        return {"error": str(exc)}
    return {"worker": ac.worker, "api_url": ac.api_url, "batch": ac.batch}


def system_view(api: Api) -> dict[str, Any]:
    cfg = sysconfig.load(api.data_dir)
    stage = release.stage_of(controls(api))
    return {
        "config": cfg.to_dict(),
        "proxy": _proxy_view(cfg),
        "providers": providers(api, cfg),
        "workers": workers(api, cfg),
        "agents": _agents(api),
        "control": {
            "host": platform.node(),
            "env": api.env,
            "data_dir": str(api.data_dir),
            "config_file": str(sysconfig.path(api.data_dir)),
            "commit": code_state().commit,
        },
        "release": {"id": stage.id, "name": stage.name, "page": "/release"},
        "audit": _audit(api),
        "write": {"header": TOKEN_HEADER, "token_configured": token_file(api).is_file()},
    }


def write_system(api: Api, b: SystemBody) -> dict[str, Any]:
    human = operate.operator(api)
    cfg = sysconfig.save(api.data_dir, b.changes, b.expected, human.name, api.now(), b.reason)
    params = {"revision": cfg.revision, "changes": b.changes}
    action = operate.write_action(api, VERB, b.reason, None, params)
    return {"revision": cfg.revision, "action": action}


def proxy_test(api: Api, b: ProxyTestBody) -> dict[str, Any]:
    """The preflight from the control host, one Report per provider (evidence); a pass
    clears that provider's breaker (D585). A failed Test does not trip it: launches do."""
    cfg = sysconfig.load(api.data_dir)
    required = [n for n, p in cfg.providers.items() if p.proxy == "required"]
    names = [b.provider] if b.provider else required
    found = proxy.resolve(cfg.proxy.override)
    out: dict[str, Any] = {"proxy": found.to_dict(), "from": "control host", "results": {}}
    for name in names:
        res = proxy.preflight(found.endpoint, cfg.provider(name).check_hosts)
        data = {**res, "from": "control host", "source": found.source, "trigger": "settings test"}
        body = Report(
            title=proxyplan.REPORT_TITLE,
            subject=name,
            summary=f"{name}: proxy preflight {'passed' if res['ok'] else 'failed'} "
            f"({found.endpoint.url}, {found.source})",
            data=data,
        )
        rid = api.store.put(Record.new("Report", body, TEST_ACTOR, api.now()))
        out["results"][name] = {**res, "report": rid}
        if res["ok"] and name in required:
            proxyplan.mark(api.ops, name, True, api.now(), cleared_by="settings test", report=rid)
    out["ok"] = all(r["ok"] for r in out["results"].values())
    return out


def worker_test(api: Api, name: str) -> dict[str, Any]:
    """One trivial job on ``name`` through the Executor (not usable: no job, say why)."""
    if name not in worker_configs(api.ctx):
        raise ConfigError(f"no [workers.{name}] in the configuration")
    ex = JobExecutor(api.ctx, api.store, api.ops, code_state().commit)
    if name not in [w.name for w in ex.workers]:
        reg = latest_registration(api.store, name)
        why = (
            "never registered (kta worker register)"
            if reg is None
            else f"registered with commit {reg.worker.commit[:12]}, control plane runs "
            f"{code_state().commit[:12]} (scripts/worker-setup deploys the current commit)"
        )
        return {"ok": False, "worker": name, "ran": False, "error": f"not usable: {why}"}
    job = Job(kind="shell", argv=["{python}", "-c", WORKER_PROBE], requires={"worker": name})
    try:
        res = ex.run(job)
    except KtaError as exc:
        return {"ok": False, "worker": name, "ran": False, "error": str(exc)}
    return {
        "ok": res.ok,
        "worker": name,
        "ran": True,
        "job": res.job_id,
        "exit_code": res.exit_code,
        "error_kind": res.error_kind,
        "wall_s": res.wall_s,
        "queue_s": res.queue_s,
        "stdout": ex.read_blob(res.stdout_blob).decode("utf-8", "replace")[-2000:],
    }


def router(api: Api) -> APIRouter:
    r = APIRouter()

    @r.get("/system")
    def get_system() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return system_view(api)

    @r.post("/system")
    def post_system(  # pyright: ignore[reportUnusedFunction]
        b: SystemBody, token: str | None = Header(None, alias=TOKEN_HEADER)
    ) -> dict[str, Any]:
        check_operator(api, token)
        return write_system(api, b)

    @r.post("/system/proxy/test")
    def post_proxy_test(b: ProxyTestBody) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return proxy_test(api, b)

    @r.post("/system/workers/{name}/test")
    def post_worker_test(name: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return worker_test(api, name)

    return r
