"""The summarize process (docs/design/11 §1): one improvement Report per batch.

A batch closes, and ``improve.summarize`` runs over the window since the previous one,
when (D406):

- an experiment result Report (``title = experiment``) appeared since the last summary:
  the batch is named after that Protocol;
- you asked for one (``kta improve run``, pointer ``summarize:request``);
- ``[summarize] every_s`` (default 86400) passed since the last summary.

The end of the last window is the pointer ``summarize:last``.
"""

from __future__ import annotations

from datetime import datetime

from kta.core.errors import ConfigError
from kta.orchestrator import improve
from kta.orchestrator.proc_common import ProcReport, action, due, get_state, put_state
from kta.orchestrator.runner import Runner
from kta.store.sqlite_ops import SqliteOps

LAST = "summarize:last"
REQUEST = "summarize:request"
DEFAULT_EVERY_S = 86400.0


def request(ops: SqliteOps, label: str, now: datetime) -> None:
    """Ask the summarize process for a batch now (``kta improve run``)."""
    put_state(ops, REQUEST, {"label": label, "at": now.isoformat()})


def _trigger(runner: Runner, since: datetime | None) -> str | None:
    svc, now = runner.svc, runner.ctx.clock.now()
    asked = get_state(svc.ops, REQUEST)
    if asked.get("label"):
        return str(asked["label"])
    for rec in svc.store.query("Report", {"title": "experiment"}, order="-created_at", limit=5):
        if since is None or rec.created_at > since:
            proto = svc.store.get(str(rec.refs["protocol"])).body
            return f"experiment {proto['name']} v{proto['version']}"
    every = float(runner.ctx.config.section("summarize").get("every_s", DEFAULT_EVERY_S))
    if since is not None and due(svc.ops, LAST, now, every):
        return f"period to {now.date().isoformat()}"
    return None


def summarize_tick(runner: Runner) -> ProcReport:
    rep = ProcReport("summarize")
    svc, now = runner.svc, runner.ctx.clock.now()
    raw = svc.ops.get_pointer(LAST)
    since = datetime.fromisoformat(raw) if raw else None
    if raw is None:
        # The first tick only opens the first window, unless a batch was asked for.
        label = _trigger(runner, None) if get_state(svc.ops, REQUEST) else None
        if label is None:
            if not svc.ops.set_pointer(LAST, None, now.isoformat()):
                raise ConfigError(f"{LAST} changed concurrently")
            return rep
    else:
        label = _trigger(runner, since)
    if label is None:
        return rep
    with action(runner, "summarize", f"improvement {label}", label) as out:
        rid = improve.summarize(runner, label, since, now)
        out["report"] = rid
    rep.add("improvement", rid)
    if not svc.ops.set_pointer(LAST, raw, now.isoformat()):
        raise ConfigError(f"{LAST} changed concurrently")
    if get_state(svc.ops, REQUEST):
        put_state(svc.ops, REQUEST, {})
    return rep
